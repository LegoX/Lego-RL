# ============================================================================
# Mirror the official SWE-bench per-instance images (Verified / Multilingual:
# swebench/sweb.eval.x86_64.*; Pro: one tag per instance under jefzda/sweap-images)
# from Docker Hub into a local registry, so sandbox pods with no Docker Hub egress
# can still pull them. No docker build and no local pull: a
# registry-to-registry copy of the linux/amd64 manifest via
# `docker buildx imagetools create` (~20 s/image instead of ~220 s, and nothing
# is staged on the local disk).
#
# Two registry endpoints over the SAME storage, which is the usual layout:
#   CHECK_REG  read-only view the cluster nodes pull from   (default 127.0.0.1:5001)
#   WRITER     writable endpoint this script pushes to      (default 127.0.0.1:5002)
# If they are the same endpoint, set both to it. REG_BIN + RW_CONF are optional:
# when set, a writer that is not listening is started from them.
#
# Images keep their Docker Hub names, so the HARBOR_NYDUS_MIRROR rewrite in
# src/harbor_patch/environments/kubernetes/kubernetes.py resolves to this copy:
#   docker.io/swebench/sweb.eval.x86_64.<id>:latest
#     -> $HARBOR_NYDUS_MIRROR/swebench/sweb.eval.x86_64.<id>:latest
#   docker.io/jefzda/sweap-images:<tag>
#     -> $HARBOR_NYDUS_MIRROR/jefzda/sweap-images:<tag>
#
# Docker Hub rate limits (anonymous: 100 pulls/hour per egress IP) are guarded in
# three layers, because this egress is usually shared with other jobs:
#   1. a free ratelimit probe before each pull: pause while the sampled bucket has
#      fewer than RATE_RESERVE pulls left;
#   2. a local ledger capping pulls at HOURLY_CAP across concurrent runs;
#   3. a real 429 backs that worker off for RATE_BACKOFF_SEC.
# With a paid `docker login` on the host the probe sees no ratelimit headers and
# never pauses; pass HOURLY_CAP=100000 to lift the ledger too.
# PARALLEL (default 12) bounds egress bandwidth (~9 MB/s per stream measured).
#
# Usage:
#   LIST=/path/to/data/harbor_swebench_multilingual_300_images.txt \
#     LOGDIR=/path/to/data/swebench-hub-mirror utils/populate_swebench_hub.sh
#   ONLY="img1 img2" LIST=... LOGDIR=...        # smoke test
#   PARALLEL=12 HOURLY_CAP=300 RATE_RESERVE=15 ...
#
# The image list is one docker.io reference per line; for a converted dataset:
#   sed -n 's/^docker_image = "\(.*\)"/\1/p' <dataset>/*/task.toml | sort -u > images.txt
#
# Re-runs skip images already present in CHECK_REG, so it is resumable.
# Stop: pkill -f populate_swebench_hub.sh; pkill -f 'xargs -P'
# Output: $LOGDIR/<list>.status.tsv (ts  image  OK|SKIP|FAIL_COPY  seconds),
#         $LOGDIR/<list>.log, per-image logs in $LOGDIR/img/.
# Check how many of a list are in place:
#   awk -F"\t" '$3=="OK"||$3=="SKIP"' $LOGDIR/<list>.status.tsv | wc -l
# ============================================================================
set -uo pipefail

LIST="${LIST:?set LIST=<file with one docker.io image ref per line>}"
WRITER="${WRITER:-127.0.0.1:5002}"
CHECK_REG="${CHECK_REG:-127.0.0.1:5001}"
LOGDIR="${LOGDIR:?set LOGDIR=<dir for status/logs>}"
PARALLEL="${PARALLEL:-12}"
HOURLY_CAP="${HOURLY_CAP:-300}"
RATE_RESERVE="${RATE_RESERVE:-15}"
PULL_RETRIES="${PULL_RETRIES:-3}"
PUSH_RETRIES="${PUSH_RETRIES:-4}"
RATE_BACKOFF_SEC="${RATE_BACKOFF_SEC:-900}"
REG_BIN="${REG_BIN:-}"        # optional: registry binary, to start the writer if it is down
RW_CONF="${RW_CONF:-}"        # optional: its config file (same storage as CHECK_REG)
RW_OUT="${RW_OUT:-$LOGDIR/writer-registry.out}"

name="$(basename "$LIST" .txt)"
STATUS="$LOGDIR/$name.status.tsv"
LOG="$LOGDIR/$name.log"
PULL_LEDGER="$LOGDIR/.pull_ledger"      # one epoch timestamp per Docker Hub pull (shared across runs)
mkdir -p "$LOGDIR/img"
touch "$STATUS" "$PULL_LEDGER"
export WRITER CHECK_REG LOGDIR STATUS LOG PULL_LEDGER HOURLY_CAP RATE_RESERVE PULL_RETRIES PUSH_RETRIES RATE_BACKOFF_SEC

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# ---- writer registry -------------------------------------------------------
if ! curl -fsS -o /dev/null --max-time 10 "http://$WRITER/v2/" 2>/dev/null; then
    [ -n "$REG_BIN" ] && [ -n "$RW_CONF" ] || {
        log "FATAL writer registry $WRITER is not listening, and REG_BIN/RW_CONF are unset"
        exit 2
    }
    log "writer registry $WRITER not listening; starting: $REG_BIN serve $RW_CONF"
    nohup "$REG_BIN" serve "$RW_CONF" >> "$RW_OUT" 2>&1 &
    for _ in $(seq 1 30); do
        curl -fsS -o /dev/null --max-time 5 "http://$WRITER/v2/" 2>/dev/null && break
        sleep 1
    done
    curl -fsS -o /dev/null --max-time 5 "http://$WRITER/v2/" || { log "FATAL writer registry did not come up"; exit 2; }
fi
curl -fsS -o /dev/null --max-time 10 "http://$CHECK_REG/v2/" || { log "FATAL read-only registry $CHECK_REG unreachable"; exit 2; }

ACCEPT='Accept: application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json'
export ACCEPT

# ---- hourly Docker Hub budget ---------------------------------------------
hub_remaining() {
    # Docker Hub's documented free probe; returns the remaining pulls of the bucket the
    # NAT happens to route this request through (empty on any failure -> caller ignores).
    # Uses the `docker login` credentials from ~/.docker/config.json when present (buildx
    # pulls with the same ones): a paid account gets NO ratelimit headers -> empty -> never pause.
    local tok basic hdr
    basic=$(python3 -c "import json,os; d=json.load(open(os.path.expanduser('~/.docker/config.json'))); print(d.get('auths',{}).get('https://index.docker.io/v1/',{}).get('auth',''))" 2>/dev/null)
    hdr=(); [ -n "$basic" ] && hdr=(-H "Authorization: Basic $basic")
    tok=$(curl -s --max-time 15 "${hdr[@]}" "https://auth.docker.io/token?service=registry.docker.io&scope=repository:ratelimitpreview/test:pull" \
        | python3 -c "import json,sys; print(json.load(sys.stdin).get('token',''))" 2>/dev/null)
    [ -n "$tok" ] || return 0
    curl -sI --max-time 15 -H "Authorization: Bearer $tok" https://registry-1.docker.io/v2/ratelimitpreview/test/manifests/latest \
        | awk 'tolower($1)=="ratelimit-remaining:" {split($2,a,";"); print a[1]}'
}
export -f hub_remaining

wait_for_budget() {
    while true; do
        local now used rem
        now=$(date +%s)
        used=$(flock "$PULL_LEDGER.lock" awk -v t="$((now - 3600))" '$1 > t' "$PULL_LEDGER" | wc -l)
        if [ "$used" -lt "$HOURLY_CAP" ]; then
            rem=$(hub_remaining)
            if [ -n "$rem" ] && [ "$rem" -lt "$RATE_RESERVE" ]; then
                echo "$(date '+%F %T') hub bucket low (remaining=$rem < $RATE_RESERVE); pausing 60s" >> "$LOG"
                sleep 60; continue
            fi
            flock "$PULL_LEDGER.lock" bash -c "echo $now >> '$PULL_LEDGER'"
            return 0
        fi
        sleep 60
    done
}
export -f wait_for_budget

do_one() {
    local src="$1" t0 repo tag ilog attempt out raw src_ref
    t0=$(date +%s)
    src="${src#docker.io/}"
    repo="${src%:*}"; tag="${src##*:}"; [ "$tag" = "$src" ] && tag=latest
    ilog="$LOGDIR/img/$(echo "$repo:$tag" | tr '/:' '__').log"

    if curl -fsS -o /dev/null --max-time 60 -H "$ACCEPT" "http://$CHECK_REG/v2/$repo/manifests/$tag" 2>/dev/null; then
        printf '%s\t%s\t%s\t%s\n' "$(date +%s)" "$repo:$tag" SKIP 0 >> "$STATUS"
        echo "$(date '+%F %T') SKIP $repo:$tag" | tee -a "$LOG"; return 0
    fi

    # Registry-to-registry copy with buildx (no local pull/extract/push: ~20 s instead of
    # ~220 s per image, nothing staged on /root/storage, no gzip load). For a multi-arch
    # index only the linux/amd64 manifest is copied (attestations skipped), because
    # dockerd's own `docker push` of a half-present index fails with
    # "NotFound: content digest ... not all ... available locally".
    attempt=0; out=1
    while :; do
        attempt=$((attempt + 1))
        wait_for_budget
        # Match the rate-limit string in THIS attempt's output only. Appending to $ilog and
        # grepping the whole file makes one historical 429 match forever: the worker then sleeps
        # RATE_BACKOFF_SEC, retries, re-matches the old line, and loops without ever recording a
        # FAIL_COPY -- the run looks alive with zero throughput.
        alog=$(mktemp)
        raw=$(docker buildx imagetools inspect --raw "docker.io/$repo:$tag" 2>"$alog")
        cat "$alog" >> "$ilog"
        if [ -z "$raw" ]; then
            if grep -qi "toomanyrequests\|rate limit" "$alog"; then
                rm -f "$alog"
                echo "$(date '+%F %T') 429 on $repo:$tag; backing off ${RATE_BACKOFF_SEC}s" | tee -a "$LOG"
                sleep "$RATE_BACKOFF_SEC"; continue
            fi
            rm -f "$alog"
            [ "$attempt" -ge "$PULL_RETRIES" ] && break
            sleep $((30 * attempt)); continue
        fi
        src_ref=$(printf '%s' "$raw" | python3 -c '
import json, sys
repo = sys.argv[1]
d = json.load(sys.stdin)
ms = d.get("manifests")
if not ms:
    print(""); sys.exit()          # single-platform manifest: copy the tag as-is
for m in ms:
    pl = m.get("platform") or {}
    if pl.get("os") == "linux" and pl.get("architecture") == "amd64" and not (m.get("annotations") or {}).get("vnd.docker.reference.type"):
        print(repo + "@" + m["digest"]); sys.exit()
print("NONE")' "$repo")
        if [ "$src_ref" = "NONE" ]; then
            echo "no linux/amd64 manifest in index" >> "$ilog"; break
        fi
        : > "$alog"
        if docker buildx imagetools create --tag "$WRITER/$repo:$tag" "docker.io/${src_ref:-$repo:$tag}" > "$alog" 2>&1; then
            cat "$alog" >> "$ilog"; rm -f "$alog"; out=0; break
        fi
        cat "$alog" >> "$ilog"
        if grep -qi "toomanyrequests\|rate limit" "$alog"; then
            rm -f "$alog"
            echo "$(date '+%F %T') 429 on $repo:$tag; backing off ${RATE_BACKOFF_SEC}s" | tee -a "$LOG"
            sleep "$RATE_BACKOFF_SEC"; continue
        fi
        rm -f "$alog"
        [ "$attempt" -ge "$PUSH_RETRIES" ] && break
        sleep $((20 * attempt))
    done
    if [ "$out" -eq 0 ]; then
        printf '%s\t%s\t%s\t%s\n' "$(date +%s)" "$repo:$tag" OK "$(( $(date +%s) - t0 ))" >> "$STATUS"
        echo "$(date '+%F %T') OK $repo:$tag ($(( $(date +%s) - t0 ))s)" | tee -a "$LOG"
    else
        printf '%s\t%s\t%s\t%s\n' "$(date +%s)" "$repo:$tag" FAIL_COPY "$(( $(date +%s) - t0 ))" >> "$STATUS"
        echo "$(date '+%F %T') FAIL_COPY $repo:$tag (see $ilog)" | tee -a "$LOG"; return 1
    fi
}
export -f do_one

# ---- run -------------------------------------------------------------------
if [ -n "${ONLY:-}" ]; then
    printf '%s\n' $ONLY
else
    grep -v '^\s*#' "$LIST" | grep -v '^\s*$'
fi > "$LOGDIR/$name.queue"
total=$(wc -l < "$LOGDIR/$name.queue")
log "START list=$LIST images=$total parallel=$PARALLEL hourly_cap=$HOURLY_CAP writer=$WRITER"
xargs -a "$LOGDIR/$name.queue" -P "$PARALLEL" -I{} bash -c 'do_one "$1"' _ {}
ok=$(awk -F'\t' '$3=="OK"' "$STATUS" | wc -l); skip=$(awk -F'\t' '$3=="SKIP"' "$STATUS" | wc -l)
fail=$(awk -F'\t' '$3 ~ /^FAIL/' "$STATUS" | wc -l)
log "DONE ok=$ok skip=$skip fail=$fail (cumulative in $STATUS)"
