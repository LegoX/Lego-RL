#!/usr/bin/env bash
# Build one offline Go module proxy per SWE-bench Pro **go** task.
#
#   utils/build_pro_goproxy.sh <dataset_dir> <out_dir> [instance_id ...]
#
# With no instance_id it picks every go task whose GOLD PATCH touches go.mod / go.sum -- 40 of
# the 731, verified equal to the hand-built list. The other ~240 go tasks need nothing.
#
# WHY. ~40 of the 731 Pro tasks are dependency-bump issues: the fix edits go.mod. The agent
# sandbox has no public egress, so `go build` / `go test` cannot fetch the new module versions
# and the task is unsolvable no matter what the model writes. Giving the whole sandbox egress
# would also hand the agent the upstream repo (and its fix) — so instead each such task gets a
# file:// module proxy holding exactly the modules its GOLD go.mod resolves to:
#
#   <out_dir>/<instance_id>/cache/download     a GOMODCACHE download tree
#   <out_dir>/<instance_id>/files.lst          its file list (cheap completeness check)
#
# Mount <out_dir> read-only at /opt/goproxy and convert the dataset with
# `--goproxy-mount /opt/goproxy`, which writes into each go task's task.toml:
#   [environment.env]
#   GOPROXY = "file:///opt/goproxy/<instance_id>/cache/download"
#   GOSUMDB = "off"
# harbor prefixes [environment.env] onto every exec, so agent and verifier both see it.
#
# KNOWN LIMIT: the cache is resolved from the gold go.mod. A correct fix that bumps a module to
# a DIFFERENT version than gold still cannot download it. That is a deliberate trade — these
# tasks are otherwise not solvable at all — but it does mean a go-subset score is a slight
# under-count, not a clean measurement.
#
# Cost: ~700 MB and a few minutes per instance (one `go mod download all` over the module
# graph), so ~28 GB for the whole go subset. Idempotent: an instance with a populated
# cache/download is skipped, so an interrupted run resumes.
#
# Needs: docker with egress on THIS host, and the task images pullable.
#
# VERIFY before spending a cluster on it -- oracle applies the gold patch (the one that bumps
# go.mod), so reward 1 means the proxy resolved:
#   GRADING_TOOLCHAIN=<toolchain> bash utils/oracle_smoke_task.sh <dataset_dir>/<a go instance> oracle
# A `dial tcp: lookup proxy.golang.org` in the verifier log means GOPROXY did not take effect
# (task.toml has no [environment.env] block, or <out_dir> is not mounted at /opt/goproxy).
set -uo pipefail

DATASET_DIR="${1:?usage: build_pro_goproxy.sh <dataset_dir> <out_dir> [instance_id ...]}"
OUT_DIR="${2:?usage: build_pro_goproxy.sh <dataset_dir> <out_dir> [instance_id ...]}"
shift 2 || true
DATASET_DIR="$(cd "$DATASET_DIR" && pwd)"
mkdir -p "$OUT_DIR"; OUT_DIR="$(cd "$OUT_DIR" && pwd)"

# GOFLAGS=-mod=mod: several of these repos vendor or pin, and we want the graph resolved.
GO_DOWNLOAD_CMD="${GO_DOWNLOAD_CMD:-go mod download all}"

if [ "$#" -gt 0 ]; then
    instances=("$@")
else
    # Only the go tasks whose GOLD PATCH edits go.mod / go.sum — 40 of the 731, verified equal
    # to the hand-built list. The other ~240 go tasks resolve every import from the module cache
    # the image already ships, so a proxy for them would cost ~170 GB and buy nothing.
    mapfile -t instances < <(
        for t in "$DATASET_DIR"/*/; do
            grep -q '"go"\]' "$t/task.toml" 2>/dev/null || continue
            grep -qE '^(\+\+\+|---) .*/go\.(mod|sum)$' "$t/solution/solve.sh" 2>/dev/null || continue
            basename "$t"
        done | sort
    )
fi
echo "[goproxy] ${#instances[@]} go instances from $DATASET_DIR -> $OUT_DIR"

ok=0; skip=0; fail=0
for iid in "${instances[@]}"; do
    task="$DATASET_DIR/$iid"
    dest="$OUT_DIR/$iid"
    if [ -d "$dest/cache/download" ] && [ -s "$dest/files.lst" ]; then
        echo "[goproxy] SKIP $iid (already built)"; skip=$((skip + 1)); continue
    fi
    [ -f "$task/task.toml" ] || { echo "[goproxy] FAIL $iid: no task.toml"; fail=$((fail + 1)); continue; }
    image="$(sed -n 's/^docker_image = "\(.*\)"/\1/p' "$task/task.toml")"
    # The verifier checks out the gold TEST files too, and those can pull their own modules.
    gold_checkout="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['gold_checkout_cmd'])" \
        "$task/tests/expected.json" 2>/dev/null || true)"
    [ -n "$image" ] || { echo "[goproxy] FAIL $iid: no docker_image"; fail=$((fail + 1)); continue; }

    rm -rf "$dest"; mkdir -p "$dest"
    echo "[goproxy] building $iid ($image)"
    docker run --rm --entrypoint bash \
        -v "$task/solution:/solution:ro" \
        -v "$dest:/goproxy-out" \
        -e GOFLAGS=-mod=mod \
        -e GOMODCACHE=/goproxy-out \
        -e GOPROXY="${UPSTREAM_GOPROXY:-https://proxy.golang.org,direct}" \
        -e GOLD_CHECKOUT="$gold_checkout" \
        "$image" -c '
set -x
cd /app || exit 1
# gold state: the fix (which is what bumps go.mod) plus the gold test files
bash /solution/solve.sh || echo "WARN: gold patch did not apply cleanly"
[ -n "$GOLD_CHECKOUT" ] && { eval "$GOLD_CHECKOUT" || echo "WARN: gold test checkout failed"; }
command -v go >/dev/null 2>&1 || { echo "ERROR: no go in image"; exit 2; }
go version
'"$GO_DOWNLOAD_CMD"' || exit 3
# test dependencies are not always in the module graph until something imports them
go list -deps -test ./... > /dev/null 2>&1 || true
(cd /goproxy-out/cache/download && find . -type f) > /goproxy-out/files.lst 2>/dev/null || true
chmod -R a+rX /goproxy-out 2>/dev/null || true
' > "$dest/build.log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ] && [ -s "$dest/files.lst" ]; then
        echo "[goproxy] OK   $iid ($(wc -l < "$dest/files.lst") files, $(du -sh "$dest" 2>/dev/null | cut -f1))"
        ok=$((ok + 1))
    else
        echo "[goproxy] FAIL $iid (rc=$rc, see $dest/build.log)"; fail=$((fail + 1))
    fi
done
echo "[goproxy] done ok=$ok skip=$skip fail=$fail -> $OUT_DIR"
[ "$fail" -eq 0 ]
