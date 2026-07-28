#!/usr/bin/env bash
# =============================================================================
# scripts/lib/live_probe.sh — read-only snapshot of THIS machine before a launch
# =============================================================================
# Usage:   bash scripts/lib/live_probe.sh [train|eval|infer]
#
# preflight.sh answers "is the config right?" by reading config only. This answers
# "is the box free?" by reading the process table only. Strictly read-only — it
# never kills, cleans, or edits anything, and it never decides whether a process
# is *yours*; it prints facts and /harbor:check does the ownership judgement.
#
# Line format (stable, one fact per line, parseable):
#     <OK|WARN|INFO>  <name>  <detail>
# OK   = nothing in the way        WARN = needs a human/agent to classify
# INFO = context, never blocking
# Exit code is always 0 — a probe is not a verdict.
# =============================================================================
KIND="${1:-train}"
SCRIPT_PATH="$(readlink -f "$0")"
REPO_ROOT="${REPO_ROOT:-$(dirname "$(dirname "$(dirname "$SCRIPT_PATH")")")}"

_p()      { printf '%-5s %-16s %s\n' "$1" "$2" "$3"; }
_ok()     { _p OK   "$1" "$2"; }
_warn()   { _p WARN "$1" "$2"; }
_info()   { _p INFO "$1" "$2"; }
# pgrep that never matches this probe or its own shell
_pg()     { pgrep -af "$1" 2>/dev/null | grep -v -e live_probe -e "^$$ " ; }

echo "───────────────────────── live probe (local host only) ─────────────────────────"
_info host "$(hostname -s 2>/dev/null) $(hostname -I 2>/dev/null | awk '{print $1}')  uptime=$(uptime -p 2>/dev/null | sed 's/^up //')"

# --- 1. is a run already in flight? -------------------------------------------
# A second run on the same GPUs OOMs or corrupts both. Trainer entrypoints first,
# then the runner wrappers (a wrapper alive without a trainer = still booting).
_found_job=0
_probe_job() {   # $1=label  $2=pgrep pattern
    local label="$1" pat="$2" line pid et cmd
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        pid="${line%% *}"; cmd="${line#* }"
        et="$(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ')"
        _warn "job:$label" "pid=$pid etime=${et:-?} :: $(printf '%s' "$cmd" | cut -c1-110)"
        _found_job=1
    done < <(_pg "$pat")
}
_probe_job trainer  'fully_async_main|verl\.trainer\.main_ppo'
_probe_job runner   'scripts/train/train\.sh|scripts/eval/eval\.sh|scripts/infer/infer\.sh'
_probe_job vllm     'vllm serve|VLLM::|EngineCore'
_probe_job harbor   'eval_swerebench_filtered\.py|harbor run'
[ "$_found_job" -eq 0 ] && _ok job:none "no trainer / runner / vllm / harbor process on this host"

# --- 2. ray ------------------------------------------------------------------
if command -v ray >/dev/null 2>&1; then
    if _pg 'raylet|gcs_server' >/dev/null 2>&1 && [ -n "$(_pg 'raylet|gcs_server')" ]; then
        _nodes="$(ray status 2>/dev/null | grep -c 'node_' || echo '?')"
        _warn ray:up "raylet alive, ray status reports ${_nodes} node(s) — leftover cluster, or the live run's"
    else
        _ok ray:down "no raylet/gcs_server (train.sh runs 'ray stop --force' at bring-up anyway)"
    fi
else
    _info ray:absent "ray not on PATH in this shell (the runner activates .venv itself)"
fi

# --- 3. GPUs -----------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    _gpu_total="$(nvidia-smi --list-gpus 2>/dev/null | wc -l)"
    _apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null)"
    if [ -z "$_apps" ]; then
        _ok gpu:idle "${_gpu_total} GPU(s), no compute process holding memory"
    else
        printf '%s\n' "$_apps" | while IFS=, read -r _gpid _gmem; do
            _gpid="$(printf '%s' "$_gpid" | tr -d ' ')"; _gmem="$(printf '%s' "$_gmem" | tr -d ' ')"
            [ -z "$_gpid" ] && continue
            _gcomm="$(ps -p "$_gpid" -o comm= 2>/dev/null | tr -d ' ')"
            _get="$(ps -p "$_gpid" -o etime= 2>/dev/null | tr -d ' ')"
            _warn gpu:busy "pid=$_gpid comm=${_gcomm:-<gone>} mem=${_gmem}MiB etime=${_get:-?}"
        done
        _info gpu:count "${_gpu_total} GPU(s) total on this host"
    fi
    _info gpu:mem "$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', *' '{printf "%s:%.0f/%.0fG ", $1, $2/1024, $3/1024}')"
else
    _info gpu:absent "nvidia-smi not available on this host (dev/CPU box?)"
fi

# --- 4. ports ----------------------------------------------------------------
# Only the ports our own stack binds. A stale listener here is the classic
# "every trial times out but nothing looks broken" cause.
_port_owner() {   # $1=port -> "comm pid" or empty
    if command -v ss >/dev/null 2>&1; then
        ss -lntp 2>/dev/null | awk -v p=":$1\$" '$4 ~ p {print $NF}' | head -1 \
            | sed -E 's/.*users:\(\("([^"]+)",pid=([0-9]+).*/\1 \2/'
    fi
}
for _port in 6379 8265 8000 8011 8002 8090; do
    _own="$(_port_owner "$_port")"
    [ -n "$_own" ] && _warn "port:$_port" "in use by ${_own% *} pid ${_own##* }"
done
_ok port:scanned "checked 6379(ray) 8265(ray-dash) 8000/8011(vllm) 8002(litellm) 8090(webui)"

# --- 5. /dev/shm + disk ------------------------------------------------------
# vLLM SIGBUSes on a full /dev/shm; a full root disk is the etcd/emptyDir killer
# (val-burst-env-setup-failed-reward-dip) and shows up as random pod deaths.
_shm="$(df -h /dev/shm 2>/dev/null | awk 'NR==2{print $3"/"$2" used="$5}')"
_shm_pct="$(df /dev/shm 2>/dev/null | awk 'NR==2{gsub(/%/,"",$5); print $5}')"
if [ "${_shm_pct:-0}" -ge 50 ] 2>/dev/null; then
    _warn shm "/dev/shm $_shm — leftover vllm/ray segments; train.sh cleans them at bring-up"
else
    _ok shm "/dev/shm $_shm"
fi
# Judge on absolute free space as well as percent: the shared /mnt store sits at 90%+
# with hundreds of TB free (harmless), while a 98G root at 85% is the real killer.
for _mnt in / "$REPO_ROOT"; do
    _du="$(df -h "$_mnt" 2>/dev/null | awk 'NR==2{print $5" of "$2" (avail "$4")"}')"
    _dp="$(df "$_mnt" 2>/dev/null | awk 'NR==2{gsub(/%/,"",$5); print $5}')"
    _dfree_g="$(df -BG "$_mnt" 2>/dev/null | awk 'NR==2{gsub(/G/,"",$4); print $4}')"
    if [ "${_dp:-0}" -ge 85 ] 2>/dev/null && [ "${_dfree_g:-99999}" -lt 500 ] 2>/dev/null; then
        _warn "disk:$_mnt" "$_du — nearly full, pods get evicted / logs stop writing"
    else
        _ok "disk:$_mnt" "$_du"
    fi
done

# --- 6. python / venv / imports ---------------------------------------------
# preflight checks config; this checks that the interpreter the runner will pick
# can actually import what the run needs.
_venv="${VENV_PATH:-$REPO_ROOT/.venv}"
if [ -x "$_venv/bin/python3" ]; then
    _py="$_venv/bin/python3"; _ok venv "$_venv ($("$_py" --version 2>&1))"
else
    _py="$(command -v python3)"; _warn venv "no $_venv/bin/python3 — runner falls back to ${_py:-<none>}"
fi
if [ -n "$_py" ]; then
    for _mod in veomni verl ray vllm wandb; do
        _org="$("$_py" -c "import importlib.util as u;s=u.find_spec('$_mod');print(s.origin if s else '')" 2>/dev/null)"
        [ -n "$_org" ] && _ok "import:$_mod" "$_org" || _warn "import:$_mod" "not importable by $_py"
    done
fi

# --- 7. recent logs (is the previous run's tail still moving?) ---------------
_newest="$(ls -t "$REPO_ROOT"/logs/*.log 2>/dev/null | head -1)"
if [ -n "$_newest" ]; then
    _age=$(( ( $(date +%s) - $(stat -c %Y "$_newest" 2>/dev/null || echo 0) ) / 60 ))
    _info log:newest "$(basename "$_newest")  size=$(du -h "$_newest" 2>/dev/null | cut -f1)  last_write=${_age}min ago"
else
    _info log:newest "no logs/*.log yet"
fi

echo "──────────────────────────────────────────────────────────────────────"
echo "[live_probe] facts only — ownership (mine vs foreign) is decided by the caller."
exit 0
