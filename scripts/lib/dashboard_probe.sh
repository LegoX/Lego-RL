#!/usr/bin/env bash
# =============================================================================
# scripts/lib/dashboard_probe.sh — read-only snapshot before serving the webui
# =============================================================================
# Usage:   bash scripts/lib/dashboard_probe.sh
#
# live_probe.sh answers "is this box free for a training run?". This one answers
# "what would it take to serve the dashboard *here*?" — where the repo is, which
# directories actually hold run logs, whether dist/ needs a rebuild, which port
# is free, and whether an instance is already up. Strictly read-only: it never
# builds, starts, kills, or edits anything, and it never decides which log dir
# the user meant; it prints facts and /rl:dashboard does the judgement.
#
# Line format (stable, one fact per line, parseable):
#     <OK|WARN|INFO>  <name>  <detail>
# OK   = usable as-is           WARN = needs a human/agent to classify
# INFO = context, never blocking
# Exit code is always 0 — a probe is not a verdict.
# =============================================================================
SCRIPT_PATH="$(readlink -f "$0")"
REPO_ROOT="${REPO_ROOT:-$(dirname "$(dirname "$(dirname "$SCRIPT_PATH")")")}"
WEBUI_DIR="${WEBUI_DIR:-$REPO_ROOT/webui}"

_p()    { printf '%-5s %-16s %s\n' "$1" "$2" "$3"; }
_ok()   { _p OK   "$1" "$2"; }
_warn() { _p WARN "$1" "$2"; }
_info() { _p INFO "$1" "$2"; }
# pgrep that never matches this probe or its own shell
_pg()   { pgrep -af "$1" 2>/dev/null | grep -v -e dashboard_probe -e "^$$ " ; }

_age_min() {  # $1=path -> minutes since last write, or "?"
    local t; t="$(stat -c %Y "$1" 2>/dev/null)" || { echo '?'; return; }
    echo $(( ( $(date +%s) - t ) / 60 ))
}

echo "───────────────────── dashboard probe (local host only) ─────────────────────"
_info host "$(hostname -s 2>/dev/null) $(hostname -I 2>/dev/null | awk '{print $1}')"

# --- 1. layout ---------------------------------------------------------------
# Everything downstream is relative to these two paths, so a wrong repo root is
# the one failure that makes every other line meaningless.
_info repo "$REPO_ROOT"
if [ -f "$WEBUI_DIR/server.py" ]; then
    _ok webui "$WEBUI_DIR"
else
    _warn webui "no server.py under $WEBUI_DIR — set WEBUI_DIR= to the real location"
fi
for _f in server.py start_dashboard.sh package.json; do
    [ -f "$WEBUI_DIR/$_f" ] || _warn "missing:$_f" "$WEBUI_DIR/$_f not found"
done

# --- 2. is a dashboard already serving? --------------------------------------
# Re-launching on a taken port just dies; re-launching on a free one leaves two
# boards showing different log dirs, which is worse.
_found_srv=0
while IFS= read -r _line; do
    [ -z "$_line" ] && continue
    _spid="${_line%% *}"; _scmd="${_line#* }"
    _sport="$(printf '%s' "$_scmd" | grep -oE '\--port +[0-9]+' | awk '{print $2}')"
    _sdir="$(printf '%s' "$_scmd"  | grep -oE '\--log-dir +[^ ]+'  | awk '{print $2}')"
    _set="$(ps -p "$_spid" -o etime= 2>/dev/null | tr -d ' ')"
    _warn srv:running "pid=$_spid port=${_sport:-?} log-dir=${_sdir:-?} etime=${_set:-?}"
    _found_srv=1
done < <(_pg 'server\.py.*--port|webui/server\.py')
[ "$_found_srv" -eq 0 ] && _ok srv:none "no webui server.py process on this host"

# --- 3. tunnels --------------------------------------------------------------
# quick-tunnel URLs are ephemeral and live only in the process's own stdout, so
# recover them from the fd the process was started with rather than guessing.
_found_tun=0
while IFS= read -r _line; do
    [ -z "$_line" ] && continue
    _tpid="${_line%% *}"; _tcmd="${_line#* }"
    _turl="$(printf '%s' "$_tcmd" | grep -oE 'https?://[^ ]+|127\.0\.0\.1:[0-9]+|localhost:[0-9]+' | head -1)"
    _pub=""
    for _fd in 1 2; do
        _t="$(readlink "/proc/$_tpid/fd/$_fd" 2>/dev/null)"
        case "$_t" in /*) _pub="$(grep -aoE 'https://[a-z0-9-]+\.trycloudflare\.com' "$_t" 2>/dev/null | tail -1)";; esac
        [ -n "$_pub" ] && break
    done
    _warn tunnel "pid=$_tpid -> ${_turl:-?} public=${_pub:-<unknown, check its stdout>}"
    _found_tun=1
done < <(_pg 'cloudflared tunnel')
[ "$_found_tun" -eq 0 ] && _ok tunnel:none "no cloudflared process on this host"

# --- 4. ports ----------------------------------------------------------------
_port_owner() {   # $1=port -> "comm pid" or empty
    if command -v ss >/dev/null 2>&1; then
        ss -lntp 2>/dev/null | awk -v p=":$1\$" '$4 ~ p {print $NF}' | head -1 \
            | sed -E 's/.*users:\(\("([^"]+)",pid=([0-9]+).*/\1 \2/'
    fi
}
_free_port=""
for _port in 8090 8091 8092 8093 8094 8095; do
    _own="$(_port_owner "$_port")"
    if [ -n "$_own" ]; then
        _warn "port:$_port" "in use by ${_own% *} pid ${_own##* }"
    else
        [ -z "$_free_port" ] && _free_port="$_port"
    fi
done
[ -n "$_free_port" ] && _ok port:free "$_free_port is the lowest free port in 8090-8095" \
                     || _warn port:none "8090-8095 all taken — pass PORT= explicitly"

# --- 5. toolchain ------------------------------------------------------------
# server.py is stdlib-only, so serving needs no venv; only a *rebuild* needs node.
if command -v python3 >/dev/null 2>&1; then
    _ok python3 "$(command -v python3) ($(python3 --version 2>&1))  [server.py is stdlib-only]"
else
    _warn python3 "not on PATH — cannot serve at all"
fi
if command -v node >/dev/null 2>&1; then
    _nmaj="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)"
    if [ "${_nmaj:-0}" -ge 20 ] 2>/dev/null; then
        _ok node "$(node -v) — new enough to rebuild the frontend"
    else
        _warn node "$(node -v) is < 20 — too old to rebuild; nvm has: $(ls "$HOME/.nvm/versions/node" 2>/dev/null | tr '\n' ' ')"
    fi
else
    _warn node "not on PATH — serving a prebuilt dist/ still works, rebuilding does not"
fi
# A quick tunnel (`cloudflared tunnel --url`) needs no Cloudflare account, no
# login and no token — but it does need egress to Cloudflare's edge on :443.
# On an air-gapped training box that is the step that fails, not the install.
if command -v cloudflared >/dev/null 2>&1; then
    _ok cloudflared "$(command -v cloudflared) $(cloudflared --version 2>/dev/null | awk '{print $3}')"
    if [ -f "$HOME/.cloudflared/cert.pem" ]; then
        _info cf:account "logged in (cert.pem present) — named tunnels available too"
    else
        _info cf:account "no login — quick tunnels only, which need no account"
    fi
    if curl -sS -m 6 -o /dev/null https://www.cloudflare.com/cdn-cgi/trace 2>/dev/null; then
        _ok egress "cloudflare edge reachable — a quick tunnel can be opened"
    else
        _warn egress "cannot reach cloudflare edge in 6s — a quick tunnel will not come up; use ssh -L"
    fi
else
    _info cloudflared "not installed — local access only; no account needed, just the binary"
fi

# --- 6. frontend build state -------------------------------------------------
# start_dashboard.sh only rebuilds when dist/index.html is *absent*, so a dist
# older than src/ is served silently. That is the "my fix isn't showing" case.
if [ -f "$WEBUI_DIR/dist/index.html" ]; then
    _dist_t="$(stat -c %Y "$WEBUI_DIR/dist/index.html" 2>/dev/null || echo 0)"
    _src_t="$(find "$WEBUI_DIR/src" -type f \( -name '*.ts' -o -name '*.tsx' -o -name '*.css' \) \
                -printf '%T@\n' 2>/dev/null | sort -rn | head -1 | cut -d. -f1)"
    if [ -n "$_src_t" ] && [ "${_src_t:-0}" -gt "${_dist_t:-0}" ] 2>/dev/null; then
        _warn dist:stale "dist/ built $(_age_min "$WEBUI_DIR/dist/index.html")min ago, but src/ changed since — rebuild to see recent edits"
    else
        _ok dist:fresh "dist/index.html is newer than src/ (built $(_age_min "$WEBUI_DIR/dist/index.html")min ago)"
    fi
    _info dist:size "$(du -sh "$WEBUI_DIR/dist" 2>/dev/null | cut -f1)"
else
    _warn dist:absent "no dist/index.html — the frontend must be built before serving"
fi
_nbak="$(find "$WEBUI_DIR" -maxdepth 1 -name 'dist.bak-*' -type d 2>/dev/null | wc -l)"
[ "${_nbak:-0}" -gt 0 ] && _info dist:backups "$_nbak dist.bak-* dirs left in webui/ (not served; safe to remove)"

# --- 7. log directories — the part that actually moves between machines ------
# server.py treats every .log/.out under a dir as a run (skipping *_vllm.log).
# Candidates are printed with evidence so the caller can pick, never guessed at.
_seen_dirs=""
_probe_logdir() {   # $1=dir  $2=why-it-is-a-candidate
    local d="$1" why="$2" real n newest
    real="$(readlink -f "$d" 2>/dev/null)" || return
    [ -d "$real" ] || return
    case " $_seen_dirs " in *" $real "*) return;; esac
    _seen_dirs="$_seen_dirs $real"
    n="$(find "$real" -maxdepth 1 \( -name '*.log' -o -name '*.out' \) ! -name '*_vllm.log' 2>/dev/null | wc -l)"
    newest="$(ls -t "$real"/*.log "$real"/*.out 2>/dev/null | grep -v '_vllm\.log$' | head -1)"
    if [ "${n:-0}" -gt 0 ]; then
        _ok "logdir:$why" "$real — ${n} run log(s), newest $(basename "${newest:-?}") $(_age_min "${newest:-/nonexistent}")min ago"
    else
        _info "logdir:$why" "$real — exists but holds no .log/.out the dashboard would show"
    fi
}
[ -n "${LOG_DIR:-}" ] && _probe_logdir "$LOG_DIR" env
_probe_logdir "$REPO_ROOT/logs" default
# whatever an already-running instance chose is by definition a working answer
while IFS= read -r _line; do
    _d="$(printf '%s' "$_line" | grep -oE '\--log-dir +[^ ]+' | awk '{print $2}')"
    [ -n "$_d" ] && _probe_logdir "$_d" running
done < <(_pg 'server\.py.*--log-dir')
# sibling checkouts on the same box often hold the runs people actually want
for _sib in "$(dirname "$REPO_ROOT")"/*/logs; do
    [ -d "$_sib" ] && _probe_logdir "$_sib" sibling
done
# Where the RUNNER actually writes, which since the config/template refactor is
# usually NOT <repo>/logs: templates/verl/common.env derives
#   TRAIN_LOG=${HARBOR_LOG_DIR}/${TRAINER_EXPERIMENT_NAME}.log
# and every real config overrides HARBOR_LOG_DIR to a per-exp dir under the shared
# harbor_trials root. server.py globs one level only, so a board serving
# <repo>/logs shows nothing for those runs even though training is perfectly fine.
# Report those dirs as candidates so the caller can serve/link them deliberately.
_trials_roots=""
for _cfg in "$REPO_ROOT"/scripts/*/configs/*.env; do
    [ -f "$_cfg" ] || continue
    while IFS= read -r _v; do
        # keep the literal prefix up to the first ${...} expansion
        _root="${_v%%\$\{*}"; _root="${_root%/}"
        case "$_root" in
            ""|"$REPO_ROOT"/logs) continue;;
        esac
        case " $_trials_roots " in *" $_root "*) continue;; esac
        [ -d "$_root" ] || continue
        _trials_roots="$_trials_roots $_root"
    done < <(grep -hoE '^[A-Z_]*LOG_DIR=[^ ]*' "$_cfg" 2>/dev/null | cut -d= -f2- | tr -d '"'"'")
done
# One pass over the candidate dirs' symlinks (a single find per dir, no per-entry
# subprocess): these are how an off-repo run gets surfaced to a served dir.
_linked_targets=""
if [ -n "$_trials_roots" ]; then
    for _sd in $_seen_dirs; do
        _linked_targets="$_linked_targets
$(find "$_sd" -maxdepth 1 -type l -printf '%l\n' 2>/dev/null)"
    done
fi
_n_off=0
for _root in $_trials_roots; do
    _info logdir:trialsroot "$_root — per-exp run logs live under <project>/<exp>/logs (configs point TRAIN_LOG here)"
    # Bounded: only dirs written in the last 3 days, freshest 8. The shared FS
    # makes an unbounded walk here cost minutes.
    while IFS= read -r _ld; do
        [ -d "$_ld" ] || continue
        _lnew="$(ls -t "$_ld"/*.log "$_ld"/*.out 2>/dev/null | grep -v -e '_vllm\.log$' -e '_train_gpu_wandb\.log$' | head -1)"
        [ -n "$_lnew" ] || continue
        _n_off=$((_n_off + 1))
        # Is it reachable from any dir the probe already listed as a candidate —
        # either because that dir IS this one, or via a symlink pointing at it?
        _linked=no
        case " $_seen_dirs " in *" $_ld "*) _linked=yes;; esac
        case "
$_linked_targets
" in *"
$_lnew
"*) _linked=yes;; esac
        if [ "$_linked" = yes ]; then
            _ok "logdir:trials" "$_ld — $(basename "$_lnew") $(_age_min "$_lnew")min ago, already visible via a link under a served dir"
        else
            _warn "logdir:offrepo" "$_ld — $(basename "$_lnew") $(_age_min "$_lnew")min ago, NOT visible from any candidate dir above"
        fi
    done < <(find "$_root" -mindepth 3 -maxdepth 3 -type d -name logs -mtime -3 2>/dev/null | head -8)
done
[ "$_n_off" -gt 0 ] && _info logdir:hint \
    "off-repo run logs exist: either --extra-log-dir them, or ln -s each into the served dir (server.py globs one level, no recursion)"

[ -z "$_seen_dirs" ] && _warn logdir:none "no candidate log directory found — pass LOG_DIR= explicitly"

echo "──────────────────────────────────────────────────────────────────────"
echo "[dashboard_probe] facts only — which log dir to serve is decided by the caller."
exit 0
