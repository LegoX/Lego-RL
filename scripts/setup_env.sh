#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# scripts/setup_env.sh
#
# Bootstraps a training-ready environment for `Lego-RL`:
#   1. Install `uv` (if missing)
#   2. Create a virtualenv at $VENV_PATH with the requested Python version
#   3. Clone + checkout harbor & verl at the configured branch/tag
#   4. Install harbor (-e), verl (-e), veomni, vllm, flash_attn, cupy
#   5. Install this repo (-e) and pin transformers==5.4.0
#
# Everything is idempotent: re-running the script only redoes what is missing.
#
# All paths/revisions are overridable via environment variables. See the
# defaults right below.
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_PATH="$(readlink -f "$0")"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_PATH")")"

# ---------------- User-overridable configuration -----------------------------

# Where harbor / verl sources live. Defaults: siblings of this repo.
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(dirname "$REPO_ROOT")}"
HARBOR_DIR="${HARBOR_DIR:-$WORKSPACE_ROOT/harbor}"

# verl: the directory name matches the branch it tracks (ydu/swe_agent_opd_dev), and
# it MUST stay in sync with NEW_VERL_DIR in scripts/lib/common_env.sh, which prepends
# the same directory to PYTHONPATH at run time. If setup installs one verl checkout
# and common_env.sh points PYTHONPATH at another, training silently runs different
# code than what was installed -- or, on a fresh box where that directory does not
# exist, preflight aborts.
VERL_DIR="${VERL_DIR:-$WORKSPACE_ROOT/verl-swe_agent_opd_dev}"

HARBOR_REPO_URL="${HARBOR_REPO_URL:-https://github.com/SWE-Lego/harbor.git}"
HARBOR_REF="${HARBOR_REF:-${HARBOR_COMMIT:-ydu_dev}}"

# ⚠ This verl fork is NOT public yet: the clone below fails without access. It carries the
# VeOmni router-replay, R3 and fully-async fixes this repo depends on. Until it is published,
# override both variables — upstream verl 0.8.0 runs the synchronous path but not those features:
#   VERL_REPO_URL=https://github.com/verl-project/verl.git VERL_REF=v0.8.0 bash scripts/setup_env.sh
VERL_REPO_URL="${VERL_REPO_URL:-https://github.com/Elvin-Yiming-Du/verl.git}"
# This is the tree training actually loads (NEW_VERL_DIR in scripts/lib/common_env.sh).
# ydu/swe_agent_opd_dev is kept fast-forwarded to the same commit, so either name
# resolves to the same code; this one is pinned because it names the merge explicitly.
VERL_REF="${VERL_REF:-${VERL_COMMIT:-ydu/merge-yt-20260729}}"
# Existing clones on the training boxes have origin=verl-project/verl and carry the
# ydu/* branches on a fork remote instead, so `git fetch origin` alone will not find
# VERL_REF there. These are added + fetched by ensure_repo when missing.
VERL_EXTRA_REMOTES="${VERL_EXTRA_REMOTES:-elvin=https://github.com/Elvin-Yiming-Du/verl.git yt0428=https://github.com/yt0428/verl.git}"

# Python / venv
PYTHON_VERSION="${PYTHON_VERSION:-3.12.3}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"

# CUDA (flash_attn build needs this)
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-90}"
FLASH_ATTN_MAX_JOBS="${FLASH_ATTN_MAX_JOBS:-8}"

# Package versions
VLLM_VERSION="${VLLM_VERSION:-0.19.0}"
VEOMNI_VERSION="${VEOMNI_VERSION:-0.1.11}"

TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.4.0}"
# Set to 1 to skip pinning transformers (e.g. when managing deps elsewhere).
SKIP_TRANSFORMERS="${SKIP_TRANSFORMERS:-0}"

# Skip flags for debugging/partial runs
SKIP_CLONE="${SKIP_CLONE:-0}"
SKIP_FLASH_ATTN="${SKIP_FLASH_ATTN:-0}"
SKIP_CUPY="${SKIP_CUPY:-0}"
SKIP_VEOMNI="${SKIP_VEOMNI:-0}"

# ---------------- Helpers ----------------------------------------------------

log()  { printf '\033[1;32m[setup_env]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup_env]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[setup_env ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

# Resolve a commit-ish to a full SHA without a working tree checkout.
resolve_sha() {
    local url="$1" rev="$2"
    git ls-remote "$url" "$rev" 2>/dev/null | awk '{print $1; exit}' || true
}

ensure_repo() {
    local name="$1" dir="$2" url="$3" ref="$4" extra_remotes="${5:-}"

    if [ "$SKIP_CLONE" = "1" ]; then
        log "[$name] SKIP_CLONE=1, assuming $dir is already prepared"
        [ -d "$dir/.git" ] || die "[$name] SKIP_CLONE=1 but $dir is not a git repo"
        return
    fi

    if [ ! -d "$dir/.git" ]; then
        log "[$name] cloning $url -> $dir"
        mkdir -p "$(dirname "$dir")"
        git clone "$url" "$dir"
    else
        log "[$name] reuse existing clone at $dir"
    fi

    # The pinned ref may live on a fork rather than origin.
    local spec remote_name remote_url
    for spec in $extra_remotes; do
        remote_name="${spec%%=*}"; remote_url="${spec#*=}"
        if ! git -C "$dir" remote get-url "$remote_name" >/dev/null 2>&1; then
            log "[$name] adding remote $remote_name -> $remote_url"
            git -C "$dir" remote add "$remote_name" "$remote_url"
        fi
    done

    log "[$name] fetching and checking out $ref"
    git -C "$dir" fetch --tags --prune origin
    for spec in $extra_remotes; do
        git -C "$dir" fetch --prune "${spec%%=*}" 2>/dev/null || \
            warn "[$name] could not fetch remote ${spec%%=*} (continuing)"
    done
    # `ref` may be a branch, tag, or commit sha. Resolve to a full sha for
    # deterministic checkout. Try bare, then origin/, then each extra remote.
    local full_sha
    full_sha="$(git -C "$dir" rev-parse "$ref^{commit}" 2>/dev/null || true)"
    if [ -z "$full_sha" ]; then
        full_sha="$(git -C "$dir" rev-parse "origin/$ref^{commit}" 2>/dev/null || true)"
    fi
    for spec in $extra_remotes; do
        [ -n "$full_sha" ] && break
        full_sha="$(git -C "$dir" rev-parse "${spec%%=*}/$ref^{commit}" 2>/dev/null || true)"
    done
    if [ -z "$full_sha" ]; then
        warn "[$name] ref '$ref' not found on any configured remote."
        warn "[$name] if this is a local-only branch, push it first so other machines"
        warn "[$name] can reproduce this environment, or pass ${name^^}_REF=<pushed-ref>."
        die "[$name] cannot resolve ref '$ref' in $dir"
    fi
    git -C "$dir" -c advice.detachedHead=false checkout --force "$full_sha"
    log "[$name] HEAD is now $(git -C "$dir" rev-parse --short HEAD)"
}

# ---------------- 1. uv ------------------------------------------------------

if ! command -v uv >/dev/null 2>&1; then
    log "installing uv via pip"
    python3 -m pip install --upgrade uv
else
    log "uv already installed: $(uv --version)"
fi

# ---------------- 2. virtualenv ---------------------------------------------

if [ ! -f "$VENV_PATH/bin/activate" ]; then
    log "creating virtualenv at $VENV_PATH (python $PYTHON_VERSION)"
    uv venv --python "$PYTHON_VERSION" "$VENV_PATH"
else
    log "reusing virtualenv at $VENV_PATH"
fi

# Activate so that `uv pip install` targets this venv.
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"
export VIRTUAL_ENV="$VENV_PATH"
log "python: $(python --version)  ($(command -v python))"

# Dependency overrides for every `uv pip install` below. uv only reads pyproject.toml's
# [tool.uv] block when the working directory happens to be the repo root, so rely on
# the env var instead -- this script may be invoked from anywhere.
UV_OVERRIDE_FILE="${UV_OVERRIDE_FILE:-$REPO_ROOT/overrides.txt}"
if [ -f "$UV_OVERRIDE_FILE" ]; then
    export UV_OVERRIDE="$UV_OVERRIDE_FILE"
    log "dependency overrides: $UV_OVERRIDE_FILE"
else
    die "missing override file: $UV_OVERRIDE_FILE (needed to resolve veomni against datasets 4.x)"
fi

# ---------------- 3. harbor & verl source trees -----------------------------

ensure_repo "harbor" "$HARBOR_DIR" "$HARBOR_REPO_URL" "$HARBOR_REF"
ensure_repo "verl"   "$VERL_DIR"   "$VERL_REPO_URL"   "$VERL_REF" "$VERL_EXTRA_REMOTES"

# ---------------- 4. install core deps --------------------------------------

log "installing harbor (-e) from $HARBOR_DIR"
uv pip install -e "$HARBOR_DIR"

log "installing verl (-e) from $VERL_DIR"
uv pip install -e "$VERL_DIR"

# veomni is a hard dependency of the default modeling backend: verl's
# workers/engine/veomni/transformer_impl.py imports veomni.{arguments,distributed,
# models.auto,optim,utils} at module import time, and MODELING_BACKEND defaults to
# veomni. Without it every veomni-backend run dies with ImportError on startup.
#
# Resolving it needs the datasets override exported above -- see overrides.txt.
if [ "$SKIP_VEOMNI" != "1" ]; then
    log "installing veomni==$VEOMNI_VERSION"
    uv pip install "veomni==$VEOMNI_VERSION"
else
    warn "SKIP_VEOMNI=1, leaving veomni untouched"
fi

log "installing vllm==$VLLM_VERSION"
uv pip install "vllm==$VLLM_VERSION"

# ---------------- 5. CUDA ext deps (flash_attn, cupy) -----------------------

export CUDA_HOME
export PATH="$CUDA_HOME/bin:${PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

if [ "$SKIP_FLASH_ATTN" != "1" ]; then
    if python -c "import flash_attn" >/dev/null 2>&1; then
        log "flash_attn already installed, skipping"
    else
        log "installing flash_attn (MAX_JOBS=$FLASH_ATTN_MAX_JOBS, ARCHS=$FLASH_ATTN_CUDA_ARCHS)"
        MAX_JOBS="$FLASH_ATTN_MAX_JOBS" FLASH_ATTN_CUDA_ARCHS="$FLASH_ATTN_CUDA_ARCHS" \
            uv pip install flash_attn --no-build-isolation --no-cache
    fi
else
    warn "SKIP_FLASH_ATTN=1, leaving flash_attn untouched"
fi

if [ "$SKIP_CUPY" != "1" ]; then
    if python -c "import cupy" >/dev/null 2>&1; then
        log "cupy already importable, skipping"
    else
        log "installing cupy-cuda12x (prebuilt wheel)"
        uv pip install cupy-cuda12x
    fi
else
    warn "SKIP_CUPY=1, leaving cupy untouched"
fi

# ---------------- 6. this repo ----------------------------------------------

log "installing Lego-RL (-e) from $REPO_ROOT"
uv pip install -e "$REPO_ROOT"

# ---------------- 7. transformers pin ---------------------------------------

if [ "$SKIP_TRANSFORMERS" != "1" ]; then
    log "pinning transformers==$TRANSFORMERS_VERSION"
    uv pip install "transformers==$TRANSFORMERS_VERSION"
else
    warn "SKIP_TRANSFORMERS=1, leaving transformers untouched"
fi

# ---------------- 8. verify the veomni backend imports ----------------------

# Cheap smoke test: vllm/transformers land after veomni and have been known to pull
# incompatible versions of its transitive deps. Fail here rather than 20 minutes into
# a run.
if [ "$SKIP_VEOMNI" != "1" ]; then
    log "verifying veomni backend imports"
    if python - <<'PY'
import sys

for mod in (
    "veomni.arguments",
    "veomni.distributed.parallel_state",
    "veomni.distributed.offloading",
    "veomni.distributed.torch_parallelize",
    "veomni.models.auto",
    "veomni.optim",
    "veomni.utils.seqlen_pos_transform_utils",
):
    try:
        __import__(mod)
    except Exception as exc:
        print(f"  FAIL {mod}: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
PY
    then
        log "veomni backend OK"
    else
        die "veomni is installed but its training-path modules do not import; the veomni backend will not run"
    fi
fi

# ---------------- 9. verl / harbor wiring check -----------------------------

# The single most damaging silent failure here is installing one verl checkout while
# scripts/lib/common_env.sh puts a different one on PYTHONPATH: training then runs
# code that was never installed, and nothing says so. Assert the two agree.
log "verifying verl / harbor resolve to the expected checkouts"

_resolved_verl="$(python -c 'import importlib.util as u; s=u.find_spec("verl"); print(s.origin or "")' 2>/dev/null || true)"
_resolved_harbor="$(python -c 'import importlib.util as u; s=u.find_spec("harbor"); print(s.origin or "")' 2>/dev/null || true)"
[ -n "$_resolved_verl" ]   || die "verl is not importable after setup"
[ -n "$_resolved_harbor" ] || die "harbor is not importable after setup"

case "$_resolved_verl" in
    "$(readlink -f "$VERL_DIR")"/*) log "  verl   -> $_resolved_verl" ;;
    *) die "verl resolves to $_resolved_verl but was installed from $VERL_DIR" ;;
esac
case "$_resolved_harbor" in
    "$(readlink -f "$HARBOR_DIR")"/*) log "  harbor -> $_resolved_harbor" ;;
    *) die "harbor resolves to $_resolved_harbor but was installed from $HARBOR_DIR" ;;
esac

# common_env.sh prepends NEW_VERL_DIR to PYTHONPATH at run time. If it names a
# different tree than what we just installed, every run silently uses that one.
_cenv="$REPO_ROOT/scripts/lib/common_env.sh"
if [ -f "$_cenv" ]; then
    # Evaluate the assignment rather than pattern-matching it: the default is derived
    # from REPO_ROOT via $(dirname ...), so a plain sed would capture the literal text.
    _new_verl_dir="$(
        REPO_ROOT="$REPO_ROOT" NEW_VERL_DIR="" bash -c '
            eval "$(grep -m1 "NEW_VERL_DIR=\"\${NEW_VERL_DIR:-" "$0" | sed "s/^[[:space:]]*//")"
            printf "%s" "${NEW_VERL_DIR:-}"
        ' "$_cenv" 2>/dev/null || true
    )"
    if [ -n "$_new_verl_dir" ] && [ "$(readlink -f "$_new_verl_dir" 2>/dev/null)" != "$(readlink -f "$VERL_DIR")" ]; then
        warn "MISMATCH: installed verl   = $(readlink -f "$VERL_DIR")"
        warn "          common_env.sh NEW_VERL_DIR = $_new_verl_dir"
        warn "Training prepends NEW_VERL_DIR to PYTHONPATH and would run that tree instead."
        warn "Fix one of the two, or export NEW_VERL_DIR=$VERL_DIR for every run."
    fi
fi

# ---------------- 10. summary -----------------------------------------------

log "======================================================================"
log "Setup complete."
log "  harbor           : $HARBOR_DIR  @ $(git -C "$HARBOR_DIR" rev-parse --short HEAD) ($HARBOR_REF)"
log "  verl             : $VERL_DIR    @ $(git -C "$VERL_DIR"   rev-parse --short HEAD) ($VERL_REF)"
log "  Lego-RL: $REPO_ROOT"
log "  venv             : $VENV_PATH"
if [ "$SKIP_VEOMNI" = "1" ]; then
    log "  veomni           : (skipped)"
else
    log "  veomni           : ==$VEOMNI_VERSION"
fi
if [ "$SKIP_TRANSFORMERS" = "1" ]; then
    log "  transformers     : (skipped)"
else
    log "  transformers     : ==$TRANSFORMERS_VERSION"
fi
log ""
log "Next step:"
log "  source \"$VENV_PATH/bin/activate\""
log "  bash scripts/sync.sh   # or sync_1nodes.sh / sync_2nodes.sh / fully_async.sh"
log "======================================================================"
