#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# scripts/setup_env.sh
#
# Bootstraps a training-ready environment for `SWE-Lego-RL`:
#   1. Install `uv` (if missing)
#   2. Create a virtualenv at $VENV_PATH with the requested Python version
#   3. Clone + checkout harbor & verl at the configured branch/tag
#   4. Install harbor (-e), verl (-e), vllm, flash_attn, cupy
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
VERL_DIR="${VERL_DIR:-$WORKSPACE_ROOT/verl}"

HARBOR_REPO_URL="${HARBOR_REPO_URL:-https://github.com/SWE-Lego/harbor.git}"
HARBOR_REF="${HARBOR_REF:-${HARBOR_COMMIT:-ydu_dev}}"

# ⚠ This verl fork is NOT public yet: the clone below fails without access. It carries the
# VeOmni router-replay, R3 and fully-async fixes this repo depends on. Until it is published,
# override both variables — upstream verl 0.8.0 runs the synchronous path but not those features:
#   VERL_REPO_URL=https://github.com/verl-project/verl.git VERL_REF=v0.8.0 bash scripts/setup_env.sh
VERL_REPO_URL="${VERL_REPO_URL:-https://github.com/Elvin-Yiming-Du/verl.git}"
VERL_REF="${VERL_REF:-${VERL_COMMIT:-ydu/swe_agent_opd_dev}}"

# Python / venv
PYTHON_VERSION="${PYTHON_VERSION:-3.12.3}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"

# CUDA (flash_attn build needs this)
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-90}"
FLASH_ATTN_MAX_JOBS="${FLASH_ATTN_MAX_JOBS:-8}"

# Package versions
VLLM_VERSION="${VLLM_VERSION:-0.19.0}"

TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.4.0}"
# Set to 1 to skip pinning transformers (e.g. when managing deps elsewhere).
SKIP_TRANSFORMERS="${SKIP_TRANSFORMERS:-0}"

# Skip flags for debugging/partial runs
SKIP_CLONE="${SKIP_CLONE:-0}"
SKIP_FLASH_ATTN="${SKIP_FLASH_ATTN:-0}"
SKIP_CUPY="${SKIP_CUPY:-0}"

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
    local name="$1" dir="$2" url="$3" ref="$4"

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

    log "[$name] fetching and checking out $ref"
    git -C "$dir" fetch --tags --prune origin
    # `ref` may be a branch, tag, or commit sha. Resolve to a full sha for
    # deterministic checkout.
    local full_sha
    full_sha="$(git -C "$dir" rev-parse "$ref^{commit}" 2>/dev/null || true)"
    if [ -z "$full_sha" ]; then
        full_sha="$(git -C "$dir" rev-parse "origin/$ref^{commit}" 2>/dev/null || true)"
    fi
    [ -n "$full_sha" ] || die "[$name] cannot resolve ref '$ref' in $dir"
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

# ---------------- 3. harbor & verl source trees -----------------------------

ensure_repo "harbor" "$HARBOR_DIR" "$HARBOR_REPO_URL" "$HARBOR_REF"
ensure_repo "verl"   "$VERL_DIR"   "$VERL_REPO_URL"   "$VERL_REF"

# ---------------- 4. install core deps --------------------------------------

log "installing harbor (-e) from $HARBOR_DIR"
uv pip install -e "$HARBOR_DIR"

log "installing verl (-e) from $VERL_DIR"
uv pip install -e "$VERL_DIR"

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

log "installing SWE-Lego-RL (-e) from $REPO_ROOT"
uv pip install -e "$REPO_ROOT"

# ---------------- 7. transformers pin ---------------------------------------

if [ "$SKIP_TRANSFORMERS" != "1" ]; then
    log "pinning transformers==$TRANSFORMERS_VERSION"
    uv pip install "transformers==$TRANSFORMERS_VERSION"
else
    warn "SKIP_TRANSFORMERS=1, leaving transformers untouched"
fi

# ---------------- 8. summary ------------------------------------------------

log "======================================================================"
log "Setup complete."
log "  harbor           : $HARBOR_DIR  @ $(git -C "$HARBOR_DIR" rev-parse --short HEAD) ($HARBOR_REF)"
log "  verl             : $VERL_DIR    @ $(git -C "$VERL_DIR"   rev-parse --short HEAD) ($VERL_REF)"
log "  SWE-Lego-RL: $REPO_ROOT"
log "  venv             : $VENV_PATH"
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
