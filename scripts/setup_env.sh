#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# scripts/setup_env.sh
#
# Bootstraps a training-ready environment for `Lego-RL`:
#   1. Install `uv` (if missing)
#   2. Create a virtualenv at $VENV_PATH with the requested Python version
#   3. Install base dependencies from requirements.txt
#   4. Clone + checkout harbor, verl & vllm into third_party
#   5. Apply local monkey patches to harbor (optional), verl and vllm
#   6. Install patched harbor (-e --no-deps), patched verl (-e --no-deps),
#      patched vllm (-e --no-deps)
#   7. Install this repo (-e --no-deps)
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

# Managed third-party source trees. setup_env.sh owns these checkouts: every run
# checks out the configured ref and reapplies the patch before editable install.
THIRD_PARTY_DIR="${THIRD_PARTY_DIR:-$REPO_ROOT/third_party}"
HARBOR_DIR="${HARBOR_DIR:-$THIRD_PARTY_DIR/harbor}"
VERL_DIR="${VERL_DIR:-$THIRD_PARTY_DIR/verl}"
VLLM_DIR="${VLLM_DIR:-$THIRD_PARTY_DIR/vllm}"

HARBOR_REPO_URL="${HARBOR_REPO_URL:-https://github.com/LegoX/harbor-internal.git}"
HARBOR_REF="${HARBOR_REF:-${HARBOR_COMMIT:-main}}"

VERL_REPO_URL="${VERL_REPO_URL:-https://github.com/verl-project/verl.git}"
VERL_REF="${VERL_REF:-${VERL_COMMIT:-v0.8.0}}"

VLLM_REPO_URL="${VLLM_REPO_URL:-https://github.com/vllm-project/vllm.git}"
VLLM_REF="${VLLM_REF:-${VLLM_COMMIT:-v0.19.0}}"

# Harbor has no repo patch checked in today. Drop one at patches/harbor.patch or
# export HARBOR_PATCH=/path/to/patch to enable the same monkey-patch step.
HARBOR_PATCH="${HARBOR_PATCH:-$REPO_ROOT/patches/harbor.patch}"
VERL_PATCH="${VERL_PATCH:-$REPO_ROOT/patches/verl_7aed6b23.patch}"
VLLM_PATCH="${VLLM_PATCH:-$REPO_ROOT/patches/vllm_2a69949b.patch}"

# Python / venv / base dependency lock
PYTHON_VERSION="${PYTHON_VERSION:-3.12.3}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$REPO_ROOT/requirements.txt}"
MAX_JOBS="${MAX_JOBS:-1}"

# Skip flags for debugging/partial runs
SKIP_REQUIREMENTS="${SKIP_REQUIREMENTS:-0}"
SKIP_CLONE="${SKIP_CLONE:-0}"
SKIP_PATCHES="${SKIP_PATCHES:-0}"
INIT_SUBMODULES="${INIT_SUBMODULES:-1}"
SKIP_VEOMNI_CHECK="${SKIP_VEOMNI_CHECK:-0}"

# Dependency policy for editable source trees. requirements.txt is the lock source;
# editable installs should normally only register the patched source package itself.
# Set INSTALL_SOURCE_DEPS=1 only when intentionally debugging upstream metadata.
INSTALL_SOURCE_DEPS="${INSTALL_SOURCE_DEPS:-0}"
SKIP_REQUIREMENTS_PIN_CHECK="${SKIP_REQUIREMENTS_PIN_CHECK:-0}"

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
    full_sha="$(git -C "$dir" rev-parse --verify -q "$ref^{commit}" 2>/dev/null || true)"
    if [ -z "$full_sha" ]; then
        full_sha="$(git -C "$dir" rev-parse --verify -q "origin/$ref^{commit}" 2>/dev/null || true)"
    fi
    for spec in $extra_remotes; do
        [ -n "$full_sha" ] && break
        full_sha="$(git -C "$dir" rev-parse --verify -q "${spec%%=*}/$ref^{commit}" 2>/dev/null || true)"
    done
    if [ -z "$full_sha" ]; then
        warn "[$name] ref '$ref' not found on any configured remote."
        warn "[$name] if this is a local-only branch, push it first so other machines"
        warn "[$name] can reproduce this environment, or pass ${name^^}_REF=<pushed-ref>."
        die "[$name] cannot resolve ref '$ref' in $dir"
    fi
    git -C "$dir" -c advice.detachedHead=false checkout --force "$full_sha"
    log "[$name] HEAD is now $(git -C "$dir" rev-parse --short HEAD)"

    if [ "$INIT_SUBMODULES" = "1" ] && [ -f "$dir/.gitmodules" ]; then
        log "[$name] updating git submodules"
        git -C "$dir" submodule update --init --recursive
    fi
}

apply_repo_patch() {
    local name="$1" dir="$2" patch_file="$3" required="${4:-1}"

    if [ "$SKIP_PATCHES" = "1" ]; then
        warn "[$name] SKIP_PATCHES=1, leaving $dir unpatched"
        return
    fi
    if [ -z "$patch_file" ] || [ ! -f "$patch_file" ]; then
        if [ "$required" = "1" ]; then
            die "[$name] missing patch file: ${patch_file:-<unset>}"
        fi
        warn "[$name] no patch file found (${patch_file:-<unset>}); skipping patch"
        return
    fi
    [ -d "$dir/.git" ] || die "[$name] cannot apply patch; $dir is not a git repo"

    log "[$name] applying monkey patch: $patch_file"
    if git -C "$dir" apply --check "$patch_file"; then
        git -C "$dir" apply "$patch_file"
        log "[$name] patch applied"
    elif git -C "$dir" apply -R --check "$patch_file"; then
        log "[$name] patch already applied"
    else
        warn "[$name] patch does not apply cleanly to $dir"
        git -C "$dir" apply --stat "$patch_file" || true
        die "[$name] patch failed; check ${name^^}_REF and ${name^^}_PATCH"
    fi
}

uv_install_editable() {
    local name="$1" dir="$2"

    if [ "$INSTALL_SOURCE_DEPS" = "1" ]; then
        warn "[$name] INSTALL_SOURCE_DEPS=1, resolving source deps constrained by $REQUIREMENTS_FILE"
        uv pip install -c "$REQUIREMENTS_FILE" -e "$dir"
    else
        log "installing $name (-e --no-deps) from $dir"
        uv pip install --no-deps -e "$dir"
    fi
}

verify_requirements_pins() {
    if [ "$SKIP_REQUIREMENTS" = "1" ]; then
        warn "SKIP_REQUIREMENTS=1, not checking pinned package versions"
        return
    fi
    if [ "$SKIP_REQUIREMENTS_PIN_CHECK" = "1" ]; then
        warn "SKIP_REQUIREMENTS_PIN_CHECK=1, not checking pinned package versions"
        return
    fi

    log "verifying requirements pins were not changed by editable installs"
    if python - "$REQUIREMENTS_FILE" <<'PY'
import importlib.metadata as md
import re
import sys
from pathlib import Path

try:
    from packaging.version import InvalidVersion, Version
except Exception:  # pragma: no cover - packaging is in requirements.txt.
    InvalidVersion = ValueError
    Version = None


def versions_equal(left: str, right: str) -> bool:
    if Version is None:
        return left == right
    try:
        return Version(left) == Version(right)
    except InvalidVersion:
        return left == right


requirements = []
for raw in Path(sys.argv[1]).read_text().splitlines():
    line = raw.split("#", 1)[0].strip()
    if not line or line.startswith(("-", "git+", "http://", "https://")):
        continue
    match = re.match(r"^([A-Za-z0-9_.-]+)==([^;\s]+)", line)
    if match:
        requirements.append(match.groups())

problems = []
for name, expected in requirements:
    try:
        installed = md.version(name)
    except md.PackageNotFoundError:
        problems.append(f"{name}: missing, expected {expected}")
        continue
    if not versions_equal(installed, expected):
        problems.append(f"{name}: installed {installed}, expected {expected}")

if problems:
    print("Requirement pin drift detected:", file=sys.stderr)
    for problem in problems[:80]:
        print(f"  {problem}", file=sys.stderr)
    if len(problems) > 80:
        print(f"  ... {len(problems) - 80} more", file=sys.stderr)
    sys.exit(1)
PY
    then
        log "requirements pins unchanged"
    else
        die "editable installs changed or removed a package pinned by $REQUIREMENTS_FILE"
    fi
}

install_flash_attn_build_prereqs() {
    local prereq_file pkg

    if ! grep -qE '^flash-attn==' "$REQUIREMENTS_FILE"; then
        return
    fi

    log "preinstalling flash-attn build prerequisites from $REQUIREMENTS_FILE"
    prereq_file="$(mktemp)"
    for pkg in torch packaging setuptools ninja; do
        grep -m1 -E "^${pkg}==" "$REQUIREMENTS_FILE" >> "$prereq_file" || true
    done

    [ -s "$prereq_file" ] || die "could not find flash-attn build prerequisites in $REQUIREMENTS_FILE"
    uv pip install -r "$prereq_file"
    rm -f "$prereq_file"
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
export MAX_JOBS
log "python: $(python --version)  ($(command -v python))"
log "build parallelism: MAX_JOBS=$MAX_JOBS"

# Dependency overrides for the locked requirements install and any opt-in resolver
# installs. uv only reads pyproject.toml's [tool.uv] block when the working
# directory happens to be the repo root, so rely on the env var instead -- this
# script may be invoked from anywhere.
UV_OVERRIDE_FILE="${UV_OVERRIDE_FILE:-$REPO_ROOT/overrides.txt}"
if [ -f "$UV_OVERRIDE_FILE" ]; then
    export UV_OVERRIDE="$UV_OVERRIDE_FILE"
    log "dependency overrides: $UV_OVERRIDE_FILE"
else
    die "missing override file: $UV_OVERRIDE_FILE (needed to resolve veomni against datasets 4.x)"
fi

# ---------------- 3. install requirements -----------------------------------

if [ "$SKIP_REQUIREMENTS" != "1" ]; then
    [ -f "$REQUIREMENTS_FILE" ] || die "missing requirements file: $REQUIREMENTS_FILE"
    install_flash_attn_build_prereqs
    log "installing base dependencies from $REQUIREMENTS_FILE"
    uv pip install --no-build-isolation flash-attn -r "$REQUIREMENTS_FILE"
else
    warn "SKIP_REQUIREMENTS=1, leaving base dependencies untouched"
fi

# ---------------- 4. harbor / verl / vllm source trees ----------------------

ensure_repo "harbor" "$HARBOR_DIR" "$HARBOR_REPO_URL" "$HARBOR_REF"
apply_repo_patch "harbor" "$HARBOR_DIR" "$HARBOR_PATCH" 0
ensure_repo "verl"   "$VERL_DIR"   "$VERL_REPO_URL"   "$VERL_REF"
apply_repo_patch "verl" "$VERL_DIR" "$VERL_PATCH"
ensure_repo "vllm"   "$VLLM_DIR"   "$VLLM_REPO_URL"   "$VLLM_REF"
apply_repo_patch "vllm" "$VLLM_DIR" "$VLLM_PATCH"

# ---------------- 5. install patched source packages ------------------------

uv_install_editable "harbor" "$HARBOR_DIR"
uv_install_editable "verl" "$VERL_DIR"
uv_install_editable "vllm" "$VLLM_DIR"

# ---------------- 6. this repo ----------------------------------------------

uv_install_editable "Lego-RL" "$REPO_ROOT"

verify_requirements_pins

# ---------------- 7. verify the veomni backend imports ----------------------

# Cheap smoke test: requirements/source installs can pull incompatible versions
# of veomni's transitive deps. Fail here rather than 20 minutes into a run.
if [ "$SKIP_VEOMNI_CHECK" != "1" ]; then
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
else
    warn "SKIP_VEOMNI_CHECK=1, not verifying veomni backend imports"
fi

# ---------------- 8. harbor / verl / vllm wiring check ----------------------

# The single most damaging silent failure here is installing one checkout while
# Python imports another. Assert the editable installs resolve to the trees this
# script just prepared.
log "verifying harbor / verl / vllm resolve to the expected checkouts"

_resolved_verl="$(python -c 'import importlib.util as u; s=u.find_spec("verl"); print(s.origin or "")' 2>/dev/null || true)"
_resolved_harbor="$(python -c 'import importlib.util as u; s=u.find_spec("harbor"); print(s.origin or "")' 2>/dev/null || true)"
_resolved_vllm="$(python -c 'import importlib.util as u; s=u.find_spec("vllm"); print(s.origin or "")' 2>/dev/null || true)"
[ -n "$_resolved_verl" ]   || die "verl is not importable after setup"
[ -n "$_resolved_harbor" ] || die "harbor is not importable after setup"
[ -n "$_resolved_vllm" ]   || die "vllm is not importable after setup"

case "$_resolved_verl" in
    "$(readlink -f "$VERL_DIR")"/*) log "  verl   -> $_resolved_verl" ;;
    *) die "verl resolves to $_resolved_verl but was installed from $VERL_DIR" ;;
esac
case "$_resolved_harbor" in
    "$(readlink -f "$HARBOR_DIR")"/*) log "  harbor -> $_resolved_harbor" ;;
    *) die "harbor resolves to $_resolved_harbor but was installed from $HARBOR_DIR" ;;
esac
case "$_resolved_vllm" in
    "$(readlink -f "$VLLM_DIR")"/*) log "  vllm   -> $_resolved_vllm" ;;
    *) die "vllm resolves to $_resolved_vllm but was installed from $VLLM_DIR" ;;
esac

# ---------------- 9. summary ------------------------------------------------

log "======================================================================"
log "Setup complete."
log "  harbor           : $HARBOR_DIR  @ $(git -C "$HARBOR_DIR" rev-parse --short HEAD) ($HARBOR_REF)"
log "  verl             : $VERL_DIR    @ $(git -C "$VERL_DIR"   rev-parse --short HEAD) ($VERL_REF)"
log "  vllm             : $VLLM_DIR    @ $(git -C "$VLLM_DIR"   rev-parse --short HEAD) ($VLLM_REF)"
if [ "$SKIP_PATCHES" = "1" ]; then
    log "  monkey patches   : (skipped)"
else
    if [ -f "$HARBOR_PATCH" ]; then
        log "  harbor patch     : $HARBOR_PATCH"
    else
        log "  harbor patch     : (none)"
    fi
    log "  verl patch       : $VERL_PATCH"
    log "  vllm patch       : $VLLM_PATCH"
fi
log "  Lego-RL: $REPO_ROOT"
log "  venv             : $VENV_PATH"
if [ "$SKIP_REQUIREMENTS" = "1" ]; then
    log "  requirements     : (skipped)"
else
    log "  requirements     : $REQUIREMENTS_FILE"
fi
log ""
log "Next steps:"
log "  source \"$VENV_PATH/bin/activate\""
log "  cp scripts/train/_template.env scripts/train/configs/my_run.env"
log "  \$EDITOR scripts/train/configs/my_run.env   # fill CHANGEME values"
log "  bash scripts/train/train.sh --dry-run scripts/train/configs/my_run.env"
log "  bash scripts/train/train.sh scripts/train/configs/my_run.env"
log "  # eval/infer use the same pattern with scripts/eval/eval.sh or scripts/infer/infer.sh"
log "======================================================================"
