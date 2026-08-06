#!/usr/bin/env bash
# lib/common_env.sh — process env + venv + python + verl selection (shared by train/infer/eval)
# Contract: caller sets REPO_ROOT first, then sources this file. Pitfalls & rationale: the Troubleshooting section of the docs site
# shellcheck disable=SC1090,SC1091

# Network / logging (identical on all nodes, else multi-node NCCL uses the wrong NIC)
export GLOO_SOCKET_IFNAME=eth0 TP_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}" DOCKER_BUILDKIT=1
export TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
# Disable proxy mismatch-logprob recompute (fragmented vLLM allocation → weight-sync NCCL starvation → step1 OOM)
export HARBOR_RECOMPUTE_MISMATCH_LOGPROBS="${HARBOR_RECOMPUTE_MISMATCH_LOGPROBS:-0}"

# site/infra layer: cluster-specific (registry / nydus / mounts / kubeconfig / acceleration / MODEL_ROOT / NEW_VERL_DIR).
# Default = portable vanilla; local acceleration only turns on when the site file exists. For another cluster, copy from site.example.env first.
# Multiple clusters can coexist as lib/site.<name>.env; select one per run via SITE_ENV_FILE (absolute or repo-relative path).
SITE_ENV_FILE="${SITE_ENV_FILE:-$REPO_ROOT/scripts/lib/site.env}"
[ -f "$SITE_ENV_FILE" ] || SITE_ENV_FILE="$REPO_ROOT/$SITE_ENV_FILE"
[ -f "$SITE_ENV_FILE" ] && source "$SITE_ENV_FILE"

# wandb key: environment > optional local secrets file. Never commit a key.
# Put `export WANDB_API_KEY=...` in scripts/lib/.secrets.sh (gitignored) if you
# do not want to export it in your shell.
[ -f "$REPO_ROOT/scripts/lib/.secrets.sh" ] && source "$REPO_ROOT/scripts/lib/.secrets.sh"

export LEGO_RL_ROOT="${LEGO_RL_ROOT:-$REPO_ROOT}"
# SWE_LEGO_RL_ROOT is the pre-rename name, kept as an alias so an older config or
# hydra file that still reads it keeps resolving.
export SWE_LEGO_RL_ROOT="${SWE_LEGO_RL_ROOT:-$REPO_ROOT}"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

# new-verl selection: veomni router_replay / R3 / async fixes live only in this worktree. Default ON.
USE_NEW_VERL="${USE_NEW_VERL:-1}"
if [ "$USE_NEW_VERL" = "1" ]; then
    # Derived from REPO_ROOT so a fresh checkout anywhere works: this must resolve to
    # the same tree scripts/setup_env.sh installs with -e (its VERL_DIR default is
    # $WORKSPACE_ROOT/verl-swe_agent_opd_dev, WORKSPACE_ROOT=$(dirname REPO_ROOT)).
    # Override only if you deliberately keep the two apart -- setup_env.sh warns when
    # this names a different tree than the one it installed.
    NEW_VERL_DIR="${NEW_VERL_DIR:-$(dirname "$REPO_ROOT")/verl-swe_agent_opd_dev}"
    export PYTHONPATH="$NEW_VERL_DIR:$PYTHONPATH"
fi

# venv activation + python resolution (venv → conda → system)
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv}"
[ -f "$VENV_PATH/bin/activate" ] && source "$VENV_PATH/bin/activate"
_resolve_python() { local c; for c in "$@"; do "$c" -c '' >/dev/null 2>&1 && { printf '%s' "$c"; return 0; }; done; return 1; }
PYTHON_BIN="${PYTHON_BIN:-$(_resolve_python \
    "$VENV_PATH/bin/python3" /opt/conda/bin/python3 /usr/local/bin/python3 /usr/bin/python3 python3 || true)}"
[ -n "${PYTHON_BIN:-}" ] || { echo "[FATAL] no usable python3 found" >&2; exit 1; }
export PYTHON_BIN VENV_PATH USE_NEW_VERL
echo "[python] $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"
echo "[verl]   $("$PYTHON_BIN" -c 'import importlib.util as u;print(u.find_spec("verl").origin)' 2>&1)"
