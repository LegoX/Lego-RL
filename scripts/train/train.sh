#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
Usage:
  scripts/train/train.sh [--dry-run|--preflight-only] path_to_config

The config is sourced first, then its TEMPLATE_MODULES are sourced from
scripts/templates in order. Config values should be explicit experiment choices;
templates provide module defaults and derived values.

  --preflight-only   resolve the config, run scripts/lib/preflight.sh, exit.
                     Equivalent to PREFLIGHT_ONLY=1. Launches nothing.
  --dry-run          the above, plus the fully expanded launch command.
USAGE
}

DRY_RUN=0
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run)        DRY_RUN=1; shift ;;
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        *) break ;;
    esac
done

[ "$#" -eq 1 ] || { usage; exit 2; }

SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEMPLATE_ROOT="$REPO_ROOT/scripts/templates"
TRAIN_LIB_DIR="$SCRIPT_DIR/lib"
CONFIG_PATH="$1"
if [[ "$CONFIG_PATH" != /* ]]; then
    CONFIG_PATH="$(readlink -f "$CONFIG_PATH")"
fi
[ -f "$CONFIG_PATH" ] || { echo "[FATAL] config not found: $CONFIG_PATH" >&2; exit 1; }

export REPO_ROOT TEMPLATE_ROOT TRAIN_LIB_DIR CONFIG_PATH DRY_RUN PREFLIGHT_ONLY

# shellcheck disable=SC1091
source "$TRAIN_LIB_DIR/runtime.sh"
# shellcheck disable=SC1091
source "$TRAIN_LIB_DIR/hydra_args.sh"
# shellcheck disable=SC1091
source "$TRAIN_LIB_DIR/ray.sh"

source_config_and_templates
validate_runtime_config
initialize_runtime
print_launch_summary
print_final_environment
print_run_configuration

# Config-only health check. Runs after the summary so a failing config is still
# shown in full, and before anything expensive: every rule here is a mistake that
# otherwise surfaces minutes into a run, or not at all.
run_preflight
if [ "$PREFLIGHT_ONLY" = "1" ]; then
    echo "[preflight-only] checks passed; not launching (PREFLIGHT_ONLY=1)."
    exit 0
fi

build_hydra_args
build_verl_command

echo "=== Launch Command ==="
print_cmd "${cmd[@]}" "${hydra_args[@]}"

if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] not launching"
    exit 0
fi

rm -f "$VLLM_LOG"
start_ray_cluster_or_exit_worker

{
    "${cmd[@]}" "${hydra_args[@]}"
} 2>&1 | tee "$TRAIN_LOG" >(stdbuf -oL -eL grep --line-buffered -E "vLLMHttpServer|throughput|tokens/s|token/s|prompt throughput|generation throughput" >> "$VLLM_LOG")
