#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
Usage:
  scripts/infer/infer.sh [--dry-run] path_to_config

The config is sourced first, then its TEMPLATE_MODULES are sourced from
scripts/templates in order. Config values should be explicit experiment choices;
templates provide module defaults and derived values.
USAGE
}

DRY_RUN=0
if [ "$#" -gt 0 ] && [ "$1" = "--dry-run" ]; then
    DRY_RUN=1
    shift
fi

[ "$#" -eq 1 ] || { usage; exit 2; }

SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEMPLATE_ROOT="$REPO_ROOT/scripts/templates"
INFER_LIB_DIR="$SCRIPT_DIR/lib"
CONFIG_PATH="$1"
if [[ "$CONFIG_PATH" != /* ]]; then
    CONFIG_PATH="$(readlink -f "$CONFIG_PATH")"
fi
[ -f "$CONFIG_PATH" ] || { echo "[FATAL] config not found: $CONFIG_PATH" >&2; exit 1; }

export REPO_ROOT TEMPLATE_ROOT INFER_LIB_DIR CONFIG_PATH DRY_RUN

# shellcheck disable=SC1091
source "$INFER_LIB_DIR/runtime.sh"
# shellcheck disable=SC1091
source "$INFER_LIB_DIR/commands.sh"
# shellcheck disable=SC1091
source "$INFER_LIB_DIR/vllm.sh"

source_config_and_templates
validate_runtime_config
initialize_runtime
print_launch_summary
print_final_environment

build_commands
print_launch_commands

if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] not launching"
    exit 0
fi

preflight_vllm_port
start_vllm
wait_for_vllm
run_inference
