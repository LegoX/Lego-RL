#!/usr/bin/env bash

require_var() {
    local name="$1"
    if [ -z "${!name+x}" ] || [ -z "${!name}" ]; then
        echo "[FATAL] $name is required after sourcing config/templates" >&2
        exit 1
    fi
}

is_true() {
    case "${1:-}" in
        1|true|True|TRUE|yes|Yes|YES|on|On|ON) return 0 ;;
        *) return 1 ;;
    esac
}

print_cmd() {
    local arg
    for arg in "$@"; do
        printf '%q ' "$arg"
    done
    printf '
'
}

resolve_python() {
    local cand
    for cand in "$@"; do
        if "$cand" -c '' >/dev/null 2>&1; then
            printf '%s' "$cand"
            return 0
        fi
    done
    return 1
}

source_config_and_templates() {
    local module module_path

    set -a
    # shellcheck disable=SC1090
    source "$CONFIG_PATH"
    set +a

    if ! declare -p TEMPLATE_MODULES >/dev/null 2>&1; then
        echo "[FATAL] config must define TEMPLATE_MODULES" >&2
        exit 1
    fi

    set -a
    for module in "${TEMPLATE_MODULES[@]}"; do
        if [[ "$module" = /* ]]; then
            module_path="$module"
        else
            module_path="$TEMPLATE_ROOT/$module"
        fi
        [ -f "$module_path" ] || { echo "[FATAL] template module not found: $module_path" >&2; exit 1; }
        # shellcheck disable=SC1090
        source "$module_path"
    done
    set +a
}

validate_runtime_config() {
    local name
    local required_vars=(
        PROJECT_NAME EXP_NAME VENV_PATH MODEL_PATH SERVED_MODEL_NAME TOOL_CALL_PARSER
        N_CONCURRENT MAX_RETRIES EVAL_TEMPERATURE MAX_INPUT_TOKENS MAX_OUTPUT_TOKENS MAX_MODEL_LEN
        GEN_TP GPU_MEM_UTIL VLLM_PORT VLLM_BIND_HOST EVAL_ENABLE_EXPERT_PARALLEL EVAL_VLLM_READY_ITERS
        API_BASE LLM_BASE_URL LLM_API_KEY EVAL_AGENT_NAME
        HARBOR_AGENT_IMPORT_PATH HARBOR_AGENT_RUNTIME_IMAGE HARBOR_AGENT_RUNTIME_MOUNT_PATH HARBOR_AGENT_RUNTIME_IMAGE_SUBPATH
        HARBOR_ENVIRONMENT_IMPORT_PATH HARBOR_ENVIRONMENT_OVERRIDE_CPUS HARBOR_ENVIRONMENT_OVERRIDE_MEMORY_MB
        HARBOR_AGENT_MAX_TIMEOUT_SEC HARBOR_AGENT_OVERRIDE_TIMEOUT_SEC HARBOR_AGENT_MODEL_INFO
        HARBOR_MAX_RETRIES HARBOR_POD_NAME_PREFIX EVAL_LOG_DIR EVAL_JOBS_DIR EVAL_LOG VLLM_LOG JOB_CFG
    )

    for name in "${required_vars[@]}"; do
        require_var "$name"
    done

    if [ -n "$DATASET_PATH" ] && [ -n "$DATASET_NAME" ]; then
        echo "[FATAL] set only one of DATASET_PATH or DATASET_NAME" >&2
        exit 2
    fi
    if [ -z "$DATASET_PATH" ] && [ -z "$DATASET_NAME" ]; then
        echo "[FATAL] set DATASET_PATH or DATASET_NAME" >&2
        exit 2
    fi
    if [ "$GEN_TP" -lt 1 ]; then
        echo "[FATAL] GEN_TP must be positive" >&2
        exit 1
    fi
    if [ "$MAX_MODEL_LEN" -lt 1 ]; then
        echo "[FATAL] MAX_MODEL_LEN must be positive" >&2
        exit 1
    fi
}

initialize_runtime() {
    unset MODELING_BACKEND 2>/dev/null || true

    export LEGO_RL_ROOT="$REPO_ROOT"
    # SWE_LEGO_RL_ROOT is the pre-rename name, kept as an alias so an older config or
    # hydra file that still reads it keeps resolving.
    export SWE_LEGO_RL_ROOT="$REPO_ROOT"
    export PYTHONPATH="${EVAL_EXTRA_PYTHONPATH:+${EVAL_EXTRA_PYTHONPATH}:}${PYTHONPATH:+${PYTHONPATH}:}$REPO_ROOT/src"
    export SERVED_MODEL_NAME TOOL_CALL_PARSER LLM_BASE_URL LLM_API_KEY

    if [ -f "$VENV_PATH/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_PATH/bin/activate"
    fi

    PYTHON_BIN="$(resolve_python         "$VENV_PATH/bin/python3"         /opt/conda/bin/python3         /usr/local/bin/python3         /usr/bin/python3         python3 || true)"
    [ -n "$PYTHON_BIN" ] || { echo "[FATAL] no usable python3 found" >&2; exit 1; }
    export PYTHON_BIN VENV_PATH

    if [ "$DRY_RUN" != "1" ]; then
        mkdir -p "$EVAL_LOG_DIR" "$EVAL_JOBS_DIR" "$(dirname "$JOB_CFG")"
    fi

    echo "[config] $CONFIG_PATH"
    echo "[python] $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"
    echo "[vllm]   $(command -v vllm || true)"
    echo "[harbor] $(command -v harbor || true)"
}

print_launch_summary() {
    echo "=== Launching Harbor eval ==="
    echo "project:       $PROJECT_NAME"
    echo "experiment:    $EXP_NAME"
    echo "templates:     ${TEMPLATE_MODULES[*]}"
    echo "model:         $MODEL_PATH"
    echo "dataset:       ${DATASET_PATH:-${DATASET_NAME}}"
    echo "n_tasks:       ${N_TASKS:-<all>}"
    echo "concurrent:    $N_CONCURRENT"
    echo "max retries:   $MAX_RETRIES"
    echo "temperature:   $EVAL_TEMPERATURE"
    echo "context:       in=$MAX_INPUT_TOKENS out=$MAX_OUTPUT_TOKENS model_len=$MAX_MODEL_LEN"
    echo "tool parser:   $TOOL_CALL_PARSER"
    echo "vLLM topo:     tp=$GEN_TP expert_parallel=$EVAL_ENABLE_EXPERT_PARALLEL"
    echo "vLLM serve:    bind=${VLLM_BIND_HOST}:${VLLM_PORT} api=${API_BASE}"
    echo "eval log:      $EVAL_LOG"
    echo "vLLM log:      $VLLM_LOG"
    echo "job config:    $JOB_CFG"
}

print_final_environment() {
    echo "=== Final Environment ==="
    env | LC_ALL=C sort
}
