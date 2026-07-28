#!/usr/bin/env bash

require_var() {
    local name="$1"
    if [ -z "${!name+x}" ] || [ -z "${!name}" ]; then
        echo "[FATAL] $name is required after sourcing config/templates" >&2
        exit 1
    fi
}

print_cmd() {
    local arg
    for arg in "$@"; do
        printf '%q ' "$arg"
    done
    printf '\n'
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
        INDEX_FILE RESULTS_DIR OUTPUT_INDEX N_TRIALS N_CONCURRENT TEMPERATURE TRIAL_HARD_TIMEOUT_SEC
        GEN_TP GPUS_PER_NODE VLLM_PORT VLLM_BIND_HOST VLLM_API_SERVER_COUNT
        VLLM_MAX_MODEL_LEN GPU_MEM_UTIL PREFIX_CACHING_HASH_ALGO API_BASE AGENT_MODEL_NAME
        INFER_LOG VLLM_LOG
    )

    for name in "${required_vars[@]}"; do
        require_var "$name"
    done

    if [ "$GEN_TP" -lt 1 ]; then
        echo "[FATAL] GEN_TP must be positive" >&2
        exit 1
    fi
    if [ "$GEN_TP" -gt "$GPUS_PER_NODE" ]; then
        echo "[FATAL] GEN_TP=$GEN_TP exceeds GPUS_PER_NODE=$GPUS_PER_NODE" >&2
        exit 1
    fi
    if [ "$VLLM_MAX_MODEL_LEN" -lt 1 ]; then
        echo "[FATAL] VLLM_MAX_MODEL_LEN must be positive" >&2
        exit 1
    fi
}

initialize_runtime() {
    export SWE_LEGO_RL_ROOT="$REPO_ROOT"
    export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$REPO_ROOT/src"
    export SERVED_MODEL_NAME TOOL_CALL_PARSER LLM_BASE_URL LLM_API_KEY

    if [ -f "$VENV_PATH/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_PATH/bin/activate"
    fi

    PYTHON_BIN="$(resolve_python \
        "$VENV_PATH/bin/python3" \
        /opt/conda/bin/python3 \
        /usr/local/bin/python3 \
        /usr/bin/python3 \
        python3 || true)"
    [ -n "$PYTHON_BIN" ] || { echo "[FATAL] no usable python3 found" >&2; exit 1; }
    export PYTHON_BIN VENV_PATH

    if [ "$DRY_RUN" != "1" ]; then
        mkdir -p "$INFER_LOG_DIR"
    fi

    echo "[config] $CONFIG_PATH"
    echo "[python] $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"
    echo "[vllm]   $(command -v vllm || true)"
}

print_launch_summary() {
    echo "=== Launching Harbor infer ==="
    echo "project:       $PROJECT_NAME"
    echo "experiment:    $EXP_NAME"
    echo "templates:     ${TEMPLATE_MODULES[*]}"
    echo "model:         $MODEL_PATH"
    echo "index:         $INDEX_FILE"
    echo "instances:     ${INSTANCES_FILE:-<all>}"
    echo "results:       $RESULTS_DIR"
    echo "output index:  $OUTPUT_INDEX"
    echo "trials:        $N_TRIALS"
    echo "concurrent:    $N_CONCURRENT"
    echo "temperature:   $TEMPERATURE"
    echo "tool parser:   $TOOL_CALL_PARSER"
    echo "vLLM topo:     tp=$GEN_TP"
    echo "vLLM serve:    bind=${VLLM_BIND_HOST}:${VLLM_PORT} api=${API_BASE}"
    echo "infer log:     $INFER_LOG"
    echo "vLLM log:      $VLLM_LOG"
}

print_final_environment() {
    echo "=== Final Environment ==="
    env | LC_ALL=C sort
}
