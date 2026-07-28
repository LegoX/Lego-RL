#!/usr/bin/env bash

require_var() {
    local name="$1"
    if [ -z "${!name+x}" ] || [ -z "${!name}" ]; then
        echo "[FATAL] $name is required after sourcing config/templates" >&2
        exit 1
    fi
}

var_is_set() {
    local name="$1"
    [ -n "${!name+x}" ] && [ -n "${!name}" ]
}

is_true() {
    case "$1" in
        1|true|True|TRUE|yes|Yes|YES|on|On|ON) return 0 ;;
        *) return 1 ;;
    esac
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

    # Source config + selected templates with allexport so Ray workers inherit the
    # final Harbor/K8s/runtime environment. set -a also exports variables assigned
    # by template defaults such as : "${VAR:=default}".
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
        PROJECT_NAME EXP_NAME TRAINER_PROJECT_NAME TRAINER_EXPERIMENT_NAME
        VERL_ENTRY_MODULE VERL_CONFIG_NAME TRAINING_MODE MODEL_ENGINE
        MODEL_PATH TRAIN_FILES VAL_FILES VENV_PATH HARBOR_LOG_DIR
        NNODES NGPUS_PER_NODE RAY_PORT RAY_DASHBOARD_PORT RAY_OBJECT_STORE_MEMORY
        SERVED_MODEL_NAME TOOL_CALL_PARSER
    )

    for name in "${required_vars[@]}"; do
        require_var "$name"
    done
}

initialize_runtime() {
    project_name="$TRAINER_PROJECT_NAME"
    exp_name="$TRAINER_EXPERIMENT_NAME"
    export project_name exp_name HARBOR_EXP_NAME="$exp_name"
    export NGPUS_PER_NODE RAY_PORT RAY_DASHBOARD_PORT RAY_OBJECT_STORE_MEMORY \
           SERVED_MODEL_NAME TOOL_CALL_PARSER

    export SWE_LEGO_RL_ROOT="$REPO_ROOT"
    export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$REPO_ROOT/src"

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

    echo "[config] $CONFIG_PATH"
    echo "[python] $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"
    echo "[verl]   $("$PYTHON_BIN" -c 'import importlib.util as u; spec=u.find_spec("verl"); print(spec.origin if spec else "<not found>")' 2>&1)"

    if [ "$DRY_RUN" != "1" ] && [ "$MODEL_ENGINE" = "veomni" ]; then
        "$PYTHON_BIN" -c 'import veomni' 2>/dev/null || {
            echo "[FATAL] 'veomni' not importable under $PYTHON_BIN" >&2
            exit 1
        }
    fi

    require_var TRAIN_LOG
    require_var VLLM_LOG
    export TRAIN_LOG VLLM_LOG
    if [ "$DRY_RUN" != "1" ]; then
        mkdir -p "$HARBOR_LOG_DIR"
    fi
}


redact_env_value_if_needed() {
    local name="$1" value="$2"
    case "$name" in
        *API_KEY*|*SECRET*|*PASSWORD*|*PASSWD*|*CREDENTIAL*)
            if [ -n "$value" ]; then
                printf '<redacted>'
            fi
            ;;
        *)
            printf '%s' "$value"
            ;;
    esac
}

print_final_environment() {
    local line name value rendered

    echo "=== Final Environment ==="
    while IFS= read -r line; do
        name="${line%%=*}"
        value="${line#*=}"
        rendered="$(redact_env_value_if_needed "$name" "$value")"
        printf '%s=%s\n' "$name" "$rendered"
    done < <(env | LC_ALL=C sort)
}

print_launch_summary() {
    echo "=== Launching Harbor verl training ==="
    echo "project:       $project_name"
    echo "experiment:    $exp_name"
    echo "mode/engine:   $TRAINING_MODE/$MODEL_ENGINE"
    echo "templates:     ${TEMPLATE_MODULES[*]}"
    echo "train log:     $TRAIN_LOG"
    echo "vLLM log:      $VLLM_LOG"
}

