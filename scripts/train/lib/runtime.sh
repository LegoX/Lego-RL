#!/usr/bin/env bash

# Checkpoint-derived facts (model_is_moe / model_moe_verdict). Sourced here rather
# than from a template so it is available while the templates themselves are being
# sourced — verl/common.env derives ENABLE_R3 from it.
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/lib/model_traits.sh"

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

    export LEGO_RL_ROOT="$REPO_ROOT"
    # SWE_LEGO_RL_ROOT is the pre-rename name, kept as an alias so an older config or
    # hydra file that still reads it keeps resolving.
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

# Run scripts/lib/preflight.sh against this config.
#
# preflight predates the config/template refactor and still reads the canonical
# names the old launch scripts used (ENGINE, TOOL_PARSER, TRAIN_INDEX, ...). Rather
# than rename its ~30 rules — every one of them a pitfall that cost a real run — map
# the new names onto the old ones here, in one place, and keep preflight as the
# single source of truth for what a bad config looks like.
#
# USE_NEW_VERL has no equivalent any more: the new runner uses whatever verl the venv
# installed instead of prepending NEW_VERL_DIR to PYTHONPATH. Probe the actual tree
# for the router-replay module instead of asserting a flag that no longer exists.
run_preflight() {
    local verl_origin verl_root use_new_verl=0

    verl_origin="$("$PYTHON_BIN" -c 'import importlib.util as u; s=u.find_spec("verl"); print(s.origin or "")' 2>/dev/null || true)"
    if [ -n "$verl_origin" ]; then
        verl_root="$(dirname "$verl_origin")"
        [ -f "$verl_root/utils/veomni/router_replay.py" ] && use_new_verl=1
    fi

    PREFLIGHT_EMBEDDED=1 PF_KIND=train \
    ENGINE="$MODEL_ENGINE" \
    SCAFFOLD="${SCAFFOLD:-}" BACKEND="${BACKEND:-}" PROFILE="${PROFILE:-grpo}" \
    TOOL_PARSER="${TOOL_CALL_PARSER:-}" AGENT_NAME="${HARBOR_AGENT_NAME:-}" \
    NNODES="${NNODES:-}" N_NODES_TRAIN="${N_NODES_TRAIN:-}" N_NODES_ROLLOUT="${N_NODES_ROLLOUT:-}" \
    NGPUS_PER_NODE="${NGPUS_PER_NODE:-}" SP_SIZE="${SP_SIZE:-}" \
    MAX_PROMPT="${MAX_PROMPT:-}" MAX_RESP="${MAX_RESP:-}" \
    USE_NEW_VERL="$use_new_verl" NEW_VERL_DIR="${NEW_VERL_DIR:-}" ENABLE_R3="${ENABLE_R3:-}" \
    FUSED_KERNELS="${FUSED_KERNELS:-}" ACTIVATION_OFFLOAD="${ENABLE_ACTIVATION_OFFLOAD:-}" \
    LR_SCHEDULER="${LR_SCHEDULER:-}" ROLLOUT_IS="${ROLLOUT_IS:-}" \
    VAL_TIMEOUT="${HARBOR_VAL_AGENT_MAX_TIMEOUT_SEC:-}" \
    IMAGE_REGISTRY="${HARBOR_OPENSWE_IMAGE_REGISTRY:-}" INLINE_BUILD="${HARBOR_K8S_INLINE_BUILD:-}" \
    NYDUS_MIRROR="${HARBOR_NYDUS_MIRROR:-}" K8S_KUBECONFIG="${K8S_KUBECONFIG:-}" \
    MODEL_PATH="${MODEL_PATH:-}" TRAIN_INDEX="${TRAIN_FILES:-}" VAL_INDEX="${VAL_FILES:-}" \
        source "$REPO_ROOT/scripts/lib/preflight.sh" \
        || { echo "[FATAL] preflight failed — refusing to launch." >&2; exit 1; }
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

# The parameters that decide the outcome of this run, in one quotable block.
#
# Purely a formatter over values the runner has already resolved — it derives
# nothing, so what it prints is what will be launched. The format is stable so
# /rl:check can quote it verbatim instead of re-reading the config. Printed before
# preflight, so a config that fails its checks is still shown in full.
print_run_configuration() {
    local kv gpn train_world dp window trials moe
    kv() { printf '  %-12s %s\n' "$1" "$2"; }
    gpn="${NGPUS_PER_NODE:-8}"
    if [ "$TRAINING_MODE" = sync ]; then
        train_world=$(( ${NNODES:-0} * gpn ))
    else
        train_world=$(( ${N_NODES_TRAIN:-0} * gpn ))
    fi
    dp=$(( ${SP_SIZE:-1} > 0 ? train_world / ${SP_SIZE:-1} : 0 ))
    window=$(( ${MAX_PROMPT:-0} + ${MAX_RESP:-0} ))
    trials=$(( ${TRAIN_BSZ:-0} * ${N_RESP:-0} ))
    moe="$(model_moe_verdict "${MODEL_PATH:-}")"

    printf '\n═════════════════════ run configuration (train) ═════════════════════\n'
    kv config     "$CONFIG_PATH"
    kv experiment "project=$project_name  exp=$exp_name"
    kv axes       "mode=$TRAINING_MODE engine=$MODEL_ENGINE scaffold=${SCAFFOLD:-?} backend=${BACKEND:-?}"
    kv model      "${MODEL_PATH:-?}  [$moe]"
    kv ""         "served=${SERVED_MODEL_NAME:-?}  tool_parser=${TOOL_CALL_PARSER:-?}"
    kv data       "train=${TRAIN_FILES:-?}"
    kv ""         "val  =${VAL_FILES:-?}"
    if [ "$TRAINING_MODE" = sync ]; then
        kv topology "${NNODES:-?} nodes × $gpn gpu (colocated)  SP=${SP_SIZE:-?} → dp=$dp  vllm_TP=${GEN_TP:-?}"
    else
        kv topology "${NNODES:-?} nodes × $gpn gpu = train ${N_NODES_TRAIN:-?} / rollout ${N_NODES_ROLLOUT:-?}  SP=${SP_SIZE:-?} → dp=$dp  vllm_TP=${GEN_TP:-?}"
    fi
    kv batch      "${TRAIN_BSZ:-?} prompts × ${N_RESP:-?} resp = $trials trials/step  mini=${TRAIN_MINI_BSZ:-?}  dynamic_bsz=${USE_DYNAMIC_BSZ:-?}"
    kv context    "prompt=${MAX_PROMPT:-?} + resp=${MAX_RESP:-?} = $window"
    kv algorithm  "${ADV_ESTIMATOR:-?}  lr=${ACTOR_LR:-?}  lr_sched=${LR_SCHEDULER:-?}  kl_loss_coef=${KL_LOSS_COEF:-?}"
    kv rollout    "temp=${TEMPERATURE:-?} top_p=${TOP_P:-?}  val_temp=${VAL_TEMPERATURE:-?}  val_n=${VAL_N:-?}"
    kv ""         "R3=${ENABLE_R3:-?} (model is $moe)  rollout_is=${ROLLOUT_IS:-?}  max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-<verl default>}"
    kv schedule   "save_freq=${TRAINER_SAVE_FREQ:-?}  test_freq=${TRAINER_TEST_FREQ:-?}  val_before_train=${TRAINER_VAL_BEFORE_TRAIN:-?}  epochs=${TOTAL_EPOCHS:-?}"
    kv logs       "train=$TRAIN_LOG"
    kv ""         "trials=${HARBOR_TRIALS_DIR:-?}"
    printf '══════════════════════════════════════════════════════════════════════\n\n'
    unset -f kv
}

