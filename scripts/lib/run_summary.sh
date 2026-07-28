#!/usr/bin/env bash
# =============================================================================
# scripts/lib/run_summary.sh — "what am I about to launch?"  (train | eval | infer)
# =============================================================================
# Usage (sourced by a runner AFTER every axis / preset / harbor var is resolved):
#     source "$LIB/run_summary.sh"; print_run_summary train
#
# Prints one fenced block listing every parameter that decides the outcome of a
# run. Read-only and derivation-free: it only formats what the runner already
# resolved, so the block is exactly what will be launched — never a re-guess.
# The format is stable so /harbor:check can quote it verbatim.
# Printed BEFORE preflight, so a failing config is still shown in full.
# =============================================================================

_rs_kv()  { printf '  %-11s %s\n' "$1" "$2"; }
_rs_val() { [ -n "${1:-}" ] && printf '%s' "$1" || printf '%s' "${2:-<unset>}"; }
_rs_rule() { printf '%s\n' "──────────────────────────────────────────────────────────────────────"; }

print_run_summary() {
    local kind="${1:-train}"
    printf '\n'
    printf '═════════════════════ run configuration (%s) ═════════════════════\n' "$kind"
    _rs_kv config "$(_rs_val "${CONFIG:-}")"
    case "$kind" in
        train) _rs_summary_train ;;
        eval)  _rs_summary_eval  ;;
        infer) _rs_summary_infer ;;
        *)     _rs_kv kind "unknown kind '$kind' — nothing to summarize" ;;
    esac
    _rs_summary_common
    printf '══════════════════════════════════════════════════════════════════════\n\n'
}

# ---------------------------------------------------------------- train ------
_rs_summary_train() {
    local gpn="${NGPUS_PER_NODE:-8}" sp="${sp_size:-1}" tw dp win trials
    tw=$(( ${n_nodes_train:-0} * gpn ))
    [ "${TRAIN_MODE:-async}" = sync ] && tw=$(( ${NNODES:-0} * gpn ))
    dp=$(( sp > 0 ? tw / sp : 0 ))
    win=$(( ${max_prompt_length:-0} + ${max_response_length:-0} ))
    trials=$(( ${train_bsz:-0} * ${n_resp_per_prompt:-0} ))

    _rs_kv experiment "project=$(_rs_val "${project_name:-}")  exp=$(_rs_val "${exp_name:-}")"
    _rs_kv axes       "mode=${TRAIN_MODE:-?} engine=${ENGINE:-?} scaffold=${SCAFFOLD:-?} backend=${BACKEND:-?} profile=${PROFILE:-grpo}"
    _rs_kv model      "$(_rs_val "${MODEL_PATH:-}")"
    _rs_kv ""         "served=$(_rs_val "${SERVED_MODEL_NAME:-}")  preset=$(_rs_val "${MODEL_PRESET:-}" '<explicit>')"
    _rs_kv data       "train=$(_rs_val "${TRAIN_INDEX:-}")"
    _rs_kv ""         "val  =$(_rs_val "${VAL_INDEX:-}")"
    if [ "${TRAIN_MODE:-async}" = sync ]; then
        _rs_kv topology "${NNODES:-?} nodes × ${gpn} gpu = $(( ${NNODES:-0} * gpn ))  (colocated)  SP=${sp} → dp=${dp}  vllm_TP=${gen_tp:-?}"
    else
        _rs_kv topology "${NNODES:-?} nodes × ${gpn} gpu  = train ${n_nodes_train:-?} / rollout ${n_nodes_rollout:-?}  SP=${sp} → dp=${dp}  vllm_TP=${gen_tp:-?}"
    fi
    _rs_kv batch      "${train_bsz:-?} prompts × ${n_resp_per_prompt:-?} resp = ${trials} trials/step   mini=${train_prompt_mini_bsz:-?}  dynamic_bsz=${use_dynamic_bsz:-?}"
    _rs_kv context    "prompt=${max_prompt_length:-?} + resp=${max_response_length:-?} = ${win}   ppo_max_token_len/gpu=${actor_ppo_max_token_len:-?}"
    _rs_kv algorithm  "${adv_estimator:-?} / gspo  lr=${actor_lr:-?}  kl_loss_coef=${kl_loss_coef:-?}  clip=${clip_ratio_low:-?}/${clip_ratio_high:-?}"
    _rs_kv ""         "loss_agg=${loss_agg_mode:-?}  lr_sched=$([ "${TRAIN_MODE:-async}" = async ] && echo constant || echo '<yaml default>')"
    _rs_kv rollout    "temp=${temperature:-?} top_p=${top_p:-?} top_k=${top_k:-?}   val_temp=${val_temperature:-?}  gpu_mem_util=${gpu_mem_util:-?}"
    _rs_kv ""         "R3=${enable_rollout_routing_replay:-?}  rollout_is=${rollout_is:-?}(thr=${rollout_is_threshold:-?})  max_num_seqs=${rollout_max_num_seqs:-?}"
    _rs_kv filters    "trajectory_filter=${traj_filter_enable:-?} drop=[${traj_filter_drop_reasons:-}]  seq_dist_metrics=${seq_dist_metrics:-?}"
    [ "${TRAIN_MODE:-async}" = async ] && \
    _rs_kv async      "staleness=${staleness_threshold:-?} partial_rollout=${partial_rollout:-?} require_batches=${require_batches:-?} sync_every=${trigger_parameter_sync_step:-?} steps"
    _rs_kv cadence    "epochs=${total_epochs:-?}  save_freq=${save_freq:-?}  test_freq=${test_freq:-?}  val_before_train=${val_before_train:-?}"
    _rs_kv resume     "$(_rs_val "${RESUME_FROM:-}" '<auto — resumes latest ckpt in the exp dir if any>')"
    _rs_kv logs       "$(_rs_val "${REPO_ROOT:-.}")/logs/${exp_name:-<exp>}.log"
}

# ----------------------------------------------------------------- eval ------
_rs_summary_eval() {
    _rs_kv experiment "project=$(_rs_val "${project_name:-}")  exp=$(_rs_val "${exp_name:-}")"
    _rs_kv axes       "scaffold=${SCAFFOLD:-?} backend=${BACKEND:-?}  (plain vLLM, real weights — not verl val_only)"
    _rs_kv model      "$(_rs_val "${MODEL_PATH:-}")"
    _rs_kv ""         "served=$(_rs_val "${SERVED_MODEL_NAME:-}")  preset=$(_rs_val "${MODEL_PRESET:-}" '<explicit>')"
    _rs_kv dataset    "$(_rs_val "${DATASET_PATH:-}" "${DATASET_NAME:-<unset>}")   n_tasks=$(_rs_val "${N_TASKS:-}" '<all>')"
    _rs_kv serving    "TP=${GEN_TP:-?}  expert_parallel=${EVAL_ENABLE_EXPERT_PARALLEL:-?}  gpu_mem=${GPU_MEM_UTIL:-?}  port=${VLLM_PORT:-?}"
    _rs_kv ""         "extra_args=$(_rs_val "${EVAL_VLLM_EXTRA_ARGS:-}" '<none>')"
    _rs_kv context    "in=${MAX_INPUT_TOKENS:-?} + out=${MAX_OUTPUT_TOKENS:-?}  max_model_len=${MAX_MODEL_LEN:-?}"
    _rs_kv sampling   "temp=${EVAL_TEMPERATURE:-?}   (temp=0 greedy loops on long agent tasks — 0.7 is the tested default)"
    _rs_kv concurrency "n_concurrent=${N_CONCURRENT:-?}  max_retries=${MAX_RETRIES:-?}  per-task budget=${HARBOR_AGENT_MAX_TIMEOUT_SEC:-?}s"
    _rs_kv endpoint   "$(_rs_val "${LLM_BASE_URL:-}")"
    _rs_kv logs       "$(_rs_val "${REPO_ROOT:-.}")/logs/${exp_name:-<exp>}_vllm.log"
}

# ---------------------------------------------------------------- infer ------
_rs_summary_infer() {
    _rs_kv model      "$(_rs_val "${MODEL_PATH:-}")"
    _rs_kv ""         "served=$(_rs_val "${SERVED_MODEL_NAME:-}")  preset=$(_rs_val "${MODEL_PRESET:-}" '<explicit>')"
    _rs_kv axes       "scaffold=${SCAFFOLD:-?} backend=${BACKEND:-?}"
    _rs_kv data       "index=$(_rs_val "${TRAIN_INDEX:-}")"
    _rs_kv ""         "shard=$(_rs_val "${INSTANCES_FILE:-}" '<all instances — this node takes the whole index>')"
    _rs_kv output     "results=$(_rs_val "${RESULTS_DIR:-}")"
    _rs_kv ""         "index  =$(_rs_val "${OUTPUT_INDEX:-}")   (completed trials are skipped on resume)"
    _rs_kv rollout    "n_trials=${N_TRIALS:-?}  n_concurrent=${N_CONCURRENT:-?}  temp=${TEMPERATURE:-?}  trial_hard_timeout=${TRIAL_HARD_TIMEOUT_SEC:-?}s"
    _rs_kv "vllm topo" "tp=${GEN_TP:-?} dp=${GEN_DP:-?} nnodes=${VLLM_NNODES:-?} local_dp=${VLLM_DP_LOCAL:-?} node_rank=${VLLM_NODE_RANK:-?}$([ "${VLLM_NODE_RANK:-0}" = 0 ] && echo ' (HEAD)' || echo ' (worker — serves DP shard only)')"
    _rs_kv ""         "head=${VLLM_HEAD_IP:-?}:${VLLM_PORT:-?}  gpu_mem=${GPU_MEM_UTIL:-?}  max_model_len=${VLLM_MAX_MODEL_LEN:-?} (prompt ${MAX_PROMPT_LEN:-?} / resp ${MAX_RESPONSE_LEN:-?})"
    _rs_kv retries    "env_start_max_attempts=${HARBOR_ENV_START_MAX_ATTEMPTS:-?}  (1 = no env-start retry, keeps the long tail bounded)"
    _rs_kv logs       "$(_rs_val "${EVAL_LOG:-}")"
}

# ---------------------------------------------------------------- common -----
# Everything below decides whether the agent can actually reach an environment.
# Same lines for all three kinds — these are the fields that cause silent
# infra-shaped failures (reward=0, env_setup_failed, ImagePullBackOff).
_rs_summary_common() {
    _rs_rule
    _rs_kv agent      "tool_parser=$(_rs_val "${HARBOR_TOOL_PARSER:-}")  agent_name=$(_rs_val "${HARBOR_AGENT_NAME:-}")  loop=$(basename "$(_rs_val "${AGENT_LOOP_CONFIG:-}" '<n/a>')")"
    _rs_kv ""         "max_iterations=${HARBOR_AGENT_MAX_ITERATIONS:-?}  timeout=${HARBOR_AGENT_MAX_TIMEOUT_SEC:-?}s  val_timeout=$(_rs_val "${HARBOR_VAL_AGENT_MAX_TIMEOUT_SEC:-}" '<n/a>')s  retries=${HARBOR_MAX_RETRIES:-?}"
    [ -n "${NUM_WORKERS:-}" ] && _rs_kv "" "num_workers=${NUM_WORKERS}"
    if [ "$(printf '%s' "${BACKEND:-k8s}" | tr '[:upper:]' '[:lower:]')" = docker ]; then
        _rs_kv backend "docker daemon=$(_rs_val "${DOCKER_HOST:-}" '<local>')  force_build=${HARBOR_ENVIRONMENT_FORCE_BUILD:-?}"
    else
        _rs_kv images "registry=$(_rs_val "${HARBOR_OPENSWE_IMAGE_REGISTRY:-}" '<none — in-pod build>')  inline_build=$(_rs_val "${HARBOR_K8S_INLINE_BUILD:-}" 'false')  nydus=$(_rs_val "${HARBOR_NYDUS_MIRROR:-}" '<none — val swebench images must be pullable>')"
        _rs_kv k8s    "kubeconfig=$(_rs_val "${K8S_KUBECONFIG:-}")  pod_prefix=$(_rs_val "${HARBOR_POD_NAME_PREFIX:-}")  cpus=${HARBOR_ENVIRONMENT_OVERRIDE_CPUS:-?} mem=${HARBOR_ENVIRONMENT_OVERRIDE_MEMORY_MB:-?}MB"
    fi
    _rs_kv trials     "$(_rs_val "${HARBOR_TRIALS_DIR:-}")"
    _rs_kv code       "python=$(_rs_val "${PYTHON_BIN:-}")  use_new_verl=${USE_NEW_VERL:-?}  verl_dir=$(_rs_val "${NEW_VERL_DIR:-}" '<installed verl>')"
    _rs_kv wandb      "$([ -n "${WANDB_API_KEY:-}" ] && echo 'key=set' || echo 'key=MISSING')  mode=$(_rs_val "${WANDB_MODE:-}" online)"
}
