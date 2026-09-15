#!/usr/bin/env bash

init_hydra_args() {
    hydra_args=()
}

add() {
    hydra_args+=("$1=$2")
}

add_plus() {
    hydra_args+=("+$1=$2")
}

# Set a key whether or not it already exists in the config tree. Plain `add`
# fails on keys hydra has never seen, `add_plus` fails on keys it has.
add_force() {
    hydra_args+=("++$1=$2")
}

add_if_set() {
    local key="$1" name="$2"
    if var_is_set "$name"; then
        hydra_args+=("$key=${!name}")
    fi
    return 0
}

append_common_hydra_args() {
    if [ "$MODEL_ENGINE" = "veomni" ]; then
        hydra_args+=("model_engine=veomni")
    fi

    add_plus actor_rollout_ref.rollout.enable_sleep_mode "$ENABLE_SLEEP_MODE"
    add actor_rollout_ref.actor.clip_ratio_high "$CLIP_HIGH"
    add actor_rollout_ref.actor.clip_ratio_low "$CLIP_LOW"
    add actor_rollout_ref.actor.entropy_coeff "$ENTROPY_COEFF"
    add actor_rollout_ref.actor.kl_loss_coef "$KL_LOSS_COEF"
    add actor_rollout_ref.actor.kl_loss_type "$KL_LOSS_TYPE"
    add actor_rollout_ref.actor.loss_agg_mode "$LOSS_AGG_MODE"
    add actor_rollout_ref.actor.optim.lr "$ACTOR_LR"
    add actor_rollout_ref.actor.optim.lr_scheduler_type "$LR_SCHEDULER"
    # Resume restores optimizer + lr_scheduler state wholesale, so a config LR
    # change is silently clobbered on restart (observed on the SAO r7 restart:
    # critic/lr stayed 5e-6 despite CRITIC_LR=2.5e-6). Set this to
    # [model,optimizer] to skip the 'extra' payload (lr_scheduler + rng): the
    # freshly built scheduler then re-asserts the configured LR. Adam moments
    # still load; only rng reproducibility is lost.
    add_if_set actor_rollout_ref.actor.checkpoint.load_contents ACTOR_CKPT_LOAD_CONTENTS
    add actor_rollout_ref.actor.policy_loss.loss_mode "$POLICY_LOSS_MODE"
    add actor_rollout_ref.actor.ppo_max_token_len_per_gpu "$ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU"
    add actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu "$ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU"
    add actor_rollout_ref.actor.ppo_mini_batch_size "$ACTOR_PPO_MINI_BATCH_SIZE"
    add actor_rollout_ref.actor.use_dynamic_bsz "$USE_DYNAMIC_BSZ"
    add actor_rollout_ref.actor.use_kl_loss "$USE_KL_LOSS"
    add actor_rollout_ref.hybrid_engine "$HYBRID_ENGINE"
    add actor_rollout_ref.nccl_timeout "$NCCL_TIMEOUT"
    add actor_rollout_ref.model.enable_gradient_checkpointing "$ENABLE_GRADIENT_CHECKPOINTING"
    add actor_rollout_ref.model.enable_activation_offload "$ENABLE_ACTIVATION_OFFLOAD"
    add actor_rollout_ref.model.use_fused_kernels "$FUSED_KERNELS"
    add actor_rollout_ref.model.path "$MODEL_PATH"
    add actor_rollout_ref.model.use_remove_padding "$USE_REMOVE_PADDING"

    add actor_rollout_ref.ref.log_prob_use_dynamic_bsz "$REF_LOG_PROB_USE_DYNAMIC_BSZ"
    add actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu "$REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
    add actor_rollout_ref.ref.log_prob_max_token_len_per_gpu "$REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU"

    add actor_rollout_ref.rollout.agent.agent_loop_config_path "$AGENT_LOOP_CONFIG_PATH"
    add actor_rollout_ref.rollout.agent.num_workers "$AGENT_NUM_WORKERS"
    add actor_rollout_ref.rollout.dtype "$ROLLOUT_DTYPE"
    add actor_rollout_ref.rollout.calculate_log_probs "$CALCULATE_LOG_PROBS"
    add actor_rollout_ref.rollout.enforce_eager "$ENFORCE_EAGER"
    add actor_rollout_ref.rollout.enable_chunked_prefill "$ENABLE_CHUNKED_PREFILL"
    add actor_rollout_ref.rollout.engine_kwargs.vllm.enable-expert-parallel "$VLLM_ENABLE_EXPERT_PARALLEL"
    add actor_rollout_ref.rollout.engine_kwargs.vllm.served-model-name "$SERVED_MODEL_NAME"
    add actor_rollout_ref.rollout.engine_kwargs.vllm.tool-call-parser "$TOOL_CALL_PARSER"
    add actor_rollout_ref.rollout.disable_log_stats "$DISABLE_LOG_STATS"
    add actor_rollout_ref.rollout.gpu_memory_utilization "$GPU_MEM_UTIL"
    add actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu "$ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU"
    add actor_rollout_ref.rollout.log_prob_use_dynamic_bsz "$ROLLOUT_LOG_PROB_USE_DYNAMIC_BSZ"
    add actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu "$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
    add actor_rollout_ref.rollout.max_model_len "$ROLLOUT_MAX_MODEL_LEN"
    # Opt-in only: unset leaves verl's own default (1024) untouched for existing configs.
    add_if_set actor_rollout_ref.rollout.max_num_seqs ROLLOUT_MAX_NUM_SEQS
    add actor_rollout_ref.rollout.enable_rollout_routing_replay "$ENABLE_ROLLOUT_ROUTING_REPLAY"
    add actor_rollout_ref.rollout.mode "$ROLLOUT_MODE"
    add actor_rollout_ref.rollout.n "$N_RESP"
    add actor_rollout_ref.rollout.name "$ROLLOUT_NAME"
    add actor_rollout_ref.rollout.temperature "$TEMPERATURE"
    add actor_rollout_ref.rollout.tensor_model_parallel_size "$GEN_TP"
    add actor_rollout_ref.rollout.top_k "$TOP_K"
    add actor_rollout_ref.rollout.top_p "$TOP_P"
    add actor_rollout_ref.rollout.val_kwargs.do_sample "$VAL_DO_SAMPLE"
    add actor_rollout_ref.rollout.val_kwargs.n "$VAL_N"
    add actor_rollout_ref.rollout.val_kwargs.temperature "$VAL_TEMPERATURE"
    add actor_rollout_ref.rollout.val_kwargs.top_k "$VAL_TOP_K"
    add actor_rollout_ref.rollout.val_kwargs.top_p "$VAL_TOP_P"

    add algorithm.adv_estimator "$ADV_ESTIMATOR"
    add algorithm.gae_whiten_advantages "$GAE_WHITEN_ADVANTAGES"
    add algorithm.gamma "$GAMMA"
    add algorithm.kl_ctrl.kl_coef "$KL_COEF"
    add algorithm.lam "$LAM"
    add algorithm.lam_critic "$LAM_CRITIC"
    add algorithm.length_adaptive_lam_alpha "$LENGTH_ADAPTIVE_LAM_ALPHA"
    add algorithm.rollout_correction.bypass_mode "$ROLLOUT_CORRECTION_BYPASS"
    add algorithm.rollout_correction.loss_type "$ROLLOUT_CORRECTION_LOSS_TYPE"
    add algorithm.rollout_correction.rollout_is "$ROLLOUT_IS"
    add algorithm.rollout_correction.rollout_is_threshold "$ROLLOUT_IS_THRESHOLD"
    add algorithm.rollout_correction.seq_dist_metrics "$SEQ_DIST_METRICS"

    # >>> LOAD-BEARING MIRROR. Do not delete as "redundant". <<<
    # algorithm.rollout_correction is NOT what the policy loss reads. In the actor,
    # compute_policy_loss_bypass_mode() (core_algos.py) does:
    #     rollout_corr_config = config.policy_loss.get("rollout_correction", None)
    # and PolicyLossConfig (workers/config/actor.py) declares
    #     rollout_correction: RolloutCorrectionConfig = field(default_factory=...)
    # so that key ALWAYS resolves and the "not configured" ValueError never fires.
    # apply_bypass_mode() does copy algorithm.rollout_correction into
    # actor.policy_loss.rollout_correction -- but it runs in the TRAINER process,
    # while the loss runs inside the actor WorkerDict ray actors, which froze their
    # config at construction and never see that mutation.
    # Without these lines a bypass-mode run silently trains on
    # RolloutCorrectionConfig's DEFAULTS -- rollout_is="sequence",
    # rollout_is_threshold=2.0 (TIS, not IcePop), loss_type="ppo_clip" -- i.e. not
    # SAO's DIS at all, with no error anywhere and metrics that look plausible
    # because they describe the wrong quantity. Diagnosed 2026-08-09 on the 8B SAO
    # smoke (rollout_is_mean 0.023 -> 0.9997 after the fix). `++` because these
    # keys are absent from the yaml tree.
    add_force actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode "$ROLLOUT_CORRECTION_BYPASS"
    add_force actor_rollout_ref.actor.policy_loss.rollout_correction.loss_type "$ROLLOUT_CORRECTION_LOSS_TYPE"
    add_force actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is "$ROLLOUT_IS"
    add_force actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_threshold "$ROLLOUT_IS_THRESHOLD"
    add_force actor_rollout_ref.actor.policy_loss.rollout_correction.seq_dist_metrics "$SEQ_DIST_METRICS"

    add algorithm.trajectory_filter.enable "$TRAJ_FILTER_ENABLE"
    # Trajectory filter v6 (patches/verl_sao_7aed6b23.patch): one drop_reasons
    # list replaces the per-reason booleans (filter_overlong, ...) of the
    # earlier schema. TRAJ_FILTER_FILTER_OVERLONG is no longer read.
    add algorithm.trajectory_filter.drop_reasons "$TRAJ_FILTER_DROP_REASONS"
    add algorithm.use_kl_in_reward "$USE_KL_IN_REWARD"

    # data.gen_batch_size and the top-level rollout.* group exist only in
    # lego_rl_fully_async_*.yaml. lego_rl_sync.yaml derives from ppo_trainer,
    # which has neither, so emitting them in sync mode makes Hydra abort with
    # "Could not override ... No match in the config".
    if [ "$TRAINING_MODE" = async ]; then
        add data.gen_batch_size "$GEN_PROMPT_BSZ"
    fi
    add data.max_prompt_length "$MAX_PROMPT"
    add data.max_response_length "$MAX_RESP"
    add data.prompt_key "$PROMPT_KEY"
    add data.return_raw_chat "$RETURN_RAW_CHAT"
    add data.train_batch_size "$TRAIN_PROMPT_BSZ"
    add data.train_files "$TRAIN_FILES"
    add data.truncation "$DATA_TRUNCATION"
    add data.val_files "$VAL_FILES"
    add_if_set data.val_batch_size VAL_BSZ

    if [ "$TRAINING_MODE" = async ]; then
        add rollout.n_gpus_per_node "$ROLLOUT_N_GPUS_PER_NODE"
        add rollout.nnodes "$ROLLOUT_NNODES"
        add_if_set rollout.total_rollout_steps TOTAL_ROLLOUT_STEPS
    fi

    add trainer.experiment_name "$exp_name"
    add trainer.logger "$TRAINER_LOGGER"
    add trainer.n_gpus_per_node "$TRAINER_N_GPUS_PER_NODE"
    add trainer.nnodes "$TRAINER_NNODES"
    add trainer.project_name "$project_name"
    add trainer.save_freq "$TRAINER_SAVE_FREQ"
    add trainer.max_actor_ckpt_to_keep "$MAX_ACTOR_CKPT_TO_KEEP"
    add trainer.max_critic_ckpt_to_keep "$MAX_CRITIC_CKPT_TO_KEEP"
    add trainer.test_freq "$TRAINER_TEST_FREQ"
    add trainer.total_epochs "$TRAINER_TOTAL_EPOCHS"
    add trainer.val_before_train "$TRAINER_VAL_BEFORE_TRAIN"
}

append_engine_hydra_args() {
    case "$MODEL_ENGINE" in
        veomni)
            add actor_rollout_ref.actor.veomni.param_offload "$ACTOR_VEOMNI_PARAM_OFFLOAD"
            add actor_rollout_ref.actor.veomni.optimizer_offload "$ACTOR_VEOMNI_OPTIMIZER_OFFLOAD"
            add actor_rollout_ref.actor.veomni.enable_full_shard "$ACTOR_VEOMNI_ENABLE_FULL_SHARD"
            add actor_rollout_ref.actor.veomni.fsdp_size "$ACTOR_VEOMNI_FSDP_SIZE"
            add actor_rollout_ref.actor.veomni.ulysses_parallel_size "$ACTOR_VEOMNI_ULYSSES_PARALLEL_SIZE"
            add actor_rollout_ref.actor.veomni.expert_parallel_size "$ACTOR_VEOMNI_EXPERT_PARALLEL_SIZE"
            add_plus actor_rollout_ref.actor.veomni.entropy_from_logits_with_chunking "$ACTOR_VEOMNI_ENTROPY_FROM_LOGITS_WITH_CHUNKING"
            add_plus actor_rollout_ref.actor.veomni.entropy_checkpointing "$ACTOR_VEOMNI_ENTROPY_CHECKPOINTING"
            add actor_rollout_ref.ref.veomni.param_offload "$REF_VEOMNI_PARAM_OFFLOAD"
            if is_true "$ENABLE_ROLLOUT_ROUTING_REPLAY"; then
                add actor_rollout_ref.actor.veomni.router_replay.mode "$R3_VEOMNI_ROUTER_REPLAY_MODE"
            fi
            ;;
        fsdp)
            add actor_rollout_ref.actor.strategy "$ACTOR_STRATEGY"
            add actor_rollout_ref.actor.fsdp_config.strategy "$ACTOR_FSDP_STRATEGY"
            add actor_rollout_ref.actor.fsdp_config.param_offload "$ACTOR_FSDP_PARAM_OFFLOAD"
            add actor_rollout_ref.actor.fsdp_config.optimizer_offload "$ACTOR_FSDP_OPTIMIZER_OFFLOAD"
            add actor_rollout_ref.actor.fsdp_config.fsdp_size "$FSDP_SIZE"
            add actor_rollout_ref.actor.ulysses_sequence_parallel_size "$SP_SIZE"
            add actor_rollout_ref.ref.strategy "$REF_STRATEGY"
            add actor_rollout_ref.ref.fsdp_config.strategy "$REF_FSDP_STRATEGY"
            add actor_rollout_ref.ref.fsdp_config.param_offload "$REF_FSDP_PARAM_OFFLOAD"
            add actor_rollout_ref.ref.ulysses_sequence_parallel_size "$SP_SIZE"
            if is_true "$ENABLE_ROLLOUT_ROUTING_REPLAY"; then
                add actor_rollout_ref.actor.fsdp_config.router_replay.mode "$R3_FSDP_ROUTER_REPLAY_MODE"
            fi
            ;;
        *)
            echo "[FATAL] unsupported MODEL_ENGINE='$MODEL_ENGINE'" >&2
            exit 1
            ;;
    esac
}

# Critic (value model) overrides, for value-based runs such as SAO
# (scripts/templates/verl/sao.env). Gated on CRITIC_ENABLE so GRPO-style runs --
# which have no critic worker at all -- do not carry dead critic.* args.
#
# Cross-module derivations live here rather than in verl/sao.env: sao.env has to be
# sourced BEFORE verl/common.env to win the `: "${VAR:=default}"` race, so it cannot
# see common.env's derived values (ACTOR_PPO_*, SP_SIZE). By the time this function
# runs, every template has been sourced.
append_critic_hydra_args() {
    is_true "${CRITIC_ENABLE:-False}" || return 0

    add critic.enable True
    # verl's default critic.model.path is ~/models/deepseek-llm-7b-chat, which
    # loads without error and trains the wrong value model. Always be explicit.
    add critic.model.path "${CRITIC_MODEL_PATH:-$MODEL_PATH}"
    add critic.optim.lr "$CRITIC_LR"
    add critic.optim.lr_scheduler_type "$LR_SCHEDULER"
    add critic.optim.lr_warmup_steps "$CRITIC_LR_WARMUP_STEPS"
    # Same clobber-on-resume story as ACTOR_CKPT_LOAD_CONTENTS above.
    add_if_set critic.checkpoint.load_contents CRITIC_CKPT_LOAD_CONTENTS
    add critic.ppo_epochs "$CRITIC_PPO_EPOCHS"
    add critic.ppo_mini_batch_size "${CRITIC_PPO_MINI_BATCH_SIZE:-$ACTOR_PPO_MINI_BATCH_SIZE}"
    add critic.ppo_micro_batch_size_per_gpu "$CRITIC_MICRO_BSZ_PER_GPU"
    # This budget, not critic.use_dynamic_bsz, is what sets the critic's
    # micro-batch memory and its grad_norm scale: a smaller pack budget means more
    # micro-batches per mini-batch. (The value loss is globally normalized in the
    # patched verl, so the loss itself no longer scales with the pack count.)
    add critic.ppo_max_token_len_per_gpu "${CRITIC_PPO_MAX_TOKEN_LEN_PER_GPU:-$ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}"
    add critic.use_dynamic_bsz "$CRITIC_USE_DYNAMIC_BSZ"
    add critic.cliprange_value "$CRITIC_CLIPRANGE_VALUE"
    # optim.clip_grad, NOT critic.grad_clip. Two separate reasons, either sufficient:
    #   1. `grad_clip` is declared only in dp_critic.yaml, so on model_engine=veomni
    #      hydra aborts composition with "Key 'grad_clip' is not in struct" before
    #      anything launches. VeOmniCriticConfig has a grad_clip *field*, which is
    #      what makes this look like it should work; the yaml group does not expose it.
    #   2. Even where it composes, it is dead. Nothing in verl/workers/ reads
    #      `.grad_clip` at runtime -- both engines clip with optimizer_config.clip_grad.
    add critic.optim.clip_grad "$CRITIC_GRAD_CLIP"
    add critic.loss_agg_mode "$CRITIC_LOSS_AGG_MODE"
    add trainer.critic_warmup "$CRITIC_WARMUP"

    case "$MODEL_ENGINE" in
        veomni)
            # veomni_critic.yaml already sets strategy=veomni; only engine knobs here.
            add critic.veomni.param_offload "$CRITIC_VEOMNI_PARAM_OFFLOAD"
            add critic.veomni.optimizer_offload "$CRITIC_VEOMNI_OPTIMIZER_OFFLOAD"
            add critic.veomni.enable_full_shard "$CRITIC_VEOMNI_ENABLE_FULL_SHARD"
            add critic.veomni.fsdp_size "$CRITIC_VEOMNI_FSDP_SIZE"
            add critic.veomni.ulysses_parallel_size \
                "${CRITIC_VEOMNI_ULYSSES_PARALLEL_SIZE:-$ACTOR_VEOMNI_ULYSSES_PARALLEL_SIZE}"
            add critic.veomni.expert_parallel_size \
                "${CRITIC_VEOMNI_EXPERT_PARALLEL_SIZE:-$ACTOR_VEOMNI_EXPERT_PARALLEL_SIZE}"
            add critic.veomni.freeze_param_patterns "$CRITIC_FREEZE_PARAM_PATTERNS"
            ;;
        fsdp)
            # The critic mounts the fsdp engine group at `fsdp`, NOT at
            # `fsdp_config` like the actor does -- see dp_critic.yaml's
            # `../engine@fsdp: fsdp` and FSDPCriticConfig.fsdp.
            add critic.strategy "$CRITIC_FSDP_STRATEGY"
            add critic.fsdp.strategy "$CRITIC_FSDP_STRATEGY"
            add critic.fsdp.param_offload "$CRITIC_FSDP_PARAM_OFFLOAD"
            add critic.fsdp.optimizer_offload "$CRITIC_FSDP_OPTIMIZER_OFFLOAD"
            add critic.fsdp.fsdp_size "$CRITIC_FSDP_SIZE"
            add critic.fsdp.freeze_param_patterns "$CRITIC_FREEZE_PARAM_PATTERNS"
            add critic.fsdp.use_orig_params "$CRITIC_FSDP_USE_ORIG_PARAMS"
            # BOTH keys, and the engine one is the load-bearing half. FSDPActorConfig
            # forwards its deprecated top-level ulysses_sequence_parallel_size into the
            # engine config (actor.py __post_init__); FSDPCriticConfig does NOT -- it
            # only reads that field in validate(). Setting the top-level one alone
            # silently leaves the critic engine at SP=1 while the actor runs SP=N, and
            # the mismatched device meshes deadlock _compute_values with one rank never
            # entering the dp all_reduce.
            add critic.fsdp.ulysses_sequence_parallel_size "${CRITIC_SP_SIZE:-$SP_SIZE}"
            add critic.ulysses_sequence_parallel_size "${CRITIC_SP_SIZE:-$SP_SIZE}"
            ;;
    esac
}

append_mode_hydra_args() {
    case "$TRAINING_MODE" in
        async)
            add async_training.partial_rollout "$PARTIAL_ROLLOUT"
            add async_training.require_batches "$REQUIRE_BATCHES"
            add async_training.staleness_threshold "$STALENESS"
            add async_training.trigger_parameter_sync_step "$TRIGGER_PARAMETER_SYNC_STEP"
            add_plus async_training.validation_rollout_min_idle_workers "$VALIDATION_ROLLOUT_MIN_IDLE_WORKERS"
            ;;
        sync)
            ;;
        *)
            echo "[FATAL] unsupported TRAINING_MODE='$TRAINING_MODE'" >&2
            exit 1
            ;;
    esac
}

build_hydra_args() {
    init_hydra_args
    append_common_hydra_args
    append_engine_hydra_args
    append_critic_hydra_args
    append_mode_hydra_args
}

build_verl_command() {
    cmd=("$PYTHON_BIN" -m "$VERL_ENTRY_MODULE" --config-name="$VERL_CONFIG_NAME" --config-path="$REPO_ROOT/src/verl_patch/config")
}
