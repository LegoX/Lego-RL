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
    add actor_rollout_ref.rollout.engine_kwargs.vllm.served-model-name "$SERVED_MODEL_NAME"
    add actor_rollout_ref.rollout.engine_kwargs.vllm.tool-call-parser "$TOOL_CALL_PARSER"
    add actor_rollout_ref.rollout.disable_log_stats "$DISABLE_LOG_STATS"
    add actor_rollout_ref.rollout.gpu_memory_utilization "$GPU_MEM_UTIL"
    add actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu "$ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU"
    add actor_rollout_ref.rollout.log_prob_use_dynamic_bsz "$ROLLOUT_LOG_PROB_USE_DYNAMIC_BSZ"
    add actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu "$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
    add actor_rollout_ref.rollout.max_model_len "$ROLLOUT_MAX_MODEL_LEN"
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
    add algorithm.kl_ctrl.kl_coef "$KL_COEF"
    add algorithm.rollout_correction.bypass_mode "$ROLLOUT_CORRECTION_BYPASS"
    add algorithm.rollout_correction.rollout_is "$ROLLOUT_IS"
    add algorithm.rollout_correction.rollout_is_threshold "$ROLLOUT_IS_THRESHOLD"
    add algorithm.rollout_correction.seq_dist_metrics "$SEQ_DIST_METRICS"
    add algorithm.trajectory_filter.enable "$TRAJ_FILTER_ENABLE"
    add algorithm.trajectory_filter.filter_overlong "$TRAJ_FILTER_FILTER_OVERLONG"
    add algorithm.use_kl_in_reward "$USE_KL_IN_REWARD"

    add data.gen_batch_size "$GEN_PROMPT_BSZ"
    add data.max_prompt_length "$MAX_PROMPT"
    add data.max_response_length "$MAX_RESP"
    add data.prompt_key "$PROMPT_KEY"
    add data.return_raw_chat "$RETURN_RAW_CHAT"
    add data.train_batch_size "$TRAIN_PROMPT_BSZ"
    add data.train_files "$TRAIN_FILES"
    add data.truncation "$DATA_TRUNCATION"
    add data.val_files "$VAL_FILES"
    add_if_set data.val_batch_size VAL_BSZ

    add rollout.n_gpus_per_node "$ROLLOUT_N_GPUS_PER_NODE"
    add rollout.nnodes "$ROLLOUT_NNODES"
    add_if_set rollout.total_rollout_steps TOTAL_ROLLOUT_STEPS

    add trainer.experiment_name "$exp_name"
    add trainer.logger "$TRAINER_LOGGER"
    add trainer.n_gpus_per_node "$TRAINER_N_GPUS_PER_NODE"
    add trainer.nnodes "$TRAINER_NNODES"
    add trainer.project_name "$project_name"
    add trainer.save_freq "$TRAINER_SAVE_FREQ"
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
    append_mode_hydra_args
}

build_verl_command() {
    cmd=("$PYTHON_BIN" -m "$VERL_ENTRY_MODULE" --config-name="$VERL_CONFIG_NAME" --config-path="$REPO_ROOT/src/verl_patch/config")
}
