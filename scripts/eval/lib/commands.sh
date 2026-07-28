#!/usr/bin/env bash

build_vllm_command() {
    local extra_args=()
    if [ -n "${EVAL_VLLM_EXTRA_ARGS:-}" ]; then
        # shellcheck disable=SC2206
        extra_args=($EVAL_VLLM_EXTRA_ARGS)
    fi

    vllm_cmd=(
        env -u MODELING_BACKEND vllm serve "$MODEL_PATH"
        --served-model-name "$SERVED_MODEL_NAME"
        --tensor-parallel-size "$GEN_TP"
        --port "$VLLM_PORT"
        --host "$VLLM_BIND_HOST"
        --max-model-len "$MAX_MODEL_LEN"
        --gpu-memory-utilization "$GPU_MEM_UTIL"
    )

    if is_true "$EVAL_ENABLE_EXPERT_PARALLEL"; then
        vllm_cmd+=(--enable-expert-parallel)
    fi

    vllm_cmd+=("${extra_args[@]}")
    vllm_cmd+=(
        --enable-auto-tool-choice
        --tool-call-parser "$TOOL_CALL_PARSER"
    )
}

build_harbor_command() {
    harbor_cmd=(
        harbor run
        --config "$JOB_CFG"
        --job-name "$EXP_NAME"
        --jobs-dir "$EVAL_JOBS_DIR"
        --n-concurrent "$N_CONCURRENT"
        --max-retries "$MAX_RETRIES"
    )
}

build_commands() {
    build_vllm_command
    build_harbor_command
}

print_launch_commands() {
    echo "=== vLLM Command ==="
    print_cmd "${vllm_cmd[@]}"
    echo "=== Harbor Command ==="
    print_cmd "${harbor_cmd[@]}"
}
