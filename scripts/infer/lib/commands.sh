#!/usr/bin/env bash

build_vllm_command() {
    vllm_cmd=(
        vllm serve "$MODEL_PATH"
        --served-model-name "$SERVED_MODEL_NAME"
        --tensor-parallel-size "$GEN_TP"
        --max-model-len "$VLLM_MAX_MODEL_LEN"
        --gpu-memory-utilization "$GPU_MEM_UTIL"
        --port "$VLLM_PORT"
        --host "$VLLM_BIND_HOST"
        --api-server-count "$VLLM_API_SERVER_COUNT"
        --enable-prefix-caching
        --prefix-caching-hash-algo "$PREFIX_CACHING_HASH_ALGO"
        --enable-chunked-prefill
        --enable-auto-tool-choice
        --tool-call-parser "$TOOL_CALL_PARSER"
        --dtype bfloat16
    )
}

build_infer_command() {
    infer_cmd=(
        "$PYTHON_BIN" "$REPO_ROOT/utils/eval_swerebench_filtered.py"
        --index "$INDEX_FILE"
        --results-dir "$RESULTS_DIR"
        --output-index "$OUTPUT_INDEX"
        --api-base "$API_BASE"
        --model-name "$AGENT_MODEL_NAME"
        --temperature "$TEMPERATURE"
        --n-trials "$N_TRIALS"
        --n-concurrent "$N_CONCURRENT"
        --trial-timeout "$TRIAL_HARD_TIMEOUT_SEC"
    )

    if [ -n "${INSTANCES_FILE:-}" ]; then
        infer_cmd+=(--instances-file "$INSTANCES_FILE")
    fi
}

build_commands() {
    build_vllm_command
    build_infer_command
}

print_launch_commands() {
    echo "=== vLLM Command ==="
    print_cmd "${vllm_cmd[@]}"
    echo "=== Infer Command ==="
    print_cmd "${infer_cmd[@]}"
}
