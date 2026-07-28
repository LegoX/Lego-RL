#!/usr/bin/env bash

port_in_use() {
    "$PYTHON_BIN" - "$1" <<'PYCHECK'
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("0.0.0.0", int(sys.argv[1])))
    sys.exit(1)
except OSError:
    sys.exit(0)
finally:
    s.close()
PYCHECK
}

preflight_vllm_port() {
    if port_in_use "$VLLM_PORT"; then
        echo "[FATAL] vLLM port already in use on $(hostname): $VLLM_PORT" >&2
        echo "[FATAL] change VLLM_PORT or free it" >&2
        exit 1
    fi
    echo "[Preflight] vLLM port free: $VLLM_PORT"
}

cleanup_vllm() {
    echo "Cleaning up..."
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}

start_vllm() {
    VLLM_PID=""
    trap cleanup_vllm EXIT

    echo "[vLLM] starting single-node model server..."
    "${vllm_cmd[@]}" > "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    echo "[vLLM] PID=$VLLM_PID"
}

wait_for_vllm() {
    local i
    echo "[vLLM] waiting for endpoint to be ready (up to ${VLLM_READY_TIMEOUT}s)..."
    for i in $(seq 1 "$VLLM_READY_TIMEOUT"); do
        if curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then
            echo "[vLLM] ready after ${i}s"
            return 0
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[vLLM] process died. Check $VLLM_LOG" >&2
            exit 1
        fi
        sleep 1
    done

    echo "[vLLM] failed to start within ${VLLM_READY_TIMEOUT}s" >&2
    exit 1
}

run_inference() {
    export LLM_BASE_URL LLM_API_KEY
    echo "[Infer] starting rollout..."
    echo "[Infer] LLM_BASE_URL: $LLM_BASE_URL"
    echo "[Infer] API base:     $API_BASE"
    echo "[Infer] model name:   $AGENT_MODEL_NAME"
    "${infer_cmd[@]}" 2>&1 | tee "$INFER_LOG"
    echo "=== Inference complete ==="
    echo "Results:       $RESULTS_DIR"
    echo "Summary:       $RESULTS_DIR/summary.csv"
    echo "Selected idx:  $OUTPUT_INDEX"
}
