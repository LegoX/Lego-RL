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

acquire_eval_lock() {
    # Reserve the port even while the model is loading and has no listener yet.
    local lock_path="${TMPDIR:-/tmp}/lego-rl-eval-${UID}-${VLLM_PORT}.lock"
    exec {EVAL_LOCK_FD}>"$lock_path"
    if ! flock -n "$EVAL_LOCK_FD"; then
        echo "[FATAL] another eval owns vLLM port $VLLM_PORT on $(hostname) ($lock_path)" >&2
        exit 1
    fi
}

kill_tree() {
    local p="$1" c
    for c in $(pgrep -P "$p" 2>/dev/null); do
        kill_tree "$c"
    done
    kill -9 "$p" 2>/dev/null || true
}

kill_vllm_and_wait() {
    if [ -n "${VLLM_PID:-}" ]; then
        kill_tree "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
    fi
}

cleanup_vllm() {
    echo "Cleaning up..."
    kill_vllm_and_wait
}

start_vllm() {
    VLLM_PID=""
    trap cleanup_vllm EXIT

    echo "[vLLM] starting plain model server..."
    (
        # Only the runner owns the lock; workers must not keep it after exit.
        exec {EVAL_LOCK_FD}>&-
        exec "${vllm_cmd[@]}"
    ) > "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    echo "[vLLM] PID=$VLLM_PID"
}

wait_for_vllm() {
    local i
    echo "[vLLM] waiting for endpoint to list $SERVED_MODEL_NAME (up to $((EVAL_VLLM_READY_ITERS * 10))s)..."
    for i in $(seq 1 "$EVAL_VLLM_READY_ITERS"); do
        if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null | grep -q "$SERVED_MODEL_NAME"; then
            echo "[vLLM] ready after $((i * 10))s"
            return 0
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[vLLM] process died. Tail of $VLLM_LOG:" >&2
            tail -30 "$VLLM_LOG" >&2 || true
            exit 1
        fi
        sleep 10
    done

    echo "[vLLM] failed to start within $((EVAL_VLLM_READY_ITERS * 10))s" >&2
    tail -30 "$VLLM_LOG" >&2 || true
    exit 1
}

run_vllm_sanity() {
    echo "[vLLM] coherence sanity check:"
    curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$SERVED_MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reverse a string in Python.\"}],\"max_tokens\":60,\"temperature\":0}" \
        2>/dev/null \
        | "$PYTHON_BIN" -c "import json,sys; print('  ', repr(json.load(sys.stdin)['choices'][0]['message']['content'][:120]))" \
        2>/dev/null \
        || echo "  (sanity query failed)"
}
