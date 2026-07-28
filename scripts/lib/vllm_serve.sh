#!/usr/bin/env bash
# lib/vllm_serve.sh — vLLM lifecycle helpers (shared by eval/infer): whole-tree kill + readiness wait.
# Only provides functions, doesn't start the service itself (each runner assembles its own serve command; tp-only vs DP differ a lot).
# shellcheck disable=SC2009

# Whole-tree kill of vLLM (tp>1 forks EngineCore/Worker child processes that pkill pattern-match misses) + wait for VRAM drain.
_vllm_kill_tree() { local p="$1" c; for c in $(pgrep -P "$p" 2>/dev/null); do _vllm_kill_tree "$c"; done; kill -9 "$p" 2>/dev/null || true; }
vllm_kill_and_wait() {
    [ -n "${VLLM_PID:-}" ] && _vllm_kill_tree "$VLLM_PID"
    pkill -9 -f "vllm serve" 2>/dev/null || true
    local i max
    for i in $(seq 1 60); do
        max=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
        [ "${max:-9999}" -lt 2000 ] && break
        sleep 2
    done
    echo "[vLLM] GPU drained (max used ${max:-?} MiB)"
}

# Wait for /v1/models or /health readiness. Usage: vllm_wait_ready <port> <served_name> <max_iters> <pid> [step_sec]
vllm_wait_ready() {
    local port="$1" served="$2" iters="${3:-120}" pid="$4" step="${5:-10}" i
    echo "[vLLM] waiting for readiness (PID $pid, probe every ${step}s, up to ${iters} times)..."
    for i in $(seq 1 "$iters"); do
        if curl -sf "http://127.0.0.1:${port}/v1/models" 2>/dev/null | grep -q "$served"; then echo "[vLLM] ready (~$((i*step))s)"; return 0; fi
        kill -0 "$pid" 2>/dev/null || { echo "[vLLM] process died"; return 1; }
        sleep "$step"
    done
    echo "[vLLM] not ready within ${iters}×${step}s"; return 1
}

# Single-prompt direct self-check, fail-fast when serialization is broken (garbled). Usage: vllm_sanity <port> <served_name>
vllm_sanity() {
    local port="$1" served="$2"
    echo "[vLLM] coherence self-check:"
    curl -s "http://127.0.0.1:${port}/v1/chat/completions" -H "Content-Type: application/json" \
      -d "{\"model\":\"$served\",\"messages\":[{\"role\":\"user\",\"content\":\"Reverse a string in Python.\"}],\"max_tokens\":60,\"temperature\":0}" 2>/dev/null \
      | python3 -c "import json,sys;print('  ',repr(json.load(sys.stdin)['choices'][0]['message']['content'][:120]))" 2>/dev/null || echo "  (self-check query failed)"
}
