#!/usr/bin/env bash
# =============================================================================
# scripts/lib/ray_bringup.sh — ### CANONICAL (sourced by train.sh, not standalone)
# section 10: /dev/shm cleanup + ray head/worker bring-up. Worker branch exits 0
# (does not proceed to launch).
# Requires from the caller: RAY_PORT NGPUS_PER_NODE NNODES PYTHON_BIN REPO_ROOT
#              exp_name project_name  (optional MASTER_ADDR RAY_OBJECT_STORE_MEMORY)
# >>> THE ENV-BEFORE-RAY ANCHOR: every export in train.sh is now visible to workers.
# =============================================================================
ray stop --force
# Clear leftover vLLM/ray + /dev/shm. ORDER MATTERS: kill first and wait for fds to
# drop before rm, else the next vLLM SIGBUSes on a full /dev/shm. The pattern must NOT
# match this launcher's own cmdline (bash train.sh).
echo "[shm] before cleanup:"; df -h /dev/shm 2>/dev/null || true
pgrep -f "VLLM|vllm|EngineCore|raylet|gcs_server|FullyAsync|WorkerDict|fully_async_main|fully_async_policy" 2>/dev/null \
    | grep -vw -e "$$" -e "${PPID:-0}" | xargs -r kill -9 2>/dev/null || true
for _ in $(seq 1 15); do
    pgrep -f "VLLM::Worker|EngineCore|WorkerDict" 2>/dev/null | grep -vqw -e "$$" -e "${PPID:-0}" || break
    sleep 1
done
rm -rf /tmp/ray/* 2>/dev/null || true
rm -f /dev/shm/vllm* /dev/shm/psm_* /dev/shm/plasma* /dev/shm/cuda.shm.* \
      /dev/shm/nccl-* /dev/shm/torch_* /dev/shm/sem.* 2>/dev/null || true
sleep 2
echo "[shm] after cleanup:"; df -h /dev/shm 2>/dev/null || true
IP_LOCAL=$(hostname -I | awk '{print $1}')
MASTER_ADDR="${MASTER_ADDR:-$IP_LOCAL}"
IP_HEAD=$(getent hosts "$MASTER_ADDR" | awk '{print $1}'); IP_HEAD="${IP_HEAD:-$MASTER_ADDR}"
echo "HEAD=$IP_HEAD  LOCAL=$IP_LOCAL  NNODES=$NNODES"

if [ "$IP_LOCAL" = "$IP_HEAD" ]; then
    echo "[HEAD] starting ray head"
    ray start --head --node-ip-address="$IP_HEAD" --port="$RAY_PORT" \
        --object-store-memory="${RAY_OBJECT_STORE_MEMORY:-7000000000}" \
        --num-gpus="$NGPUS_PER_NODE" --disable-usage-stats
else
    echo "[WORKER] waiting for head $IP_HEAD:$RAY_PORT"
    for _ in $(seq 1 30); do
        "$PYTHON_BIN" -c "import socket;s=socket.socket();s.settimeout(2);s.connect(('$IP_HEAD',$RAY_PORT))" \
            >/dev/null 2>&1 && break
        sleep 2
    done
    ray start --address="$IP_HEAD:$RAY_PORT" \
        --object-store-memory="${RAY_OBJECT_STORE_MEMORY:-7000000000}" \
        --num-gpus="$NGPUS_PER_NODE" --disable-usage-stats
fi

# Head waits for all NNODES nodes to join, then returns to train.sh's launch; a worker
# starts a GPU monitor and exits 0 (in a sourced context this exits the whole runner,
# so it never reaches launch).
if [ "$IP_LOCAL" = "$IP_HEAD" ]; then
    for _ in $(seq 1 60); do
        up=$(ray status 2>/dev/null | grep -c "node_" || true)
        [ "${up:-0}" -ge "$NNODES" ] && break
        sleep 5
    done
    ray status || true
else
    # Training-node GPU monitor: an idle wandb run so _System/gpu.* is reported ~every 10s.
    GPU_WANDB_PID_FILE="/tmp/gpu_wandb_${IP_LOCAL}.pid"
    GPU_WANDB_LOG="$REPO_ROOT/logs/${exp_name}_train_gpu_wandb.log"
    if [ -f "$GPU_WANDB_PID_FILE" ]; then
        OLD_GPID=$(cat "$GPU_WANDB_PID_FILE" 2>/dev/null || true)
        [ -n "$OLD_GPID" ] && kill "$OLD_GPID" 2>/dev/null || true
        rm -f "$GPU_WANDB_PID_FILE"
    fi
    GPU_WANDB_PROJECT="$project_name" \
    GPU_WANDB_NAME="${exp_name}-trainnode" \
    GPU_WANDB_GROUP="$exp_name" \
    nohup "$PYTHON_BIN" -u -c '
import os, time, wandb
wandb.init(
    project=os.environ["GPU_WANDB_PROJECT"],
    name=os.environ["GPU_WANDB_NAME"],
    group=os.environ["GPU_WANDB_GROUP"],
    job_type="train",
)
print(f"[gpu_wandb] trainnode run started: {wandb.run.url}", flush=True)
while True:
    time.sleep(60)
' > "$GPU_WANDB_LOG" 2>&1 &
    GPU_WANDB_PID=$!
    echo "$GPU_WANDB_PID" > "$GPU_WANDB_PID_FILE"
    disown "$GPU_WANDB_PID" 2>/dev/null || true
    echo "[GPU wandb] PID=${GPU_WANDB_PID} -> ${GPU_WANDB_LOG}"
    echo "[WORKER] joined; head drives training. exiting worker script."
    exit 0
fi
