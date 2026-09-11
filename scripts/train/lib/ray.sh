#!/usr/bin/env bash

cleanup_ray_and_shm() {
    ray stop --force --grace-period=60 || true
    rm -rf /tmp/ray/* 2>/dev/null || true
    sleep 5

    echo "[shm] before cleanup:"
    df -h /dev/shm 2>/dev/null || true
    pgrep -f "VLLM|vllm|EngineCore|raylet|gcs_server|FullyAsync|WorkerDict|fully_async_main|fully_async_policy" 2>/dev/null \
        | grep -vw -e "$$" -e "$PPID" | xargs -r kill -9 2>/dev/null || true
    for _ in $(seq 1 15); do
        pgrep -f "VLLM::Worker|EngineCore|WorkerDict" 2>/dev/null | grep -vqw -e "$$" -e "$PPID" || break
        sleep 1
    done
    rm -rf /tmp/ray/* 2>/dev/null || true
    rm -f /dev/shm/vllm* /dev/shm/psm_* /dev/shm/plasma* /dev/shm/cuda.shm.* \
          /dev/shm/nccl-* /dev/shm/torch_* /dev/shm/sem.* 2>/dev/null || true
    sleep 2
    echo "[shm] after cleanup:"
    df -h /dev/shm 2>/dev/null || true
}

resolve_ray_addresses() {
    # Use the same interface as NCCL/GLOO/TP. `hostname -I` does not guarantee
    # interface order, so resolve the local Ray address explicitly.
    IP_LOCAL=$(ip -4 -o addr show dev "$TRAIN_NETWORK_INTERFACE" scope global 2>/dev/null \
        | awk 'NR == 1 {sub(/\/.*/, "", $4); print $4}')
    [ -n "$IP_LOCAL" ] || {
        echo "[FATAL] could not determine an IPv4 address for TRAIN_NETWORK_INTERFACE=${TRAIN_NETWORK_INTERFACE}." >&2
        exit 1
    }
    : "${MASTER_ADDR:=$IP_LOCAL}"
    IP_HEAD=$(getent ahostsv4 "$MASTER_ADDR" 2>/dev/null | awk 'NR == 1 {print $1; exit}' || true)
    if [ -z "$IP_HEAD" ]; then
        IP_HEAD="$MASTER_ADDR"
    fi
    RAY_NODE_ROLE=worker
    if ip -4 -o addr show scope global 2>/dev/null \
        | awk -v head="$IP_HEAD" '{sub(/\/.*/, "", $4); if ($4 == head) found=1} END {exit !found}'; then
        RAY_NODE_ROLE=head
    fi
    export IP_LOCAL MASTER_ADDR IP_HEAD RAY_NODE_ROLE
    echo "HEAD=$IP_HEAD  MASTER_ADDR=$MASTER_ADDR  LOCAL=$IP_LOCAL  IFACE=$TRAIN_NETWORK_INTERFACE  ROLE=$RAY_NODE_ROLE  NNODES=$NNODES"
}

stop_ray_head_on_exit() {
    local exit_status="$1"

    # Prevent recursion because this function exits with the trainer's original
    # status after stopping the head runtime.
    trap - EXIT
    echo "[HEAD] runner exiting; stopping Ray so worker launchers can finish."
    ray stop --force --grace-period=5 || true
    exit "$exit_status"
}

ray_head_is_reachable() {
    "$PYTHON_BIN" - "$IP_HEAD" "$RAY_PORT" <<'PY'
import socket
import sys

with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=2):
    pass
PY
}

cleanup_worker_on_exit() {
    local exit_status="$1"

    trap - EXIT
    if [ -n "${GPU_WANDB_PID:-}" ]; then
        kill "$GPU_WANDB_PID" 2>/dev/null || true
        wait "$GPU_WANDB_PID" 2>/dev/null || true
    fi
    if [ -n "${GPU_WANDB_PID_FILE:-}" ]; then
        rm -f "$GPU_WANDB_PID_FILE"
    fi
    ray stop --force --grace-period=5 || true
    exit "$exit_status"
}

wait_for_ray_head_shutdown() {
    local failed_probes=0
    local max_failed_probes=6

    echo "[WORKER] joined; keeping this launcher alive until the Ray head stops."
    while [ "$failed_probes" -lt "$max_failed_probes" ]; do
        if ray_head_is_reachable >/dev/null 2>&1; then
            failed_probes=0
        else
            failed_probes=$((failed_probes + 1))
            echo "[WORKER] head probe failed (${failed_probes}/${max_failed_probes}): ${IP_HEAD}:${RAY_PORT}"
        fi
        [ "$failed_probes" -ge "$max_failed_probes" ] || sleep 10
    done
    echo "[WORKER] Ray head is no longer reachable; stopping the local runtime."
}

start_ray_node() {
    if [ "$RAY_NODE_ROLE" = head ]; then
        echo "[HEAD] starting ray head"
        ray start --head --node-ip-address="$IP_LOCAL" --port="$RAY_PORT" \
            --object-store-memory="$RAY_OBJECT_STORE_MEMORY" \
            --num-gpus="$NGPUS_PER_NODE" --disable-usage-stats --dashboard-host=0.0.0.0
        trap 'stop_ray_head_on_exit "$?"' EXIT
    else
        echo "[WORKER] waiting for head $IP_HEAD:$RAY_PORT"
        for _ in $(seq 1 30); do
            "$PYTHON_BIN" -c "import socket;s=socket.socket();s.settimeout(2);s.connect(('$IP_HEAD',$RAY_PORT))" \
                >/dev/null 2>&1 && break
            sleep 2
        done
        ray start --address="$IP_HEAD:$RAY_PORT" --node-ip-address="$IP_LOCAL" \
            --object-store-memory="$RAY_OBJECT_STORE_MEMORY" \
            --num-gpus="$NGPUS_PER_NODE" --disable-usage-stats
    fi
}

wait_for_ray_cluster_or_exit_worker() {
    local up old_gpid

    if [ "$RAY_NODE_ROLE" = head ]; then
        up=0
        for _ in $(seq 1 60); do
            up=$(ray status 2>/dev/null | grep -c "node_" || true)
            [ "$up" -ge "$NNODES" ] && break
            sleep 5
        done
        if [ "$up" -lt "$NNODES" ]; then
            echo "[FATAL] only ${up}/${NNODES} nodes joined the ray cluster after 300s." >&2
            echo "[FATAL] every node must run this script with MASTER_ADDR=${IP_HEAD};" >&2
            echo "[FATAL] check that missing nodes can reach ${IP_HEAD}:${RAY_PORT}." >&2
            ray status || true
            exit 1
        fi
        ray status || true
        return 0
    fi

    GPU_WANDB_PID_FILE="/tmp/gpu_wandb_${IP_LOCAL}.pid"
    GPU_WANDB_LOG="$HARBOR_LOG_DIR/${exp_name}_train_gpu_wandb.log"
    if [ -f "$GPU_WANDB_PID_FILE" ]; then
        old_gpid=$(cat "$GPU_WANDB_PID_FILE" 2>/dev/null || true)
        [ -n "$old_gpid" ] && kill "$old_gpid" 2>/dev/null || true
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
    echo "[GPU wandb] PID=${GPU_WANDB_PID} -> ${GPU_WANDB_LOG}"

    # Ray daemonizes its worker processes.  On managed training platforms the
    # shell is the pod's main process, so exiting here tears the pod down and
    # silently removes this node's GPUs from the cluster.  Keep the launcher in
    # the foreground until the head runner closes the GCS port.
    trap 'cleanup_worker_on_exit "$?"' EXIT
    wait_for_ray_head_shutdown
    exit 0
}

start_ray_cluster_or_exit_worker() {
    cleanup_ray_and_shm
    resolve_ray_addresses
    start_ray_node
    wait_for_ray_cluster_or_exit_worker
}
