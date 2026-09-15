#!/usr/bin/env bash

write_job_config() {
    export EXP_NAME N_CONCURRENT MAX_RETRIES SERVED_MODEL_NAME LLM_BASE_URL LLM_API_KEY
    export DATASET_PATH DATASET_NAME N_TASKS EVAL_AGENT_NAME
    export HARBOR_ENVIRONMENT_IMPORT_PATH HARBOR_ENVIRONMENT_FORCE_BUILD HARBOR_ENVIRONMENT_DELETE
    export HARBOR_ENVIRONMENT_OVERRIDE_CPUS HARBOR_ENVIRONMENT_OVERRIDE_MEMORY_MB HARBOR_HOSTPATH_MOUNTS
    export HARBOR_AGENT_IMPORT_PATH HARBOR_AGENT_RUNTIME_IMAGE HARBOR_AGENT_RUNTIME_MOUNT_PATH HARBOR_AGENT_RUNTIME_IMAGE_SUBPATH
    export K8S_POD_ACTIVE_DEADLINE_SECONDS HARBOR_EXCLUDE_NODES
    export HARBOR_AGENT_TIMEOUT_MULTIPLIER HARBOR_VERIFIER_TIMEOUT_MULTIPLIER HARBOR_AGENT_MAX_ITERATIONS
    export LLM_TIMEOUT LLM_NUM_RETRIES EVAL_AGENT_EXTRA_ENV OH_SDK_PATCH_ROOT OH_SDK_RESTORE_TASK_ENV

    "$PYTHON_BIN" - "$JOB_CFG" <<'PYJOB'
import json
import os
import sys

job_cfg_path = sys.argv[1]
served = os.environ["SERVED_MODEL_NAME"]

def truthy(value):
    return str(value).lower() in {"1", "true", "yes", "on"}

raw_mounts = os.environ.get("HARBOR_HOSTPATH_MOUNTS", "")
host_path_mounts = None
try:
    if raw_mounts and raw_mounts != "null":
        parsed = json.loads(raw_mounts)
        if isinstance(parsed, list):
            host_path_mounts = parsed
except Exception:
    host_path_mounts = None

env_kwargs = {
    "agent_runtime_image": os.environ.get("HARBOR_AGENT_RUNTIME_IMAGE"),
    "agent_runtime_mount_path": os.environ.get("HARBOR_AGENT_RUNTIME_MOUNT_PATH"),
    "agent_runtime_image_subpath": os.environ.get("HARBOR_AGENT_RUNTIME_IMAGE_SUBPATH"),
    "agent_runtime_image_pull_policy": os.environ.get("HARBOR_AGENT_RUNTIME_IMAGE_PULL_POLICY", "IfNotPresent"),
    "pod_active_deadline_seconds": int(os.environ.get("K8S_POD_ACTIVE_DEADLINE_SECONDS", "6000")),
}
if host_path_mounts is not None:
    env_kwargs["host_path_mounts"] = host_path_mounts
# Same knob as the training agent loop (agent_loop_config_*.yaml -> kubernetes.py:exclude_nodes):
# keep sandbox pods off nodes that cannot reach the vLLM host, where every LLM call times out
# and the trial is lost. Comma-separated; empty = no node affinity at all.
if os.environ.get("HARBOR_EXCLUDE_NODES", "").strip():
    env_kwargs["exclude_nodes"] = os.environ["HARBOR_EXCLUDE_NODES"].strip()

agent_env = {
    "LLM_BASE_URL": os.environ["LLM_BASE_URL"],
    "LLM_API_KEY": os.environ.get("LLM_API_KEY", "dummy"),
    "LLM_MODEL": f"hosted_vllm/{served}",
}
# The OpenHands SDK's LLM.timeout defaults to 300 s, which is short for a benchmark whose
# agent budget is measured in tens of minutes. These are honoured by the runtime entrypoint,
# so they are a no-op on a runtime image that does not read them.
for key in ("LLM_TIMEOUT", "LLM_NUM_RETRIES", "OH_SDK_RESTORE_TASK_ENV"):
    if os.environ.get(key):
        agent_env[key] = os.environ[key]

# OH_SDK_PATCH_ROOT selects a PATCHED runtime entrypoint without rebuilding the runtime image:
# harbor resolves the entrypoint under $CUSTOM_AGENT_RUNTIME_ROOT, so pointing that at a hostPath
# laid out like the real runtime root (every sibling file included -- the image's runtime-env.sh
# resolves ALL of them from the root, so a partial copy fails detection outright) swaps the
# entrypoint while CUSTOM_AGENT_PYTHON still points at the real interpreter.
# OH_SDK_RESTORE_TASK_ENV=1 above is read by that patched entrypoint: it strips the runtime's own
# bin dirs from PATH and unsets PYTHONSAFEPATH, so the `python`/`pip` the agent types in the tool
# shell are the TASK image's again. Both are inert with a stock runtime image.
patch_root = os.environ.get("OH_SDK_PATCH_ROOT", "").rstrip("/")
if patch_root:
    real_root = os.environ.get(
        "HARBOR_AGENT_RUNTIME_MOUNT_PATH", "/opt/custom-agent-runtime/oh-sdk"
    ).rstrip("/")
    agent_env["CUSTOM_AGENT_RUNTIME_ROOT"] = patch_root
    agent_env["CUSTOM_AGENT_PYTHON"] = f"{real_root}/bin/python"
    agent_env["OH_SDK_REAL_RUNTIME_ROOT"] = real_root

# EVAL_AGENT_EXTRA_ENV: JSON object of extra env vars for the agent container, e.g. a
# GOPROXY/GOSUMDB pair pointing at an offline module proxy for tasks whose fix edits go.mod.
extra_env_raw = os.environ.get("EVAL_AGENT_EXTRA_ENV", "").strip()
if extra_env_raw:
    extra_env = json.loads(extra_env_raw)
    if not isinstance(extra_env, dict):
        raise SystemExit("EVAL_AGENT_EXTRA_ENV must be a JSON object")
    agent_env.update({str(k): str(v) for k, v in extra_env.items()})

# Timeouts: harbor enforces min(override_timeout_sec, max_timeout_sec) * multiplier, with the
# first two defaulting to the task's own task.toml values. HARBOR_AGENT_MAX_TIMEOUT_SEC alone
# does NOT reach a harbor job, so without the multiplier a config asking for 7200 s still got
# AgentTimeoutError at the task's declared 3000 s. Iterations: the custom agent only caps turns
# when max_iterations is passed as an agent kwarg.
agent_kwargs = {}
if os.environ.get("HARBOR_AGENT_MAX_ITERATIONS"):
    agent_kwargs["max_iterations"] = int(os.environ["HARBOR_AGENT_MAX_ITERATIONS"])

cfg = {
    "job_name": os.environ["EXP_NAME"],
    "n_attempts": 1,
    "agent_timeout_multiplier": float(os.environ.get("HARBOR_AGENT_TIMEOUT_MULTIPLIER", "1.0")),
    "verifier_timeout_multiplier": float(os.environ.get("HARBOR_VERIFIER_TIMEOUT_MULTIPLIER", "1.0")),
    "n_concurrent_trials": int(os.environ["N_CONCURRENT"]),
    "retry": {"max_retries": int(os.environ["MAX_RETRIES"])},
    "environment": {
        "import_path": os.environ["HARBOR_ENVIRONMENT_IMPORT_PATH"],
        "force_build": truthy(os.environ.get("HARBOR_ENVIRONMENT_FORCE_BUILD", "False")),
        "delete": truthy(os.environ.get("HARBOR_ENVIRONMENT_DELETE", "True")),
        "override_cpus": int(os.environ.get("HARBOR_ENVIRONMENT_OVERRIDE_CPUS", "1")),
        "override_memory_mb": int(os.environ.get("HARBOR_ENVIRONMENT_OVERRIDE_MEMORY_MB", "4096")),
        "kwargs": {k: v for k, v in env_kwargs.items() if v not in (None, "")},
    },
    "verifier": {"disable": False},
    "agents": [{
        "name": os.environ.get("EVAL_AGENT_NAME", "ohsdk"),
        "import_path": os.environ["HARBOR_AGENT_IMPORT_PATH"],
        "model_name": f"hosted_vllm/{served}",
        "kwargs": agent_kwargs,
        "env": agent_env,
    }],
    "datasets": [{k: v for k, v in {
        "path": os.environ.get("DATASET_PATH") or None,
        "name": os.environ.get("DATASET_NAME") or None,
        "n_tasks": int(os.environ["N_TASKS"]) if os.environ.get("N_TASKS") else None,
    }.items() if v is not None}],
}

from harbor.job import JobConfig
JobConfig.model_validate(cfg)
with open(job_cfg_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PYJOB

    echo "[harbor] job config -> $JOB_CFG"
    sed 's/^/    /' "$JOB_CFG"
}

run_harbor_eval() {
    echo "[harbor] starting run..."
    "${harbor_cmd[@]}" 2>&1 | tee "$EVAL_LOG"
}

print_eval_tally() {
    local solved scored
    solved=$(find "$EVAL_JOBS_DIR/$EXP_NAME" -maxdepth 3 -name reward.txt -exec cat {} \; 2>/dev/null | grep -c '^1$' || true)
    scored=$(find "$EVAL_JOBS_DIR/$EXP_NAME" -maxdepth 3 -name reward.txt 2>/dev/null | wc -l)
    echo "=== DONE: solved=$solved / scored=$scored  ($EVAL_JOBS_DIR/$EXP_NAME) ==="
}
