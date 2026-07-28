#!/usr/bin/env bash

write_job_config() {
    export EXP_NAME N_CONCURRENT MAX_RETRIES SERVED_MODEL_NAME LLM_BASE_URL LLM_API_KEY
    export DATASET_PATH DATASET_NAME N_TASKS EVAL_AGENT_NAME
    export HARBOR_ENVIRONMENT_IMPORT_PATH HARBOR_ENVIRONMENT_FORCE_BUILD HARBOR_ENVIRONMENT_DELETE
    export HARBOR_ENVIRONMENT_OVERRIDE_CPUS HARBOR_ENVIRONMENT_OVERRIDE_MEMORY_MB HARBOR_HOSTPATH_MOUNTS
    export HARBOR_AGENT_IMPORT_PATH HARBOR_AGENT_RUNTIME_IMAGE HARBOR_AGENT_RUNTIME_MOUNT_PATH HARBOR_AGENT_RUNTIME_IMAGE_SUBPATH
    export K8S_POD_ACTIVE_DEADLINE_SECONDS

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

cfg = {
    "job_name": os.environ["EXP_NAME"],
    "n_attempts": 1,
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
        "env": {
            "LLM_BASE_URL": os.environ["LLM_BASE_URL"],
            "LLM_API_KEY": os.environ.get("LLM_API_KEY", "dummy"),
            "LLM_MODEL": f"hosted_vllm/{served}",
        },
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
