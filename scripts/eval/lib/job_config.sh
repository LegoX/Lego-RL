#!/usr/bin/env bash

write_job_config() {
    export EXP_NAME N_CONCURRENT MAX_RETRIES SERVED_MODEL_NAME LLM_BASE_URL LLM_API_KEY
    export DATASET_PATH DATASET_NAME N_TASKS EVAL_AGENT_NAME
    # Harness-specific endpoint/model/env declarations come from the scaffold
    # module.  The generated JobConfig is consumed by a separate `harbor run`
    # process, so render those declarations into each agent trial here.
    export SCAFFOLD HARBOR_AGENT_NAME HARBOR_AGENT_TEMPERATURE
    export HARBOR_HARNESS_PROTOCOL HARBOR_HARNESS_MODEL_TEMPLATE
    export HARBOR_HARNESS_ENDPOINT_TEMPLATE HARBOR_HARNESS_ENV_MAP
    export HARBOR_HARNESS_KWARGS_MAP HARBOR_HARNESS_ALLOW_ENDPOINT_HOST
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
    "agents": [],
    "datasets": [{k: v for k, v in {
        "path": os.environ.get("DATASET_PATH") or None,
        "name": os.environ.get("DATASET_NAME") or None,
        "n_tasks": int(os.environ["N_TASKS"]) if os.environ.get("N_TASKS") else None,
    }.items() if v is not None}],
}

agent_import_path = os.environ["HARBOR_AGENT_IMPORT_PATH"]

llm_base = os.environ["LLM_BASE_URL"]
llm_key = os.environ.get("LLM_API_KEY", "dummy")
harness_protocol = os.environ.get("HARBOR_HARNESS_PROTOCOL", "")
anthropic_key = os.environ.get("ANTHROPIC_API_KEY", llm_key)
openai_key = os.environ.get("OPENAI_API_KEY", llm_key)
temperature = float(
    os.environ.get(
        "HARBOR_AGENT_TEMPERATURE",
        os.environ.get("EVAL_TEMPERATURE", "0.7"),
    )
)
from verl_patch.agent_loop.harness import render_harness_contract
contract_raw = os.environ.get("HARBOR_HARNESS_RENDER_CONTRACT", "")
if contract_raw:
    contract = json.loads(contract_raw)
else:
    # Compatibility for callers that source an older scaffold without the
    # aggregate contract variable.
    contract = {
        "model_template": os.environ.get("HARBOR_HARNESS_MODEL_TEMPLATE", "hosted_vllm/{served}"),
        "endpoint_template": os.environ.get("HARBOR_HARNESS_ENDPOINT_TEMPLATE", "{llm_base}"),
        "kwargs": json.loads(os.environ.get("HARBOR_HARNESS_KWARGS_MAP", "{}")),
        "env": json.loads(os.environ.get("HARBOR_HARNESS_ENV_MAP", "{}")),
        "allow_endpoint_host": os.environ.get("HARBOR_HARNESS_ALLOW_ENDPOINT_HOST", "true").lower() in {"1", "true", "yes", "on"},
    }
from urllib.parse import urlsplit, urlunsplit
parsed = urlsplit(llm_base)
path = parsed.path.rstrip("/")
if path == "/v1": path = ""
elif path.endswith("/v1"): path = path[:-3]
anthropic_base = urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment))
rendered = render_harness_contract(contract, openai_base=llm_base, anthropic_base=anthropic_base, openai_api_key=llm_key, anthropic_api_key=anthropic_key, served=served, temperature=temperature)

agent_name_raw = os.environ.get("HARBOR_AGENT_NAME")
if not agent_name_raw or agent_name_raw.strip().lower() == "null":
    agent_name = None
else:
    agent_name = agent_name_raw
agent_cfg = {
    # A null name lets Harbor dispatch any custom import_path.  A scaffold may
    # explicitly set HARBOR_AGENT_NAME when it wants a built-in Harbor agent.
    "name": agent_name,
    "import_path": agent_import_path,
    "model_name": rendered["model_name"],
    "kwargs": rendered["kwargs"],
    "env": rendered["env"],
    "extra_allowed_hosts": [],
}

allow_endpoint_host = os.environ.get(
    "HARBOR_HARNESS_ALLOW_ENDPOINT_HOST", "true"
).lower() in {"1", "true", "yes", "on"}
if contract["allow_endpoint_host"]:
    agent_cfg["extra_allowed_hosts"].append(rendered["endpoint_host"])
cfg["agents"] = [agent_cfg]

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
