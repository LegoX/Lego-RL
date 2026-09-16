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
