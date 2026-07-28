# Template Runners

This directory contains reusable `.env` template modules for the train, infer,
and eval runners.

The train entrypoint is:

```bash
scripts/train/train.sh [--dry-run] scripts/train/configs/<experiment>.env
```

`train.sh` does not generate another bash script. It sources one complete
experiment config, sources the template modules selected by that config, prints
the final environment and launch command, then starts the Ray cluster and verl
training. `--dry-run` stops after printing the final environment and command.

## Files

| Path | Purpose |
| --- | --- |
| `scripts/train/_template.env` | User-facing skeleton for a new train config. Every `CHANGEME` should be filled explicitly. |
| `scripts/train/configs/*.env` | Complete experiment configs. These choose module variants and set experiment-specific values. |
| `scripts/train/train.sh` | Stable train entrypoint and high-level flow orchestration. |
| `scripts/train/lib/runtime.sh` | Config/template sourcing, required-variable checks, Python setup, log setup, and printed summaries. |
| `scripts/train/lib/hydra_args.sh` | Assembly of verl Hydra overrides from the final environment. |
| `scripts/train/lib/ray.sh` | Ray cleanup, head/worker startup, worker-side W&B helper, and head-only training launch gate. |
| `scripts/templates/**.env` | Reusable defaults for runtime, backend, Harbor, scaffold, train mode/engine, infer serving, and eval serving/job setup. |

## Config Contract

A train config is a normal bash `.env` file. It is sourced first, then the
modules listed in `TEMPLATE_MODULES` are sourced in order.

Config files should set concrete experiment choices directly. Templates provide
module defaults and derived values using the default-assignment form:

```bash
: "${VAR:=default}"
```

That means a value explicitly set in the config remains authoritative even when
templates are sourced afterwards. Derived values in templates can still see the
explicit config values.

The common train config shape is:

```bash
# --- template selection -----------------------------------------------
BACKEND=k8s              # k8s | docker
SCAFFOLD=ohsdk           # ohsdk | oh | cc | oc
TRAINING_MODE=async      # async | sync
MODEL_ENGINE=veomni      # veomni | fsdp

# --- identity ----------------------------------------------------------
PROJECT_NAME=...
EXP_NAME=...

# --- runtime -----------------------------------------------------------
VENV_PATH=/path/to/venv
WANDB_API_KEY=...

# Optional; defaults to ${REPO_ROOT}/logs when unset.
HARBOR_LOG_DIR=...

# --- model -------------------------------------------------------------
MODEL_PATH=...
TOOL_CALL_PARSER=...

# Optional; defaults in verl/common.env.
SERVED_MODEL_NAME=vllm_model

# --- data --------------------------------------------------------------
TRAIN_FILES=/path/to/train.parquet
VAL_FILES=/path/to/val.parquet

# --- cluster topology --------------------------------------------------
NNODES=3
N_NODES_TRAIN=2
N_NODES_ROLLOUT=1
K8S_KUBECONFIG=/path/to/kubeconfig   # required when BACKEND=k8s

# --- cadence -----------------------------------------------------------
TOTAL_EPOCHS=3

# --- overrides ---------------------------------------------------------
MAX_PROMPT=30000
MAX_RESP=170000
SP_SIZE=8
EP_SIZE=1
ENABLE_R3=True

# --- template composition ---------------------------------------------
# Keep this after explicit settings because module paths and derived defaults
# depend on values above.
TEMPLATE_MODULES=(
  runtime/process.env
  "backend/${BACKEND}.env"
  harbor/common.env
  "scaffold/${SCAFFOLD}.env"
  verl/common.env
  "verl/${TRAINING_MODE}.env"
  "verl/${MODEL_ENGINE}.env"
)
```

Use `scripts/train/_template.env` as the authoritative skeleton. A filled example
mirroring the legacy 3-node Qwen3.5 OH-SDK VeOmni script lives at:

```bash
scripts/train/configs/fully_async_3nodes_qwen35_ohsdk_veomni.env
```

## Template Modules

| Module | Owns |
| --- | --- |
| `runtime/process.env` | Process-level environment such as socket interfaces, NCCL/logging defaults, tokenizer/thread knobs, Ray ports, and Ray object store memory. |
| `backend/k8s.env` | Harbor Kubernetes backend defaults, including environment import/type and image strategy defaults. |
| `backend/docker.env` | Harbor Docker backend selector. Site-specific Docker host values stay in configs or caller environment. |
| `harbor/common.env` | Harbor agent, trial, validation, retry, resource, verifier, and timeout defaults shared across scaffolds/backends. |
| `scaffold/ohsdk.env` | OpenHands SDK agent identity and runtime image settings. |
| `scaffold/oh.env` | OpenHands agent identity and runtime image settings. |
| `scaffold/cc.env` | Claude Code agent identity and runtime image settings. |
| `scaffold/oc.env` | OpenCode agent identity and runtime image settings. |
| `verl/common.env` | Verl-native defaults shared by sync/async and VeOmni/FSDP: data, model, actor, rollout, ref, algorithm, topology, experiment/log defaults. |
| `verl/async.env` | Fully-async entry/config selection and `async_training.*` defaults. |
| `verl/sync.env` | Sync entry/config selection and sync-specific train batch defaults. |
| `verl/veomni.env` | VeOmni model-engine-specific actor/ref/router-replay overrides. |
| `verl/fsdp.env` | FSDP model-engine-specific actor/ref/router-replay overrides. |

## Runner Behavior

The runner performs this sequence:

1. Resolve `REPO_ROOT`, `TEMPLATE_ROOT`, and the absolute config path.
2. Source the config with `set -a` so assigned variables are exported.
3. Source each item in `TEMPLATE_MODULES` from `scripts/templates` with `set -a`.
4. Validate required final variables, including `TRAIN_FILES`, `VAL_FILES`,
   `MODEL_PATH`, topology values, and verl entry/config names.
5. Append `${REPO_ROOT}/src` to any existing `PYTHONPATH`.
6. Print launch summary and `=== Final Environment ===`.
7. Build and print `=== Launch Command ===`.
8. In `--dry-run`, stop before Ray startup.
9. In normal mode, clean old Ray/shared-memory state, start or join Ray, and run
   the verl command on the head node only.

The final environment print is important because Harbor configuration is consumed
through environment variables and does not appear in the Hydra command line.

## Adding Or Changing Templates

Prefer adding defaults to the most specific module that owns the setting:

| Setting type | Put it in |
| --- | --- |
| Process/runtime knobs | `runtime/process.env` |
| Harbor backend knobs | `backend/<backend>.env` |
| Harbor agent/common behavior | `harbor/common.env` |
| Agent scaffold identity/runtime | `scaffold/<scaffold>.env` |
| Verl shared native config | `verl/common.env` |
| Async vs sync training behavior | `verl/async.env` or `verl/sync.env` |
| VeOmni vs FSDP engine behavior | `verl/veomni.env` or `verl/fsdp.env` |
| Infer vLLM serving behavior | `infer/vllm.env` |
| Infer rollout/data/log behavior | `infer/common.env` |
| Eval Harbor-native job/data/log behavior | `eval/common.env` |
| Eval plain-vLLM serving behavior | `eval/vllm.env` |
| Train experiment-specific paths/names/site overrides | `scripts/train/configs/<experiment>.env` |
| Infer experiment-specific paths/names/site overrides | `scripts/infer/configs/<experiment>.env` |
| Eval experiment-specific paths/names/site overrides | `scripts/eval/configs/<experiment>.env` |

Template defaults should normally use:

```bash
: "${VAR:=default}"
```

Required values that must come from the final config can use:

```bash
: "${VAR:?VAR is required}"
```

Keep `TEMPLATE_MODULES` at the end of configs so module names and derived values
can depend on earlier explicit settings.

## Quick Checks

Syntax check:

```bash
bash -n scripts/train/train.sh scripts/train/lib/*.sh
bash -n scripts/infer/infer.sh scripts/infer/lib/*.sh scripts/templates/infer/*.env
bash -n scripts/eval/eval.sh scripts/eval/lib/*.sh scripts/templates/eval/*.env
```

Dry-run a config:

```bash
scripts/train/train.sh --dry-run scripts/train/configs/fully_async_3nodes_qwen35_ohsdk_veomni.env
```

Run training:

```bash
scripts/train/train.sh scripts/train/configs/fully_async_3nodes_qwen35_ohsdk_veomni.env
```

Dry-run infer:

```bash
scripts/infer/infer.sh --dry-run scripts/infer/configs/infer_openswe_filtered_ohsdk_qwen3_6_27b_completed4trials.env
```

Dry-run eval:

```bash
scripts/eval/eval.sh --dry-run scripts/eval/configs/qwen36-27b_official_verified.env
```

## Infer Runner

The infer entrypoint follows the same config/template pattern:

```bash
scripts/infer/infer.sh [--dry-run] scripts/infer/configs/<experiment>.env
```

The runner starts one local vLLM server, waits for its health endpoint, then runs:

```bash
utils/eval_swerebench_filtered.py
```

A filled config converted from
`scripts/infer_openswe_filtered_ohsdk_qwen3_6_27b_completed4trials.sh` lives at:

```bash
scripts/infer/configs/infer_openswe_filtered_ohsdk_qwen3_6_27b_completed4trials.env
```

Infer configs use `INDEX_FILE` for the parquet passed to `--index`. This avoids
confusing infer base indexes with train-side `TRAIN_FILES`.

Infer-specific reusable modules are:

| Module | Owns |
| --- | --- |
| `infer/vllm.env` | Single-node vLLM serving defaults: tensor parallel size, GPU count, port, API base, and local LLM env defaults. |
| `infer/common.env` | Infer identity, model/data/output paths, rollout concurrency/trials, context split, and log defaults. |

The infer runner also prints `=== Final Environment ===`, `=== vLLM Command ===`,
and `=== Infer Command ===` before launching. Infer currently assumes single-node
serving; distributed vLLM launch flags are intentionally not part of the template
contract.


## Eval Runner

The eval entrypoint follows the same config/template pattern:

```bash
scripts/eval/eval.sh [--dry-run] scripts/eval/configs/<experiment>.env
```

Eval is Harbor-native: the runner starts a local plain vLLM server that loads HF
or merged HF weights directly, writes a strict Harbor `JobConfig` JSON file, and
then launches `harbor run`. This avoids the verl `val_only` path and its training
engine weight-sync assumptions.

A filled config converted from the Qwen3.6 official-verified wrapper lives at:

```bash
scripts/eval/configs/qwen36-27b_official_verified.env
```

Eval configs should set `MODEL_PATH` directly; the old `MODEL_PRESET`-based
`scripts/eval/templates/*.env` files are no longer part of the runner contract. Set
exactly one of `DATASET_PATH` or `DATASET_NAME`.

Eval-specific reusable modules are:

| Module | Owns |
| --- | --- |
| `eval/common.env` | Eval identity, dataset, sampling/context, Harbor job defaults, logs, and job-config path. |
| `eval/vllm.env` | Single-node plain-vLLM serving defaults, expert-parallel flag, readiness budget, and pod-reachable `LLM_BASE_URL`. |

The eval runner prints `=== Final Environment ===`, `=== vLLM Command ===`, and
`=== Harbor Command ===` before launching. In `--dry-run`, it stops before writing
the Harbor job config, starting vLLM, or running Harbor.
