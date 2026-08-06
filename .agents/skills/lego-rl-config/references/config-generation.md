# Lego-RL Config Generation Reference

This reference summarizes the current repository-specific config model for
creating or refactoring Lego-RL train, infer, and eval configs.

## Current Runner Contract

Run kinds:

```bash
bash scripts/train/train.sh [--dry-run] scripts/train/configs/<config>.env
bash scripts/infer/infer.sh [--dry-run] scripts/infer/configs/<config>.env
bash scripts/eval/eval.sh [--dry-run] scripts/eval/configs/<config>.env
```

Each runner:

1. resolves `REPO_ROOT`, `TEMPLATE_ROOT`, and the absolute config path
2. sources the config with `set -a`
3. sources each `TEMPLATE_MODULES` item from `scripts/templates`
4. validates required final variables
5. appends `${REPO_ROOT}/src` to `PYTHONPATH`
6. prints a launch summary and `=== Final Environment ===`
7. builds and prints the workload command block
8. exits before launch when `--dry-run` is set

The runner output is the source of truth for resolved parameters. Skills should
read runner output rather than recomputing hidden defaults.

## Config Layers

| Layer | Location | Ownership |
| --- | --- | --- |
| Site/caller values | site env, user env, or explicit run config | cluster/user-specific values |
| Template modules | `scripts/templates/**.env` | reusable defaults and derived values |
| Generated config | `scripts/<kind>/configs/*.env` | one concrete experiment |
| Lib helpers | `scripts/<kind>/lib/*.sh` | executable workload logic |
| Entrypoints | `scripts/<kind>/<kind>.sh` | stable runner contract |

Do not mix these layers. A shared template must not contain a user's kubeconfig,
registry credential, local checkpoint root, or secret. A generated config should
not inline every default from the template modules.

## Workload Shapes

### Train

Entrypoint:

```bash
scripts/train/train.sh [--dry-run] scripts/train/configs/<experiment>.env
```

Authoritative skeleton:

```bash
scripts/train/_template.env
```

Required experiment axes:

- `BACKEND`: `k8s` or `docker`
- `SCAFFOLD`: `ohsdk`, `oh`, `cc`, or `oc`
- `TRAINING_MODE`: `async` or `sync`
- `MODEL_ENGINE`: `veomni` or `fsdp`
- `PROJECT_NAME`, `EXP_NAME`
- `VENV_PATH`, `MODEL_PATH`, `TOOL_CALL_PARSER`
- `TRAIN_FILES`, `VAL_FILES`
- `NNODES`, `N_NODES_TRAIN`, `N_NODES_ROLLOUT`
- `TOTAL_EPOCHS`
- `K8S_KUBECONFIG` when `BACKEND=k8s`

Common train modules:

```bash
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

The train runner starts or joins Ray only outside `--dry-run`, and runs the verl
command on the head node only.

### Infer

Entrypoint:

```bash
scripts/infer/infer.sh [--dry-run] scripts/infer/configs/<experiment>.env
```

Authoritative skeleton:

```bash
scripts/infer/_template.env
```

Infer currently assumes one local vLLM server. Distributed vLLM launch flags are
not part of the infer template contract.

Required experiment axes:

- `BACKEND`, `SCAFFOLD`
- `PROJECT_NAME`, `EXP_NAME`
- `VENV_PATH`, `MODEL_PATH`, `TOOL_CALL_PARSER`
- `INDEX_FILE`, `RESULTS_DIR`, `OUTPUT_INDEX`
- serving and rollout values, usually from `infer/vllm.env` and
  `infer/common.env`: `GEN_TP`, `GPUS_PER_NODE`, `VLLM_PORT`,
  `VLLM_MAX_MODEL_LEN`, `N_TRIALS`, `N_CONCURRENT`
- `K8S_KUBECONFIG` when `BACKEND=k8s`

Common infer modules:

```bash
TEMPLATE_MODULES=(
  runtime/process.env
  infer/vllm.env
  infer/common.env
  "backend/${BACKEND}.env"
  harbor/common.env
  "scaffold/${SCAFFOLD}.env"
)
```

The infer runner prints `=== Final Environment ===`, `=== vLLM Command ===`, and
`=== Infer Command ===`; `--dry-run` exits before vLLM startup and inference.

### Eval

Entrypoint:

```bash
scripts/eval/eval.sh [--dry-run] scripts/eval/configs/<experiment>.env
```

Authoritative skeleton:

```bash
scripts/eval/_template.env
```

Eval is Harbor-native. It starts a local plain vLLM server, writes a strict
Harbor `JobConfig` JSON file, then launches `harbor run`. It does not use the
verl `val_only` path.

Required experiment axes:

- `BACKEND`, `SCAFFOLD`
- `PROJECT_NAME`, `EXP_NAME`
- `VENV_PATH`, `MODEL_PATH`
- exactly one of `DATASET_PATH` or `DATASET_NAME`
- serving/job values, usually from `eval/common.env` and `eval/vllm.env`:
  `N_CONCURRENT`, `MAX_RETRIES`, `EVAL_TEMPERATURE`, `MAX_INPUT_TOKENS`,
  `MAX_OUTPUT_TOKENS`, `MAX_MODEL_LEN`, `GEN_TP`, `VLLM_PORT`
- `K8S_KUBECONFIG` when `BACKEND=k8s`

Common eval modules:

```bash
TEMPLATE_MODULES=(
  runtime/process.env
  eval/common.env
  eval/vllm.env
  "backend/${BACKEND}.env"
  harbor/common.env
  "scaffold/${SCAFFOLD}.env"
)
```

The eval runner prints `=== Final Environment ===`, `=== vLLM Command ===`, and
`=== Harbor Command ===`; `--dry-run` exits before writing the Harbor job config,
starting vLLM, or running Harbor.

## Template Module Ownership

| Module | Owns |
| --- | --- |
| `runtime/process.env` | Process-level environment such as socket interfaces, NCCL/logging defaults, tokenizer/thread knobs, Ray ports, and Ray object store memory. |
| `backend/k8s.env` | Harbor Kubernetes backend defaults, environment import/type, and image strategy defaults. |
| `backend/docker.env` | Harbor Docker backend selector. Site-specific Docker host values stay in configs or caller environment. |
| `harbor/common.env` | Harbor agent, trial, validation, retry, resource, verifier, and timeout defaults shared across scaffolds/backends. |
| `scaffold/ohsdk.env` | OpenHands SDK agent identity and runtime image defaults. |
| `scaffold/oh.env` | OpenHands agent identity and runtime image defaults. |
| `scaffold/cc.env` | Claude Code agent identity and runtime image defaults. |
| `scaffold/oc.env` | OpenCode agent identity and runtime image defaults. |
| `verl/common.env` | Verl-native defaults shared by sync/async and VeOmni/FSDP: data, model, actor, rollout, ref, algorithm, topology, experiment/log defaults. |
| `verl/async.env` | Fully-async entry/config selection and `async_training.*` defaults. |
| `verl/sync.env` | Sync entry/config selection and sync-specific train batch defaults. |
| `verl/veomni.env` | VeOmni model-engine-specific actor/ref/router-replay overrides. |
| `verl/fsdp.env` | FSDP model-engine-specific actor/ref/router-replay overrides. |
| `infer/vllm.env` | Infer single-node vLLM serving defaults: tensor parallel size, GPU count, port, API base, and local LLM env defaults. |
| `infer/common.env` | Infer identity, model/data/output paths, rollout concurrency/trials, context split, and log defaults. |
| `eval/common.env` | Eval identity, dataset, sampling/context, Harbor job defaults, logs, and job-config path. |
| `eval/vllm.env` | Eval single-node plain-vLLM serving defaults, expert-parallel flag, readiness budget, and pod-reachable `LLM_BASE_URL`. |

## Parameter Tiers

T0 template axes:

- runtime process defaults
- backend: `BACKEND`
- scaffold: `SCAFFOLD`
- train mode: `TRAINING_MODE`
- train engine: `MODEL_ENGINE`
- workload-specific serving modules for infer/eval

T1 required experiment values:

- train: identity, runtime, model, train/val files, cluster topology, cadence
- infer: identity, runtime, model, index/results/output paths, rollout shape
- eval: identity, runtime, model, dataset selector, sampling/context/job shape

T2/T3 optional overrides:

- cadence: `SAVE_FREQ`, `TEST_FREQ`, `TOTAL_EPOCHS`
- rollout scale: `TRAIN_BSZ`, `N_RESP`, `N_CONCURRENT`, `N_TRIALS`
- context: `MAX_PROMPT`, `MAX_RESP`, `MAX_INPUT_TOKENS`,
  `MAX_OUTPUT_TOKENS`, `MAX_MODEL_LEN`
- algorithm: advantage estimator, policy loss mode, learning rate, scheduler
- advanced behavior: R3, importance sampling, trajectory filters, GPU memory
  utilization, vLLM extra args

T4 site values:

- kubeconfig paths, Docker host, node/path rewrites, registry/image mirrors,
  hostpath mounts, W&B keys, local model/checkpoint roots, and credentials

T4 values should stay in site env, caller env, or explicit run configs when a
site requires them. Never add them to shared templates.

## Validation Checklist

After creating or changing a generated config:

```bash
bash scripts/<kind>/<kind>.sh --dry-run scripts/<kind>/configs/<name>.env
```

Treat any non-zero exit or `[FATAL]` line as blocking. A successful dry-run must
print the final environment and command block while avoiding the real launch.

For syntax checks:

```bash
bash -n scripts/train/train.sh scripts/train/lib/*.sh
bash -n scripts/infer/infer.sh scripts/infer/lib/*.sh scripts/templates/infer/*.env
bash -n scripts/eval/eval.sh scripts/eval/lib/*.sh scripts/templates/eval/*.env
```

Only claim success after reporting which commands were run. If validation cannot
run on the current host, explain why and list the exact command for the user.
