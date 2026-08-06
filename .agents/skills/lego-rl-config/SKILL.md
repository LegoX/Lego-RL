---
name: lego-rl-config
description: Compose, edit, refactor, and validate Lego-RL train/eval/infer .env configs and reusable scripts/templates modules. Use when Codex is asked to generate an experiment config, migrate legacy wrappers into configs, edit template modules, dry-run a train/eval/infer workload, or explain the runner/template/site-env contract for this repository.
---

# Lego-RL Config

Use this skill for Lego-RL configuration work. The current design is a
single runner per workload plus small `.env` experiment configs composed from
reusable modules under `scripts/templates`.

This skill follows the Claude `/rl:*` plugin's layering rule:

> Scripts own deterministic behavior. Skills own orchestration, judgement, and
> the final report.

## Start Here

1. Find the repository root: it contains `scripts/train/train.sh`,
   `scripts/infer/infer.sh`, `scripts/eval/eval.sh`, and `scripts/templates/README.md`.
2. Read `scripts/templates/README.md` first. It is the authoritative runner and
   template contract.
3. Read `references/config-generation.md` before creating or refactoring configs.
4. Inspect only the files relevant to the requested workload:
   `scripts/<kind>/<kind>.sh`, `scripts/<kind>/_template.env`,
   `scripts/<kind>/configs/*.env`, `scripts/<kind>/lib/*.sh`, and the selected
   `scripts/templates/**.env` modules.
5. Preserve user-owned run configs. Do not rewrite unrelated configs, logs,
   checkpoints, trial outputs, or cluster state.

## Workflow

### 1. Resolve The Workload

Classify the request as `train`, `infer`, or `eval`.

- `train`: verl policy training, sync/async mode, VeOmni/FSDP engine,
  `TRAIN_FILES`, `VAL_FILES`, `NNODES`, `N_NODES_TRAIN`, `N_NODES_ROLLOUT`.
- `infer`: batch trajectory generation through `utils/eval_swerebench_filtered.py`,
  `INDEX_FILE`, optional `INSTANCES_FILE`, `RESULTS_DIR`, `OUTPUT_INDEX`,
  single-node vLLM serving knobs such as `GEN_TP`, `GPUS_PER_NODE`, and
  `VLLM_PORT`.
- `eval`: Harbor-native scoring, exact `MODEL_PATH`, exactly one of
  `DATASET_PATH` or `DATASET_NAME`, local plain-vLLM serving, generated
  Harbor `JobConfig`, and `harbor run`.

If a config path is provided, infer the kind from `scripts/<kind>/...`. If only a
bare name is provided, search `scripts/{train,infer,eval}/configs/`. Ask only
when multiple plausible configs match.

### 2. Compose Template Modules

Templates live under `scripts/templates/**.env`, and a config chooses them with
`TEMPLATE_MODULES`. The config is sourced first, then modules are sourced in
order from `scripts/templates`. Template defaults should use:

```bash
: "${VAR:=default}"
```

That means explicit config values are authoritative, while modules provide
defaults and derived values.

Use the current module ownership model:

- `runtime/process.env`: process-level env, sockets, NCCL/logging defaults,
  tokenizer/thread knobs, Ray ports, and Ray object store memory.
- `backend/k8s.env` and `backend/docker.env`: Harbor backend selectors and
  backend defaults.
- `harbor/common.env`: Harbor agent, trial, validation, retry, resource,
  verifier, and timeout defaults shared across workloads.
- `scaffold/{ohsdk,oh,cc,oc}.env`: agent identity and runtime image defaults.
- `verl/common.env`: shared train-side verl data/model/actor/rollout/ref/
  algorithm/topology/log defaults.
- `verl/{async,sync}.env`: train mode entrypoint/config and mode-specific
  defaults.
- `verl/{veomni,fsdp}.env`: train model-engine-specific actor/ref/router-replay
  overrides.
- `infer/{vllm,common}.env`: infer single-node vLLM serving plus infer
  rollout/data/output/log defaults.
- `eval/{common,vllm}.env`: Harbor-native eval job/data/log defaults plus
  single-node plain-vLLM serving defaults.

Keep `TEMPLATE_MODULES` at the end of configs so module names and derived
defaults can depend on earlier explicit settings.

### 3. Generate Or Edit The Config

Write generated configs to exactly one of:

- `scripts/train/configs/<name>.env`
- `scripts/infer/configs/<name>.env`
- `scripts/eval/configs/<name>.env`

Use the workload skeleton as the starting point:

- `scripts/train/_template.env`
- `scripts/infer/_template.env`
- `scripts/eval/_template.env`

Keep configs readable as experiment records: template selection first, identity,
runtime, model, data/output, topology or serving, optional overrides, then
`TEMPLATE_MODULES`. Keep generated configs small; do not copy every template
default into the config.

Important current variable names:

- Use `EXP_NAME`, not `EXP_TAG`.
- Train uses `TRAIN_FILES` and `VAL_FILES`.
- Infer uses `INDEX_FILE` for the parquet passed to `--index`.
- Eval uses `MODEL_PATH` directly; old `MODEL_PRESET`-based eval templates are
  not part of the current runner contract.
- Eval must set exactly one of `DATASET_PATH` or `DATASET_NAME`.

### 4. Validate Through The Runner

Do not re-implement runner checks. Use the workload runner's dry-run path:

```bash
bash scripts/<kind>/<kind>.sh --dry-run scripts/<kind>/configs/<config>.env
```

Dry-run sources the config and modules, validates required variables, initializes
local runtime state, prints `=== Final Environment ===`, prints the launch
command block, then exits before Ray startup, vLLM startup, Harbor job writing,
or training/eval/infer execution.

For static syntax checks, use the commands in `scripts/templates/README.md`, for
example:

```bash
bash -n scripts/train/train.sh scripts/train/lib/*.sh
bash -n scripts/infer/infer.sh scripts/infer/lib/*.sh scripts/templates/infer/*.env
bash -n scripts/eval/eval.sh scripts/eval/lib/*.sh scripts/templates/eval/*.env
```

If validation fails, report the exact fatal/error lines and adjust only the
config or template layer that owns the value.

### 5. Report

Answer in Chinese unless the user asked otherwise. Include:

- config path or template path created/changed
- template modules used or introduced
- key resolved axes: kind, backend, scaffold, model, and topology/serving; for
  train also mode and engine
- validation commands run and their result
- any manual values still needed, especially data paths, checkpoint/model paths,
  kubeconfig/backend/site values, registry/image/mount values, and multi-node
  host/rank values

## Refactor Rules

- Keep existing runners as the execution contract:
  `scripts/train/train.sh`, `scripts/infer/infer.sh`, and `scripts/eval/eval.sh`.
- Move reusable defaults to `scripts/templates`, not `scripts/lib`.
- Keep `scripts/lib` for executable shell helpers and workload orchestration.
- Generated configs belong under the workload's `configs/` directory, never
  under `scripts/templates`.
- Site-specific paths, kubeconfigs, registries, mounts, Docker hosts, and
  secrets remain in site env, caller env, or explicit run configs when the site
  requires them. Do not bake them into shared templates.
- Do not delete legacy monolithic scripts unless the user explicitly asks. When
  migrating one, preserve behavior with one generated config plus reusable
  modules, then validate with `--dry-run`.
- Prefer the existing module tree over introducing new dimensions. Add a module
  only when an existing module has the wrong ownership boundary.

## Guardrails

- Do not launch training/eval/infer unless the user explicitly asks and confirms.
- Do not kill processes, clear `/dev/shm`, run `ray stop`, delete logs, delete
  checkpoints, or mutate the cluster.
- Do not SSH to worker nodes; for multi-node flows, print the commands the user
  must run on each node.
- Do not claim a config is validated without runner output.
- Do not hardcode secrets such as `WANDB_API_KEY`, kubeconfig contents, registry
  credentials, or personal tokens into templates or generated configs.
