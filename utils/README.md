# utils/ — Python helper scripts

Standalone Python tools used by the Lego-RL training / evaluation flow. `scripts/` holds
only the `.sh` runners, which call into this directory as `$SCRIPT_DIR/../utils/xxx.py` or
`$REPO_ROOT/utils/xxx.py`.

> Path convention: `SCRIPT_DIR` = `scripts/`, `REPO_ROOT` = the repository root, so
> `$REPO_ROOT/utils/` is this directory.

---

## DeepSWE

`scripts/eval/configs/qwen38-27b_deepswe.env` evaluates the original
[Datacurve DeepSWE](https://github.com/datacurve-ai/deep-swe) tasks using this
repository's Harbor fork, Claude Code and Kubernetes. No dataset converter or
Pier installation is needed. The official leaderboard's Pier + mini-swe-agent
scaffold is different.

The task schema and collect commands were checked against all 113 tasks at
`0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea` (v1.1). Keep the dataset revision
with your experiment record:

```bash
git clone https://github.com/datacurve-ai/deep-swe /path/to/deep-swe
git -C /path/to/deep-swe checkout 0b9fabbb63b9104d678fe965e1632f2dd9eaa2ea
```

### Harbor support

`scripts/setup_env.sh` pins the ydu Harbor fork to
`77282683ba93315ae1175b34230e67c1cd233255` and applies `patches/harbor.patch`.
For an existing checkout at that revision, apply the patch once before launching
new jobs (do not replace source while a job is running):

```bash
git -C third_party/harbor apply --check "$PWD/patches/harbor.patch"
git -C third_party/harbor apply "$PWD/patches/harbor.patch"
```

The patch adds `verifier.environment_mode`, `verifier.environment`,
`verifier.collect` (main service only), and task-level `artifacts`. Existing
shared-environment tasks keep their original verification path. Separate tasks
run collect hooks before verification, copy declared artifacts to the host, and
upload them into a fresh verifier environment. Explicit verifier resource/env
fields override the task environment; omitted fields, including the prebuilt
image, inherit from it. Agent runtime and host mounts are not inherited by the
verifier. Each verifier retry uses a new environment and log directory.

Collect/copy errors are reported as infrastructure errors, rather than silently
grading an empty submission. A successfully collected zero-byte patch remains
a valid empty submission. This error reporting is intentionally stricter than
Pier's best-effort collect hooks. Verifier logs are retained on errors/timeouts
and the separate environment is cleaned up. Canonical scores and logs remain
under `<trial>/verifier/`; individual attempts live under
`<trial>/verifier-environments/`.

### Configure and validate

`qwen38-27b_deepswe.env` selects Qwen3.8-27B with
Claude Code and site-specific model, dataset and Kubernetes paths.
Before launching, set `MODEL_PATH`, `DATASET_PATH`, `K8S_KUBECONFIG` and
`HARBOR_NETADMIN_IMAGE`. Workers must be able to pull the configured task images
and Claude Code runtime image.
No SWE-rebench/Pro grading-toolchain host mount is needed. The endpoint in
`LLM_BASE_URL` must be reachable from the task pods.

The [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) reports
DeepSWE 1.1 at 42.2 with Claude Code, temperature 1.0, top_p 0.95 and a 256K
context window. The config sets these sampling defaults in vLLM and sets
`CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144` in the actual agent environment.
Thinking and preserved reasoning history stay enabled with Qwen's default
xhigh effort; `qwen3` parses reasoning and `qwen3_coder` parses tool calls.
CC 2.1.118, 32768 output tokens, one attempt per task and task-defined timeouts
are local choices: the card does not specify those details for DeepSWE. Its
avg@3/8-hour/32768 note belongs to QwenSWEBench, not DeepSWE. A model evaluation
is still needed to measure whether this setup reaches the reported score.

Important settings in this config:

- `HARBOR_K8S_SKIP_DOCKERFILE=true` consumes the complete prebuilt task image;
  it preserves the image's PATH and /tmp (which can contain offline caches),
  and skips Dockerfile RUN replay and extra musl setup. It requires `docker_image`.
- `HARBOR_BLOCK_GIT_HISTORY_LEAK=0` preserves the official image's pruned Git
  history. DeepSWE collects `git diff --binary <base_commit> HEAD`, so the agent
  must commit its changes as requested by the original instruction. Rebuilding
  or restoring `.git` would break this submission protocol.
- `HARBOR_EVAL_CC_SUBMISSION_GUARD=1` in `EVAL_AGENT_EXTRA_ENV` initializes
  missing Git author fields locally in `/app` before CC runs (existing identity
  is preserved), and fails explicitly if setup fails. It leaves the original
  task instruction unchanged; the task already requests a commit. Training does
  not enable it.
- `python -m harbor_patch.eval_tools tally <job-dir>` reports job completion,
  pending final results and aggregate exceptions, including failures without a
  task result file. `EVAL_REWARD_KEY` and `EVAL_REWARD_VALUES` select the score;
  `solved` counts the highest allowed value (binary defaults: `reward`, `[0,1]`).
  `python -m harbor_patch.eval_tools diagnose <job-dir>` lists missing test IDs,
  other failed tests, empty patches, and raw-log storage/build/timeout/report
  signatures without modifying scores. Build failures are not automatically
  classified as infrastructure failures. Missing IDs
  can reflect renamed tests, compilation failures or interrupted test runs;
  inspect the raw logs before attributing a cause.
- Eval acquires a host-local lock for its vLLM port before launching. Duplicate
  launches fail even during model loading. Cleanup terminates only this runner's
  server process tree, rather than all `vllm serve` processes on the host.
- `EVAL_TASK_NAMES` accepts a JSON array of exact task names or globs for
  diagnostic subsets. Local preflight rejects unmatched filters. Subset scores
  are diagnostic and do not replace a full-benchmark score.
- `HARBOR_K8S_STRICT_NETWORK_POLICY=true` makes network-policy installation
  failures fatal. `HARBOR_K8S_ALLOW_PRIVATE_NETWORK=false` restricts agent
  access to explicitly allowed hosts (the LLM endpoint), loopback and cluster
  DNS; it disables the backend's legacy blanket RFC1918 exception. Verifier networking
  follows the task's `no-network` policy. Offline external connections are
  rejected immediately so optional web assets cannot stall local browser tests.
- Agent timeout and CPU/memory overrides are `null`, retaining each task's
  own budgets. The Pod deadline is a separate backstop; adjust it if changing
  the task budget. Start with low concurrency because verifier pods also use
  cluster resources.
- Memory requests remain 8 GiB; `HARBOR_ENVIRONMENT_MEMORY_LIMIT_FACTOR=1.5`
  sets a 12 GiB hard limit. This is a local resource difference, not an exact
  reproduction of an 8 GiB limit.
- `HARBOR_ENVIRONMENT_OVERRIDE_CPU_LIMIT=2` enforces v1.1's declared CPU quota.
  CPU requests alone let joblib detect the whole node and can exhaust memory
  in tasks such as `skrub-duration-encoding`.
- `HARBOR_K8S_NOFILE=65536` raises the soft open-file limit for commands in
  the task/verifier container. MongoDB in `eicrud-keyset-pagination-cursor`
  exhausts containerd's default 1024 descriptors. The container's hard limit
  must permit the configured value; an unsuccessful `ulimit` fails the command.
- `EVAL_REQUIRE_SEPARATE_VERIFIER=true` rejects an unpatched Harbor before
  starting vLLM. When the dataset is present, preflight validates every task's
  schema. A dry-run with a placeholder dataset only checks runner capability.
- `EVAL_REWARD_KEY=reward` computes the job mean from the binary score while
  retaining DeepSWE's other reward fields (`f2p`, `p2p`, `partial`, and counts)
  for diagnostics. Standalone Harbor jobs should set
  `metrics: [{type: mean, kwargs: {reward_key: reward}}]` as well.
- `EVAL_REWARD_VALUES='[0,1]'` sets `verifier.allowed_rewards.reward` to
  `[0,1]`. A crash sentinel (`reward=-1`) or missing/invalid primary score
  raises `InvalidRewardError` before aggregation. Verification retries once
  using the same collected submission in a fresh container. Persistent errors
  remain unscored, with logs retained; the default job retry policy excludes
  this exception to avoid regenerating the model answer. As with other missing
  rewards, the job mean counts this trial as zero; report errors alongside it.
  Standalone Harbor jobs should set `verifier: {allowed_rewards: {reward: [0,1]}}`.

```bash
bash scripts/eval/eval.sh --dry-run scripts/eval/configs/qwen38-27b_deepswe.env
```

Before a full model run, validate the configured backend with official oracle
and nop agents on a small subset: oracle should apply/commit the solution and
score 1, while nop should produce an empty patch and score 0. Check
`artifacts/model.patch`, verifier logs, image pulls and endpoint/network access.
Then run a small Claude Code subset using `N_TASKS` before increasing concurrency.
The runner's final tally reads JSON or text rewards, counts only top-level
trials, and reports infrastructure errors separately. Agent errors with a valid
verifier reward are both scored and included in the error count.

Local regression tests (no GPU, containers or cluster required):

```bash
.venv/bin/python -m pytest -c third_party/harbor/pyproject.toml \
  third_party/harbor/tests/unit/test_separate_verifier.py tests/eval/test_deepswe.py
```

## Called from the shell runners

Changing these affects training / evaluation runs already in flight.

### `eval_swerebench_filtered.py`

The offline-evaluation core for the SWE-rebench filtered subset. Runs N trials per instance
against an already-serving vLLM, writes `<instance_id>/trial_<i>.json` (trajectory + reward),
and resumes cleanly (instances that already have their full trial count are skipped). On
completion it emits a summary CSV and a filtered index parquet.

### `apply_r3_vllm_patches.py`

Re-applies the R3 (Rollout Routing Replay) patches to the *installed* vLLM in site-packages —
which is outside any git repository, so the patches are lost on every `pip install`, venv
rebuild, or move to a new machine. Idempotent; `--check` reports without writing. Only needed
on older vLLM (<0.19, i.e. before the built-in routed-experts capturer).

### `apply_veomni_valuehead_patch.py`

Makes VeOmni able to build a **value model (critic)** at all. VeOmni's per-family model
registry dispatches `"<Family>ForTokenClassification"` by substring and, unpatched, falls
through to the CausalLM class — so a critic (SAO / PPO) is silently built as a language model
and GAE sees vocab-sized "values" (surfaced as `size of tensor a (8) must match ... (151936)`).
veomni is a wheel, not a managed checkout, so this edits site-packages: `setup_env.sh` runs it
once after installing, and `train.sh` re-runs it before any `CRITIC_ENABLE=True` VeOmni launch.
It also installs the Qwen3.5-MoE `ForTokenClassification` sidecar from
`src/verl_patch/models/` (Qwen3.5-MoE ships no such class). Idempotent; `--check` reports
without writing and exits non-zero when a patch is missing.

### `merge_critic_ckpt.sh` / `merge_critic_ckpt_streaming.py`

Merge a sharded **critic** checkpoint (`global_step_N/critic/`, which has no loadable weights)
into a HuggingFace directory that `CRITIC_MODEL_PATH` can load — the warm-start critic that
stabilised the SAO 35B runs. Both fix a verl trap first: the critic's saved `config.json` says
`architectures: [...ForCausalLM]`, so the stock merger drops the value head. The streaming
variant bounds host memory (a 30B critic needs >120 GB with the stock path) and **refuses
Qwen3.5-MoE checkpoints**, whose fused-expert layout it would silently corrupt; for those use
`merge_qwen35_fsdp_to_hf.py` on the critic directory and rewrite the architecture by hand.

Usage: `bash utils/merge_critic_ckpt.sh <step>/critic <out_dir>` (fp32 shards → bf16).

### `merge_veomni_fsdp_to_hf.py`

Merges a veomni-FSDP2 actor checkpoint (`model_world_size_N_rank_*.pt`, whose 1-D mesh is named
`dp_shard_sp`) into HuggingFace safetensors. It monkeypatches verl's `FSDPModelMerger` to accept
that mesh name and handles the veomni fused-MoE → HF per-expert layout conversion.

Usage: `python merge_veomni_fsdp_to_hf.py --local_dir <step>/actor --target_dir <out>`

> ⚠️ **Qwen3-MoE / dense only. On Qwen3.5-MoE it silently corrupts the weights** — exit code 0,
> plausible file sizes, but 1811 keys become 31666 and the experts end up in the per-expert
> layout. For Qwen3.5 use `merge_qwen35_fsdp_to_hf.py` below.

### `merge_qwen35_fsdp_to_hf.py`

For Qwen3.5-MoE (`Qwen3_5MoeForConditionalGeneration`) specifically. Its shards are already in
the target HF layout (fused experts, single-level nesting), so after reusing verl's shard
aggregation it **writes safetensors directly, preserving key names verbatim**, bypassing HF's
`save_pretrained()` which would rewrite them. It also backfills `mtp.*` from the base model
(785 tensors for the untrained speculative head, which veomni does not save). A layout mismatch
fails an assertion rather than writing bad weights.

Usage: `python merge_qwen35_fsdp_to_hf.py --local_dir <step>/actor --target_dir <out> --base_dir <base_hf>`

### `verify_merged_hf_keys.py`

**Run this after every merge, before the weights touch a GPU.** It diffs key by key against the
base model's `model.safetensors.index.json` (pure CPU, a few seconds) and asserts
`UNEXPECTED == 0`, `SHAPE MISMATCH == 0`, and that `MISSING` contains only `mtp.*`. A merge with
wrong keys does not crash, does not warn, and produces normal-looking file sizes — it just makes
vLLM emit degenerate text. This is the only reliable place to catch it.

Usage: `python verify_merged_hf_keys.py <merged_hf_dir> <base_hf_dir>` (exit 0 = PASS)

---

## Standalone data / ops tools

Run by hand; not on the critical path of the training runners.

### `create_task_index.py`

Builds the task-index parquet for online RL training — a task *registry*, **not** training data.
Each row tells the trainer which task to run; the actual trajectories are generated live against
the current model at every training step. This is the one up-front data-preparation step online
RL needs.

### `convert_swerebench_filtered.py`

Converts the `nebius/SWE-rebench` filtered subset into Harbor task directories plus an index
parquet (one directory per instance).

Usage: `--output-dir ... --index-dir ...`

### Setting up an official SWE-bench benchmark

Both Multilingual and Pro are *pinned-image* benchmarks: nothing is built in-pod, everything is
pulled. The order matters, and skipping a step produces `reward 0` on a correct patch rather than
an error.

```bash
# 1. convert           -> Harbor task dirs + index parquet
python utils/convert_swebench_multilingual.py --output-dir <ds> --index-path-prefix <as the training host sees it>

# 2. offline grader    -> mounted at /opt/grading (Multilingual REQUIRES it, see below)
OUT=<toolchain> bash utils/build_grading_toolchain_v2.sh

# 3. mirror the images -> only if the sandbox nodes cannot reach Docker Hub
sed -n 's/^docker_image = "\(.*\)"/\1/p' <ds>/*/task.toml | sort -u > images.txt
LIST=images.txt LOGDIR=<logs> bash utils/populate_swebench_hub.sh    # then set HARBOR_NYDUS_MIRROR

# 4. smoke-test        -> oracle must score 1, empty must score 0, on a few tasks per language
GRADING_TOOLCHAIN=<toolchain> bash utils/oracle_smoke_task.sh <ds>/<instance> oracle

# 5. run               -> scripts/eval/configs/qwen35-35b_official_{multilingual,pro}.env
```

**Multilingual cannot be scored without step 2.** Its verifier ends in `uv run parser.py`, which
resolves swebench from PyPI; sandboxes have no egress, so `test.sh` takes its fallback branch and
writes `reward 0` unconditionally — and most of these images ship no Python to run a grader with.

**Pro has two subset-specific traps**, both covered by `convert_swebench_pro.py` /
`build_pro_goproxy.sh` below: its 266 python tasks need the task image's own python in the agent's
tool shell (`OH_SDK_RESTORE_TASK_ENV`, ~8 points on that subset), and 40 of its go tasks need an
offline module proxy or they are unsolvable outright.

> ⚠️ **Do not train on Pro as it stands.** The verifier restores exactly the pathspec of
> `before_repo_set_cmd`'s last line, as the official harness does — but for **440 of the 731**
> instances the test files it actually runs are a superset of that, and the extra ones are left as
> the agent wrote them. Harmless for evaluation; as an RL reward signal it is an open
> reward-hacking channel. Extend the gold checkout to cover `selected_test_files_to_run` first.

**Using Multilingual as the validation set of a training run:**

```bash
VAL_FILES=<ds>_index.parquet
HARBOR_HOSTPATH_MOUNTS='[{"host_path":"<toolchain>","mount_path":"/opt/grading","read_only":true}]'
HARBOR_ENVIRONMENT_OVERRIDE_CPUS=2        # cargo / maven / go test starve at one core
HARBOR_AGENT_OVERRIDE_TIMEOUT_SEC=4800    # the BASE budget, shared by train and val
HARBOR_VAL_AGENT_MAX_TIMEOUT_SEC=4800     # val-only cap, independent of the train cap
```

`min(override, max) * multiplier` is what harbor enforces, and val replaces only the cap — so a
3600 s train cap and a 4800 s val cap coexist: rollouts stay short while validation is measured at
the budget the benchmark needs (p99 of agent wall-clock over the 300 tasks is 4800 s; a 3600 s cap
costs ~2 points of reported score). Note `agent_loop_config_oh.yaml` had no `override_timeout_sec`
until recently, which made `HARBOR_AGENT_OVERRIDE_TIMEOUT_SEC` inert for the ohsdk scaffold.

### `convert_swebench_multilingual.py`

Converts `SWE-bench/SWE-bench_Multilingual` (300 instances, 9 languages) into Harbor task
directories in the same layout as the SWE-bench Verified val set, plus the index parquet. The
verifier is the official `make_test_spec(datum).eval_script` from swebench 4.1.0 (with the three
instance-specific fixes the upstream Harbor adapter carries), and grading goes
**offline grader first, `uv run` online only as a fallback** — see
`grading_toolchain_v2/` below for why that ordering is the whole point.

Needs `swebench>=4.1,<5` **on the converter host** (5.x changed the `make_test_spec` API); the
task images themselves need nothing.

Usage:

```bash
python utils/convert_swebench_multilingual.py \
    --output-dir /path/to/data/harbor_swebench_multilingual_300 \
    --index-path-prefix /path/to/shared/harbor_swebench_multilingual_300   # path as the TRAINING host sees it
```

### `convert_swebench_pro.py`

Converts `ScaleAI/SWE-bench_Pro` (731 public instances) plus the per-instance `run_script.sh` /
`parser.py` from `scaleapi/SWE-bench_Pro-os` into Harbor task dirs and an index parquet. The repo
lives at `/app`; the verifier reproduces the official
`swe_bench_pro_eval.py::create_entryscript` (Dockerfile `ENV` exports → gold test-file checkout →
`run_script.sh` → `parser.py` → resolved iff every `fail_to_pass` **and** `pass_to_pass` test
passed). Gold test files are checked out by the verifier, so the agent never sees them.

15 instances have an upstream gold patch that fails its own tests (oracle reward 0); they are
written to `<output-dir>_known_invalid_gold_ids.txt`.

Usage:

```bash
python utils/convert_swebench_pro.py \
    --output-dir /path/to/data/harbor_swebench_pro_731 \
    --index-path-prefix /path/to/shared/harbor_swebench_pro_731 \
    [--pro-repo <SWE-bench_Pro-os checkout>]   # else cloned to a temp dir
    [--goproxy-mount /opt/goproxy]             # see build_pro_goproxy.sh
```

### `build_pro_goproxy.sh`

Builds one offline Go module proxy per Pro task whose **gold patch bumps `go.mod`** (40 of 731).
Those tasks are otherwise unsolvable: the fix needs module versions the image's cache does not
have, and the sandbox has no egress — while giving it egress would hand the agent the upstream
repo and its fix. Each cache is a `GOMODCACHE` download tree consumed through
`GOPROXY=file://…` + `GOSUMDB=off`, which `convert_swebench_pro.py --goproxy-mount` writes into
the affected `task.toml`s. ~700 MB per instance, idempotent, resumable. The other ~240 go tasks
need nothing.

Usage: `bash utils/build_pro_goproxy.sh <dataset_dir> <out_dir> [instance_id ...]`

### `grading_toolchain_v2/parser_offline_v2.py` + `build_grading_toolchain_v2.sh`

A self-contained offline SWE-bench grader, mounted read-only at `/opt/grading` in the verifier
pod. **Without it the multilingual set cannot be scored at all**: the upstream verifier ends in
`uv run parser.py`, which resolves `swebench` from PyPI, and sandboxes have no egress — so
`tests/test.sh` takes its fallback branch and writes `reward 0` unconditionally, a structural
zero that reads as a bad model. Most of these images ship no Python at all, so the toolchain
brings its own: python-build-standalone 3.11 + `swebench==4.1.0`.

The parser uses only the log parser + `get_eval_tests_report` (no `make_test_spec`, no network),
and prints the `SWEBench results ends here` marker **only on a valid verdict** — a crashed
grader therefore falls back online instead of being recorded as a failed task.

```bash
OUT=/path/to/data/grading-toolchain-v2 bash utils/build_grading_toolchain_v2.sh
# then, in the eval/train config (host_path is resolved NODE-side):
# HARBOR_HOSTPATH_MOUNTS='[{"host_path":"/path/to/data/grading-toolchain-v2","mount_path":"/opt/grading","read_only":true}]'
```

### `harbor_image_env.py`

Bakes each task image's own `ENV` (`PATH`, `CARGO_HOME`, `JAVA_HOME`, …) into
`tests/image_env.sh` and makes `tests/test.sh` source it. The Kubernetes backend sets its own
`PATH` on the pod spec, which **replaces** the image's `ENV PATH` for every exec. The agent
survives (its shell is a login shell that sources `/root/.cargo/env` and friends), but the
verifier runs `bash /tests/test.sh` non-login and loses every toolchain the image only exposes
through `ENV` — `cargo: command not found`, scored as a legitimate 0. The converter calls this
automatically; run it by hand to backfill an existing dataset (idempotent, and safe to run while
an eval is in flight, since `tests/` is uploaded at verifier start).

Usage: `python utils/harbor_image_env.py <dataset_dir> [more dirs] [--registry host:port]`

### `populate_swebench_hub.sh`

Mirrors the official per-instance images (`swebench/sweb.eval.x86_64.*` for Verified /
Multilingual, `jefzda/sweap-images:*` for Pro) from Docker Hub into a
local registry with `docker buildx imagetools create` (registry-to-registry, nothing staged on
disk), so no-egress sandbox nodes can pull them. Images keep their Docker Hub names, which is
what lets `HARBOR_NYDUS_MIRROR` rewrite them transparently. Guards the shared Docker Hub rate
limit three ways (free ratelimit probe, local hourly ledger, 429 backoff) and is resumable.

Usage: `LIST=<images.txt> LOGDIR=<logs> bash utils/populate_swebench_hub.sh`

### `oracle_smoke_task.sh`

Validates a converted task with plain `docker` — no harbor, no k8s. `oracle` applies
`solution/solve.sh` then runs the verifier (expect `reward 1`); `empty` runs the verifier on the
untouched checkout (expect `reward 0`). Run it on a handful of instances per language before
spending a cluster on a new dataset: it separates "the model failed" from "the task is broken"
in about a minute.

Usage: `GRADING_TOOLCHAIN=<dir> bash utils/oracle_smoke_task.sh <task_dir> [oracle|empty]`

### `purge_infra_failed_trials.py`

Removes trials that failed for infrastructure / verifier reasons but were recorded as
`status==completed`, so the inference runner's resume logic re-runs them instead of baking a
false `reward=0` into the dataset. Dry-run by default (writes a manifest only); `APPLY=1`
actually deletes, and `INCLUDE_EXIT1=1` also clears the ambiguous "agent cmd exit1" bucket.

Usage: `APPLY=1 python3 utils/purge_infra_failed_trials.py [RESULTS_DIR]`
