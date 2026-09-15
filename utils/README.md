# utils/ — Python helper scripts

Standalone Python tools used by the Lego-RL training / evaluation flow. `scripts/` holds
only the `.sh` runners, which call into this directory as `$SCRIPT_DIR/../utils/xxx.py` or
`$REPO_ROOT/utils/xxx.py`.

> Path convention: `SCRIPT_DIR` = `scripts/`, `REPO_ROOT` = the repository root, so
> `$REPO_ROOT/utils/` is this directory.

---

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

### `purge_infra_failed_trials.py`

Removes trials that failed for infrastructure / verifier reasons but were recorded as
`status==completed`, so the inference runner's resume logic re-runs them instead of baking a
false `reward=0` into the dataset. Dry-run by default (writes a manifest only); `APPLY=1`
actually deletes, and `INCLUDE_EXIT1=1` also clears the ambiguous "agent cmd exit1" bucket.

Usage: `APPLY=1 python3 utils/purge_infra_failed_trials.py [RESULTS_DIR]`
