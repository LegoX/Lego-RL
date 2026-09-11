#!/usr/bin/env bash
# Merge a verl FSDP *critic* checkpoint into a HuggingFace directory that
# critic.model.path can load.
#
# WHY THIS EXISTS. checkpoints/<proj>/<exp>/global_step_N/critic/ is NOT loadable:
#   model_world_size_4_rank_*.pt   <- the weights, 4 DTensor shards, fp32
#   optim_world_size_4_rank_*.pt   <- optimizer state, not used here
#   huggingface/                   <- config.json + tokenizer ONLY, NO weights
# Pointing critic.model.path at it makes from_pretrained fail (no weights found).
#
# AND THERE IS A TRAP, on the FSDP path. verl saves the critic's
# huggingface/config.json with
#     "architectures": ["Qwen3ForCausalLM"]
# even though the checkpoint is a token-classification model: its state dict ends in
# score.weight (1, hidden) + score.bias (1,) and has NO lm_head. The merger dispatches
# on architectures[0] (verl/model_merger/base_model_merger.py:209-214), so run as-is it
# picks AutoModelForCausalLM and the value head does not survive.
# This script fixes that in a STAGING COPY -- the original checkpoint is never touched.
# On the VeOmni path the architecture is already written correctly (measured on a
# Qwen3-30B-A3B critic: ["Qwen3MoeForTokenClassification"]) and the rewrite is a no-op
# -- but "num_labels": null still needs fixing, and VeOmni's device mesh needs the
# merger patch further down, so run this script for both backends.
#
# USAGE
#   bash utils/merge_critic_ckpt.sh <critic-ckpt-dir> <output-dir>
# EXAMPLE
#   bash utils/merge_critic_ckpt.sh \
#     checkpoints/sao-8b-long/sao-1node-qwen3-8b-long-r3/global_step_122/critic \
#     checkpoints/sao-8b-long/critic_hf_r3_step122
#
# NOTE ON PRECISION: the merger casts shards to bfloat16 (fsdp_model_merger.py:169).
# The source is fp32. This is the same path used to export actors, and the critic
# trains in bf16 mixed precision anyway, but it is a real one-way narrowing -- keep
# the original checkpoint.
set -euo pipefail

SRC="${1:?usage: merge_critic_ckpt.sh <critic-ckpt-dir> <output-dir>}"
OUT="${2:?usage: merge_critic_ckpt.sh <critic-ckpt-dir> <output-dir>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

SRC="$(cd "$SRC" && pwd)"
[ -f "$SRC/huggingface/config.json" ] || { echo "[FATAL] no huggingface/config.json under $SRC"; exit 1; }
ls "$SRC"/model_world_size_*_rank_0.pt >/dev/null 2>&1 || { echo "[FATAL] no model_world_size_*_rank_0.pt under $SRC"; exit 1; }

STAGE="$(mktemp -d "${TMPDIR:-/tmp}/critic_merge_stage.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
echo "[stage] $STAGE"

# Symlink the shards -- they are tens of GB and we only need them read-only.
# fsdp_config.json is required, not optional: _get_world_size() reads it first and
# raises FileNotFoundError before touching a single shard.
for f in "$SRC"/model_world_size_*.pt "$SRC"/extra_state_world_size_*.pt "$SRC"/fsdp_config.json; do
    [ -e "$f" ] && ln -s "$f" "$STAGE/$(basename "$f")"
done
[ -e "$STAGE/fsdp_config.json" ] || { echo "[FATAL] no fsdp_config.json under $SRC"; exit 1; }
mkdir -p "$STAGE/huggingface"
for f in "$SRC"/huggingface/*; do ln -s "$f" "$STAGE/huggingface/$(basename "$f")"; done
# ...except config.json, which we replace with a corrected real file.
rm -f "$STAGE/huggingface/config.json"

"$PYTHON_BIN" - "$SRC/huggingface/config.json" "$STAGE/huggingface/config.json" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
cfg = json.load(open(src))
old = cfg.get("architectures")
arch = (old or ["Qwen3ForCausalLM"])[0]
# Qwen3ForCausalLM -> Qwen3ForTokenClassification, Qwen3MoeForCausalLM -> ...MoeFor...,
# Qwen3_5MoeForConditionalGeneration -> Qwen3_5MoeForTokenClassification.
for suffix in ("ForConditionalGeneration", "ForCausalLM"):
    if suffix in arch:
        arch = arch.replace(suffix, "ForTokenClassification")
        break
cfg["architectures"] = [arch]
cfg["num_labels"] = 1
# Dropout must be off or the merged head is not the head that was trained.
cfg["classifier_dropout"] = 0.0
cfg["hidden_dropout"] = "0"
cfg["summary_dropout_prob"] = 0.0
json.dump(cfg, open(dst, "w"), indent=2)
print(f"[config] architectures {old} -> {cfg['architectures']}, num_labels=1")
PY

echo "[merge] running verl.model_merger (this reads all shards; expect several minutes)"
mkdir -p "$OUT"
# Not `python -m verl.model_merger` -- VeOmni checkpoints trip an over-tight assert in
# it. See the driver's own comment for why widening it is safe. Everything else about
# the merge is stock verl.
"$PYTHON_BIN" - "$STAGE" "$OUT" <<'PY'
import sys
from verl.model_merger.base_model_merger import generate_config_from_args, parse_args
from verl.model_merger.fsdp_model_merger import FSDPModelMerger

# WHY THIS PATCH. VeOmni saves FSDP2 shards on a 1-D device mesh named
# ('dp_shard_sp',) -- data-parallel shard and sequence-parallel flattened into one
# dim. verl's _calculate_shard_configuration whitelists only ("fsdp",) and
# ("ddp","fsdp") and asserts on anything else (fsdp_model_merger.py:120), so a VeOmni
# critic dies there with `Unsupported mesh_dim_names ('dp_shard_sp',)` before reading
# a single shard.
#
# The assert is the ONLY thing that does not handle this layout. Checked against
# global_step_75/critic (Qwen3-30B-A3B, world_size 16), every downstream branch is
# already correct for it:
#   - all 532 keys are Shard(dim=0) on a (16,) mesh, no Replicate/Partial, no
#     non-DTensor keys, so the 1-D `torch.cat(shards, dim=0)` path is the right one;
#   - "tp" is not in mesh_dim_names, so total_shards = mesh.shape[-1] = 16, correct;
#   - mesh_dim_names[0] is neither "dp" nor "ddp" (:173), so no placement gets
#     stripped -- which is what we want, there is no dp dim to strip on a 1-D mesh.
# So widen the whitelist here rather than editing the shared verl checkout.
#
# NOTE on uneven shards: score.weight is global (1, hidden) split over 16 ranks, i.e.
# rank 0 holds (1, hidden) and ranks 1-15 hold (0, hidden). torch.cat handles that and
# reproduces (1, hidden). The verify step at the end of this script is what actually
# proves the head survived -- do not skip it.
_orig = FSDPModelMerger._calculate_shard_configuration


def _calculate_shard_configuration(self, mesh, mesh_dim_names):
    if mesh_dim_names in (("fsdp",), ("ddp", "fsdp")):
        return _orig(self, mesh, mesh_dim_names)
    if len(mesh_dim_names) != 1:
        raise NotImplementedError(
            f"only 1-D non-standard meshes are supported, got {mesh_dim_names}"
        )
    if "tp" in mesh_dim_names:
        raise NotImplementedError(f"tensor parallelism is not supported, got {mesh_dim_names}")
    print(f"[patch] treating 1-D mesh {mesh_dim_names} as plain FSDP sharding")
    return int(mesh.shape[-1]), (int(mesh.shape[-1]),)


FSDPModelMerger._calculate_shard_configuration = _calculate_shard_configuration

stage, out = sys.argv[1], sys.argv[2]
sys.argv = ["verl.model_merger", "merge", "--backend", "fsdp",
            "--local_dir", stage, "--target_dir", out]
config = generate_config_from_args(parse_args())
print(f"config: {config}")
merger = FSDPModelMerger(config)
merger.merge_and_save()
merger.cleanup()
PY

# The merger re-serialises the config from the loaded model and DROPS num_labels.
# verl does not care -- load_valuehead_model forces hf_config.num_labels = 1 before
# from_pretrained (workers/engine/fsdp/transformer_impl.py:264) -- but transformers
# defaults num_labels to 2 when it is absent, so anyone loading this directory with a
# plain AutoModelForTokenClassification.from_pretrained() would build a (2, hidden)
# head against a (1, hidden) checkpoint. Write it back so the dir is self-describing.
"$PYTHON_BIN" - "$OUT/config.json" <<'PY'
import json, sys
p = sys.argv[1]
cfg = json.load(open(p))
if cfg.get("num_labels") != 1:
    cfg["num_labels"] = 1
    json.dump(cfg, open(p, "w"), indent=2)
    print("[config] wrote num_labels=1 back into the merged config")
PY

# The whole point of the exercise. A random head would also "merge" fine, so check
# the head is (a) present and (b) not the all-zero/untrained thing.
"$PYTHON_BIN" - "$OUT" <<'PY'
import sys, glob, json, os
from safetensors import safe_open
out = sys.argv[1]
files = sorted(glob.glob(os.path.join(out, "*.safetensors")))
if not files:
    sys.exit(f"[FATAL] no safetensors written to {out}")
found = {}
for f in files:
    with safe_open(f, framework="pt") as fh:
        for k in fh.keys():
            if k.startswith("score.") or k.startswith("lm_head."):
                found[k] = fh.get_tensor(k)
if "score.weight" not in found:
    sys.exit(f"[FATAL] score.weight missing from the merge -- got {sorted(found)}. "
             "The architecture rewrite did not take effect; do NOT use this checkpoint.")
w = found["score.weight"].float()
print(f"[verify] score.weight shape={tuple(w.shape)} dtype={found['score.weight'].dtype} "
      f"absmean={w.abs().mean():.6f} std={w.std():.6f}")
if w.abs().max() == 0:
    sys.exit("[FATAL] score.weight is all zeros -- the trained head did not survive.")
cfg = json.load(open(os.path.join(out, "config.json")))
print(f"[verify] config architectures={cfg.get('architectures')} num_labels={cfg.get('num_labels')}")
print("[verify] OK")
PY

echo "[done] merged critic -> $OUT"
echo "        set CRITIC_MODEL_PATH=$OUT"
