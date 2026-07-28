#!/usr/bin/env python3
"""
Merge a veomni-FSDP2 actor checkpoint (model_world_size_N_rank_*.pt, saved with a
1-D device mesh named 'dp_shard_sp') into a HuggingFace safetensors model.

verl's stock FSDPModelMerger hard-asserts mesh_dim_names in (('fsdp',),('ddp','fsdp')).
veomni saves the (functionally identical) 1-D fully-sharded mesh under the name
'dp_shard_sp', so we monkeypatch _calculate_shard_configuration to accept it and
reuse verl's tested load/merge/save path unchanged.

Usage:
    python merge_veomni_fsdp_to_hf.py --local_dir <step>/actor --target_dir <out>
"""
import argparse
import os
import re
import sys

import torch

from verl.model_merger.base_model_merger import ModelMergerConfig
from verl.model_merger.fsdp_model_merger import FSDPModelMerger

# 1-D meshes that are pure FSDP sharding under a different label.
_FSDP_EQUIVALENT_1D = {("fsdp",), ("dp_shard_sp",), ("dp_shard",)}

# ---------------------------------------------------------------------------
# veomni fused-MoE  ->  HF per-expert  (the bug fix)
# ---------------------------------------------------------------------------
# veomni stores Qwen3-MoE experts in a FUSED/grouped v5 layout:
#     model.layers.{i}.mlp.experts.gate_up_proj   [E, 2*I, H]   (gate=[:I], up=[I:])
#     model.layers.{i}.mlp.experts.down_proj      [E, H, I]
#     model.layers.{i}.mlp.gate.weight            [E, H]        (router, HF-named already)
# HF Qwen3MoeForCausalLM instead wants PER-EXPERT 2-D weights:
#     model.layers.{i}.mlp.experts.{j}.gate_proj.weight  [I, H]
#     model.layers.{i}.mlp.experts.{j}.up_proj.weight    [I, H]
#     model.layers.{i}.mlp.experts.{j}.down_proj.weight  [H, I]
# verl's stock merger gathers the FSDP shards correctly but writes the fused keys
# verbatim -> vLLM/HF can't find the per-expert keys -> experts stay random-init
# -> the served model emits token salad. This is the exact inverse of veomni's
# load-time converter (qwen3_moe/checkpoint_tensor_converter.py): pure split +
# unstack, NO transpose. E is read from the tensor itself (dim 0) so it is
# self-validating and needs no config lookup.
_FUSED_EXPERT_PATTERN = re.compile(r"^(.+\.mlp)\.experts\.(gate_up_proj|down_proj)$")


def _unfuse_moe_experts(state_dict: dict) -> dict:
    out: dict = {}
    n_converted = 0
    for k, v in state_dict.items():
        m = _FUSED_EXPERT_PATTERN.match(k)
        if not m:
            out[k] = v
            continue
        prefix, which = m.groups()  # prefix == "model.layers.{i}.mlp"
        e = v.shape[0]
        if which == "gate_up_proj":  # [E, 2I, H] -> per-expert gate [I,H] + up [I,H]
            i = v.shape[1] // 2
            for j in range(e):
                out[f"{prefix}.experts.{j}.gate_proj.weight"] = v[j, :i, :].clone()
                out[f"{prefix}.experts.{j}.up_proj.weight"] = v[j, i:, :].clone()
        else:  # down_proj: [E, H, I] -> per-expert down [H, I]
            for j in range(e):
                out[f"{prefix}.experts.{j}.down_proj.weight"] = v[j].clone()
        n_converted += 1
    if n_converted:
        print(f"[merge_veomni] unfused {n_converted} grouped-expert tensors "
              f"-> {sum(1 for k in out if '.experts.' in k and k.endswith('.weight'))} per-expert HF keys")
    return out


# ---------------------------------------------------------------------------
# veomni module-nesting -> HF key names (qwen3_5 / Qwen3_5ForConditionalGeneration)
# ---------------------------------------------------------------------------
# veomni's qwen3_5 wrapper nests the text tower under extra language_model levels
# and hangs the vision tower off language_model; its state-dict keys leak that
# nesting verbatim (verl's merger save_pretrained(state_dict=...) writes keys
# as-is). Observed on the 937 27b run:
#     model.language_model.language_model.language_model.X -> model.language_model.X
#     model.language_model.visual.Y                        -> model.visual.Y
# lm_head.weight is already correct. Idempotent no-op for models that don't
# match (e.g. qwen3-moe 30b, whose keys are flat model.layers.*).
# NOTE: veomni checkpoints also omit the (untrained) mtp.* speculative head;
# if you need a byte-complete HF dir, backfill those 15 tensors from the base
# model afterwards.
def _normalize_veomni_nesting(state_dict: dict) -> dict:
    out = {}
    n = 0
    for k, v in state_dict.items():
        k2 = re.sub(r"^model\.(language_model\.)+", "model.language_model.", k)
        k2 = re.sub(r"^model\.language_model\.visual\.", "model.visual.", k2)
        if k2 != k:
            n += 1
        assert k2 not in out, f"key collision after normalization: {k2}"
        out[k2] = v
    if n:
        print(f"[merge_veomni] normalized {n} veomni-nested key names to HF layout")
    return out


_orig_load_and_merge = FSDPModelMerger._load_and_merge_state_dicts


def _patched_load_and_merge(self, *args, **kwargs):
    merged = _orig_load_and_merge(self, *args, **kwargs)
    return _normalize_veomni_nesting(_unfuse_moe_experts(merged))


FSDPModelMerger._load_and_merge_state_dicts = _patched_load_and_merge


def _patched_calculate_shard_configuration(self, mesh, mesh_dim_names):
    names = tuple(mesh_dim_names)
    assert names in _FSDP_EQUIVALENT_1D or names == ("ddp", "fsdp"), (
        f"Unsupported mesh_dim_names {names}"
    )
    if "tp" in names:
        total_shards = mesh.shape[-1] * mesh.shape[-2]
        mesh_shape = (mesh.shape[-2], mesh.shape[-1])
    else:
        total_shards = mesh.shape[-1]
        mesh_shape = (mesh.shape[-1],)
    return total_shards, mesh_shape


FSDPModelMerger._calculate_shard_configuration = _patched_calculate_shard_configuration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", required=True, help="<global_step_N>/actor dir")
    ap.add_argument("--target_dir", required=True, help="output HF dir")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    cfg = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=args.local_dir,
        target_dir=args.target_dir,
        hf_model_config_path=os.path.join(args.local_dir, "huggingface"),
        trust_remote_code=args.trust_remote_code,
        use_cpu_initialization=True,
    )
    merger = FSDPModelMerger(cfg)
    merger.merge_and_save()
    print(f"[merge_veomni] saved HF model -> {args.target_dir}")


if __name__ == "__main__":
    sys.exit(main())
