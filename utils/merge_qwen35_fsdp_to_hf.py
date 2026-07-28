"""Merge veomni-FSDP2 shards -> HF safetensors for Qwen3.5-35B-A3B, writing the
state dict VERBATIM.

Why not utils/merge_veomni_fsdp_to_hf.py: that script is written for Qwen3-MoE
(its only callers are a Qwen3-30B-A3B run and a dense 27B run). For
Qwen3.5-35B-A3B the shards already carry exactly the target HF layout --
`model.language_model.layers.N.mlp.experts.gate_up_proj` [E,2I,H], fused and
single-nested -- so no key transform is wanted. Routing that state dict through
verl's save_hf_model_and_tokenizer() -> HF save_pretrained() instead rewrites it
into 30,720 per-expert keys under a triple-nested prefix, which no longer matches
the model definition. We therefore reuse verl's (correct) shard gathering and
write the tensors ourselves.

mtp.* (the untrained speculative head) is absent from veomni checkpoints and is
backfilled from the base model so the directory is complete.
"""
import argparse, glob, json, os, shutil, sys

import torch
from safetensors.torch import save_file

from verl.model_merger.base_model_merger import ModelMergerConfig
from verl.model_merger.fsdp_model_merger import FSDPModelMerger

_FSDP_EQUIVALENT_1D = {("fsdp",), ("dp_shard_sp",), ("dp_shard",)}


def _patched_calc(self, mesh, mesh_dim_names):
    names = tuple(mesh_dim_names)
    assert names in _FSDP_EQUIVALENT_1D or names == ("ddp", "fsdp"), names
    return mesh.shape[-1], (mesh.shape[-1],)


FSDPModelMerger._calculate_shard_configuration = _patched_calc

SHARD_BYTES = 40 * 1024**3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", required=True)
    ap.add_argument("--target_dir", required=True)
    ap.add_argument("--base_dir", required=True, help="for mtp.* backfill + config")
    args = ap.parse_args()

    cfg = ModelMergerConfig(
        operation="merge", backend="fsdp", local_dir=args.local_dir,
        target_dir=args.target_dir,
        hf_model_config_path=os.path.join(args.local_dir, "huggingface"),
        trust_remote_code=True, use_cpu_initialization=True,
    )
    merger = FSDPModelMerger(cfg)
    ws = merger._get_world_size()
    rz = merger._load_rank_zero_state_dict(ws)
    mesh, names = merger._extract_device_mesh_info(rz, ws)
    total, shape = merger._calculate_shard_configuration(mesh, names)
    print(f"[direct] gathering {total} shards, mesh {names}")
    sd = merger._load_and_merge_state_dicts(ws, total, shape, names)
    print(f"[direct] gathered {len(sd)} keys")

    fused = [k for k in sd if k.endswith(("experts.gate_up_proj", "experts.down_proj"))]
    perexp = [k for k in sd if ".experts." in k and k.endswith(
        ("gate_proj.weight", "up_proj.weight", "down_proj.weight"))]
    nested = [k for k in sd if k.startswith("model.language_model.language_model")]
    print(f"[direct] fused={len(fused)} per-expert={len(perexp)} triple-nested={len(nested)}")
    assert fused and not perexp and not nested, (
        "gathered state dict is not in the expected Qwen3.5 layout")

    # Backfill the untrained mtp.* head from the base model.
    from safetensors import safe_open
    added = 0
    for f in sorted(glob.glob(os.path.join(args.base_dir, "*.safetensors"))):
        with safe_open(f, framework="pt") as fh:
            for k in fh.keys():
                if (k.startswith("mtp") or ".mtp" in k) and k not in sd:
                    sd[k] = fh.get_tensor(k)
                    added += 1
    print(f"[direct] backfilled {added} mtp.* tensors from base")

    os.makedirs(args.target_dir, exist_ok=True)
    items = sorted(sd.items())
    shards, cur, cur_bytes = [], {}, 0
    for k, v in items:
        v = v.contiguous()
        nb = v.numel() * v.element_size()
        if cur and cur_bytes + nb > SHARD_BYTES:
            shards.append(cur); cur, cur_bytes = {}, 0
        cur[k] = v; cur_bytes += nb
    if cur:
        shards.append(cur)

    weight_map, total_size = {}, 0
    n = len(shards)
    for i, part in enumerate(shards, 1):
        name = f"model-{i:05d}-of-{n:05d}.safetensors"
        print(f"[direct] writing {name}  ({len(part)} tensors)")
        save_file(part, os.path.join(args.target_dir, name),
                  metadata={"format": "pt"})
        for k, v in part.items():
            weight_map[k] = name
            total_size += v.numel() * v.element_size()
    json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map},
              open(os.path.join(args.target_dir, "model.safetensors.index.json"), "w"),
              indent=2)

    hf = os.path.join(args.local_dir, "huggingface")
    for fn in os.listdir(hf):
        if not fn.endswith(".safetensors"):
            shutil.copy2(os.path.join(hf, fn), os.path.join(args.target_dir, fn))
    print(f"[direct] wrote {n} shards + index + config -> {args.target_dir}")


if __name__ == "__main__":
    sys.exit(main())
