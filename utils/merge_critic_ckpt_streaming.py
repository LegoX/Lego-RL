#!/usr/bin/env python3
"""Merge a sharded FSDP critic checkpoint with bounded host memory.

The stock verl merger loads every fp32 rank shard at once.  A 30B critic needs
well over 120 GB for that step, which does not fit in small utility cgroups.
This implementation first converts one rank at a time to temporary bf16
safetensors, then reconstructs and writes one global tensor at a time.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForTokenClassification
from transformers.core_model_loading import revert_weight_conversion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="sharded critic checkpoint directory")
    parser.add_argument("output", type=Path, help="HuggingFace output directory")
    parser.add_argument(
        "--max-shard-size-gb",
        type=float,
        default=2.0,
        help="maximum buffered output shard size (default: 2 GiB)",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="temporary filesystem; needs roughly 61 GB free for a 30B bf16 critic",
    )
    return parser.parse_args()


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def remove_incomplete_outputs(output: Path) -> None:
    patterns = (
        ".model-part-*.tmp",
        "model-part-*.safetensors",
        "model-*-of-*.safetensors",
        "model.safetensors.index.json",
    )
    for pattern in patterns:
        for path in output.glob(pattern):
            path.unlink()


def copy_huggingface_metadata(source: Path, output: Path) -> None:
    for item in source.iterdir():
        if item.name == "config.json":
            continue
        destination = output / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)

    config = json.loads((source / "config.json").read_text())
    architecture = (config.get("architectures") or ["Qwen3ForCausalLM"])[0]
    for suffix in ("ForConditionalGeneration", "ForCausalLM"):
        if suffix in architecture:
            architecture = architecture.replace(suffix, "ForTokenClassification")
            break
    config["architectures"] = [architecture]
    config["num_labels"] = 1
    config["classifier_dropout"] = 0.0
    config["hidden_dropout"] = "0"
    config["summary_dropout_prob"] = 0.0
    temporary = output / ".config.json.tmp"
    temporary.write_text(json.dumps(config, indent=2) + "\n")
    os.replace(temporary, output / "config.json")


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    hf_source = source / "huggingface"
    fsdp_config_path = source / "fsdp_config.json"

    if not (hf_source / "config.json").is_file():
        raise SystemExit(f"missing {hf_source / 'config.json'}")
    _model_type = str(json.loads((hf_source / "config.json").read_text()).get("model_type", ""))
    if _model_type.startswith("qwen3_5"):
        # revert_weight_conversion() splits Qwen3.5's fused expert tensors into
        # 31k per-expert keys under a triple-nested `model.language_model.` prefix
        # and exits 0 -- silent corruption. Verified 2026-08-27 on the r6 critic.
        raise SystemExit(
            f"model_type={_model_type}: this merger corrupts the Qwen3.5 fused-expert "
            "layout; use utils/merge_qwen35_fsdp_to_hf.py instead."
        )
    if not fsdp_config_path.is_file():
        raise SystemExit(f"missing {fsdp_config_path}")
    if args.max_shard_size_gb <= 0:
        raise SystemExit("--max-shard-size-gb must be positive")

    world_size = int(json.loads(fsdp_config_path.read_text())["world_size"])
    rank_paths = [source / f"model_world_size_{world_size}_rank_{rank}.pt" for rank in range(world_size)]
    missing = [str(path) for path in rank_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"missing model shards: {missing}")

    output.mkdir(parents=True, exist_ok=True)
    completed_files = list(output.glob("model-*-of-*.safetensors"))
    if completed_files or (output / "model.safetensors.index.json").exists():
        raise SystemExit(f"output already contains model weights: {output}")
    remove_incomplete_outputs(output)

    torch.set_num_threads(1)
    shard_limit = int(args.max_shard_size_gb * 1024**3)
    tmp_parent = str(args.tmp_dir.resolve()) if args.tmp_dir else None
    global_shapes: dict[str, tuple[int, ...]] = {}
    part_files: list[Path] = []

    try:
        with tempfile.TemporaryDirectory(prefix="critic-merge-", dir=tmp_parent) as temporary_dir:
            temporary_root = Path(temporary_dir)
            local_shards: list[Path] = []
            expected_keys: list[str] | None = None

            # Phase 1: torch.load one fp32 rank at a time.  Keeping all original
            # shards resident is what makes verl's stock merger exceed 120 GB.
            for rank, rank_path in enumerate(rank_paths):
                print(f"[stage {rank + 1}/{world_size}] loading {rank_path.name}", flush=True)
                state_dict = torch.load(rank_path, map_location="cpu", weights_only=False)
                keys = sorted(state_dict)
                if expected_keys is None:
                    expected_keys = keys
                elif keys != expected_keys:
                    raise RuntimeError(f"rank {rank} parameter keys differ from rank 0")

                local_state: dict[str, torch.Tensor] = {}
                for key, value in state_dict.items():
                    if not hasattr(value, "_local_tensor"):
                        raise TypeError(f"{key} is not a DTensor")
                    placements = tuple(value.placements)
                    if len(placements) != 1 or not placements[0].is_shard() or placements[0].dim != 0:
                        raise NotImplementedError(f"{key} has unsupported placements {placements}")
                    if rank == 0:
                        global_shapes[key] = tuple(value.shape)
                    local_state[key] = value._local_tensor.detach().to(torch.bfloat16).contiguous()

                local_path = temporary_root / f"rank-{rank:05d}.safetensors"
                save_file(local_state, str(local_path), metadata={"format": "pt"})
                local_shards.append(local_path)
                del local_state, state_dict
                gc.collect()

            config = AutoConfig.from_pretrained(hf_source)
            config.num_labels = 1
            config.classifier_dropout = 0.0
            config.hidden_dropout = "0"
            config.summary_dropout_prob = 0.0
            try:
                from verl_patch.models.qwen3_5_moe_token_classification import (
                    register_qwen3_5_moe_token_classification,
                )

                register_qwen3_5_moe_token_classification()
            except Exception as exc:
                print(f"[warn] Qwen3.5 token-classification register skipped: {exc!r}")
            with init_empty_weights():
                model = AutoModelForTokenClassification.from_config(config, dtype=torch.bfloat16)

            weight_map: dict[str, str] = {}
            total_parameters = 0
            total_size = 0
            output_buffer: dict[str, torch.Tensor] = {}
            output_buffer_bytes = 0

            def flush_output_buffer() -> None:
                nonlocal output_buffer, output_buffer_bytes
                if not output_buffer:
                    return
                part_number = len(part_files) + 1
                temporary_part = output / f".model-part-{part_number:05d}.tmp"
                final_part = output / f"model-part-{part_number:05d}.safetensors"
                print(
                    f"[write {part_number}] {len(output_buffer)} tensors, "
                    f"{output_buffer_bytes / 1e9:.2f} GB",
                    flush=True,
                )
                save_file(output_buffer, str(temporary_part), metadata={"format": "pt"})
                os.replace(temporary_part, final_part)
                for key in output_buffer:
                    weight_map[key] = final_part.name
                part_files.append(final_part)
                output_buffer = {}
                output_buffer_bytes = 0
                gc.collect()

            # Phase 2: memory-map the bf16 local shards and reconstruct one
            # global parameter at a time.  Revert the Transformers in-memory
            # fused-expert format to the standard on-disk expert keys.
            with contextlib.ExitStack() as stack:
                handles = [
                    stack.enter_context(safe_open(path, framework="pt", device="cpu")) for path in local_shards
                ]
                reference_keys = list(handles[0].keys())
                for rank, handle in enumerate(handles[1:], start=1):
                    if list(handle.keys()) != reference_keys:
                        raise RuntimeError(f"temporary rank {rank} keys differ from rank 0")

                for index, key in enumerate(reference_keys, start=1):
                    local_tensors = [handle.get_tensor(key) for handle in handles]
                    merged = torch.cat(local_tensors, dim=0).contiguous()
                    expected_shape = global_shapes[key]
                    if tuple(merged.shape) != expected_shape:
                        raise RuntimeError(
                            f"{key} merged shape {tuple(merged.shape)} != expected {expected_shape}"
                        )

                    merged_bytes = tensor_nbytes(merged)
                    if output_buffer and output_buffer_bytes + merged_bytes > shard_limit:
                        flush_output_buffer()

                    converted = revert_weight_conversion(model, {key: merged})
                    converted_bytes = 0
                    for output_key, tensor in converted.items():
                        if output_key in weight_map or output_key in output_buffer:
                            raise RuntimeError(f"duplicate output parameter {output_key}")
                        # Expert splits are views into one fused tensor.  Give each
                        # safetensor entry independent storage.
                        owned = tensor.detach().clone().contiguous()
                        output_buffer[output_key] = owned
                        size = tensor_nbytes(owned)
                        converted_bytes += size
                        total_parameters += owned.numel()
                        total_size += size
                    if converted_bytes != merged_bytes:
                        raise RuntimeError(
                            f"{key} conversion changed byte count: {merged_bytes} -> {converted_bytes}"
                        )
                    output_buffer_bytes += converted_bytes

                    del converted, merged, local_tensors
                    if index % 25 == 0 or index == len(reference_keys):
                        print(f"[merge {index}/{len(reference_keys)}] {key}", flush=True)

                flush_output_buffer()

            part_count = len(part_files)
            rename_map: dict[str, str] = {}
            for index, part_path in enumerate(part_files, start=1):
                final_name = f"model-{index:05d}-of-{part_count:05d}.safetensors"
                final_path = output / final_name
                os.replace(part_path, final_path)
                rename_map[part_path.name] = final_name
            weight_map = {key: rename_map[name] for key, name in weight_map.items()}

            index_data = {
                "metadata": {
                    "total_parameters": total_parameters,
                    "total_size": total_size,
                },
                "weight_map": weight_map,
            }
            temporary_index = output / ".model.safetensors.index.json.tmp"
            temporary_index.write_text(json.dumps(index_data, indent=2) + "\n")
            os.replace(temporary_index, output / "model.safetensors.index.json")
            copy_huggingface_metadata(hf_source, output)

        print(
            f"[done] {len(weight_map)} tensors, {total_parameters} parameters, "
            f"{total_size / 1e9:.2f} GB -> {output}",
            flush=True,
        )
    except BaseException:
        remove_incomplete_outputs(output)
        raise


if __name__ == "__main__":
    main()
