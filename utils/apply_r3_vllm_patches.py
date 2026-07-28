#!/usr/bin/env python3
"""Re-apply the R3 (Rollout Routing Replay) vLLM patches to the installed vLLM.

These patches live in vLLM's site-packages (NOT in any git repo) and are LOST on
every `pip install`/venv rebuild/machine switch. Run this script after any vLLM
(re)install so R3 routing capture works correctly.

What it patches (PRODUCTION fixes only — no diagnostic probes):
  1. routed_experts_capturer.py
     - host buffer int32 -> uint8 (writer init_buffer + reader attach_buffer), so a
       correctly-sized buffer fits in /dev/shm.
     - OOB guards in save_captured_experts / get_routed_experts (skip out-of-bounds
       KV slots instead of IndexError-crashing the EngineCore).
  2. gpu_model_runner.py + scheduler.py
     - max_num_kv_tokens formula  x R3_BUFFER_FACTOR (default 5). The upstream formula
       `(num_blocks // num_groups) * min_block_size` UNDERSIZES the buffer ~4x for
       hybrid-KV models (Qwen3.5 GatedDeltaNet + full-attn); real KV slots reach ~4x
       it, so without this every long trajectory loses most of its routing.


Idempotent: each patch is skipped if its marker is already present. Run with
`--check` to report status without writing.

Usage:
    python utils/apply_r3_vllm_patches.py            # apply
    python utils/apply_r3_vllm_patches.py --check    # report only
"""

from __future__ import annotations

import os
import sys


def _vllm_dir() -> str:
    try:
        import vllm  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        sys.exit(f"[R3-patch] cannot import vllm: {e!r}")
    return os.path.dirname(os.path.abspath(vllm.__file__))


# Each patch: (name, marker, find, replace).
#   marker  -> already-applied detection (skip if present)
#   find    -> exact ORIGINAL (unpatched) vLLM text
#   replace -> patched text
CAPTURER = "model_executor/layers/fused_moe/routed_experts_capturer.py"
GPU_RUNNER = "v1/worker/gpu_model_runner.py"
SCHEDULER = "v1/core/sched/scheduler.py"

_SAVE_GUARD = '''        with _file_lock(self._lock_file):
            # R3 guard: hybrid-KV models underestimate max_num_kv_tokens, so real KV
            # slot_mapping indices can exceed the host buffer. Skip out-of-bounds slots
            # (graceful degrade) rather than crashing the EngineCore with IndexError.
            idx = indices
            if hasattr(idx, "cpu"):
                idx = idx.cpu().numpy()
            idx = np.asarray(idx)
            size = self._host_buffer_view.shape[0]
            if idx.size and int(idx.max()) >= size:
                valid = idx < size
                self._host_buffer_view[idx[valid], :, :] = data[valid]
            else:
                self._host_buffer_view[idx, :, :] = data'''

_READ_GUARD = '''        with _file_lock(self._lock_file, mode="rb+"):
            # R3 guard: mirror save_captured_experts — out-of-bounds slots were never
            # written, so return zero routing for them instead of an IndexError.
            idx = indices
            if hasattr(idx, "cpu"):
                idx = idx.cpu().numpy()
            idx = np.asarray(idx)
            size = self._host_buffer_view.shape[0]
            if idx.size and int(idx.max()) >= size:
                out = np.zeros(
                    (len(idx), *self._host_buffer_view.shape[1:]),
                    dtype=self._host_buffer_view.dtype,
                )
                valid = idx < size
                out[valid] = self._host_buffer_view[idx[valid], :, :]
                return out
            return self._host_buffer_view[indices, :, :].copy()'''

_SCHED_SIZE = '''            num_groups = len(kv_cache_config.kv_cache_groups)
            # R3 fix: x R3_BUFFER_FACTOR to cover the real KV slot range (hybrid models
            # under-size ~4x). MUST match gpu_model_runner's identical computation.
            import os

            self.max_num_kv_tokens = (
                (kv_cache_config.num_blocks // num_groups)
                * min_block_size
                * int(os.environ.get("R3_BUFFER_FACTOR", "5"))
            )'''

_RUNNER_SIZE = '''        num_groups = len(self.kv_cache_config.kv_cache_groups)
        # R3 fix: x R3_BUFFER_FACTOR to cover the real KV slot range (hybrid models
        # under-size ~4x). MUST match scheduler.py's identical computation.
        import os

        self.max_num_kv_tokens = (
            (self.kv_cache_config.num_blocks // num_groups)
            * min_block_size
            * int(os.environ.get("R3_BUFFER_FACTOR", "5"))
        )'''

PATCHES: dict[str, list[tuple[str, str, str, str]]] = {
    CAPTURER: [
        (
            "buffer uint8 (writer buffer_size)",
            "np.dtype(np.uint8).itemsize",
            "buffer_size = int(np.prod(shape)) * np.dtype(np.int32).itemsize",
            "buffer_size = int(np.prod(shape)) * np.dtype(np.uint8).itemsize  # R3 uint8",
        ),
        (
            "buffer uint8 (writer ndarray)",
            "np.ndarray(shape, dtype=np.uint8, buffer=self._shm.buf)",
            "self._host_buffer_view = np.ndarray(shape, dtype=np.int32, buffer=self._shm.buf)",
            "self._host_buffer_view = np.ndarray(shape, dtype=np.uint8, buffer=self._shm.buf)  # R3 uint8",
        ),
        (
            "buffer uint8 (reader ndarray)",
            "shape, dtype=np.uint8, buffer=self._shm.buf",
            "self._host_buffer_view = np.ndarray(\n                shape, dtype=np.int32, buffer=self._shm.buf\n            )",
            "self._host_buffer_view = np.ndarray(\n                shape, dtype=np.uint8, buffer=self._shm.buf  # R3 uint8\n            )",
        ),
        (
            "OOB guard (save_captured_experts)",
            "self._host_buffer_view[idx[valid], :, :] = data[valid]",
            "        with _file_lock(self._lock_file):\n            self._host_buffer_view[indices, :, :] = data",
            _SAVE_GUARD,
        ),
        (
            "OOB guard (get_routed_experts)",
            "out[valid] = self._host_buffer_view[idx[valid], :, :]",
            '        with _file_lock(self._lock_file, mode="rb+"):\n            return self._host_buffer_view[indices, :, :].copy()',
            _READ_GUARD,
        ),
    ],
    SCHEDULER: [
        (
            "max_num_kv_tokens x R3_BUFFER_FACTOR",
            "R3_BUFFER_FACTOR",
            "            num_groups = len(kv_cache_config.kv_cache_groups)\n"
            "            self.max_num_kv_tokens = (\n"
            "                kv_cache_config.num_blocks // num_groups\n"
            "            ) * min_block_size",
            _SCHED_SIZE,
        ),
    ],
    GPU_RUNNER: [
        (
            "max_num_kv_tokens x R3_BUFFER_FACTOR",
            "R3_BUFFER_FACTOR",
            "        num_groups = len(self.kv_cache_config.kv_cache_groups)\n"
            "        self.max_num_kv_tokens = (\n"
            "            self.kv_cache_config.num_blocks // num_groups\n"
            "        ) * min_block_size",
            _RUNNER_SIZE,
        ),
    ],
}


def main() -> int:
    check_only = "--check" in sys.argv
    root = _vllm_dir()
    print(f"[R3-patch] vLLM at: {root}")
    total_applied = total_skipped = total_missing = 0

    for rel, patches in PATCHES.items():
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            print(f"  [MISSING FILE] {rel}")
            total_missing += len(patches)
            continue
        with open(path, encoding="utf-8") as f:
            content = f.read()
        changed = False
        for name, marker, find, replace in patches:
            if marker in content:
                print(f"  [already]  {rel} :: {name}")
                total_skipped += 1
            elif find in content:
                content = content.replace(find, replace, 1)
                changed = True
                total_applied += 1
                print(f"  [{'WOULD' if check_only else 'APPLY'}]  {rel} :: {name}")
            else:
                total_missing += 1
                print(f"  [!! NOT FOUND] {rel} :: {name} "
                      f"(vLLM version mismatch? patch the find-string)")
        if changed and not check_only:
            # Atomic write (temp + rename) so running this on multiple nodes against a
            # shared venv can't leave a half-written file for a concurrent reader.
            tmp = f"{path}.r3tmp.{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp, path)

    print(f"[R3-patch] applied={total_applied} already={total_skipped} "
          f"not_found={total_missing}{' (check only)' if check_only else ''}")
    if total_missing:
        print("[R3-patch] WARNING: some patches did not match — R3 may be broken. "
              "Likely a vLLM version change; update the find-strings in this script.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
