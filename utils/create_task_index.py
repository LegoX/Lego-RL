"""
Create a Harbor task index parquet for RLLM online RL training.

This is the ONLY data-prep step needed for online RL with Harbor.

The parquet is NOT training data — it is a task registry. Each row tells RLLM
"here is one task I should run during online training". The actual trajectories
(model outputs + rewards) are generated LIVE during each training step when
HarborWorkflow calls trial.run() against the current model.

How it fits into the online RL loop
-------------------------------------
  (one-time)  create_task_index.py → task_index.parquet
                                           │
  (each step) RLLM samples batch from task_index.parquet
                    │
                    ▼  extra_info per row passed to:
              HarborWorkflow.run(task, uid)
                    │
                    ▼  Harbor trial executes:
              sandbox → Terminus-2 → calls vLLM HTTP endpoint → verifier
                    │
                    ▼  returns Episode (chat_history + reward)
              RLLM tokenises chat_completions → PPO/GRPO update

Row schema (verl-compatible)
-----------------------------
  prompt           list[dict]  placeholder prompt; content = task dir path
  reward_model     dict        {"style": "rule", "ground_truth": None}
  extra_info       dict        the dict that HarborWorkflow.run() receives as `task`
    ├─ harbor_task_path  str   absolute path to the task directory
    ├─ instance_id       str   task directory name (used as unique ID)
    └─ data_source       str   "harbor"

Usage
-----
  # Build index from a parent directory of task sub-dirs
  python examples/harbor/create_task_index.py \\
      --tasks_dir  /data/my_harbor_tasks \\
      --output     /data/harbor_task_index_train.parquet \\
      --split      train

  # With an explicit train/val split (80/20)
  python examples/harbor/create_task_index.py \\
      --tasks_dir  /data/my_harbor_tasks \\
      --output     /data/harbor_task_index \\
      --split_ratio 0.8

  # Then train with:
  #   python3 -m rllm.trainer.verl.train_workflow_pipeline \\
  #       data.train_files=/data/harbor_task_index_train.parquet \\
  #       data.val_files=/data/harbor_task_index_val.parquet \\
  #       ...
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _is_valid_task(path: Path) -> bool:
    """A valid Harbor task directory must contain instruction.md."""
    return path.is_dir() and (path / "instruction.md").exists()


def collect_task_dirs(root: str) -> list[Path]:
    """
    Return all valid Harbor task directories under *root*.

    If *root* itself contains instruction.md it is treated as a single task.
    Otherwise every sub-directory that contains instruction.md is a task.
    """
    root_path = Path(root)
    if not root_path.exists():
        print(f"[error] Path does not exist: {root}", file=sys.stderr)
        return []

    if _is_valid_task(root_path):
        return [root_path]

    tasks = sorted(p for p in root_path.iterdir() if _is_valid_task(p))
    return tasks


def task_to_row(path: Path) -> dict:
    """
    Convert one Harbor task directory into a parquet row.

    ``prompt`` holds a minimal placeholder so verl's tokenizer does not
    choke on an empty field.  The actual task content is read from
    instruction.md at runtime by Harbor.
    """
    return {
        # Placeholder prompt — verl tokenises this, but HarborWorkflow never
        # uses it; the real task text lives in instruction.md inside task_path.
        "prompt": [{"role": "user", "content": str(path)}],
        # Required by DatasetRegistry.apply_verl_postprocessing()
        "reward_model": {"style": "rule", "ground_truth": None},
        # This dict is what HarborWorkflow.run() receives as `task`
        "extra_info": {
            "harbor_task_path": str(path.resolve()),
            "instance_id":      path.name,
            "data_source":      "harbor",
        },
    }


def write_parquet(rows: list[dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    print(f"  Wrote {len(df)} tasks → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a Harbor task index parquet for RLLM online RL training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--tasks_dir", required=True,
        help="Root directory containing Harbor task sub-directories.",
    )
    ap.add_argument(
        "--output", required=True,
        help=(
            "Output path.  If --split_ratio is set, '_train.parquet' and "
            "'_val.parquet' are appended; otherwise use this exact path."
        ),
    )
    ap.add_argument(
        "--split", choices=["train", "val", "all"], default="all",
        help="Which split to write when NOT using --split_ratio (default: all).",
    )
    ap.add_argument(
        "--split_ratio", type=float, default=None,
        help=(
            "Fraction of tasks to use for training (e.g. 0.8 → 80%% train, "
            "20%% val). Writes two files: <output>_train.parquet and "
            "<output>_val.parquet."
        ),
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val split (default: 42).",
    )
    args = ap.parse_args()

    # Collect task directories
    task_dirs = collect_task_dirs(args.tasks_dir)
    if not task_dirs:
        print("[error] No valid Harbor task directories found.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(task_dirs)} valid Harbor tasks in {args.tasks_dir}")

    rows = [task_to_row(p) for p in task_dirs]

    if args.split_ratio is not None:
        # Shuffle + split
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        n_train = max(1, int(len(rows) * args.split_ratio))
        train_rows = rows[:n_train]
        val_rows   = rows[n_train:] or rows[:1]  # ensure val is non-empty

        output_base = args.output.removesuffix(".parquet")
        print(f"Split: {len(train_rows)} train / {len(val_rows)} val")
        write_parquet(train_rows, f"{output_base}_train.parquet")
        write_parquet(val_rows,   f"{output_base}_val.parquet")

    else:
        # Single file — output must end in .parquet
        output = args.output if args.output.endswith(".parquet") else args.output + ".parquet"
        write_parquet(rows, output)

    print("\nDone. Pass the parquet path(s) to RLLM:")
    print("  data.train_files=<output>_train.parquet")
    print("  data.val_files=<output>_val.parquet")


if __name__ == "__main__":
    main()
