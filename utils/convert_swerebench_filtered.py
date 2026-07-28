"""
Convert nebius/SWE-rebench 'filtered' subset to Harbor task directories + index parquet.

Usage:
    python convert_swerebench_filtered.py \
        --output-dir /path/to/data/harbor_tasks_swerebench_filtered \
        --index-dir  /path/to/data/harbor_indexes

Produces:
    <output-dir>/<instance_id>/       — one Harbor task directory per instance
    <index-dir>/train_swerebench_filtered.parquet  — training index
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import sys
from pathlib import Path
from textwrap import dedent
from typing import Optional

import pandas as pd
from datasets import load_dataset


TEMPLATE_DIR = Path("/path/to/harbor/adapters/swerebenchv2/template")

# ── helpers (adapted from harbor/adapters/swerebenchv2/utils.py) ──────────

def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    return path.read_text()


def render_literal(template_text: str, **repls: str) -> str:
    out = template_text
    for k, v in repls.items():
        out = out.replace("{" + k + "}", v)
    return out


def get_image_name(record: dict) -> str:
    if record.get("image_name"):
        return record["image_name"]
    if record.get("docker_image"):
        return record["docker_image"]
    ic = record.get("install_config") or {}
    if ic.get("image_name"):
        return ic["image_name"]
    raise ValueError(f"No image_name found for {record.get('instance_id', '?')}")


def get_workdir(record: dict) -> str:
    repo = record.get("repo", "")
    if "/" in repo:
        return f"/{repo.split('/')[1]}"
    return "/testbed"


def _extract_patch_paths(test_patch: str):
    preimage_paths = [
        p for p in re.findall(r"^--- a/(.*)$", test_patch, re.MULTILINE)
        if p != "/dev/null"
    ]
    postimage_paths = [
        p for p in re.findall(r"^\+\+\+ b/(.*)$", test_patch, re.MULTILINE)
        if p != "/dev/null"
    ]
    tracked = list(dict.fromkeys(preimage_paths))
    all_paths = list(dict.fromkeys(preimage_paths + postimage_paths))
    return tracked, all_paths


def get_test_commands(record: dict) -> str:
    test_patch = record.get("test_patch") or ""
    base_commit = record["base_commit"]
    install_config = record.get("install_config") or {}

    test_cmds = install_config.get("test_cmd", [])
    if isinstance(test_cmds, str):
        test_cmds = [test_cmds]

    workdir = get_workdir(record)
    test_patch_files, test_patch_cleanup_files = _extract_patch_paths(test_patch)
    quoted_files = " ".join(shlex.quote(p) for p in test_patch_files)
    quoted_cleanup = " ".join(shlex.quote(p) for p in test_patch_cleanup_files)

    reset_cmd = (
        f"git checkout {shlex.quote(base_commit)} -- {quoted_files}"
        if quoted_files else ":"
    )
    cleanup_cmd = (
        dedent(f"""
            # Remove only untracked files the test patch touches.
            for path in {quoted_cleanup}; do
                if [ -e "$path" ] && ! git ls-files --error-unmatch -- "$path" >/dev/null 2>&1; then
                    rm -rf -- "$path"
                fi
            done
            """).strip()
        if quoted_cleanup else ":"
    )

    newline = "\n"
    return f"""#!/bin/bash
set -uo pipefail -x

cd {shlex.quote(workdir)}

# Reset test files to base commit state
{reset_cmd}

# Remove untracked files from previous test-patch applications
{cleanup_cmd}

# Apply the test patch
echo {shlex.quote(test_patch)} > /tmp/test_patch.diff
git apply -v --3way --recount --ignore-space-change --whitespace=nowarn /tmp/test_patch.diff \\
  || git apply /tmp/test_patch.diff \\
  || patch --fuzz=5 -p1 -i /tmp/test_patch.diff

# Record terminal output in LOG_FILE
LOG_FILE=$(mktemp)
export LOG_FILE
exec 3>&1 4>&2
exec > >(tee "$LOG_FILE") 2>&1

set +x
{newline.join(test_cmds)} || true
exec 1>&3 2>&4 # stop record
exec 1>&3 2>&4 # stop record

# Reset tests back to base commit
{reset_cmd}
{cleanup_cmd}
"""


def serialize_record(raw: dict) -> dict:
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            out[k] = serialize_record(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [_serialize_item(i) for i in v]
        elif isinstance(v, (int, float, str, bool, type(None))):
            out[k] = v
        else:
            try:
                out[k] = v.item()
            except (AttributeError, ValueError):
                out[k] = str(v)
    return out


def _serialize_item(item):
    if isinstance(item, dict):
        return serialize_record(item)
    if isinstance(item, (list, tuple)):
        return [_serialize_item(i) for i in item]
    if isinstance(item, (int, float, str, bool, type(None))):
        return item
    try:
        return item.item()
    except (AttributeError, ValueError):
        return str(item)


# ── task generation ───────────────────────────────────────────────────────

def generate_task(
    raw: dict,
    out_root: Path,
    template_dir: Path,
    max_timeout: float = 3000.0,
    overwrite: bool = False,
) -> Path:
    instance_id = raw["instance_id"]
    task_dir = out_root / instance_id

    if task_dir.exists():
        if not overwrite:
            return task_dir
        shutil.rmtree(task_dir)

    env_dir = task_dir / "environment"
    tests_dir = task_dir / "tests"
    solution_dir = task_dir / "solution"
    for d in [env_dir, tests_dir, solution_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # instruction.md
    instr_tpl = read_text(template_dir / "instruction.md")
    problem = dedent(raw.get("problem_statement", "")).strip()
    instr = render_literal(instr_tpl, problem_statement=problem)
    if not instr.endswith("\n"):
        instr += "\n"
    (task_dir / "instruction.md").write_text(instr)

    # task.toml
    difficulty = "unknown"
    meta = raw.get("meta")
    if meta and isinstance(meta, dict):
        llm_meta = meta.get("llm_metadata") or meta.get("llm_score")
        if isinstance(llm_meta, dict):
            d = llm_meta.get("difficulty") or llm_meta.get("difficulty_score")
            if isinstance(d, str) and d in ("easy", "medium", "hard"):
                difficulty = d
    cfg_tpl = read_text(template_dir / "task.toml")
    cfg = render_literal(cfg_tpl, difficulty=difficulty, max_timeout=str(int(max_timeout)))
    (task_dir / "task.toml").write_text(cfg)

    # tests/config.json
    datum = serialize_record(raw)
    (tests_dir / "config.json").write_text(json.dumps(datum, indent=2))

    # tests/test.sh
    test_sh_tpl = read_text(template_dir / "test.sh")
    test_commands = get_test_commands(raw)
    test_sh = render_literal(test_sh_tpl, test_commands=test_commands)
    test_sh_path = tests_dir / "test.sh"
    test_sh_path.write_text(test_sh)
    test_sh_path.chmod(0o755)

    # environment/Dockerfile
    docker_image = get_image_name(raw)
    workdir = get_workdir(raw)
    dockerfile_tpl = read_text(template_dir / "Dockerfile")
    dockerfile = render_literal(dockerfile_tpl, docker_image=docker_image, workdir=workdir)
    (env_dir / "Dockerfile").write_text(dockerfile)

    # solution/solve.sh
    solve_tpl = read_text(template_dir / "solve.sh")
    patch = (raw.get("patch") or "").strip()
    solve_sh = render_literal(solve_tpl, patch=patch)
    solve_path = solution_dir / "solve.sh"
    solve_path.write_text(solve_sh)
    solve_path.chmod(0o755)

    return task_dir


def task_to_index_row(task_path: Path) -> dict:
    return {
        "prompt": [{"role": "user", "content": str(task_path)}],
        "reward_model": {"style": "rule", "ground_truth": None},
        "extra_info": {
            "harbor_task_path": str(task_path.resolve()),
            "instance_id": task_path.name,
            "data_source": "harbor",
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Convert SWE-rebench filtered to Harbor format")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("/path/to/data/harbor_tasks_swerebench_filtered"))
    ap.add_argument("--index-dir", type=Path,
                    default=Path("/path/to/data/harbor_indexes"))
    ap.add_argument("--index-name", type=str,
                    default="train_swerebench_filtered.parquet")
    ap.add_argument("--template-dir", type=Path, default=None)
    ap.add_argument("--timeout", type=float, default=3000.0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    template_dir = args.template_dir or TEMPLATE_DIR
    if not template_dir.exists():
        print(f"[error] Template dir not found: {template_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading nebius/SWE-rebench filtered split...")
    ds = load_dataset("nebius/SWE-rebench", split="filtered")
    print(f"Loaded {len(ds)} instances")

    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
        print(f"Limited to {len(ds)} instances")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.index_dir.mkdir(parents=True, exist_ok=True)

    success = []
    failures = []
    for idx, row in enumerate(ds):
        instance_id = row["instance_id"]
        try:
            task_path = generate_task(
                row, args.output_dir, template_dir,
                max_timeout=args.timeout, overwrite=args.overwrite,
            )
            success.append(task_path)
            if (idx + 1) % 100 == 0:
                print(f"  [{idx+1}/{len(ds)}] processed...", flush=True)
        except Exception as e:
            print(f"  [{idx+1}] FAIL {instance_id}: {e}")
            failures.append((instance_id, str(e)))

    print(f"\nTask dirs: {len(success)} success, {len(failures)} failures")

    if success:
        rows = [task_to_index_row(p) for p in success]
        index_path = args.index_dir / args.index_name
        df = pd.DataFrame(rows)
        df.to_parquet(str(index_path), index=False)
        print(f"Index written: {index_path} ({len(df)} rows)")

    if failures:
        print(f"\nFailures ({len(failures)}):")
        for iid, reason in failures[:20]:
            print(f"  {iid}: {reason}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")


if __name__ == "__main__":
    main()
