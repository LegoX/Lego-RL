"""
Evaluate Qwen3.5-35B on SWE-rebench filtered (6542 tasks × 8 trials each).

Features:
    - Resumable: skips instances that already have 8 completed results
    - Saves per-trial trajectory + reward in results_dir/<instance_id>/trial_<i>.json
    - Generates summary CSV and filtered index parquet at the end
    - Concurrent execution via asyncio (configurable parallelism)

Usage:
    # Assumes vLLM + LiteLLM proxy are already running (started by the shell wrapper)
    python eval_swerebench_filtered.py \
        --index /path/to/data/harbor_indexes/train_swerebench_filtered.parquet \
        --results-dir /path/to/data/eval_swerebench_filtered_qwen35 \
        --api-base http://localhost:8002 \
        --n-trials 8 \
        --n-concurrent 80
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Trial-level hard timeout (seconds). harbor only enforces PER-PHASE timeouts
# (env-start, agent-setup, agent-run, verify), which STACK — env-start retries +
# a 50min agent run can push one trial past 2h while the agent never even started.
# This single cap bounds the WHOLE trial regardless of which phase hangs; on
# timeout the trial's run() is cancelled (its shielded env.stop still tears the
# pod down) and the trial is recorded as reward=0. Set via --trial-timeout.
TRIAL_HARD_TIMEOUT_SEC = float(os.environ.get("HARBOR_TRIAL_HARD_TIMEOUT_SEC", "4500"))

try:
    from harbor.trial.trial import Trial
    from harbor.models.trial.config import TrialConfig
except ImportError:
    logger.error("harbor not installed. pip install harbor")
    sys.exit(1)


def load_index(path: str) -> list[dict]:
    df = pd.read_parquet(path)
    rows = []
    for _, row in df.iterrows():
        extra = row["extra_info"]
        rows.append({
            "instance_id": extra["instance_id"],
            "harbor_task_path": extra["harbor_task_path"],
        })
    return rows


def get_completed_trials(results_dir: Path, instance_id: str) -> list[dict]:
    inst_dir = results_dir / instance_id
    if not inst_dir.exists():
        return []
    results = []
    for f in sorted(inst_dir.glob("trial_*.json")):
        try:
            data = json.loads(f.read_text())
            if data.get("status") == "completed":
                results.append(data)
        except Exception:
            continue
    return results


def _parse_hostpath_mounts_env() -> list[dict] | None:
    """
    Read HARBOR_HOSTPATH_MOUNTS and return it as a list[dict] suitable for
    harbor_patch.environments.kubernetes.kubernetes.KubernetesEnvironment's
    ``host_path_mounts`` kwarg (passed via TrialConfig.environment.kwargs).

    Format (consumed by _coerce_host_path_mounts in harbor_patch k8s env,
    matches fully_async_2nodes_oh_mlg.sh:208):
        [{"host_path": str, "mount_path": str, "read_only": bool, "type"?: str}]
    """
    raw = os.environ.get("HARBOR_HOSTPATH_MOUNTS")
    if not raw or raw.lower() in ("null", "none", "[]"):
        return None
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("HARBOR_HOSTPATH_MOUNTS is not valid JSON, ignoring: %s", e)
        return None
    return entries


def build_trial_config(
    task_path: str,
    trial_name: str,
    trials_dir: str,
    api_base: str,
    model_name: str,
    temperature: float,
    agent_import_path: str,
    agent_max_iterations: int,
    agent_max_timeout_sec: int,
    agent_runtime_image: str,
    agent_runtime_mount_path: str,
    agent_runtime_image_subpath: str,
    env_import_path: str,
    env_override_cpus: int,
    env_override_memory_mb: int,
) -> dict:
    return {
        "task": {"path": task_path},
        "trial_name": trial_name,
        "trials_dir": trials_dir,
        "n_attempts": 1,
        "agent": {
            "name": os.environ.get("HARBOR_AGENT_NAME", "claude-code"),
            "import_path": agent_import_path,
            "model_name": model_name,
            "max_timeout_sec": agent_max_timeout_sec,
            "override_timeout_sec": agent_max_timeout_sec,
            "kwargs": {
                "max_turns": agent_max_iterations,
                "api_base": api_base,
                "temperature": temperature,
            },
        },
        "environment": {
            "type": None,
            "import_path": env_import_path,
            "force_build": False,
            "delete": True,
            "override_cpus": env_override_cpus,
            "override_memory_mb": env_override_memory_mb,
            "kwargs": {
                "agent_runtime_image": agent_runtime_image,
                "agent_runtime_mount_path": agent_runtime_mount_path,
                "agent_runtime_image_subpath": agent_runtime_image_subpath,
                "host_path_mounts": _parse_hostpath_mounts_env(),
            },
        },
        "verifier": {
            "enabled": True,
        },
    }


async def run_single_trial(
    instance_id: str,
    task_path: str,
    trial_idx: int,
    results_dir: Path,
    trial_cfg_kwargs: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    inst_dir = results_dir / instance_id
    inst_dir.mkdir(parents=True, exist_ok=True)
    result_file = inst_dir / f"trial_{trial_idx}.json"

    if result_file.exists():
        try:
            data = json.loads(result_file.read_text())
            if data.get("status") == "completed":
                return data
        except Exception:
            pass

    trial_name = f"{instance_id}_trial_{trial_idx}"
    cfg_dict = build_trial_config(
        task_path=task_path,
        trial_name=trial_name,
        trials_dir=str(inst_dir / "harbor_trials"),
        **trial_cfg_kwargs,
    )

    result = {
        "instance_id": instance_id,
        "trial_idx": trial_idx,
        "status": "failed",
        "reward": 0.0,
        "error": None,
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
        "chat_history_len": 0,
    }

    async with semaphore:
        try:
            trial_config = TrialConfig.model_validate(cfg_dict)
            trial = await Trial.create(trial_config)
            # Trial-level hard cap over the WHOLE run (env-start + agent + verify).
            # On timeout, run() is cancelled; its `finally: _cleanup_and_finalize`
            # runs a shielded environment.stop(delete=True), so the pod is still
            # torn down — no leaked k8s pods.
            trial_result = await asyncio.wait_for(
                trial.run(), timeout=TRIAL_HARD_TIMEOUT_SEC
            )

            result["finished_at"] = datetime.now().isoformat()

            exc_info = trial_result.exception_info
            if exc_info:
                result["error"] = f"{exc_info.exception_type}: {exc_info.exception_message}"

            if trial_result.verifier_result:
                reward = float(trial_result.verifier_result.rewards.get("reward", 0.0))
                result["reward"] = reward
                result["status"] = "completed"
            elif exc_info:
                result["status"] = "completed"
                result["reward"] = 0.0
            else:
                result["status"] = "completed"
                result["reward"] = 0.0

            if trial_result.agent_result and trial_result.agent_result.metadata:
                md = trial_result.agent_result.metadata
                # Different agents stash the conversation under different keys:
                # image_mounted_* uses "all_messages"; OpenHands-SDK uses none of
                # them in metadata. Try the known in-memory keys first.
                msgs = (
                    md.get("all_messages")
                    or md.get("messages")
                    or md.get("history")
                    or []
                )
                result["chat_history_len"] = len(msgs)

            # OpenHands-SDK does not populate metadata with the message list, so
            # the count above stays 0 for the oh-sdk agent. The real trajectory is
            # written to <trial>/agent/trajectory.json ({..., "steps": [...]}).
            # Fall back to counting its steps so chat_history_len reflects the
            # actual rollout length regardless of agent type / metadata wiring.
            if not result["chat_history_len"]:
                traj_file = (
                    inst_dir / "harbor_trials" / trial_name / "agent" / "trajectory.json"
                )
                try:
                    if traj_file.exists():
                        tdata = json.loads(traj_file.read_text())
                        steps = tdata.get("steps") if isinstance(tdata, dict) else tdata
                        if isinstance(steps, list):
                            result["chat_history_len"] = len(steps)
                except Exception:
                    pass

        except (asyncio.TimeoutError, TimeoutError):
            result["error"] = f"TrialHardTimeout: exceeded {TRIAL_HARD_TIMEOUT_SEC:.0f}s"
            result["finished_at"] = datetime.now().isoformat()
            result["status"] = "completed"
            result["reward"] = 0.0
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
            result["finished_at"] = datetime.now().isoformat()
            result["status"] = "completed"
            result["reward"] = 0.0

    result_file.write_text(json.dumps(result, indent=2, default=str))
    return result


async def evaluate_instance(
    instance_id: str,
    task_path: str,
    n_trials: int,
    results_dir: Path,
    trial_cfg_kwargs: dict,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    completed = get_completed_trials(results_dir, instance_id)
    completed_indices = {r["trial_idx"] for r in completed}
    remaining = [i for i in range(n_trials) if i not in completed_indices]

    if not remaining:
        return completed

    tasks = []
    for trial_idx in remaining:
        tasks.append(
            run_single_trial(
                instance_id=instance_id,
                task_path=task_path,
                trial_idx=trial_idx,
                results_dir=results_dir,
                trial_cfg_kwargs=trial_cfg_kwargs,
                semaphore=semaphore,
            )
        )

    new_results = await asyncio.gather(*tasks, return_exceptions=True)
    all_results = list(completed)
    for r in new_results:
        if isinstance(r, Exception):
            logger.error("Task %s exception: %s", instance_id, r)
        elif isinstance(r, dict):
            all_results.append(r)

    return all_results


def generate_summary(results_dir: Path, n_trials: int) -> pd.DataFrame:
    rows = []
    for inst_dir in sorted(results_dir.iterdir()):
        if not inst_dir.is_dir() or inst_dir.name.startswith("."):
            continue
        instance_id = inst_dir.name
        rewards = []
        for f in sorted(inst_dir.glob("trial_*.json")):
            try:
                data = json.loads(f.read_text())
                if data.get("status") == "completed":
                    rewards.append(data.get("reward", 0.0))
            except Exception:
                continue
        if rewards:
            rows.append({
                "instance_id": instance_id,
                "n_completed": len(rewards),
                "n_pass": sum(1 for r in rewards if r > 0),
                "n_fail": sum(1 for r in rewards if r == 0),
                "pass_rate": sum(1 for r in rewards if r > 0) / len(rewards),
                "rewards": rewards,
            })
    return pd.DataFrame(rows)


def generate_filtered_index(
    summary_df: pd.DataFrame,
    original_index_path: str,
    output_path: str,
    min_pass: int = 1,
) -> int:
    original_df = pd.read_parquet(original_index_path)

    original_instances = {}
    for idx, row in original_df.iterrows():
        iid = row["extra_info"]["instance_id"]
        original_instances[iid] = idx

    keep_ids = set()
    for _, row in summary_df.iterrows():
        pr = row["pass_rate"]
        if 0 < pr < 1:
            keep_ids.add(row["instance_id"])

    keep_indices = [original_instances[iid] for iid in keep_ids if iid in original_instances]
    filtered_df = original_df.loc[keep_indices].reset_index(drop=True)
    filtered_df.to_parquet(output_path, index=False)
    return len(filtered_df)


async def main_async(args):
    global TRIAL_HARD_TIMEOUT_SEC
    if args.trial_timeout:
        TRIAL_HARD_TIMEOUT_SEC = args.trial_timeout
    logger.info("Trial-level hard timeout: %.0fs", TRIAL_HARD_TIMEOUT_SEC)

    instances = load_index(args.index)
    logger.info("Loaded %d instances from %s", len(instances), args.index)

    # Optional: restrict to an explicit instance-id allowlist (one id per line).
    if args.instances_file:
        with open(args.instances_file) as f:
            wanted = {line.strip() for line in f if line.strip()
                      and not line.lstrip().startswith("#")}
        before = len(instances)
        instances = [inst for inst in instances if inst["instance_id"] in wanted]
        found = {inst["instance_id"] for inst in instances}
        missing = wanted - found
        logger.info(
            "Instance filter: %s -> kept %d/%d (matched %d/%d wanted ids, %d missing)",
            args.instances_file, len(instances), before,
            len(found), len(wanted), len(missing),
        )
        if missing:
            logger.warning(
                "%d wanted instance ids not present in index (e.g. %s)",
                len(missing), list(sorted(missing))[:5],
            )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    trial_cfg_kwargs = {
        "api_base": args.api_base,
        "model_name": args.model_name,
        "temperature": args.temperature,
        "agent_import_path": os.environ.get(
            "HARBOR_AGENT_IMPORT_PATH",
            "harbor_patch.agents.image_mounted_claude_code:ClaudeCode",
        ),
        "agent_max_iterations": int(os.environ.get("HARBOR_AGENT_MAX_ITERATIONS", "100")),
        "agent_max_timeout_sec": int(os.environ.get("HARBOR_AGENT_MAX_TIMEOUT_SEC", "1800")),
        "agent_runtime_image": os.environ.get(
            "HARBOR_AGENT_RUNTIME_IMAGE",
            "docker.io/jierun/c-cc-2.1.118:v0.1",
        ),
        "agent_runtime_mount_path": os.environ.get(
            "HARBOR_AGENT_RUNTIME_MOUNT_PATH",
            "/opt/custom-agent-runtime/claude-code",
        ),
        "agent_runtime_image_subpath": os.environ.get(
            "HARBOR_AGENT_RUNTIME_IMAGE_SUBPATH",
            "opt/custom-agent-runtime/claude-code",
        ),
        "env_import_path": os.environ.get(
            "HARBOR_ENVIRONMENT_IMPORT_PATH",
            "harbor_patch.environments.kubernetes.kubernetes:KubernetesEnvironment",
        ),
        "env_override_cpus": int(os.environ.get("HARBOR_ENVIRONMENT_OVERRIDE_CPUS", "1")),
        "env_override_memory_mb": int(os.environ.get("HARBOR_ENVIRONMENT_OVERRIDE_MEMORY_MB", "16384")),
    }

    semaphore = asyncio.Semaphore(args.n_concurrent)

    total = len(instances)
    done = 0
    pass_count = 0
    fail_count = 0
    start_time = time.time()

    # NO batch barrier: submit every instance at once and let the global
    # Semaphore(n_concurrent) be the ONLY concurrency gate. As soon as any trial
    # slot frees it is refilled by the next pending instance's trials — a 1-2
    # long-tail instance only holds its own slot instead of stalling a whole
    # `gather` wave (which previously left the freed slots idle and starved vLLM
    # to 0% GPU until the slowest member of the batch finished).
    async def _run_one(inst):
        res = await evaluate_instance(
            instance_id=inst["instance_id"],
            task_path=inst["harbor_task_path"],
            n_trials=args.n_trials,
            results_dir=results_dir,
            trial_cfg_kwargs=trial_cfg_kwargs,
            semaphore=semaphore,
        )
        return inst["instance_id"], res

    tasks = [asyncio.create_task(_run_one(inst)) for inst in instances]
    for fut in asyncio.as_completed(tasks):
        done += 1
        try:
            iid, res = await fut
        except Exception as e:
            logger.error("[%d/%d] task EXCEPTION: %s", done, total, e)
            continue
        rewards = [r.get("reward", 0.0) for r in res if isinstance(r, dict)]
        n_pass = sum(1 for r in rewards if r > 0)
        pass_count += n_pass
        fail_count += len(rewards) - n_pass
        elapsed = time.time() - start_time
        rate = done / elapsed * 3600 if elapsed > 0 else 0
        logger.info(
            "[%d/%d] %s  pass=%d/%d  (total pass_rate=%.3f, %.0f inst/hr)",
            done, total, iid, n_pass, len(rewards),
            pass_count / max(1, pass_count + fail_count),
            rate,
        )

    logger.info("Evaluation complete. Generating summary...")
    summary = generate_summary(results_dir, args.n_trials)
    summary_path = results_dir / "summary.csv"
    summary.to_csv(str(summary_path), index=False)
    logger.info("Summary written to %s", summary_path)

    all_pass = len(summary[summary["pass_rate"] == 1.0])
    all_fail = len(summary[summary["pass_rate"] == 0.0])
    partial = len(summary[(summary["pass_rate"] > 0) & (summary["pass_rate"] < 1.0)])
    logger.info(
        "Distribution: all_pass=%d  partial=%d  all_fail=%d  (total=%d)",
        all_pass, partial, all_fail, len(summary),
    )

    if args.output_index:
        n_kept = generate_filtered_index(
            summary, args.index, args.output_index,
        )
        logger.info(
            "Filtered index: %d tasks (excluded %d all-pass + %d all-fail) -> %s",
            n_kept, all_pass, all_fail, args.output_index,
        )

    pr_dist = summary["pass_rate"].describe()
    logger.info("Pass rate distribution:\n%s", pr_dist.to_string())


def main():
    ap = argparse.ArgumentParser(description="Evaluate SWE-rebench filtered with harbor trials")
    ap.add_argument("--index", type=str, required=True,
                    help="Path to task index parquet")
    ap.add_argument("--results-dir", type=str, required=True,
                    help="Directory to store per-instance results (resumable)")
    ap.add_argument("--output-index", type=str, default=None,
                    help="Path for filtered output index parquet (0 < pass_rate < 1)")
    ap.add_argument("--instances-file", type=str, default=None,
                    help="Optional path to a text file with one instance_id per "
                         "line; restricts the run to those instances (lines "
                         "starting with # are ignored).")
    ap.add_argument("--api-base", type=str, default="http://localhost:8002",
                    help="LiteLLM proxy URL")
    ap.add_argument("--model-name", type=str, default="vllm_model",
                    help="Model name for LiteLLM routing")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--n-trials", type=int, default=8,
                    help="Number of trials per instance")
    ap.add_argument("--n-concurrent", type=int, default=80,
                    help="Max concurrent trials")
    ap.add_argument("--trial-timeout", type=float, default=None,
                    help="Trial-level hard timeout in seconds (caps the WHOLE "
                         "trial: env-start + agent + verify). Overrides the "
                         "HARBOR_TRIAL_HARD_TIMEOUT_SEC env var. Default 4500.")
    args = ap.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
