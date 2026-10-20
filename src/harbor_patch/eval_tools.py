"""Read-only eval compatibility checks, score summaries and submission diagnostics."""

import argparse
import fnmatch
import json
import math
import os
import tomllib
from pathlib import Path


def require_separate_verifier() -> None:
    from harbor.models.task.config import TaskConfig, VerifierConfig
    from harbor.trial.trial import Trial

    required = {"environment_mode", "environment", "collect"}
    if (
        not required.issubset(VerifierConfig.model_fields)
        or "artifacts" not in TaskConfig.model_fields
        or not getattr(Trial, "SUPPORTS_SEPARATE_VERIFIER", False)
    ):
        raise ValueError(
            "This Harbor lacks separate-verifier/collect support. Apply "
            "patches/harbor.patch to the Harbor checkout used by VENV_PATH."
        )


def preflight() -> None:
    reward_key = os.environ.get("EVAL_REWARD_KEY", "").strip()
    reward_values = os.environ.get("EVAL_REWARD_VALUES", "").strip()
    if reward_values and reward_values.lower() != "null":
        from harbor.models.trial.config import VerifierConfig

        if not reward_key or reward_key.lower() == "null":
            raise ValueError("EVAL_REWARD_VALUES requires EVAL_REWARD_KEY")
        if "allowed_rewards" not in VerifierConfig.model_fields:
            raise ValueError(
                "This Harbor lacks verifier reward validation. Apply patches/harbor.patch."
            )
        VerifierConfig(allowed_rewards={reward_key: json.loads(reward_values)})
    if reward_key and reward_key.lower() != "null":
        from harbor.metrics.mean import Mean

        try:
            Mean(reward_key=reward_key)
        except TypeError as error:
            raise ValueError(
                "This Harbor lacks named reward metrics. Apply patches/harbor.patch "
                "to the Harbor checkout used by VENV_PATH."
            ) from error
    required = os.environ.get("EVAL_REQUIRE_SEPARATE_VERIFIER", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    dataset = os.environ.get("DATASET_PATH", "")
    root = Path(dataset) if dataset else None
    tasks = []
    if root is not None and root.is_dir():
        tasks = (
            [root / "task.toml"]
            if (root / "task.toml").is_file()
            else sorted(root.glob("*/task.toml"))
        )
        if not tasks:
            raise ValueError(f"No Harbor tasks found in {root}")
    elif required:
        if os.environ.get("DRY_RUN") != "1":
            raise ValueError(f"DeepSWE task directory does not exist: {dataset}")
        print(
            "[preflight] dataset unavailable; dry-run checks runner capabilities only"
        )

    raw_tasks = [(path, tomllib.loads(path.read_text())) for path in tasks]
    filters_raw = os.environ.get("EVAL_TASK_NAMES", "").strip()
    if filters_raw:
        filters = json.loads(filters_raw)
        if not isinstance(filters, list) or not filters or any(
            not isinstance(name, str) or not name for name in filters
        ):
            raise ValueError("EVAL_TASK_NAMES must be a nonempty JSON array of task names")
        if raw_tasks:
            unmatched = [name for name in filters if not any(
                fnmatch.fnmatchcase(path.parent.name, name) for path, _ in raw_tasks
            )]
            if unmatched:
                raise ValueError(f"EVAL_TASK_NAMES filters matched no task: {unmatched}")
            raw_tasks = [(path, raw) for path, raw in raw_tasks if any(
                fnmatch.fnmatchcase(path.parent.name, name) for name in filters
            )]
            print(f"[preflight] selected {len(raw_tasks)} tasks by EVAL_TASK_NAMES")
    needs_support = required or any(
        raw.get("verifier", {}).get("environment_mode") == "separate"
        or "environment" in raw.get("verifier", {})
        or raw.get("verifier", {}).get("collect")
        for _, raw in raw_tasks
    )
    if needs_support:
        require_separate_verifier()
        from harbor.models.task.config import TaskConfig

        for path, raw in raw_tasks:
            try:
                TaskConfig.model_validate(raw)
            except ValueError as error:
                raise ValueError(f"Invalid task {path}: {error}") from error
        print(
            f"[preflight] separate-verifier support OK; validated {len(raw_tasks)} task configs"
        )


def tally(
    job_dir: Path, reward_key: str = "reward", reward_values: list[float] | None = None
) -> dict[str, int]:
    allowed = reward_values if reward_values is not None else [0, 1]
    if not isinstance(allowed, list) or not allowed or any(
        not isinstance(v, (int, float)) or not math.isfinite(v) for v in allowed
    ):
        raise ValueError("reward_values must be a nonempty finite numeric array")
    # With a configured domain, 'solved' means the highest allowed score.
    solved_value = max(allowed)
    counts = {"solved": 0, "scored": 0, "errors": 0, "unscored": 0}
    root_path = job_dir / "result.json"
    root = json.loads(root_path.read_text()) if root_path.exists() else {}
    root_errors = {
        name
        for evaluation in root.get("stats", {}).get("evals", {}).values()
        for names in evaluation.get("exception_stats", {}).values()
        for name in names
    }
    seen = set()
    # Only top-level trials: do not count nested separate-verifier attempts.
    for trial in sorted(job_dir.iterdir()):
        if not trial.is_dir() or not (trial / "config.json").is_file():
            continue
        seen.add(trial.name)
        try:
            has_error = False
            result_path = trial / "result.json"
            if result_path.exists():
                result = json.loads(result_path.read_text())
                rewards = (result.get("verifier_result") or {}).get("rewards") or {}
                reward = rewards.get(reward_key)
                if result.get("exception_info"):
                    has_error = True
                    counts["errors"] += 1
                    # Agent timeouts can still have a valid scored submission.
                    if reward is None:
                        continue
            else:
                reward = None
                if trial.name in root_errors:
                    counts["errors"] += 1
                    continue
            if reward is None:
                text_path = trial / "verifier" / "reward.txt"
                json_path = trial / "verifier" / "reward.json"
                if json_path.exists():
                    reward = json.loads(json_path.read_text()).get(reward_key)
                elif text_path.exists() and reward_key == "reward":
                    reward = float(text_path.read_text().strip())
            if reward is None:
                counts["unscored"] += 1
            elif not isinstance(reward, (int, float)) or not math.isfinite(reward) or reward not in allowed:
                counts["errors"] += not has_error
            else:
                counts["scored"] += 1
                counts["solved"] += reward == solved_value
        except (ValueError, TypeError, OSError):
            counts["errors"] += 1
    counts["errors"] += len(root_errors - seen)
    # Completed retries can lose their local directory; preserve their recorded
    # score without overriding any available trial result or active error.
    aggregate_scores = {}
    for evaluation in root.get("stats", {}).get("evals", {}).values():
        for value, names in evaluation.get("reward_stats", {}).get(reward_key, {}).items():
            for name in names:
                aggregate_scores[name] = float(value)
    for name, reward in aggregate_scores.items():
        if name in seen or name in root_errors:
            continue
        if not math.isfinite(reward) or reward not in allowed:
            counts["errors"] += 1
        else:
            counts["scored"] += 1
            counts["solved"] += reward == solved_value
    return counts


def job_status(job_dir: Path) -> dict:
    path = job_dir / "result.json"
    root = json.loads(path.read_text()) if path.exists() else {}
    completed_names = {trial.name for trial in job_dir.iterdir() if trial.is_dir()
                       and (trial / "config.json").exists() and (trial / "result.json").exists()}
    for evaluation in root.get("stats", {}).get("evals", {}).values():
        for names in evaluation.get("reward_stats", {}).get("reward", {}).values():
            completed_names.update(names)
    completed = len(completed_names)
    total = root.get("n_total_trials")
    return {"finished": bool(root.get("finished_at")), "total": total,
            "completed": completed, "pending": max(0, total - completed) if total is not None else None}


def submission_diagnostics(job_dir: Path) -> dict:
    """Report missing test IDs separately from assertions; never alter rewards."""
    missing, assertions, empty, signals = {}, {}, [], {}
    for trial in sorted(job_dir.iterdir()):
        if not trial.is_dir() or not (trial / "config.json").exists():
            continue
        report = trial / "verifier" / "ctrf.json"
        if report.exists():
            for test in json.loads(report.read_text()).get("results", {}).get("tests", []):
                if test.get("status") != "failed":
                    continue
                bucket = missing if "missing from report" in test.get("message", "") else assertions
                bucket.setdefault(trial.name, []).append(test.get("name"))
        patch = trial / "artifacts" / "model.patch"
        if patch.exists() and patch.stat().st_size == 0:
            empty.append(trial.name)
        # These are observed log signatures, not automatic explanations or
        # permission to change a reward. Builds can fail because of agent code.
        raw = trial / "verifier" / "test-stdout.txt"
        if raw.exists():
            output = raw.read_text(errors="replace")
            observed = []
            if "no space left on device" in output.lower():
                observed.append("storage_exhausted")
            if "[build failed]" in output:
                observed.append("build_failed_cause_unresolved")
            if "panic: test timed out" in output:
                observed.append("test_process_timeout")
            if "missing or invalid JSON" in output:
                observed.append("missing_or_invalid_test_report")
            if observed:
                signals[trial.name] = observed
    return {"missing_test_ids": missing, "other_failed_tests": assertions,
            "empty_patches": empty, "verifier_signals": signals}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "tally", "diagnose"))
    parser.add_argument("job_dir", nargs="?", type=Path)
    parser.add_argument("--reward-key", default=os.environ.get("EVAL_REWARD_KEY") or "reward")
    parser.add_argument("--reward-values", default=os.environ.get("EVAL_REWARD_VALUES") or "[0,1]")
    args = parser.parse_args()
    try:
        if args.command == "preflight":
            preflight()
        else:
            if args.job_dir is None:
                parser.error(f"{args.command} requires job_dir")
            if args.command == "diagnose":
                print(json.dumps(submission_diagnostics(args.job_dir), ensure_ascii=False, indent=2))
                return
            key = "reward" if args.reward_key.lower() == "null" else args.reward_key
            values = json.loads(args.reward_values)
            counts = tally(args.job_dir, key, values)
            status = job_status(args.job_dir)
            print(
                ("=== DONE: " if status["finished"] else "=== RUNNING: ")
                + " / ".join(f"{k}={v}" for k, v in counts.items())
                + " / " + " / ".join(f"{k}={v}" for k, v in status.items())
                + f" / reward_key={key} / solved_value={max(values or [0, 1])}"
                + f" ({args.job_dir}) ==="
            )
    except (ValueError, TypeError, ImportError, OSError) as error:
        parser.exit(1, f"[FATAL] {error}\n")


if __name__ == "__main__":
    main()
