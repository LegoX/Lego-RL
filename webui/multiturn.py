"""Read optional multi-turn artifacts without changing legacy run interpretation."""
import importlib.util
import json
import math
from pathlib import Path


def read_multiturn(task_dir, details=False):
    path = Path(task_dir) / "multiturn.json"
    try:
        if path.stat().st_size > 8 * 1024 * 1024:
            return {"enabled": True, "valid": False, "error": "Multi-turn artifact exceeds read limit"}
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or not value.get("enabled"):
            return None
        # Earlier artifacts predate the explicit validity field.
        value.setdefault("valid", value.get("stop_reason") not in ("error", "cancelled", None))
        value.setdefault("task_profile", "ccbench")
        if details:
            conversation = Path(task_dir) / "conversation.json"
            if not value.get("rounds") and conversation.is_file():
                try:
                    if conversation.stat().st_size <= 8 * 1024 * 1024:
                        turns = json.loads(conversation.read_text())
                        if isinstance(turns, list):
                            value["rounds"] = [r for r in turns if isinstance(r, dict)]
                except (OSError, ValueError):
                    pass
            for row in value.get("rounds", []):
                number = row.get("round_id")
                if not isinstance(number, int) or number < 1:
                    continue
                directory = Path(task_dir) / "multiturn" / f"round-{number:03d}"
                for key, relative in (("test_output", "verification/verifier/test-stdout.txt"),
                                      ("agent_trace", "claude-code.jsonl")):
                    try:
                        with (directory / relative).open("rb") as stream:
                            stream.seek(max(0, (directory / relative).stat().st_size - 24000))
                            row[key] = stream.read(24000).decode(errors="replace")
                    except OSError:
                        row[key] = ""
        else:
            value.pop("rounds", None)
        return value
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError):
        return {"enabled": True, "valid": False, "error": "Unreadable multi-turn artifact"}


def final_reward(value):
    if not value or not value.get("valid"):
        return None
    if value.get("task_profile") == "swe_interact":
        score = value.get("training_reward", value.get("final_reward"))
        return score if isinstance(score, (float, int)) and math.isfinite(score) else None
    rounds = value.get("round_rewards", [])
    if not rounds:
        return None
    score = rounds[-1].get("training_reward")
    return score if isinstance(score, (float, int)) and math.isfinite(score) else None


def artifact_metrics(summaries):
    # Shared pure-Python aggregation, no torch/Harbor dependency in the web server.
    source = Path(__file__).resolve().parents[1] / "src/verl_patch/multiturn/metrics.py"
    spec = importlib.util.spec_from_file_location("multiturn_statistics", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.aggregate_episodes(summaries, prefix="eval-multiturn")
