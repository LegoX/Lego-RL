import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "webui"))
from multiturn import read_multiturn, artifact_metrics


def test_dashboard_final_score_and_failed_episode(tmp_path):
    path = tmp_path / "multiturn.json"
    data = {"enabled": True, "valid": True, "task_profile": "openswe", "max_rounds": 5,
            "stop_reason": "verifier_success", "round_rewards": [
                {"round_id": 1, "functional_reward": 1, "judge_score": 0.4, "training_reward": 0.7}]}
    path.write_text(json.dumps(data))
    (tmp_path / "verifier").mkdir()
    (tmp_path / "verifier/reward.txt").write_text("1")
    spec = importlib.util.spec_from_file_location("dashboard_server_test", ROOT / "webui/server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._read_reward(str(tmp_path)) == 0.7
    assert module._read_success_reward(str(tmp_path)) == 1
    assert module._has_trajectory(str(tmp_path))
    assert artifact_metrics([read_multiturn(tmp_path)])["eval-multiturn/openswe/final_reward"] == 0.7
    data["round_rewards"][0].update(functional_reward=0, training_reward=0.2)
    path.write_text(json.dumps(data))
    assert module._read_reward(str(tmp_path)) == 0.2
    assert module._read_success_reward(str(tmp_path)) == 0
    data["valid"] = False
    path.write_text(json.dumps(data))
    assert module._read_reward(str(tmp_path)) is None  # no fallback to stale reward=1
    path.unlink()
    assert module._read_reward(str(tmp_path)) == 1  # unchanged single-turn behavior


def test_round_details_read_only_bounded_artifacts(tmp_path):
    directory = tmp_path / "multiturn/round-001"
    (directory / "verification/verifier").mkdir(parents=True)
    (directory / "claude-code.jsonl").write_text("tool " * 10000)
    (directory / "verification/verifier/test-stdout.txt").write_text("FAILED")
    (tmp_path / "multiturn.json").write_text(json.dumps({"enabled": True, "valid": True, "rounds": [{"round_id": 1}]}))
    value = read_multiturn(tmp_path, details=True)
    assert len(value["rounds"][0]["agent_trace"]) == 24000
    assert value["rounds"][0]["test_output"] == "FAILED"
    assert "rounds" not in read_multiturn(tmp_path)


def test_multiturn_api_and_distribution_do_not_infer_gradient_from_test_success(tmp_path):
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location("dashboard_server_metrics_test", ROOT / "webui/server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("task-a", "task-b"):
        directory = tmp_path / "step_0001" / name
        directory.mkdir(parents=True)
        (directory / "multiturn.json").write_text(json.dumps({
            "enabled": True, "valid": True, "task_profile": "openswe", "max_rounds": 5,
            "round_rewards": [{"round_id": 1, "functional_reward": 0, "training_reward": 0.4}],
        }))
    handler = SimpleNamespace(_exp_dirs_for=lambda _: [str(tmp_path),str(tmp_path)],
                              _json=lambda payload, *args: payload)
    output = module.DashboardHandler._handle_multiturn(handler, "run", {"step": ["step_0001"]})
    assert output["metrics"]["eval-multiturn/openswe/valid_episodes"] == 2
    assert output["metrics"]["eval-multiturn/openswe/final_success_rate"] == 0
    dist = module._rollout_distribution([str(tmp_path)], expected_n=2)
    assert dist["is_multiturn"]
    assert dist["no_signal"] is None
    assert "cannot establish" in module._fmt_rollout_dist(dist, None, None)
