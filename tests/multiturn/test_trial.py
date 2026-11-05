"""Run the real Harbor lifecycle/verifier with CPU-only filesystem environments."""

import asyncio
import json
import os
import re
import shutil
from types import SimpleNamespace

import pytest
from harbor.agents.factory import AgentFactory
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import AgentInfo
from harbor_patch.multiturn.config import MultiTurnConfig
from harbor_patch.multiturn.trial import EnvironmentFactory, create_multiturn_trial


class FilesystemEnvironment:
    """Test double for container isolation; exec runs real shell/tar/test scripts."""

    is_mounted = True
    supports_dynamic_network_policy = False
    default_user = None

    def __init__(self, root, paths, workspace="app"):
        self.workspace = workspace
        self.root = root
        self.paths = paths
        self.starts = self.stops = 0
        (root / workspace).mkdir(parents=True)
        (root / "tmp").mkdir()
        (root / "logs").mkdir()
        for name in ("agent", "verifier", "artifacts"):
            (root / "logs" / name).symlink_to(getattr(paths, f"{name}_dir"))
        (root / workspace / "binary.dat").write_bytes(b"\x00\xff\x80original\x00")
        (root / workspace / "binary-link").symlink_to("binary.dat")

    def path(self, absolute):
        return self.root / str(absolute).lstrip("/")

    async def start(self, force_build=False):
        self.starts += 1

    async def stop(self, delete=True):
        self.stops += 1
        self.deleted = delete

    async def run_healthcheck(self):
        pass

    async def prepare_logs_for_host(self):
        pass

    async def apply_network_policy(self, policy):
        pass

    async def exec(self, command, user=None, env=None, **kwargs):
        command = re.sub(
            r"/(app|testbed|tmp|logs|tests)(?=/| |$)",
            lambda m: str(self.path(m.group())),
            command,
        )
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=self.root,
            env={**os.environ, **(env or {}), "SANDBOX_ROOT": str(self.root)},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return SimpleNamespace(
            return_code=process.returncode,
            stdout=stdout.decode(),
            stderr=stderr.decode(),
        )

    async def download_file(self, source_path, target_path):
        shutil.copy2(self.path(source_path), target_path)

    async def upload_file(self, source_path, target_path):
        shutil.copy2(source_path, self.path(target_path))

    async def upload_dir(self, source_dir, target_dir):
        shutil.copytree(source_dir, self.path(target_dir), dirs_exist_ok=True)


class Agent:
    def to_agent_info(self):
        return AgentInfo(name="test-native-cc", version="1")

    async def setup(self, environment):
        pass

    async def run_user_round(self, instruction, environment, context, round_id, log_path):
        app = environment.path("/" + environment.workspace)
        assert not (app / "hidden-test.txt").exists()
        assert (app / "binary.dat").read_bytes() == b"\x00\xff\x80original\x00"
        assert (app / "binary-link").is_symlink()
        previous = (app / "progress").read_text() if (app / "progress").exists() else "0"
        assert int(previous) == round_id - 1
        (app / "progress").write_text(str(round_id))
        log_path.write_text(json.dumps({"instruction": instruction, "round": round_id}))
        return {
            "session_id": "same-native-session",
            "final_message": f"Completed round {round_id}",
        }


@pytest.fixture
def setup_trial(tmp_path, monkeypatch, request):
    workspace = getattr(request, "param", "app")
    task = tmp_path / "task"
    task.mkdir()
    (task / "instruction.md").write_text("Fix the program.")
    (task / "task.toml").write_text('version = "1.0"\n[environment]\nallow_internet = true\n')
    (task / "environment").mkdir()
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\nWORKDIR /app\n")
    (task / "tests").mkdir()
    script = task / "tests" / "test.sh"
    script.write_text("""#!/bin/sh
set -eu
# Deliberately mutate the verifier's workspace: none of this may reach the agent.
echo hidden > "$SANDBOX_ROOT/app/hidden-test.txt"
rm "$SANDBOX_ROOT/app/binary.dat"
if [ "$(cat "$SANDBOX_ROOT/app/progress")" -ge 3 ]; then
  echo 1 > "$SANDBOX_ROOT/logs/verifier/reward.txt"
else
  echo 0 > "$SANDBOX_ROOT/logs/verifier/reward.txt"
fi
""")
    if workspace != "app":
        script.write_text(script.read_text().replace("/app", "/" + workspace))
        dockerfile = task / "environment/Dockerfile"
        dockerfile.write_text(dockerfile.read_text().replace("/app", "/" + workspace))
    environments = []

    def create_environment(**kwargs):
        environment = FilesystemEnvironment(tmp_path / "envs" / str(len(environments)), kwargs["trial_paths"], workspace)
        environments.append(environment)
        return environment

    monkeypatch.setattr(EnvironmentFactory, "create_environment_from_config", create_environment)
    monkeypatch.setattr(AgentFactory, "create_agent_from_config", lambda *a, **k: Agent())
    starts = []

    async def begin_user_round(sid, number):
        starts.append((sid, number))

    config = TrialConfig.model_validate(
        {
            "task": {"path": str(task)},
            "trial_name": "episode",
            "trials_dir": str(tmp_path / "trials"),
            "agent": {"import_path": "harbor_patch.agents.image_mounted_claude_code:ClaudeCode"},
            "predict_patch": {"enabled": False},
        }
    )
    options = MultiTurnConfig(
        enabled=True,
        task_profile="openswe" if workspace == "testbed" else "ccbench",
        max_rounds=5,
        simulator_mode="scripted",
        scripted_followups=["Still failing", "Try this case", "One more"],
    )
    return SimpleNamespace(
        config=config,
        options=options,
        environments=environments,
        starts=starts,
        proxy=SimpleNamespace(begin_user_round=begin_user_round),
        script=script,
    )


@pytest.mark.asyncio
async def test_three_rounds_preserve_live_files_and_use_final_reward(setup_trial):
    s = setup_trial
    trial = await create_multiturn_trial(s.config, s.options, s.proxy, "proxy-session")
    result = await trial.run()
    assert result.exception_info is None
    assert result.verifier_result.rewards == {"reward": 1.0}
    assert s.starts == [("proxy-session", n) for n in (1, 2, 3)]
    assert len(s.environments) == 4
    assert all(e.starts == e.stops == 1 for e in s.environments)
    assert all(e.deleted for e in s.environments[1:])
    assert not (s.environments[0].root / "tests").exists()
    metadata = result.agent_result.metadata["multiturn"]
    assert metadata["num_user_rounds"] == 3
    assert metadata["stop_reason"] == "verifier_success"
    sidecar = json.loads((trial.trial_dir / "multiturn.json").read_text())
    assert [r["functional_reward"] for r in sidecar["rounds"]] == [0, 0, 1]
    assert {r["session_id"] for r in sidecar["rounds"]} == {"same-native-session"}
    assert not list(trial.trial_dir.rglob("workspace.tar"))
    assert s.config.agent.import_path.endswith(":ClaudeCode")  # Original config unchanged.


@pytest.mark.asyncio
@pytest.mark.parametrize("max_rounds,stop_on_success,expected", [(1, True, 1), (5, False, 4), (2, True, 2)])
async def test_round_cap_and_no_op(setup_trial, max_rounds, stop_on_success, expected):
    s = setup_trial
    options = s.options.model_copy(update={"max_rounds": max_rounds, "stop_on_success": stop_on_success})
    trial = await create_multiturn_trial(s.config, options, s.proxy, "session")
    result = await trial.run()
    assert result.exception_info is None
    assert len(s.starts) == expected
    assert result.agent_result.metadata["multiturn"]["stop_reason"] == (
        "simulator_no_op" if expected == 4 else "max_rounds"
    )


@pytest.mark.asyncio
async def test_simulator_failure_discards_previous_reward(setup_trial):
    s = setup_trial
    trial = await create_multiturn_trial(s.config, s.options, s.proxy, "session")

    async def fail(*args):
        raise RuntimeError("simulator offline")

    trial._next_instruction = fail
    result = await trial.run()
    assert result.verifier_result is None
    assert result.exception_info.exception_type == "MultiTurnError"
    assert all(e.stops == 1 for e in s.environments)
    assert result.agent_result.metadata["multiturn"]["stop_reason"] == "error"


@pytest.mark.asyncio
async def test_browser_skip_is_invalid_not_a_success(setup_trial):
    s = setup_trial
    s.script.write_text(
        s.script.read_text()
        + '\necho "browser error treated as non-blocking"\necho 1 > "$SANDBOX_ROOT/logs/verifier/reward.txt"\n'
    )
    trial = await create_multiturn_trial(s.config, s.options, s.proxy, "session")
    result = await trial.run()
    assert result.verifier_result is None
    assert result.exception_info.exception_type == "MultiTurnError"
    assert all(e.stops == 1 for e in s.environments)


@pytest.mark.asyncio
async def test_episode_timeout_cleans_up_without_stale_reward(setup_trial):
    s = setup_trial
    trial = await create_multiturn_trial(
        s.config,
        s.options.model_copy(update={"episode_timeout_sec": 0.05}),
        s.proxy,
        "session",
    )

    async def slow(*args):
        await asyncio.sleep(10)

    trial._agent.run_user_round = slow
    result = await trial.run()
    assert result.verifier_result is None
    assert result.exception_info.exception_type == "MultiTurnError"
    assert all(e.stops == 1 for e in s.environments)


@pytest.mark.asyncio
async def test_cancellation_cleans_up(setup_trial):
    s = setup_trial
    trial = await create_multiturn_trial(s.config, s.options, s.proxy, "session")
    entered = asyncio.Event()

    async def wait(*args):
        entered.set()
        await asyncio.Event().wait()

    trial._agent.run_user_round = wait
    running = asyncio.create_task(trial.run())
    await entered.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert all(e.stops == 1 for e in s.environments)
    assert trial.result.verifier_result is None


@pytest.mark.asyncio
async def test_each_verifier_uses_pristine_environment_definition(setup_trial, monkeypatch):
    s = setup_trial
    original = EnvironmentFactory.create_environment_from_config
    images_before = []

    def mutating_factory(**kwargs):
        images_before.append(kwargs["task_env_config"].docker_image)
        kwargs["task_env_config"].docker_image = "inferred-base-image"
        return original(**kwargs)

    monkeypatch.setattr(EnvironmentFactory, "create_environment_from_config", mutating_factory)
    trial = await create_multiturn_trial(s.config, s.options, s.proxy, "session")
    result = await trial.run()
    assert result.exception_info is None
    assert images_before == [None] * 4


@pytest.mark.asyncio
@pytest.mark.parametrize("reward", ["nan", "inf", "2"])
async def test_invalid_verifier_reward_is_not_trainable(setup_trial, reward):
    s = setup_trial
    s.script.write_text(f'#!/bin/sh\necho {reward} > "$SANDBOX_ROOT/logs/verifier/reward.txt"\n')
    trial = await create_multiturn_trial(s.config, s.options, s.proxy, "session")
    result = await trial.run()
    assert result.verifier_result is None
    assert result.exception_info.exception_type == "MultiTurnError"
    assert all(e.stops == 1 for e in s.environments)


@pytest.mark.asyncio
async def test_cancellation_inside_verifier_cleans_both_environments(setup_trial, monkeypatch):
    from harbor_patch.multiturn.trial import Verifier

    s = setup_trial
    entered = asyncio.Event()

    async def wait(self):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(Verifier, "verify", wait)
    trial = await create_multiturn_trial(s.config, s.options, s.proxy, "session")
    running = asyncio.create_task(trial.run())
    await entered.wait()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert len(s.environments) == 2
    assert all(e.stops == 1 for e in s.environments)
    assert trial.result.verifier_result is None


@pytest.mark.asyncio
async def test_per_round_judge_reward_and_test_only_success_gate(setup_trial, monkeypatch):
    from harbor_patch.multiturn.round_judge import RoundJudge, UXJudgeResult

    s = setup_trial
    s.options = s.options.model_copy(
        update={"judge_enabled": True, "judge_model": "openai/test", "reward_mode": "per_round"}
    )
    calls = []
    monkeypatch.setattr(RoundJudge, "__init__", lambda self, options: None)

    async def score(self, payload, directory):
        calls.append(payload)
        value = [100, 60, 0][len(calls) - 1]
        return UXJudgeResult(
            reward=0.99,
            task_success_score=value,
            final_requirement_following_score=value,
            context_continuity_score=value,
            correction_handling_score=value,
            testing_and_verification_score=value,
            user_effort_score=value,
            rationale="checkpoint",
        )

    monkeypatch.setattr(RoundJudge, "score", score)
    trial = await create_multiturn_trial(s.config, s.options, s.proxy, "proxy-session")
    result = await trial.run()
    assert result.exception_info is None
    assert result.verifier_result.rewards["reward"] == 0.5
    mt = result.agent_result.metadata["multiturn"]
    assert mt["stop_reason"] == "verifier_success"
    assert [r["training_reward"] for r in mt["round_rewards"]] == [0.5, 0.3, 0.5]
    assert [len(c["rounds"]) for c in calls] == [1, 2, 3]
    assert len(s.starts) == 3  # Judge=1 alone didn't stop; test=1/judge=0 did.
    assert all(e.stops == 1 for e in s.environments)


@pytest.mark.asyncio
async def test_judge_failure_discards_previous_round_rewards(setup_trial, monkeypatch):
    from harbor_patch.multiturn.round_judge import RoundJudge

    s = setup_trial
    s.options = s.options.model_copy(
        update={"judge_enabled": True, "judge_model": "openai/test", "reward_mode": "per_round"}
    )
    monkeypatch.setattr(RoundJudge, "__init__", lambda self, options: None)

    async def score(self, payload, directory):
        if len(payload["rounds"]) == 2:
            raise TimeoutError("judge unavailable")
        return SimpleNamespace(task_success_score=80, model_dump=lambda: {})

    monkeypatch.setattr(RoundJudge, "score", score)
    trial = await create_multiturn_trial(s.config, s.options, s.proxy, "proxy-session")
    result = await trial.run()
    assert result.exception_info is not None
    assert result.verifier_result is None
    assert len(s.starts) == 2
    assert all(e.stops == 1 for e in s.environments)


@pytest.mark.asyncio
@pytest.mark.parametrize("setup_trial", ["testbed"], indirect=True)
async def test_openswe_rounds_use_testbed_without_changing_source_data(setup_trial):
    s = setup_trial
    task = s.config.task.path
    before = {str(p): p.read_bytes() for p in task.rglob("*") if p.is_file()}
    trial = await create_multiturn_trial(s.config, s.options, s.proxy, "openswe-session")
    result = await trial.run()
    assert result.exception_info is None
    assert result.verifier_result.rewards["reward"] == 1
    assert len(s.starts) == 3
    assert all(not (e.root / "app").exists() for e in s.environments)
    assert not (s.environments[0].root / "testbed/hidden-test.txt").exists()
    assert before == {str(p): p.read_bytes() for p in task.rglob("*") if p.is_file()}
    assert result.agent_result.metadata["multiturn"]["test_metric"] == "task_success"


@pytest.mark.asyncio
@pytest.mark.parametrize("limit_round", [1, 2])
async def test_context_limit_verifies_current_workspace_and_stops_without_stale_reward(setup_trial, monkeypatch, limit_round):
    from harbor_patch.multiturn.round_judge import RoundJudge

    s = setup_trial
    s.options = s.options.model_copy(update={"judge_enabled": True, "judge_model": "openai/test", "reward_mode": "per_round"})
    trial = await create_multiturn_trial(s.config, s.options, s.proxy, "session")
    original = trial._agent.run_user_round

    async def limited(*args):
        result = await original(*args)
        if args[3] == limit_round:
            result.update(stop_reason="context_limit", final_message="")
        return result

    trial._agent.run_user_round = limited
    monkeypatch.setattr(RoundJudge, "__init__", lambda *a: None)
    calls = []

    async def score(self, payload, directory):
        calls.append(payload["judged_round_id"])
        return SimpleNamespace(task_success_score=80 / len(calls), model_dump=lambda: {})

    monkeypatch.setattr(RoundJudge, "score", score)
    result = await trial.run()
    assert result.exception_info is None
    assert result.verifier_result.rewards == {"reward": .4 / limit_round}
    mt = result.agent_result.metadata["multiturn"]
    assert mt["valid"] and mt["stop_reason"] == "context_limit"
    assert calls == list(range(1, limit_round + 1))
    assert [r["training_reward"] for r in mt["round_rewards"]] == [.4 / n for n in calls]
    assert len(s.starts) == limit_round
    assert len(s.environments) == limit_round + 1
    assert all(e.stops == 1 for e in s.environments)


@pytest.mark.asyncio
async def test_context_limit_with_failed_verifier_is_still_invalid(setup_trial):
    s = setup_trial
    trial = await create_multiturn_trial(s.config, s.options, s.proxy, "session")

    async def limited(*args):
        return {"stop_reason": "context_limit", "final_message": ""}

    async def fail(*args):
        raise RuntimeError("verifier offline")

    trial._agent.run_user_round = limited
    trial._verify_round = fail
    result = await trial.run()
    assert result.verifier_result is None
    assert result.exception_info.exception_type == "MultiTurnError"
    assert not result.agent_result.metadata["multiturn"]["valid"]
