import asyncio
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from harbor.environments.base import ExecResult
from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from harbor.models.trial.paths import TrialPaths
from harbor.trial.trial import Trial

from harbor_patch.environments.kubernetes import kubernetes as k8s_module
from harbor_patch.environments.kubernetes.kubernetes import KubernetesEnvironment
from harbor_patch.eval_tools import preflight, tally, job_status, submission_diagnostics

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "name",
    [
        "valid_Name.123",
        "datacurve/abs-module-cache-flags",
        "task-" * 30,
        "任务/试验",
    ],
)
def test_k8s_pod_labels_accept_package_task_names(tmp_path, monkeypatch, name):
    monkeypatch.setattr(KubernetesEnvironment, "_NETADMIN_IMAGE", "test:iptables")
    monkeypatch.setenv("HARBOR_K8S_SKIP_DOCKERFILE", "true")
    env = KubernetesEnvironment(
        environment_dir=tmp_path,
        environment_name=name,
        session_id=name + "__unique-trial-id",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(docker_image="example:latest"),
    )
    pod = env._build_pod_spec()
    for value in pod.metadata.labels.values():
        assert len(value) <= 63
        assert re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?", value)
    assert pod.metadata.annotations["harbor/environment-name"] == name
    assert pod.metadata.annotations["kubectl.kubernetes.io/default-container"] == "main"
    if name == "valid_Name.123":
        assert pod.metadata.labels["environment"] == name


@pytest.mark.parametrize("prebuilt", [True, False])
def test_k8s_prebuilt_image_inherits_path(tmp_path, monkeypatch, prebuilt):
    monkeypatch.setattr(KubernetesEnvironment, "_NETADMIN_IMAGE", "test:iptables")
    monkeypatch.setenv("HARBOR_K8S_SKIP_DOCKERFILE", str(prebuilt).lower())
    env = KubernetesEnvironment(
        environment_dir=tmp_path,
        environment_name="test",
        session_id="trial",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(docker_image="example:latest"),
    )
    pod = env._build_pod_spec()
    variables = {entry.name: entry.value for entry in pod.spec.containers[0].env}
    assert ("PATH" not in variables) == prebuilt
    mounts = {entry.mount_path for entry in pod.spec.containers[0].volume_mounts}
    assert ("/tmp" not in mounts) == prebuilt
    assert "/logs" in mounts
    assert ("tmp-dir" not in {v.name for v in pod.spec.volumes}) == prebuilt


@pytest.mark.parametrize("directory", [False, True])
@pytest.mark.parametrize(
    "open_stream,exit_code,error",
    [
        (True, None, TimeoutError),
        (False, 2, RuntimeError),
        (False, None, RuntimeError),
    ],
)
def test_k8s_upload_rejects_incomplete_or_failed_extraction(
    tmp_path, monkeypatch, directory, open_stream, exit_code, error
):
    source = tmp_path / "grader.py"
    source.write_text("print('grade')")
    env = KubernetesEnvironment.__new__(KubernetesEnvironment)
    env.pod_name, env.namespace = "test", "default"
    env._core_api = MagicMock()
    env._ensure_client = AsyncMock()
    env._wait_for_container_exec_ready = AsyncMock()
    env.exec = AsyncMock(return_value=ExecResult(return_code=0))
    env.logger = logging.getLogger()
    stream = MagicMock()
    stream.is_open.return_value = open_stream
    stream.returncode = exit_code
    stream.read_stderr.return_value = "tar extraction failed"
    monkeypatch.setattr(k8s_module, "stream", MagicMock(return_value=stream))
    method = (
        KubernetesEnvironment.upload_dir
        if directory
        else KubernetesEnvironment.upload_file
    )
    with pytest.raises(error):
        asyncio.run(
            method.__wrapped__(env, tmp_path if directory else source, "/tests")
        )
    stream.close.assert_called_once()


@pytest.mark.parametrize("limit", ["", "4096", "invalid; exit 0"])
def test_k8s_exec_open_file_limit_is_inherited(monkeypatch, limit):
    monkeypatch.setenv("HARBOR_K8S_NOFILE", limit)
    env = KubernetesEnvironment.__new__(KubernetesEnvironment)
    env.pod_name, env.namespace = "test", "default"
    env._core_api = MagicMock()
    env._ensure_client = AsyncMock()
    env._resolve_user = lambda user: user
    env._merge_env = lambda variables: variables
    env._read_exec_output = lambda response: ("", "")
    response = MagicMock(returncode=0)
    captured = MagicMock(return_value=response)
    monkeypatch.setattr(k8s_module, "stream", captured)
    if limit.startswith("invalid"):
        with pytest.raises(ValueError, match="positive integer"):
            asyncio.run(env.exec("ulimit -Sn"))
        captured.assert_not_called()
        return
    asyncio.run(env.exec("ulimit -Sn"))
    command = captured.call_args.kwargs["command"]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    expected = (
        limit
        or subprocess.check_output(["bash", "-c", "ulimit -Sn"], text=True).strip()
    )
    assert result.stdout.strip() == expected


def test_offline_network_rejects_external_requests_after_loopback_allow():
    env = KubernetesEnvironment.__new__(KubernetesEnvironment)
    script = env._build_egress_iptables(
        NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
    )
    assert script.index("iptables -A OUTPUT -o lo -j ACCEPT") < script.index(
        "iptables -A OUTPUT -j REJECT"
    )
    assert script.index("ip6tables -A OUTPUT -o lo -j ACCEPT") < script.index(
        "ip6tables -A OUTPUT -j REJECT"
    )
    assert "iptables -P OUTPUT DROP" in script
    online = env._build_egress_iptables(
        NetworkPolicy(network_mode=NetworkMode.ALLOWLIST)
    )
    assert "-j REJECT" not in online


@pytest.mark.parametrize("private_lan", [None, "false"])
def test_model_allowlist_does_not_open_private_lan_when_disabled(
    monkeypatch, private_lan
):
    from harbor.models.task.config import TaskConfig
    from harbor.models.trial.config import AgentConfig
    from harbor.trial.network_policy import resolve_agent_phase_policy

    monkeypatch.delenv("HARBOR_K8S_ALLOW_PRIVATE_NETWORK", raising=False)
    if private_lan is not None:
        monkeypatch.setenv("HARBOR_K8S_ALLOW_PRIVATE_NETWORK", private_lan)
    task = TaskConfig(agent={"network_mode": "no-network"})
    policy = resolve_agent_phase_policy(
        task,
        AgentConfig(extra_allowed_hosts=["192.168.16.115"]),
        task.environment.resolve_baseline(),
    )
    env = KubernetesEnvironment.__new__(KubernetesEnvironment)
    script = env._build_egress_iptables(policy)
    assert "getent ahostsv4 192.168.16.115" in script
    assert "iptables -P OUTPUT DROP" in script
    for cidr in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"):
        assert (f"-d {cidr} -j ACCEPT" in script) == (private_lan is None)
    offline = env._build_egress_iptables(NetworkPolicy(network_mode="no-network"))
    assert "192.168.16.115" not in offline


@pytest.mark.parametrize("values", ["[]", "[NaN]", '"bad"'])
def test_preflight_rejects_invalid_reward_domain(tmp_path, monkeypatch, values):
    monkeypatch.setenv("EVAL_REWARD_KEY", "reward")
    monkeypatch.setenv("EVAL_REWARD_VALUES", values)
    with pytest.raises(ValueError):
        preflight()


def test_tally_json_text_errors_and_nested_verifiers(tmp_path):
    for name, value in [
        ("json", {"reward": 1, "partial": 1}),
        ("failed", {"reward": 0}),
        ("crash", -1),
        ("text", 1.0),
        ("invalid", "bad"),
    ]:
        trial = tmp_path / name
        (trial / "verifier").mkdir(parents=True)
        (trial / "config.json").write_text("{}")
        if isinstance(value, dict):
            (trial / "verifier" / "reward.json").write_text(json.dumps(value))
        else:
            (trial / "verifier" / "reward.txt").write_text(str(value))
        nested = trial / "verifier-environments" / "attempt" / "verifier"
        nested.mkdir(parents=True)
        (nested / "reward.txt").write_text("1")
    assert tally(tmp_path) == {"solved": 2, "scored": 3, "errors": 2, "unscored": 0}


def test_tally_infrastructure_error_does_not_count_stale_reward(tmp_path):
    (tmp_path / "trial" / "verifier").mkdir(parents=True)
    (tmp_path / "trial" / "config.json").write_text("{}")
    (tmp_path / "trial" / "result.json").write_text(
        '{"exception_info": {"exception_type": "CollectError"}}'
    )
    (tmp_path / "trial" / "verifier" / "reward.txt").write_text("1")
    assert tally(tmp_path) == {"solved": 0, "scored": 0, "errors": 1, "unscored": 0}


def test_preflight_rejects_old_harbor(tmp_path, monkeypatch):
    (tmp_path / "task.toml").write_text('[verifier]\nenvironment_mode="separate"\n')
    monkeypatch.setenv("DATASET_PATH", str(tmp_path))
    monkeypatch.setattr(Trial, "SUPPORTS_SEPARATE_VERIFIER", False)
    with pytest.raises(ValueError, match="patches/harbor.patch"):
        preflight()


@pytest.mark.parametrize("task_names", [None, ["example-task", "another-*"]])
def test_deepswe_job_preserves_task_budgets(tmp_path, task_names):
    target = tmp_path / "job.json"
    env = {
        **os.environ,
        "REPO_ROOT": str(ROOT),
        "TEMPLATE_ROOT": str(ROOT / "scripts/templates"),
        "CONFIG_PATH": str(ROOT / "scripts/eval/configs/qwen38-27b_deepswe.env"),
        "PYTHON_BIN": sys.executable,
        "TEST_JOB_CFG": str(target),
    }
    if task_names is not None:
        env["EVAL_TASK_NAMES"] = json.dumps(task_names)
    script = """
set -euo pipefail
source scripts/eval/lib/runtime.sh
source scripts/eval/lib/job_config.sh
source_config_and_templates
JOB_CFG="$TEST_JOB_CFG"
write_job_config
"""
    proc = subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    config = json.loads(target.read_text())
    assert config["environment"]["override_cpus"] is None
    assert config["environment"]["override_memory_mb"] is None
    assert "override_timeout_sec" not in config["agents"][0]
    assert "max_timeout_sec" not in config["agents"][0]
    assert config["agent_timeout_multiplier"] == 1
    assert config["verifier_timeout_multiplier"] == 1
    assert config["metrics"] == [{"type": "mean", "kwargs": {"reward_key": "reward"}}]
    assert config["verifier"]["allowed_rewards"] == {"reward": [0, 1]}
    assert config["agents"][0]["env"]["HARBOR_EVAL_CC_SUBMISSION_GUARD"] == "1"
    if task_names is not None:
        assert config["datasets"][0]["task_names"] == task_names
    else:
        assert "task_names" not in config["datasets"][0]


def test_k8s_preserves_official_image_and_git(tmp_path, monkeypatch):
    env = KubernetesEnvironment.__new__(KubernetesEnvironment)
    env.environment_dir = tmp_path
    env._skip_dockerfile_run = True
    env.exec = AsyncMock()
    (tmp_path / "Dockerfile").write_text(
        "FROM official:v1.1\nRUN git clone example /app\n"
    )
    monkeypatch.setenv("HARBOR_BLOCK_GIT_HISTORY_LEAK", "0")
    asyncio.run(env._apply_dockerfile_run_commands())
    asyncio.run(env._rebuild_git_single_commit())
    asyncio.run(env.restore_git_history())
    env.exec.assert_not_awaited()


def test_k8s_strict_network_errors_propagate(monkeypatch):
    env = KubernetesEnvironment.__new__(KubernetesEnvironment)
    env._NETADMIN_IMAGE = "test:iptables"
    env.pod_name = "test"
    env.logger = logging.getLogger()
    env.exec = AsyncMock(
        return_value=ExecResult(return_code=1, stderr="iptables failed")
    )
    monkeypatch.setenv("HARBOR_K8S_EGRESS_ISOLATION", "1")
    monkeypatch.setenv("HARBOR_K8S_STRICT_NETWORK_POLICY", "true")
    with pytest.raises(RuntimeError, match="iptables failed"):
        asyncio.run(
            env.apply_network_policy(NetworkPolicy(network_mode=NetworkMode.NO_NETWORK))
        )


def test_tally_root_errors_and_custom_score_domain(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({
        "n_total_trials": 3, "finished_at": None,
        "stats": {"evals": {"agent": {"exception_stats": {"ExitError": ["retrying"]}}}},
    }))
    for name in ["success", "retrying", "pending"]:
        trial = tmp_path / name
        (trial / "verifier").mkdir(parents=True)
        (trial / "config.json").write_text("{}")
    (tmp_path / "success/result.json").write_text(json.dumps({
        "verifier_result": {"rewards": {"score": 2, "reward": 0}},
    }))
    # An errored retry must not inherit a stale score file.
    (tmp_path / "retrying/verifier/reward.json").write_text('{"score": 2}')
    assert tally(tmp_path, "score", [0, 1, 2]) == {
        "solved": 1, "scored": 1, "errors": 1, "unscored": 1,
    }
    assert job_status(tmp_path) == {"finished": False, "total": 3, "completed": 1, "pending": 2}
    # A later successful final result supersedes a stale aggregate error.
    (tmp_path / "retrying/result.json").write_text('{"verifier_result":{"rewards":{"score":2}}}')
    assert tally(tmp_path, "score", [0, 1, 2])["errors"] == 0


def test_submission_diagnostics_distinguishes_missing_ids(tmp_path):
    trial = tmp_path / "trial"
    (trial / "verifier").mkdir(parents=True)
    (trial / "artifacts").mkdir()
    (trial / "config.json").write_text("{}")
    (trial / "artifacts/model.patch").write_text("")
    (trial / "verifier/ctrf.json").write_text(json.dumps({"results": {"tests": [
        {"name": "renamed", "status": "failed", "message": "missing from report"},
        {"name": "broken", "status": "failed", "message": "assert 1 == 2"},
    ]}}))
    assert submission_diagnostics(tmp_path) == {
        "missing_test_ids": {"trial": ["renamed"]},
        "other_failed_tests": {"trial": ["broken"]}, "empty_patches": ["trial"],
        "verifier_signals": {},
    }


def test_preflight_rejects_unmatched_task_filter(tmp_path, monkeypatch):
    task = tmp_path / "actual"
    task.mkdir()
    (task / "task.toml").write_text("")
    monkeypatch.setenv("DATASET_PATH", str(tmp_path))
    monkeypatch.setenv("EVAL_TASK_NAMES", '["typo"]')
    monkeypatch.delenv("EVAL_REWARD_KEY", raising=False)
    monkeypatch.delenv("EVAL_REWARD_VALUES", raising=False)
    with pytest.raises(ValueError, match="matched no task"):
        preflight()


@pytest.mark.parametrize("enabled", [False, True])
def test_cc_submission_guard_is_eval_only(tmp_path, monkeypatch, enabled):
    from harbor_patch.agents.image_mounted_claude_code.claude_code import ClaudeCode, _BaseCustomClaudeCode

    agent = ClaudeCode.__new__(ClaudeCode)
    agent._get_env = lambda key: "1" if enabled else None
    agent._exec_as_agent_with_env_priority = AsyncMock(return_value=ExecResult(return_code=0, stderr=""))
    parent_run = AsyncMock()
    monkeypatch.setattr(_BaseCustomClaudeCode, "run", parent_run)
    environment, context = object(), object()
    asyncio.run(agent.run("original task", environment, context))
    instruction = parent_run.call_args.args[0]
    if not enabled:
        agent._exec_as_agent_with_env_priority.assert_not_awaited()
        assert instruction == "original task"
        return
    assert instruction == "original task"
    command = agent._exec_as_agent_with_env_priority.call_args.kwargs["command"]
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    command = command.replace("cd /app", f"cd {tmp_path}")
    git_env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"}
    subprocess.run(["sh", "-c", command], check=True, env=git_env)
    assert subprocess.check_output(["git", "-C", str(tmp_path), "var", "GIT_AUTHOR_IDENT"], env=git_env, text=True).startswith("Harbor Eval <eval@harbor.invalid>")
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Existing"], check=True)
    subprocess.run(["sh", "-c", command], check=True, env=git_env)
    assert subprocess.check_output(["git", "-C", str(tmp_path), "config", "user.name"], text=True).strip() == "Existing"

    agent._exec_as_agent_with_env_priority.return_value = ExecResult(return_code=1, stderr="read-only filesystem")
    parent_run.reset_mock()
    with pytest.raises(RuntimeError, match="Git identity setup failed"):
        asyncio.run(agent.run("original task", environment, context))
    parent_run.assert_not_awaited()


def test_tally_recovers_missing_retry_directory(tmp_path):
    (tmp_path / 'result.json').write_text(json.dumps({
        'n_total_trials': 2, 'finished_at': 'done',
        'stats': {'evals': {'cc': {'reward_stats': {'reward': {'1': ['gone'], '0': ['present']}}, 'exception_stats': {}}}},
    }))
    trial = tmp_path / 'present'
    trial.mkdir()
    (trial / 'config.json').write_text('{}')
    (trial / 'result.json').write_text('{"verifier_result":{"rewards":{"reward":0}}}')
    assert tally(tmp_path) == {'solved': 1, 'scored': 2, 'errors': 0, 'unscored': 0}
    assert job_status(tmp_path)['pending'] == 0


def test_eval_duplicate_launch_lock(tmp_path):
    library = ROOT / 'scripts/eval/lib/vllm.sh'
    holder = subprocess.Popen(['bash', '-c', 'source "$1"; VLLM_PORT=19999; acquire_eval_lock; echo ready; read -r _', 'bash', str(library)],
                              env={**os.environ, 'TMPDIR': str(tmp_path)}, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == 'ready'
        second = subprocess.run(['bash', '-c', 'source "$1"; VLLM_PORT=19999; acquire_eval_lock', 'bash', str(library)],
                                env={**os.environ, 'TMPDIR': str(tmp_path)}, capture_output=True, text=True)
        assert second.returncode == 1
        assert 'another eval owns' in second.stderr
    finally:
        holder.communicate('\n', timeout=5)
    assert subprocess.run(['bash', '-c', 'source "$1"; VLLM_PORT=19999; acquire_eval_lock', 'bash', str(library)],
                          env={**os.environ, 'TMPDIR': str(tmp_path)}).returncode == 0


def test_eval_cleanup_preserves_other_server(tmp_path):
    library = ROOT / 'scripts/eval/lib/vllm.sh'
    unrelated = subprocess.Popen(['bash', '-c', 'exec -a "vllm serve unrelated" sleep 30'])
    try:
        result = subprocess.run(['bash', '-c', 'source "$1"; sleep 30 & VLLM_PID=$!; kill_vllm_and_wait; test -z "$VLLM_PID"', 'bash', str(library)], capture_output=True, text=True, timeout=5)
        assert result.returncode == 0
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_diagnostics_reports_storage_separately_from_build_failure(tmp_path):
    trial = tmp_path / 'trial'
    (trial / 'verifier').mkdir(parents=True)
    (trial / 'config.json').write_text('{}')
    raw = trial / 'verifier/test-stdout.txt'
    raw.write_text('FAIL encryption [build failed]\n')
    assert submission_diagnostics(tmp_path)['verifier_signals'] == {'trial': ['build_failed_cause_unresolved']}
    raw.write_text('mkdir: no space left on device\nFAIL encryption [build failed]\n')
    assert submission_diagnostics(tmp_path)['verifier_signals'] == {'trial': ['storage_exhausted', 'build_failed_cause_unresolved']}
