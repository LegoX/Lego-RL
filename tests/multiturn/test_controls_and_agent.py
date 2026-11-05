import json
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from harbor_patch.multiturn.benchmark import MultiTurnBehaviorConfig
from harbor_patch.multiturn.claude_code import (
    ClaudeCode,
    MultiTurnAgentError,
    MultiTurnClaudeCode,
)
from harbor_patch.multiturn.config import MultiTurnConfig
from harbor_patch.multiturn.user_simulator import UserSimulator, UserSimulatorInput
from pydantic import ValidationError


def test_config_disabled_by_default_and_rejects_invalid_enabled_config():
    assert not MultiTurnConfig().enabled
    assert MultiTurnConfig(enabled=True, max_rounds=1).max_rounds == 1
    for values in (
        {"enabled": True},
        {"max_rounds": 0},
        {"enable": True},
        {"simulator_mode": "invalid"},
    ):
        with pytest.raises(ValidationError):
            MultiTurnConfig.model_validate(values)


def test_shell_controls_are_optional_root_hydra_overrides():
    script = (
        'source scripts/train/lib/hydra_args.sh; init_hydra_args; append_multiturn_hydra_args; '
        'printf "%s\\n" "${hydra_args[@]}"'
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("HARBOR_MULTITURN_")}
    assert (
        subprocess.check_output(["bash", "-c", script], env=env, text=True).strip()
        == ""
    )
    env.update(
        HARBOR_MULTITURN_ENABLED="true",
        HARBOR_MULTITURN_TASK_PROFILE="openswe",
        HARBOR_MULTITURN_WORKSPACE_DIR="/testbed",
        HARBOR_MULTITURN_MAX_ROUNDS="8",
        HARBOR_MULTITURN_SIMULATOR_MODEL="openai/user-model",
    )
    output = subprocess.check_output(["bash", "-c", script], env=env, text=True)
    assert "++multiturn.enabled=true" in output
    assert "++multiturn.task_profile=openswe" in output
    assert "++multiturn.workspace_dir=/testbed" in output
    assert "++multiturn.max_rounds=8" in output
    assert "++multiturn.simulator_model=openai/user-model" in output


@pytest.mark.asyncio
async def test_native_cc_uses_one_session_and_does_not_duplicate_runtime_setup(
    tmp_path, monkeypatch
):
    agent = object.__new__(MultiTurnClaudeCode)
    agent._resolved_runtime_claude_path = "/runtime/bin/claude"
    commands = []

    async def parent_run(self, instruction, environment, context):
        commands.append(self._current_claude_command())

    async def download(source, target):
        assert source == "/logs/agent/claude-code.txt"
        target.write_text(
            json.dumps(
                {
                    "type": "result",
                    "session_id": agent._multiturn_session_id,
                    "is_error": False,
                    "result": "done",
                }
            )
        )

    monkeypatch.setattr(ClaudeCode, "run", parent_run)
    environment = SimpleNamespace(download_file=download)
    ids = []
    for n in range(1, 4):
        result = await agent.run_user_round(
            "feedback", environment, None, n, tmp_path / f"{n}.json"
        )
        ids.append(result["session_id"])
    assert len(set(ids)) == 1
    assert commands[0] == f"/runtime/bin/claude --session-id {ids[0]}"
    assert commands[1:] == [f"/runtime/bin/claude --resume {ids[0]}"] * 2


@pytest.mark.parametrize(
    "record",
    [
        {"type": "result", "session_id": "different", "is_error": False},
        {"type": "result", "session_id": "expected", "is_error": True},
        {"type": "system", "session_id": "expected"},
    ],
)
def test_native_cc_rejects_restart_or_missing_completion(tmp_path, record):
    agent = object.__new__(MultiTurnClaudeCode)
    agent._multiturn_session_id = "expected"
    log = tmp_path / "cc.jsonl"
    log.write_text(json.dumps(record))
    with pytest.raises(MultiTurnAgentError):
        agent._validate_round_log(log)


@pytest.mark.asyncio
async def test_simulator_keeps_benchmark_json_repair_and_language(tmp_path):
    simulator = object.__new__(UserSimulator)
    simulator._call_kwargs = {}
    simulator._router = None
    simulator._llm = SimpleNamespace(
        call=AsyncMock(
            side_effect=[
                SimpleNamespace(content="not json"),
                SimpleNamespace(
                    content=json.dumps(
                        {
                            "action": "question",
                            "instruction": "请再检查边界条件",
                            "intent": "边界检查",
                            "expected_agent_behavior": "检查代码",
                        }
                    )
                ),
            ]
        )
    )
    payload = UserSimulatorInput(
        task_name="task",
        initial_instruction="请检查并修复这个程序中的错误。",
        round_id=3,
        behavior=MultiTurnBehaviorConfig(id="debug", type="debug"),
        previous_rounds=[],
        context_summary={},
    )
    output = await simulator.generate(payload, tmp_path / "sim.json")
    assert output.instruction == "请再检查边界条件"
    assert simulator._llm.call.await_count == 2
    assert (
        "Detected response_language: Chinese"
        in simulator._llm.call.call_args_list[0].args[0]
    )


def test_simulator_credentials_are_client_scoped(monkeypatch):
    from unittest.mock import Mock

    from harbor_patch.multiturn import user_simulator

    client = Mock()
    monkeypatch.setattr(user_simulator, "LiteLLM", client)
    monkeypatch.setenv("OPENAI_API_KEY", "policy-key")
    UserSimulator(model="openai/user-model", api_key="user-simulator-key")
    assert os.environ["OPENAI_API_KEY"] == "policy-key"
    assert client.call_args.kwargs["api_key"] == "user-simulator-key"


def test_nested_agent_events_do_not_look_like_a_native_session_restart(tmp_path):
    agent = object.__new__(MultiTurnClaudeCode)
    agent._multiturn_session_id = "main"
    path = tmp_path / "cc.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {
                    "type": "assistant",
                    "session_id": "child",
                    "parent_tool_use_id": "task-1",
                },
                {
                    "type": "result",
                    "session_id": "main",
                    "is_error": False,
                    "result": "done",
                },
            ]
        )
    )
    assert agent._validate_round_log(path)["session_id"] == "main"


@pytest.mark.asyncio
async def test_real_litellm_client_uses_simulator_endpoint_and_credentials(monkeypatch):
    import asyncio

    from aiohttp import web

    payload = {
        "action": "message",
        "instruction": "Please check the edge case.",
        "intent": "debug",
        "expected_agent_behavior": "Inspect the code",
    }
    received = []

    async def complete(request):
        body = await request.json()
        received.append((request.headers.get("Authorization"), body))
        return web.json_response(
            {
                "id": "simulator-test",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(payload),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 20,
                    "total_tokens": 40,
                },
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    monkeypatch.setenv("OPENAI_API_KEY", "policy-key-not-for-the-simulator")
    try:
        simulator = UserSimulator(
            model="openai/gpt-4o-mini",
            api_base=f"http://127.0.0.1:{port}/v1",
            api_key="simulator-only",
        )
        inputs = UserSimulatorInput(
            task_name="test",
            initial_instruction="Fix a bug in the program.",
            round_id=2,
            behavior=MultiTurnBehaviorConfig(id="debug", type="debug"),
            previous_rounds=[],
            context_summary={},
        )
        output = await asyncio.wait_for(simulator.generate(inputs), timeout=15)
        assert output.instruction == payload["instruction"]
        assert received[0][0] == "Bearer simulator-only"
        assert received[0][1]["messages"]
        assert os.environ["OPENAI_API_KEY"] == "policy-key-not-for-the-simulator"
    finally:
        await runner.cleanup()


def test_profile_workspace_defaults_and_unsafe_snapshot_roots():
    from harbor_patch.multiturn.config import MultiTurnConfig
    assert MultiTurnConfig(task_profile="openswe").resolved_workspace == "/testbed"
    assert MultiTurnConfig().resolved_workspace == "/app"
    assert MultiTurnConfig(task_profile="openswe", workspace_dir="/repo").resolved_workspace == "/repo"
    for path in ("/", "/testbed/..", "relative", "/repo/"):
        with pytest.raises(ValueError):
            MultiTurnConfig(workspace_dir=path)
