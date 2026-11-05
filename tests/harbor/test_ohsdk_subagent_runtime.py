"""Mounted SDK interpreters must execute the patched, repository-owned runner."""

from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("harbor")
from harbor.agents.custom.openhands_sdk import CustomOpenHandsSDK


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_mounted_runtime_gets_overlay_only_when_subagents_enabled(
    tmp_path, enabled
):
    agent = CustomOpenHandsSDK.__new__(CustomOpenHandsSDK)
    agent.logs_dir = tmp_path
    agent._enable_subagent = enabled
    agent._detect_mounted_runtime = AsyncMock(return_value="/runtime/bin/python")
    runner = tmp_path / "repository_runtime.py"
    runner.write_text('print("patched runtime")\n')
    agent._repo_runtime_entrypoint_path = lambda: runner
    env = NS(upload_file=AsyncMock(), exec=AsyncMock())
    await agent.install(env)
    calls = env.upload_file.await_args_list
    assert any(c.kwargs["target_path"] == str(agent._LOADER_ENTRYPOINT) for c in calls)
    overlays = [
        c
        for c in calls
        if c.kwargs["target_path"] == str(agent._LEGACY_RUNTIME_ENTRYPOINT)
    ]
    assert bool(overlays) == enabled
    if enabled:
        assert Path(overlays[0].kwargs["source_path"]).read_text() == runner.read_text()
    assert agent._resolved_python_path == "/runtime/bin/python"
