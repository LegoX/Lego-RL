import shlex
import tomllib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("harbor", reason="Codex runtime tests require the Harbor dependency")
from harbor_patch.agents.image_mounted_codex.codex import Codex, _BaseCodex


def agent():
    obj = object.__new__(Codex)
    obj._get_env = lambda key: {'OPENAI_BASE_URL': 'http://proxy/sess/trial/v1'}.get(key)
    obj.logger = Mock()
    obj._version = '0.153.4'
    return obj


@pytest.mark.asyncio
async def test_mounted_runtime_version_and_provider_configuration():
    obj = agent()
    obj._detect_mounted_runtime = AsyncMock(return_value='/runtime/bin/codex')
    obj.exec_as_root = AsyncMock()
    env = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(stdout='codex-cli 0.153.4\n')))
    await obj.install(env)
    commands = [call.kwargs['command'] for call in env.exec.await_args_list]
    cmd = next(c for c in commands if 'config.toml' in c)
    tokens = shlex.split(cmd)
    config = tomllib.loads(tokens[tokens.index('echo') + 1])
    provider = config['model_providers'][config['model_provider']]
    assert provider['base_url'] == 'http://proxy/sess/trial/v1'
    assert provider['wire_api'] == 'responses'
    assert provider['supports_websockets'] is False


@pytest.mark.asyncio
async def test_version_mismatch_stops_before_running_agent():
    obj = agent()
    obj._detect_mounted_runtime = AsyncMock(return_value='/runtime/bin/codex')
    obj.exec_as_root = AsyncMock()
    env = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(stdout='codex-cli 0.1.0\n')))
    with pytest.raises(RuntimeError, match='version mismatch'):
        await obj.install(env)


@pytest.mark.asyncio
async def test_installer_fallback_also_configures_http_provider(monkeypatch):
    obj = agent()
    obj._detect_mounted_runtime = AsyncMock(return_value=None)
    obj._write_provider_config = AsyncMock()
    install = AsyncMock()
    monkeypatch.setattr(_BaseCodex, 'install', install)
    env = SimpleNamespace()
    await obj.install(env)
    install.assert_awaited_once_with(env)
    obj._write_provider_config.assert_awaited_once_with(env)


def test_mcp_registration_preserves_the_provider(monkeypatch):
    obj = agent()
    monkeypatch.setattr(_BaseCodex, '_build_register_mcp_servers_command',
                        lambda self: 'echo "[mcp_servers.test]" > "$CODEX_HOME/config.toml"')
    assert obj._build_register_mcp_servers_command().endswith(' >> "$CODEX_HOME/config.toml"')
