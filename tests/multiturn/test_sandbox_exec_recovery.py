from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock

import pytest
from websocket import WebSocketConnectionClosedException
from harbor_patch.multiturn import native_swe as native


class Stream:
    def __init__(self, code=0, error=None):
        self.code, self.error, self.closed = code, error, False

    def is_open(self):
        return False

    @property
    def returncode(self):
        if self.error:
            raise self.error
        return self.code

    def close(self):
        self.closed = True


def episode():
    e = object.__new__(native.NativeEpisode)
    e.name, e.namespace, e.summary = 'test-pod', 'default', {}
    api = SimpleNamespace(connect_get_namespaced_pod_exec=Mock(), api_client=SimpleNamespace(close=Mock()))
    e.api = Mock(return_value=api)
    return e, api


@pytest.mark.parametrize('code', [0, 1, 2, 127, 139])
def test_real_exit_codes_are_not_retried_or_reclassified(monkeypatch, code):
    e, api = episode()
    ws = Stream(code)
    call = Mock(return_value=ws)
    monkeypatch.setattr(native, 'stream', call)
    assert e.execute('command', retry_read=True) == (code, '')
    call.assert_called_once()
    assert ws.closed
    api.api_client.close.assert_called_once()


@pytest.mark.parametrize('ws', [Stream(None), Stream(error=ValueError('OCI runtime exec failed'))])
def test_missing_or_runtime_exit_status_is_transport_failure(monkeypatch, ws):
    e, api = episode()
    monkeypatch.setattr(native, 'stream', Mock(return_value=ws))
    with pytest.raises(native.SandboxExecTransportError) as caught:
        e.execute('command')
    assert native._native_failure_category(caught.value) == ('sandbox_transport', True)
    assert ws.closed
    api.api_client.close.assert_called_once()


def test_connect_failure_also_closes_client(monkeypatch):
    e, api = episode()
    call = Mock(side_effect=WebSocketConnectionClosedException('Connection to remote host was lost.'))
    monkeypatch.setattr(native, 'stream', call)
    with pytest.raises(WebSocketConnectionClosedException) as caught:
        e.execute('run-agent')
    assert native._native_failure_category(caught.value) == ('sandbox_transport', True)
    call.assert_called_once()  # Unknown execution state: never replay the agent.
    api.api_client.close.assert_called_once()


def test_readonly_retry_recovers_with_fresh_connections(monkeypatch):
    e, api = episode()
    call = Mock(side_effect=[WebSocketConnectionClosedException('lost'), Stream(0)])
    monkeypatch.setattr(native, 'stream', call)
    monkeypatch.setattr(native.time, 'sleep', lambda _: None)
    assert e.execute('cat /logs/verifier/reward.txt', retry_read=True) == (0, '')
    assert call.call_count == api.api_client.close.call_count == 2
    assert e.summary['exec_read_retries'] == 1


def test_readonly_retry_is_bounded(monkeypatch):
    e, api = episode()
    call = Mock(side_effect=WebSocketConnectionClosedException('lost'))
    monkeypatch.setattr(native, 'stream', call)
    monkeypatch.setattr(native.time, 'sleep', lambda _: None)
    with pytest.raises(WebSocketConnectionClosedException):
        e.execute('cat /logs/verifier/reward.txt', retry_read=True)
    assert call.call_count == api.api_client.close.call_count == 3


def test_test_binary_failure_is_not_transient_infrastructure():
    assert native._native_failure_category(RuntimeError('Final verifier failed to execute (127, 0)')) == ('verifier_launch_error', False)


def test_diagnostics_capture_oom_without_pod_env():
    e, api = episode()
    terminated = SimpleNamespace(reason='OOMKilled', exit_code=137, signal=9)
    status = SimpleNamespace(name='main', ready=False, restart_count=0,
        state=SimpleNamespace(terminated=terminated, waiting=None), last_state=None)
    api.read_namespaced_pod = Mock(return_value=SimpleNamespace(
        spec=SimpleNamespace(node_name='cpu-node', containers=['DO NOT SERIALIZE ENV']),
        status=SimpleNamespace(phase='Failed', reason=None, container_statuses=[status])))
    result = e.sandbox_diagnostics()
    assert result['containers'][0]['state']['reason'] == 'OOMKilled'
    assert result['node'] == 'cpu-node'
    assert 'DO NOT SERIALIZE ENV' not in str(result)
    api.api_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_failure_diagnostics_are_written_before_cleanup(tmp_path):
    e, _ = episode()
    e.run_dir, e.created = tmp_path, []
    e.gateway = AsyncMock()
    e.create_sandbox = AsyncMock()
    e.install_runtime = AsyncMock()
    e.execute_episode = AsyncMock(side_effect=WebSocketConnectionClosedException('lost'))
    e.sandbox_diagnostics = Mock(return_value={'phase': 'Failed', 'reason': 'Evicted'})
    with pytest.raises(WebSocketConnectionClosedException):
        await e.run()
    import json
    saved = json.loads((tmp_path / 'multiturn.json').read_text())
    assert saved['retryable'] is True
    assert saved['retry_category'] == 'sandbox_transport'
    assert saved['sandbox_diagnostics']['reason'] == 'Evicted'


def test_rejected_handshake_retries_without_reexecuting_command(monkeypatch):
    e,api=episode()
    e.sandbox_diagnostics=lambda:{'phase':'Running','containers':[{'name':'main','ready':True}]}
    call=Mock(side_effect=[native.client.ApiException(status=500,reason='container not found (main)'),Stream(0)])
    monkeypatch.setattr(native,'stream',call)
    monkeypatch.setattr(native.time,'sleep',lambda _:None)
    assert e.execute('run-agent')==(0,'')
    assert call.call_count==2


def test_oom_container_is_not_retried_in_place(monkeypatch):
    e,api=episode()
    e.sandbox_diagnostics=lambda:{'phase':'Failed','containers':[{'name':'main','state':{'reason':'OOMKilled','exit_code':137}}]}
    call=Mock(side_effect=native.client.ApiException(status=500,reason='container not found (main)'))
    monkeypatch.setattr(native,'stream',call)
    with pytest.raises(native.client.ApiException):e.execute('run-agent')
    call.assert_called_once()
