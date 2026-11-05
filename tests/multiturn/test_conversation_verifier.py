import base64
import io
import subprocess
import tarfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from harbor_patch.multiturn.conversation_verifier import SnapshotVerifier, parse_verdict, verify_round_snapshot


@pytest.mark.parametrize('code,read,text,guard', [(124,0,'0',0),(0,1,'',0),(0,0,'nan',0),(0,0,'0',1)])
def test_invalid_verification_is_not_zero_reward(code,read,text,guard):
    with pytest.raises((RuntimeError,ValueError)):
        parse_verdict(code,read,text,guard)


def test_test_failure_is_valid_zero():
    assert parse_verdict(101,0,'0',0)=={'reward':0.,'exit_code':101}


@pytest.mark.asyncio
@pytest.mark.parametrize('fail_restore', [False,True])
async def test_snapshot_keeps_new_files_and_hidden_tests_stay_in_clone(tmp_path,monkeypatch,fail_restore):
    workspace=tmp_path/'workspace';workspace.mkdir()
    (workspace/'new.py').write_text('new untracked content')
    run_dir=tmp_path/'run';run_dir.mkdir()
    policy_commands=[];clones=[];uploads=[];cleanups=[]
    def policy_exec(command,**kwargs):
        policy_commands.append(command)
        result=subprocess.run(command,shell=True,text=True,capture_output=True)
        return result.returncode,result.stdout
    episode=SimpleNamespace(summary={},run_dir=run_dir,name='policy-pod',row={'workdir':str(workspace)},
        execute=policy_exec,test_files=lambda _: {'tests/test.sh':(b'SECRET TEST',0o600)})
    async def create(self):clones.append(self);self.created.append('pod')
    def upload_snapshot(self,encoded):
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(encoded)),mode='r:gz') as tar:
            assert tar.extractfile('workspace/new.py').read()==b'new untracked content'
    def upload(self,files):uploads.append((self.name,files))
    def execute(self,command,**kwargs):
        if command.startswith('test -d '):return (1 if fail_restore else 0),''
        if command=='cat /logs/verifier/reward.txt':return 0,'1'
        if 't.add(' in command:return 1,'' # no artifacts in this fake pod
        return 0,''
    async def cleanup(self):cleanups.append(self.name)
    monkeypatch.setattr(SnapshotVerifier,'create_sandbox',create)
    monkeypatch.setattr(SnapshotVerifier,'upload_snapshot',upload_snapshot)
    monkeypatch.setattr(SnapshotVerifier,'upload',upload)
    monkeypatch.setattr(SnapshotVerifier,'execute',execute)
    monkeypatch.setattr(SnapshotVerifier,'cleanup',cleanup)
    if fail_restore:
        with pytest.raises(RuntimeError,match='restore'):await verify_round_snapshot(episode,1)
        assert not uploads
    else:
        result = await verify_round_snapshot(episode,1)
        assert result['reward'] == 1. and result['exit_code'] == 0
        assert result['test_progress']['available'] is False
        assert uploads[0][0] != episode.name
        assert 'tests/test.sh' in uploads[0][1]
    assert len(policy_commands)==1 and '/tests/' not in policy_commands[0]
    assert cleanups==[clones[0].name]
    assert (workspace/'new.py').read_text()=='new untracked content'

@pytest.mark.parametrize('code,output',[
 (139,'/tests/canonical_test.sh: line 181: 203 Segmentation fault (core dumped) /tmp/test_stratum'),
 (127,"undefined reference to `websock_connect_proto'\ncollect2: error: ld returned 1 exit status\n/tests/canonical_test.sh: line 13: ./build/test/retest: No such file or directory")])
def test_historical_program_failures_are_valid_zero(code,output):
    assert parse_verdict(code,0,'0',0,output)['reward']==0
    with pytest.raises(RuntimeError):parse_verdict(code,0,'1',0,output)

@pytest.mark.parametrize('code',[124,126,127,137,139,143])
def test_unproven_launch_timeout_and_kill_are_not_zero(code):
    with pytest.raises(RuntimeError):parse_verdict(code,0,'0',0,'')

@pytest.mark.asyncio
async def test_transport_retry_uses_new_snapshot_attempt(monkeypatch):
    from harbor_patch.multiturn import conversation_verifier as module
    from websocket import WebSocketConnectionClosedException
    attempts=[]
    async def verify(e,r,a):
        attempts.append(a)
        if a==0:raise WebSocketConnectionClosedException('lost')
        return {'reward':0.,'exit_code':1}
    monkeypatch.setattr(module,'_verify_round_snapshot_once',verify)
    monkeypatch.setattr(module.asyncio,'sleep',AsyncMock())
    e=SimpleNamespace(summary={})
    assert (await module.verify_round_snapshot(e,1))['reward']==0
    assert attempts==[0,1]

@pytest.mark.asyncio
async def test_real_test_failure_is_never_resampled(monkeypatch):
    from harbor_patch.multiturn import conversation_verifier as module
    call=AsyncMock(return_value={'reward':0.,'exit_code':1})
    monkeypatch.setattr(module,'_verify_round_snapshot_once',call)
    await module.verify_round_snapshot(SimpleNamespace(summary={}),1)
    call.assert_awaited_once()


def test_snapshot_upload_closes_api_after_handshake_failure(monkeypatch):
    from harbor_patch.multiturn import conversation_verifier as module
    from unittest.mock import Mock
    from websocket import WebSocketConnectionClosedException
    e=object.__new__(SnapshotVerifier);e.name='test';e.namespace='default'
    api=Mock();e.api=lambda:api
    monkeypatch.setattr(module,'stream',Mock(side_effect=WebSocketConnectionClosedException('lost')))
    with pytest.raises(WebSocketConnectionClosedException):e.upload_snapshot('AAAA')
    api.api_client.close.assert_called_once()
