import asyncio
import json
import urllib.error
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from harbor_patch.multiturn.config import MultiTurnConfig
from harbor_patch.multiturn.swe_conversation import ConversationEpisode, ConversationSimulator, task_context
from harbor_patch.multiturn.swe_conversation_driver import drive, exchange_conversation
from harbor_patch.multiturn.user_simulator import UserSimulator, UserSimulatorOutput


def episode(tmp_path, **options):
    e = object.__new__(ConversationEpisode)
    e.context = {'initial_request': 'Fix the exit code', 'task_instruction': 'Return the actual child exit code.'}
    e.pending_instruction = e.context['initial_request']
    e.turns, e.summary, e.ended, e.simulator = [], {}, False, None
    e._advance_lock, e._advance_replies = asyncio.Lock(), {}
    e.run_dir, e.session_id, e.gateway_port = tmp_path, 's', 1234
    e.host, e.token, e.key = '10.0.0.1', 'unused', 'unused'
    e.row = {'task_id': 'example', 'workdir': '/app/src'}
    e.options = MultiTurnConfig(enabled=True, task_profile='swe_interact',
        simulator_mode='scripted', scripted_followups=['Please also check a failing command.'], **options)
    e.sampling_params = {'temperature': 0.7, 'top_p': 0.9}
    e.proxy = SimpleNamespace(begin_user_round=AsyncMock(), begin_tool_handoff=AsyncMock(),
        session_anthropic_base=lambda _: 'http://10.0.0.1:1235', _sessions={'s': {}})
    return e


def test_only_original_task_prose_is_available_to_simulator(tmp_path):
    root = tmp_path / 'environment/user-server'
    root.mkdir(parents=True)
    (root / 'review_contract.json').write_text(json.dumps({
        'initial_request': 'fix it', 'task_instruction': 'original issue',
        'verification_context': 'HIDDEN_TEST', 'atomic_goals': [{'requirement': 'HIDDEN_ASSERTION'}],
        'verifier_owned_paths': ['HIDDEN_PATH'], 'disclosure_order': ['secret'],
    }))
    assert task_context(tmp_path) == {'initial_request': 'fix it', 'task_instruction': 'original issue'}


def test_conversation_504_retries_identical_payload(monkeypatch):
    from harbor_patch.multiturn import swe_conversation_driver as driver
    payloads = []
    def urlopen(request, timeout):
        payloads.append(request.data)
        if len(payloads) < 3:
            raise urllib.error.HTTPError(request.full_url, 504, 'Gateway Timeout', {}, None)
        return _JsonResponse({'action': 'stop', 'reason': 'user_done'})
    monkeypatch.setattr(driver.urllib.request, 'urlopen', urlopen)
    monkeypatch.setattr(driver.time, 'sleep', lambda _: None)
    assert exchange_conversation('http://example.test/turn', {'round': 1}, 600)['action'] == 'stop'
    assert len(payloads) == 3 and len(set(payloads)) == 1


class _JsonResponse:
    def __init__(self, value):
        self.value = value
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass
    def read(self, *_):
        return json.dumps(self.value).encode()


@pytest.mark.asyncio
async def test_conversation_replay_is_idempotent(tmp_path):
    e = episode(tmp_path)
    first, replay = await asyncio.gather(e.advance_once({'round': 0}), e.advance_once({'round': 0}))
    assert first == replay == {'action': 'reply', 'message': 'Fix the exit code', 'round': 1}
    e.proxy.begin_user_round.assert_awaited_once()
    with pytest.raises(Exception) as error:
        await e.advance_once({'round': 0, 'different': True})
    assert error.value.status == 409


@pytest.mark.asyncio
async def test_simulator_stream_requires_complete_response(monkeypatch, tmp_path):
    from harbor_patch.multiturn import user_simulator
    calls = []
    def chunk(content, finish_reason=None):
        return SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content=content), finish_reason=finish_reason)])
    class Stream:
        def __init__(self, complete):
            self.complete = complete
        def __aiter__(self):
            async def iterator():
                yield chunk('{"action":"no-op",')
                yield chunk('"instruction":"","intent":"done","expected_agent_behavior":"none"}',
                            'stop' if self.complete else None)
            return iterator()
    class Router:
        async def acompletion(self, **kwargs):
            calls.append(kwargs)
            return Stream(len(calls) == 1)
    monkeypatch.setattr(user_simulator, '_simulator_router', lambda *_: Router())
    simulator = UserSimulator(model='openai/test', api_bases=('http://sim.example/v1',))
    content = await simulator._call('prompt', tmp_path / 'stream.json')
    assert json.loads(content)['action'] == 'no-op'
    assert calls[0]['stream'] is True
    assert json.loads((tmp_path / 'stream.json').read_text())['first_chunk_sec'] >= 0
    with pytest.raises(ValueError, match='complete content'):
        await simulator._call('prompt', None)


@pytest.mark.asyncio
async def test_natural_boundaries_preserve_tools_and_sampling(tmp_path):
    e = episode(tmp_path)
    messages, tools = [{'role': 'user', 'content': 'fix'}], [{'type': 'function'}]
    for _ in range(40):
        returned, functions, body = await e.prepare(messages, tools, {'max_tokens': 9000})
        assert returned is messages and functions is tools
        assert body == {'max_tokens': 4096, 'temperature': 0.7, 'top_p': 0.9}
    e.proxy.begin_tool_handoff.assert_not_awaited()


@pytest.mark.asyncio
async def test_round_history_and_noop_stop_without_approval(tmp_path):
    e = episode(tmp_path)
    assert (await e.advance({'round': 0}))['round'] == 1
    assert (await e.advance({'round': 1, 'assistant_message': 'Fixed success path.'}))['round'] == 2
    assert await e.advance({'round': 2, 'assistant_message': 'Fixed failure too.'}) == {'action': 'stop', 'reason': 'user_done'}
    assert len(json.loads((tmp_path / 'conversation.json').read_text())) == 2
    assert [c.args for c in e.proxy.begin_user_round.await_args_list] == [('s', 1), ('s', 2)]
    with pytest.raises(ValueError):
        await e.advance({'round': 3, 'assistant_message': 'late'})


@pytest.mark.asyncio
@pytest.mark.parametrize('budget,reason', [(False, 'max_rounds'), (True, 'max_iterations')])
async def test_limits_do_not_request_another_simulator_round(tmp_path, budget, reason):
    e = episode(tmp_path, max_rounds=1)
    e.next_instruction = AsyncMock(side_effect=AssertionError('Must not call simulator'))
    await e.advance({'round': 0})
    assert (await e.advance({'round': 1, 'assistant_message': 'progress', 'budget_exhausted': budget}))['reason'] == reason
    e.next_instruction.assert_not_awaited()


@pytest.mark.asyncio
async def test_simulator_gets_public_context_and_can_end_early(tmp_path):
    e = episode(tmp_path)
    e.options = e.options.model_copy(update={'simulator_mode': 'llm'})
    e.execute = lambda *a, **kw: (0, 'public diff')
    e.simulator = SimpleNamespace(generate=AsyncMock(return_value=UserSimulatorOutput(
        action='no-op', instruction='', intent='done', expected_agent_behavior='none')))
    await e.advance({'round': 0})
    assert (await e.advance({'round': 1, 'assistant_message': 'Implemented and tested.'}))['reason'] == 'user_done'
    payload = e.simulator.generate.call_args.args[0]
    assert payload.profile_details == {'task_requirements': e.context['task_instruction']}
    prompt = ConversationSimulator._build_prompt(payload)
    assert 'public diff' in prompt and 'Implemented and tested.' in prompt
    assert 'verification_context' not in prompt and 'atomic_goals' not in prompt


def execution(e, native, agent_rc=0, final_rc=0, reward='1', overflow=False):
    actions = []
    e.proxy._sessions['s']['context_overflow'] = overflow
    def execute(command, **kwargs):
        actions.append(command)
        if command.startswith('cd /app/src && env '): return agent_rc, ''
        if command == 'cat /logs/agent/interaction-result.json': return 0, json.dumps(native)
        if command == 'cat /logs/verifier/reward.txt': return 0, reward
        if command == 'timeout -s TERM 600 bash /tests/test.sh': return final_rc, ''
        return 0, ''
    e.execute = execute
    e.test_files = lambda step: {'step': step}
    e.upload = lambda files: actions.append(files)
    return actions


@pytest.mark.asyncio
@pytest.mark.parametrize('native,code,overflow', [
    ({'stop_reason': 'user_done'}, 0, False),
    ({'stop_reason': 'max_rounds'}, 0, False),
    ({'stop_reason': 'max_iterations'}, 0, False),
    ({'error': 'context limit'}, 1, True),
    ({'error': 'timeout'}, 124, False),
])
async def test_final_verifier_runs_without_approval_on_all_valid_stops(tmp_path, native, code, overflow):
    e = episode(tmp_path)
    actions = execution(e, native, agent_rc=code, overflow=overflow)
    reward, _ = await e.execute_episode()
    assert reward == 1 and e.summary['valid']
    assert {'step': '02_final_verify'} in actions
    assert {'step': '01_interactive_develop'} not in actions
    assert 'approval_reward' not in e.summary


@pytest.mark.asyncio
async def test_user_satisfaction_cannot_override_failed_test(tmp_path):
    e = episode(tmp_path)
    execution(e, {'stop_reason': 'user_done'}, final_rc=1, reward='0')
    assert await e.execute_episode() == (0., 'agent_completed')


@pytest.mark.asyncio
@pytest.mark.parametrize('runner_exit', [2, 22, 101])
async def test_test_runner_failure_is_valid_zero_reward(tmp_path, runner_exit):
    e = episode(tmp_path)
    execution(e, {'stop_reason': 'user_done'}, final_rc=runner_exit, reward='0')
    assert await e.execute_episode() == (0., 'agent_completed')
    assert e.summary['valid'] and e.summary['final_verifier_exit_code'] == runner_exit


@pytest.mark.asyncio
async def test_simulator_or_agent_failure_is_not_a_zero_reward(tmp_path):
    e = episode(tmp_path)
    actions = execution(e, {'error': 'HTTP 500 simulator'}, agent_rc=1)
    with pytest.raises(RuntimeError, match='Conversation execution failed'):
        await e.execute_episode()
    assert not any(isinstance(a, dict) for a in actions)
    assert not e.summary.get('valid')


@pytest.mark.asyncio
@pytest.mark.parametrize('rc,reward', [(124, '0'), (0, 'nan'), (0, '0.5')])
async def test_invalid_verifier_result_is_excluded(tmp_path, rc, reward):
    e = episode(tmp_path)
    execution(e, {'stop_reason': 'user_done'}, final_rc=rc, reward=reward)
    with pytest.raises((RuntimeError, ValueError)):
        await e.execute_episode()
    assert not e.summary.get('valid')


def test_driver_preserves_followups_and_global_budget():
    calls = []
    responses = iter([{'action': 'reply', 'round': 1, 'message': 'initial'},
                      {'action': 'reply', 'round': 2, 'message': 'clarification'},
                      {'action': 'stop', 'reason': 'max_iterations'}])
    def query(prompt, turn, remaining):
        calls.append((prompt, turn, remaining))
        return 'answer', 2
    sent = []
    def exchange(payload):
        sent.append(payload)
        return next(responses)
    assert drive(query, exchange, 4) == ('max_iterations', 2)
    assert calls == [('initial', 1, 4), ('clarification', 2, 2)]
    assert sent[-1]['budget_exhausted'] is True


def test_conversation_sandbox_excludes_review_server(tmp_path):
    e = episode(tmp_path)
    pod = {'spec': {'containers': [{'name': 'main'}, {'name': 'user-server'}]}}
    e.configure_sandbox(pod)
    assert pod['spec']['containers'] == [{'name': 'main'}]


def test_conversation_metrics_do_not_report_fictitious_approval():
    from verl_patch.multiturn.metrics import aggregate_episodes
    summary = dict(enabled=True, valid=True, task_profile='swe_interact', final_reward=1,
                   interaction_mode='conversation', approval_required=False, num_user_rounds=2)
    result = aggregate_episodes([summary])
    assert result['multiturn/swe_interact/final_success_rate'] == 1
    assert result['multiturn/swe_interact/feedback_count'] == 1
    assert not any('approval' in k for k in result)


@pytest.mark.asyncio
async def test_verifier_wrapper_setup_failure_is_excluded(tmp_path):
    e = episode(tmp_path)
    execution(e, {'stop_reason': 'user_done'}, reward='0')
    original = e.execute
    def execute(command, **kwargs):
        if command.startswith('python3 -c '): return 1, ''
        return original(command, **kwargs)
    e.execute = execute
    with pytest.raises(RuntimeError, match='verifier setup failed'):
        await e.execute_episode()
    assert not e.summary.get('valid')


@pytest.mark.asyncio
@pytest.mark.parametrize('mode,failed', [('conversation', False), ('conversation', True), ('native_review', False)])
async def test_mode_dispatch_and_infrastructure_loss_exclusion(tmp_path, monkeypatch, mode, failed):
    from harbor_patch.multiturn import native_swe, swe_conversation
    monkeypatch.setenv('HARBOR_SWE_INTERACTION_MODE', mode)
    monkeypatch.setenv('HARBOR_SWE_TRIAL_MAX_RETRIES', '0')
    monkeypatch.setenv('HARBOR_SWE_INTERACT_MANIFEST', 'unused')
    monkeypatch.setattr(native_swe, 'load_manifest', lambda _: {'example': {'task_id': 'example', 'split': 'train', 'task_path': '/example'}})
    chosen = []
    class Episode:
        def __init__(self, *args):
            chosen.append(mode)
            self.summary = {'enabled': True, 'valid': not failed}
        async def run(self):
            if failed: raise RuntimeError('simulator unavailable')
            return 1., 'agent_completed'
    target = swe_conversation if mode == 'conversation' else native_swe
    monkeypatch.setattr(target, 'ConversationEpisode' if mode == 'conversation' else 'NativeEpisode', Episode)
    sessions = {}
    async def open_session(session, **kwargs): sessions[session] = {}
    async def pop_session(session):
        assert 'native_round_controller' not in sessions[session]
        sessions.pop(session)
        return {}
    proxy = SimpleNamespace(open_session=open_session, pop_session=pop_session, _sessions=sessions)
    loop = SimpleNamespace(base_trials_dir=tmp_path, multiturn=SimpleNamespace(judge_enabled=False),
                           _assign_proxy_metrics_from_session=lambda *a: None)
    reward, reason, meta = await native_swe.run_native_trial(loop, proxy, '/example', 0, {}, {}, False)
    assert chosen == [mode] and not sessions
    if failed:
        assert meta['disable_proxy_trajectory'] is True and reason == 'env_setup_failed'
    else:
        assert reward == 1 and 'disable_proxy_trajectory' not in meta

@pytest.fixture(autouse=True)
def isolated_round_verifier(monkeypatch):
    from harbor_patch.multiturn import conversation_verifier
    mock = AsyncMock(return_value={'reward': 0., 'exit_code': 1})
    monkeypatch.setattr(conversation_verifier, 'verify_round_snapshot', mock)
    return mock


@pytest.mark.asyncio
async def test_round_verification_precedes_followup_and_preserves_full_history(tmp_path, isolated_round_verifier):
    e = episode(tmp_path)
    e.options = e.options.model_copy(update={'simulator_mode': 'llm', 'simulator_max_context_chars': 1000})
    text = 'BEGIN_EVIDENCE' + 'x' * 50000 + 'END_EVIDENCE'
    e.execute = lambda *a, **kw: (0, text)
    async def generate(payload, path):
        assert isolated_round_verifier.await_count == 1
        assert payload.context_summary['conversation'][0]['final_message'] == text
        assert payload.context_summary['workspace_diff'] == text
        assert payload.context_summary['round_verifier_feedback']['outcome'] == 'not_passed'
        assert payload.context_summary['round_verifier_feedback']['history'][0]['functional_reward'] == 0
        return UserSimulatorOutput(action='no-op', instruction='', intent='done', expected_agent_behavior='none')
    e.simulator = SimpleNamespace(generate=generate)
    await e.advance_once({'round': 0})
    request = {'round': 1, 'assistant_message': text}
    first = await e.advance_once(request)
    assert await e.advance_once(request) == first
    isolated_round_verifier.assert_awaited_once_with(e, 1)
    assert e.summary['round_rewards'][0]['functional_reward'] == 0


@pytest.mark.asyncio
async def test_success_verifies_before_cap_and_stops(tmp_path, isolated_round_verifier):
    e = episode(tmp_path, max_rounds=1)
    isolated_round_verifier.return_value = {'reward': 1., 'exit_code': 0}
    e.next_instruction = AsyncMock(side_effect=AssertionError('No simulator after success'))
    await e.advance({'round': 0})
    assert await e.advance({'round': 1, 'assistant_message': 'done'}) == {'action': 'stop', 'reason': 'verifier_success'}
    isolated_round_verifier.assert_awaited_once_with(e, 1)
    e.next_instruction.assert_not_awaited()


@pytest.mark.asyncio
async def test_round_verifier_error_is_not_reward_zero(tmp_path, isolated_round_verifier):
    e = episode(tmp_path)
    isolated_round_verifier.side_effect = RuntimeError('verifier setup failed')
    await e.advance({'round': 0})
    with pytest.raises(RuntimeError, match='setup failed'):
        await e.advance({'round': 1, 'assistant_message': 'done'})
    assert not e.summary.get('round_rewards')
    assert not e.summary.get('valid')


def test_driver_accepts_verified_success():
    calls = iter([{'action': 'reply', 'round': 1, 'message': 'task'},
                  {'action': 'stop', 'reason': 'verifier_success'}])
    assert drive(lambda *_: ('done', 1), lambda _: next(calls), 10) == ('verifier_success', 1)

@pytest.mark.asyncio
async def test_final_result_reuses_last_completed_round_without_live_test_upload(tmp_path):
    e = episode(tmp_path)
    actions = execution(e, {'stop_reason': 'verifier_success'})
    e._last_round_verdict = {'reward': 1., 'exit_code': 0}
    assert await e.execute_episode() == (1., 'agent_completed')
    assert not any(isinstance(action, dict) for action in actions)
    assert 'timeout -s TERM 600 bash /tests/test.sh' not in actions

from harbor_patch.multiturn.swe_conversation_driver import ContextLimitReached, query_with_context_limit


@pytest.mark.asyncio
@pytest.mark.parametrize('final_rc', [0, 2, 8, 101])
async def test_test_exit_code_without_reward_is_excluded(tmp_path, final_rc):
    e = episode(tmp_path)
    execution(e, {'stop_reason': 'user_done'}, final_rc=final_rc, reward='0')
    original = e.execute
    def execute(command, **kwargs):
        if command == 'cat /logs/verifier/reward.txt': return 1, ''
        return original(command, **kwargs)
    e.execute = execute
    with pytest.raises(RuntimeError, match='Final verifier failed to execute'):
        await e.execute_episode()
    assert not e.summary.get('valid')


@pytest.mark.asyncio
@pytest.mark.parametrize('reward', ['0', '1'])
async def test_cli_context_limit_uses_final_test_reward(tmp_path, reward):
    e = episode(tmp_path)
    actions = execution(e, {'stop_reason': 'context_limit'}, reward=reward,
                        final_rc=1 if reward == '0' else 0)
    assert await e.execute_episode() == (float(reward), 'overlong')
    assert e.summary['valid'] and e.summary['termination_reason'] == 'overlong'
    assert {'step': '02_final_verify'} in actions


@pytest.mark.parametrize('is_error,result,terminal,expected', [
    (True, 'Prompt is too long', 'blocking_limit', ContextLimitReached),
    (False, 'Prompt is too long', 'blocking_limit', RuntimeError),
    (True, 'HTTP Error 504: Gateway Timeout', 'blocking_limit', RuntimeError),
    (True, 'Prompt is too long', 'api_error', RuntimeError),
])
def test_cli_limit_detection_requires_structured_result(tmp_path, is_error, result, terminal, expected):
    log = tmp_path / 'claude-turn-001.jsonl'
    log.write_text(json.dumps(dict(type='result', is_error=is_error, result=result,
                                  terminal_reason=terminal)) + '\n')
    def query():
        raise RuntimeError('ClaudeCodeExecutionError: exit=1')
    with pytest.raises(expected) as caught:
        query_with_context_limit(query, log)
    assert type(caught.value) is expected


def test_unrelated_error_cannot_reuse_cli_limit_log(tmp_path):
    log = tmp_path / 'claude-turn-001.jsonl'
    log.write_text(json.dumps(dict(type='result', is_error=True,
        result='Prompt is too long', terminal_reason='blocking_limit')) + '\n')
    error = RuntimeError('Simulator request failed')
    def query():
        raise error
    with pytest.raises(RuntimeError) as caught:
        query_with_context_limit(query, log)
    assert caught.value is error


@pytest.mark.parametrize('text', [None, 'not valid json'])
def test_missing_or_malformed_cli_log_preserves_error(tmp_path, text):
    log = tmp_path / 'claude-turn-001.jsonl'
    if text is not None:
        log.write_text(text)
    error = RuntimeError('ClaudeCodeExecutionError: exit=1')
    def query():
        raise error
    with pytest.raises(RuntimeError) as caught:
        query_with_context_limit(query, log)
    assert caught.value is error


def test_context_limit_metrics_use_all_attempts():
    from verl_patch.multiturn.metrics import aggregate_episodes
    base = dict(enabled=True, task_profile='swe_interact')
    values = [dict(base, valid=True, final_reward=1., termination_reason='overlong', num_user_rounds=1),
              dict(base, valid=True, final_reward=0., termination_reason='agent_completed', num_user_rounds=2),
              dict(base, valid=False, failure_stage='agent')]
    result = aggregate_episodes(values)
    prefix = 'multiturn/swe_interact/'
    assert result[prefix + 'success_rate_all_attempts'] == pytest.approx(1 / 3)
    assert result[prefix + 'final_success_rate'] == .5
    assert result[prefix + 'stop/overlong_ratio'] == pytest.approx(1 / 3)
    assert result[prefix + 'first_round_overlong_ratio'] == pytest.approx(1 / 3)


@pytest.mark.asyncio
async def test_simulator_retries_timeout_and_records_source_without_secret(tmp_path):
    import asyncio
    from harbor_patch.multiturn.swe_conversation import SimulatorUnavailable
    e = episode(tmp_path)
    e.options = e.options.model_copy(update={'simulator_mode': 'llm', 'simulator_timeout_sec': 0.01})
    e.execute = lambda *a, **kw: (0, 'diff')
    async def slow(*args):
        await asyncio.sleep(1)
    e.simulator = SimpleNamespace(generate=AsyncMock(side_effect=slow))
    await e.advance({'round': 0})
    with pytest.raises(SimulatorUnavailable):
        await e.advance({'round': 1, 'assistant_message': 'done'})
    assert e.simulator.generate.await_count == 2
    events = [json.loads(line) for line in (tmp_path / 'simulator-attempts.jsonl').read_text().splitlines()]
    assert [event['error_type'] for event in events] == ['TimeoutError', 'TimeoutError']
    assert e.summary['failure_stage'] == 'simulator'
    assert not e.ended and not e.summary.get('valid')


@pytest.mark.asyncio
async def test_simulator_retry_recovers_without_duplicate_turns(tmp_path):
    e = episode(tmp_path)
    e.options = e.options.model_copy(update={'simulator_mode': 'llm'})
    e.execute = lambda *a, **kw: (0, 'diff')
    e.simulator = SimpleNamespace(generate=AsyncMock(side_effect=[
        RuntimeError('secret-credential-must-not-be-logged'),
        UserSimulatorOutput(action='message', instruction='Check it', intent='verify', expected_agent_behavior='test')]))
    await e.advance({'round': 0})
    result = await e.advance({'round': 1, 'assistant_message': 'done'})
    assert result == {'action': 'reply', 'message': 'Check it', 'round': 2}
    assert len(e.turns) == 1
    assert 'secret-credential' not in (tmp_path / 'simulator-attempts.jsonl').read_text()


@pytest.mark.asyncio
async def test_gateway_identifies_simulator_exhaustion(tmp_path):
    import aiohttp
    from harbor_patch.multiturn.swe_conversation import SimulatorUnavailable
    e = episode(tmp_path)
    e.advance = AsyncMock(side_effect=SimulatorUnavailable('unavailable'))
    await e.gateway()
    try:
        async with aiohttp.ClientSession() as client:
            async with client.post(f'http://127.0.0.1:{e.gateway_port}/conversation/{e.token}', json={'round': 1}) as response:
                assert response.status == 503
                assert (await response.json())['error']['code'] == 'simulator_unavailable'
        assert not e.summary.get('valid')
    finally:
        await e.gateway_runner.cleanup()


@pytest.mark.asyncio
async def test_generation_budget_also_applies_to_json_repair(monkeypatch):
    from harbor_patch.multiturn import user_simulator
    from harbor_patch.multiturn.user_simulator import UserSimulator, UserSimulatorInput
    from harbor_patch.multiturn.benchmark import MultiTurnBehaviorConfig
    llm = SimpleNamespace(call=AsyncMock(side_effect=[SimpleNamespace(content='not JSON'), SimpleNamespace(
        content='{"action":"no-op","instruction":"","intent":"done","expected_agent_behavior":"none"}')]))
    constructor = {}
    def build(**kwargs):
        constructor.update(kwargs)
        return llm
    monkeypatch.setattr(user_simulator, 'LiteLLM', build)
    sim = UserSimulator('openai/model', max_tokens=32768)
    payload = UserSimulatorInput(task_name='test', initial_instruction='fix', round_id=2,
        behavior=MultiTurnBehaviorConfig(id='test', type='conversation'), previous_rounds=[], context_summary={})
    assert (await sim.generate(payload)).action == 'no-op'
    assert [call.kwargs['max_tokens'] for call in llm.call.await_args_list] == [32768, 32768]
    assert 'reasoning_effort' not in constructor


@pytest.mark.asyncio
async def test_native_failure_type_persisted_before_cleanup(tmp_path):
    from harbor_patch.multiturn.native_swe import NativeEpisode
    e = episode(tmp_path)
    e.created = []
    e.gateway = AsyncMock()
    e.create_sandbox = AsyncMock(side_effect=RuntimeError('internal detail'))
    with pytest.raises(RuntimeError):
        await NativeEpisode.run(e)
    summary = json.loads((tmp_path / 'multiturn.json').read_text())
    assert summary['error_type'] == 'RuntimeError' and summary['valid'] is False
    assert 'internal detail' not in json.dumps(summary)

@pytest.mark.asyncio
async def test_partial_verifier_failure_is_cached_not_reapplied(tmp_path,monkeypatch):
    from harbor_patch.multiturn import conversation_verifier
    from harbor_patch.multiturn.swe_conversation import ConversationTurnFailed
    e=episode(tmp_path)
    await e.advance_once({'round':0})
    verify=AsyncMock(side_effect=RuntimeError('private-error-text'))
    monkeypatch.setattr(conversation_verifier,'verify_round_snapshot',verify)
    payload={'round':1,'assistant_message':'done'}
    for _ in range(3):
        with pytest.raises(ConversationTurnFailed):await e.advance_once(payload)
    assert len(e.turns)==1
    verify.assert_awaited_once()
    assert not e.summary.get('round_rewards')
    assert 'private-error-text' not in (tmp_path/'conversation-failures.jsonl').read_text()

@pytest.mark.asyncio
async def test_partial_failure_returns_structured_503(tmp_path,monkeypatch):
    import aiohttp
    from harbor_patch.multiturn import conversation_verifier
    e=episode(tmp_path)
    await e.advance_once({'round':0})
    monkeypatch.setattr(conversation_verifier,'verify_round_snapshot',AsyncMock(side_effect=RuntimeError('failed')))
    await e.gateway()
    try:
        async with aiohttp.ClientSession() as client:
            for _ in range(2):
                async with client.post(f'http://127.0.0.1:{e.gateway_port}/conversation/{e.token}',json={'round':1,'assistant_message':'done'}) as r:
                    assert r.status==503
                    assert (await r.json())['error']['code']=='conversation_turn_failed'
        assert len(e.turns)==1
    finally:await e.gateway_runner.cleanup()
