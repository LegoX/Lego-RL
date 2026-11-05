import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from harbor_patch.multiturn.config import MultiTurnConfig
from harbor_patch.multiturn.native_budget import TurnBudget
from harbor_patch.multiturn.native_swe import NativeEpisode, classify_result, load_manifest
from verl_patch.multiturn.metrics import aggregate_episodes, terminal_score


def test_native_requires_terminal_reward_and_no_extra_judge():
    MultiTurnConfig(enabled=True, task_profile='swe_interact', max_rounds=1)
    for overrides in ({'reward_mode': 'per_round'}, {'judge_enabled': True, 'judge_model': 'judge'}):
        with pytest.raises(ValueError):
            MultiTurnConfig(enabled=True, task_profile='swe_interact', max_rounds=1, **overrides)


@pytest.mark.parametrize('native,approval,reward,rc,overflow,code,reason', [
    ({'exit_status': 'SimulatorApproved'}, 1, 1, 0, False, 0, 'agent_completed'),
    ({'exit_status': 'SimulatorApproved'}, 1, 0, 1, False, 0, 'agent_completed'),
    ({'error': 'Interaction turn limit exhausted before simulator approval'}, 0, 0, None, False, 1, 'max_turns_reached'),
    ({'error': 'maximum context length'}, 0, 0, None, True, 1, 'overlong'),
])
def test_native_keeps_true_verdict_and_budget_failures(native, approval, reward, rc, overflow, code, reason):
    assert classify_result(native, approval, reward, rc, overflow, code) == reason


@pytest.mark.parametrize('native,approval,reward,rc', [
    ({'error': 'API 504'}, 0, 0, None), ({'error': 'ModuleNotFoundError'}, 0, 0, None),
    ({'exit_status': 'SimulatorApproved'}, 1, 1, 124),
    ({'exit_status': 'SimulatorApproved'}, 0, 1, 0),
    ({'exit_status': 'SimulatorApproved'}, 1, float('nan'), 0),
])
def test_native_rejects_infrastructure_failures(native, approval, reward, rc):
    with pytest.raises((RuntimeError, ValueError)):
        classify_result(native, approval, reward, rc, False, 0)


def test_native_terminal_metrics_do_not_invent_per_round_scores():
    a = dict(enabled=True, valid=True, task_profile='swe_interact', final_reward=1, approval_reward=1, num_user_rounds=5)
    b = {**a, 'final_reward': 0, 'approval_reward': 0, 'num_user_rounds': 3}
    invalid = {**a, 'valid': False}
    result = aggregate_episodes([a, b, invalid])
    assert terminal_score(invalid) is None
    assert result['multiturn/swe_interact/final_success_rate'] == 0.5
    assert result['multiturn/swe_interact/feedback_count'] == 3
    assert result['multiturn/swe_interact/invalid_ratio'] == pytest.approx(1 / 3)
    assert not any('first_success' in k or '/round/' in k for k in result)


@pytest.mark.asyncio
async def test_native_handoff_preserves_sampling_and_original_messages():
    e = object.__new__(NativeEpisode)
    e.proxy = SimpleNamespace(begin_tool_handoff=AsyncMock())
    e.session_id = 'native-a'
    e.budget = TurnBudget(first_round_calls=2)
    e.budget.begin(1)
    e.sampling_params = {'temperature': 0.6, 'top_p': 0.95}
    messages = [{'role': 'user', 'content': 'fix bug'}]
    tools = [{'type': 'function'}]
    _, returned, body = await e.prepare(messages, tools, {'max_tokens': 10000})
    assert returned == tools and body['temperature'] == 0.6 and body['max_tokens'] == 4096
    _, returned, body = await e.prepare(messages, tools, {})
    assert returned is None and body['tool_choice'] == 'none' and 'logit_bias' not in body
    e.proxy.begin_tool_handoff.assert_awaited_once_with('native-a')
    assert messages == [{'role': 'user', 'content': 'fix bug'}]


@pytest.mark.asyncio
async def test_final_tests_are_not_run_without_approval(tmp_path):
    e = object.__new__(NativeEpisode)
    e.row, e.task, e.run_dir, e.summary = {'workdir': '/app/src'}, tmp_path, tmp_path, {}
    e.session_id, e.host, e.gateway_port, e.token, e.export_token = 'a', '10.0.0.1', 1234, 'local', 'local'
    e.budget = TurnBudget()
    e.options = SimpleNamespace(max_rounds=5, episode_timeout_sec=3600)
    e.proxy = SimpleNamespace(session_anthropic_base=lambda _: 'http://10.0.0.1/sess/a', _sessions={'a': {}})
    actions = []
    def execute(command, **kwargs):
        actions.append(command)
        if command == 'cat /logs/agent/interaction-result.json':
            return 0, json.dumps({'error': 'Interaction turn limit exhausted before simulator approval'})
        if command == 'cat /logs/verifier/reward.txt':
            return 0, '0\n'
        return 0, ''
    e.execute, e.test_files, e.upload = execute, lambda step: {'step': step}, lambda files: actions.append(files)
    reward, reason = await e.execute_episode()
    assert reward == 0 and reason == 'max_turns_reached' and e.summary['valid']
    assert actions[0].startswith('cd /app/src && env ')
    assert {'step': '01_interactive_develop'} in actions
    assert {'step': '02_final_verify'} not in actions


def test_manifest_rejects_duplicates_and_unpinned_images(tmp_path):
    row = {'task_id': 'owner__repo-1', 'images_ready': True, 'protocol': 'two-step-text@3.6',
           'images': {'main': 'r/m@sha256:a', 'user-server': 'r/u@sha256:b'}}
    path = tmp_path / 'tasks.jsonl'
    path.write_text(json.dumps(row) + '\n' + json.dumps(row) + '\n')
    with pytest.raises(ValueError): load_manifest(str(path))
    path.write_text(json.dumps({**row, 'images': {'main': 'r/m:latest'}}))
    with pytest.raises(ValueError): load_manifest(str(path))


@pytest.mark.asyncio
async def test_agent_loop_routes_native_before_legacy_trial(monkeypatch):
    from verl_patch.agent_loop.builtin_swe_agent_loop import BuiltinSWEAgentLoop
    from harbor_patch.multiturn import native_swe
    native = AsyncMock(return_value=(1.0, 'agent_completed', {'session_id': 'native'}))
    monkeypatch.setattr(native_swe, 'run_native_trial', native)
    loop = SimpleNamespace(multiturn=MultiTurnConfig(enabled=True, task_profile='swe_interact', max_rounds=1))
    selection = SimpleNamespace(definition=SimpleNamespace(name='claude_code'))
    proxy, metrics = object(), {}
    result = await BuiltinSWEAgentLoop._run_harbor_trial(loop, task_path='/task', global_steps=0,
        sampling_params={'temperature': 1}, proxy=proxy, metrics=metrics, is_val=False, harness_selection=selection)
    assert result[0] == 1.0
    native.assert_awaited_once_with(loop, proxy, '/task', 0, {'temperature': 1}, metrics, False)


def test_native_training_metrics_deduplicate_terminal_rewards():
    import numpy as np
    import torch
    from verl_patch.multiturn.metrics import training_metrics
    a = dict(enabled=True, valid=True, task_profile='swe_interact', final_reward=1, approval_reward=1, num_user_rounds=3)
    b = {**a, 'final_reward': 0, 'approval_reward': 0}
    batch = SimpleNamespace(non_tensor_batch={'multiturn_summary': [json.dumps(a), json.dumps(a), json.dumps(b)],
        'trial_id': ['a', 'a', 'b'], 'trial_gen_tokens': [4,4,2]}, batch={'response_mask': torch.ones(3,2)})
    result = training_metrics(batch, np.array([True,True,True]), torch.tensor([1.,1.,0.]))
    assert result['reward_stats/trial_count'] == 2
    assert result['reward_stats/trial_mean'] == 0.5
    assert result['multiturn/swe_interact/trainable_episodes'] == 2
