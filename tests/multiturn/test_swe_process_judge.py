import json
import pytest

from harbor_patch.multiturn.swe_process_judge import (
    JudgeScores, aggregate, blend_reward, evidence, prompt_for, schema,
)


def test_swe_judge_uses_public_actor_evidence_only(tmp_path):
    (tmp_path / 'conversation.json').write_text(json.dumps([
        {'round_id': 1, 'instruction': 'Fix parsing', 'final_message': 'Changed parse_host'}]))
    (tmp_path / 'proxy_trajectory.json').write_text(json.dumps({'messages_snapshot': [
        {'role': 'assistant', 'tool_calls': [{'id': 'a', 'function': {'name': 'Bash',
            'arguments': {'command': 'cargo test'}}}]},
        {'role': 'tool', 'tool_call_id': 'a', 'content': 'test result: ok. 12 passed\nExit code 0'},
    ], 'compaction_count': 0}))
    patch = tmp_path / 'artifacts/logs/agent/model.patch'
    patch.parent.mkdir(parents=True)
    patch.write_text('diff --git a/src.rs b/src.rs\n+fn parse_host() {}\n')
    verifier = tmp_path / 'artifacts/logs/verifier/secret.txt'
    verifier.parent.mkdir(parents=True)
    verifier.write_text('SECRET_HIDDEN_VERIFIER')
    payload = evidence(tmp_path, {'task_id': 'sample', 'initial_request': 'Fix parsing',
                                  'task_instruction': 'Handle hosts correctly'})
    prompt = prompt_for(payload, 48000)
    assert 'cargo test' in prompt and '12 passed' in prompt
    assert 'parse_host' in prompt
    assert 'SECRET_HIDDEN_VERIFIER' not in prompt
    assert schema()['json_schema']['strict'] is True


def test_swe_judge_reward_bands_and_fallback():
    scores = JudgeScores(relevant_implementation=3, tool_execution=2,
        verification_and_honesty=2, requirement_tracking=3,
        feedback_handling=None, user_effort_efficiency=2,
        evidence_refs=['patch hunk'], uncertainty='none', abstain=False)
    judge_score = aggregate(scores)
    assert 0 <= judge_score <= 1
    assert blend_reward(0.0, {'status': 'ok', 'judge_score': 1.0}, .9, .1) == .1
    assert blend_reward(1.0, {'status': 'ok', 'judge_score': 0.0}, .9, .1) == .9
    assert blend_reward(1.0, {'status': 'abstain'}, .9, .1) == 1.0


def test_swe_metrics_keep_terminal_and_blended_rewards_separate():
    from verl_patch.multiturn.metrics import aggregate_episodes
    base = {'enabled': True, 'valid': True, 'task_profile': 'swe_interact',
            'judge_status': 'ok', 'judge_usage': {'prompt_tokens': 1200,
            'completion_tokens': 100}}
    metrics = aggregate_episodes([
        {**base, 'final_reward': 1.0, 'training_reward': .96, 'judge_score': .6},
        {**base, 'final_reward': 0.0, 'training_reward': .04, 'judge_score': .4},
    ])
    assert metrics['multiturn/swe_interact/final_success_rate'] == .5
    assert metrics['multiturn/swe_interact/training_reward'] == .5
    assert metrics['multiturn/swe_interact/judge_coverage'] == 1
    assert metrics['multiturn/swe_interact/judge_prompt_tokens'] == 1200


async def _run_mock_native_judge(tmp_path, monkeypatch, judgment):
    from types import SimpleNamespace
    from harbor_patch.multiturn import native_swe, swe_conversation, swe_process_judge
    task = tmp_path / 'task'
    task.mkdir()
    row = {'task_id': 'task', 'task_path': str(task), 'split': 'train'}
    monkeypatch.setattr(native_swe, 'load_manifest', lambda path: {'task': row})
    monkeypatch.setenv('HARBOR_SWE_INTERACT_MANIFEST', str(tmp_path / 'manifest'))
    monkeypatch.setenv('HARBOR_SWE_INTERACTION_MODE', 'conversation')

    class Episode:
        def __init__(self, proxy, session_id, row, options, run_dir, sampling_params):
            self.run_dir = run_dir
            run_dir.mkdir(parents=True)
            self.context = {'initial_request': 'fix', 'task_instruction': 'fix'}
            self.summary = {'enabled': True, 'valid': True,
                            'task_profile': 'swe_interact', 'final_reward': 1.0}

        async def run(self):
            return 1.0, 'agent_completed'

    class Proxy:
        def __init__(self):
            self._sessions = {}

        async def open_session(self, session_id, **kwargs):
            self._sessions[session_id] = {}

        async def pop_session(self, session_id):
            self._sessions.pop(session_id)
            return {}

    async def fake_judge(run_dir, context, options):
        if isinstance(judgment, Exception):
            raise judgment
        return judgment

    monkeypatch.setattr(swe_conversation, 'ConversationEpisode', Episode)
    monkeypatch.setattr(swe_process_judge, 'judge', fake_judge)
    opts = SimpleNamespace(judge_enabled=True, test_weight=.9, judge_weight=.1)
    loop = SimpleNamespace(base_trials_dir=tmp_path, multiturn=opts,
                           _assign_proxy_metrics_from_session=lambda metrics, meta: None)
    reward, reason, meta = await native_swe.run_native_trial(loop, Proxy(), task, 1, {}, {}, False)
    return reward, meta['multiturn']


@pytest.mark.asyncio
async def test_native_trial_blends_judge_and_keeps_terminal_verdict(tmp_path, monkeypatch):
    reward, summary = await _run_mock_native_judge(
        tmp_path, monkeypatch, {'status': 'ok', 'model': 'gpt-5.5',
                                'judge_score': .3, 'usage': {'prompt_tokens': 100}})
    assert reward == .93
    assert summary['final_reward'] == 1.0
    assert summary['training_reward'] == .93
    assert summary['judge_status'] == 'ok'


@pytest.mark.asyncio
async def test_native_trial_judge_error_falls_back_to_terminal(tmp_path, monkeypatch):
    reward, summary = await _run_mock_native_judge(tmp_path, monkeypatch, RuntimeError('offline'))
    assert reward == 1.0
    assert summary['valid'] is True
    assert summary['judge_status'] == 'error'
