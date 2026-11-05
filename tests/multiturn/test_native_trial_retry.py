import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from kubernetes.client.exceptions import ApiException

from harbor_patch.multiturn import native_swe as native


@pytest.fixture
def trial(tmp_path, monkeypatch):
    monkeypatch.setenv('HARBOR_SWE_INTERACT_MANIFEST', 'unused')
    monkeypatch.setenv('HARBOR_SWE_INTERACTION_MODE', 'native_review')
    monkeypatch.setenv('HARBOR_SWE_TRIAL_MAX_RETRIES', '3')
    row = {'task_id': 'owner__repo-1', 'task_path': '/tasks/owner__repo-1', 'split': 'train'}
    monkeypatch.setattr(native, 'load_manifest', lambda _: {row['task_id']: row})
    sleep = AsyncMock()
    monkeypatch.setattr(native.asyncio, 'sleep', sleep)
    state = SimpleNamespace(outcomes=[], sessions=[], closed=[], metrics=[], episodes=[], row=row)
    proxy = SimpleNamespace(_sessions={})

    async def open_session(session_id, *, trajectory_dir, harness_name):
        assert not proxy._sessions, 'Previous session must be closed before a retry'
        trajectory_dir.mkdir(parents=True)
        proxy._sessions[session_id] = {}
        state.sessions.append(session_id)

    async def pop_session(session_id):
        proxy._sessions.pop(session_id)
        state.closed.append(session_id)
        return {'chosen_attempt': len(state.sessions)}

    proxy.open_session, proxy.pop_session = open_session, pop_session

    class Episode:
        def __init__(self, proxy, session_id, row, options, run_dir, sampling_params):
            self.summary = {'valid': False}
            self.run_dir = run_dir
            self.session_id = session_id
            state.episodes.append(self)

        async def run(self):
            value = state.outcomes.pop(0)
            if isinstance(value, BaseException):
                raise value
            self.summary.update(valid=True, final_reward=value)
            return value, 'agent_completed'

    monkeypatch.setattr(native, 'NativeEpisode', Episode)
    def assign(metrics, meta):
        metrics['selected'] = meta['chosen_attempt']
    loop = SimpleNamespace(base_trials_dir=tmp_path, multiturn=SimpleNamespace(),
                           _assign_proxy_metrics_from_session=assign)
    async def run(is_val=False):
        row['split'] = 'val' if is_val else 'train'
        metrics = {'outer': 99}
        result = await native.run_native_trial(loop, proxy, row['task_path'], 1282, {}, metrics, is_val)
        return result, metrics
    state.run, state.proxy, state.sleep = run, proxy, sleep
    state.records = lambda: [json.loads(line) for p in (tmp_path / '_retries').glob('*.jsonl')
                             for line in p.read_text().splitlines()]
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize('is_val', [False, True])
async def test_success_on_third_retry_has_fresh_sessions_and_only_final_metrics(trial, is_val):
    trial.outcomes = [ConnectionError('disconnected') for _ in range(3)] + [1.0]
    (reward, reason, meta), metrics = await trial.run(is_val)
    assert reward == 1 and reason == 'agent_completed'
    assert len(set(trial.sessions)) == 4 and trial.closed == trial.sessions
    assert len({e.run_dir for e in trial.episodes}) == 4
    assert metrics == {'outer': 99, 'selected': 4, 'native_retry/retries': 3.0,
                       'native_retry/exhausted': 0.0}
    assert not meta.get('disable_proxy_trajectory')
    assert meta['native_retry']['attempts'] == 4
    assert [x['will_retry'] for x in trial.records()] == [True, True, True, False]
    assert all(x['validate'] == is_val for x in trial.records())
    assert [x.args[0] for x in trial.sleep.await_args_list] == [1, 2, 4]


@pytest.mark.asyncio
async def test_exhaustion_stops_at_four_total_attempts_and_remains_invalid(trial):
    trial.outcomes = [TimeoutError('network') for _ in range(5)]
    (reward, reason, meta), metrics = await trial.run()
    assert len(trial.sessions) == 4 and len(trial.outcomes) == 1
    assert reward == 0 and reason == 'env_setup_failed'
    assert meta['disable_proxy_trajectory'] and not meta['multiturn']['valid']
    assert metrics['native_retry/exhausted'] == 1
    assert len(trial.records()) == 4 and not trial.records()[-1]['will_retry']


@pytest.mark.asyncio
async def test_valid_zero_reward_is_never_resampled(trial):
    trial.outcomes = [0.0, 1.0]
    (reward, _, meta), metrics = await trial.run()
    assert reward == 0 and meta['multiturn']['valid']
    assert len(trial.sessions) == 1 and trial.outcomes == [1.0]
    assert metrics['native_retry/retries'] == 0 and not trial.records()


@pytest.mark.asyncio
@pytest.mark.parametrize('error', [
    RuntimeError('Final verifier failed to execute (127, 0)'),
    RuntimeError("Claude's response exceeded the 4096 output token maximum"),
    ValueError('invalid manifest configuration'),
    ApiException(status=401, reason='unauthorized'),
])
async def test_nontransient_errors_are_not_retried(trial, error):
    trial.outcomes = [error]
    (_, _, meta), _ = await trial.run()
    assert len(trial.sessions) == 1 and not meta['native_retry']['retryable']
    trial.sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancellation_propagates_without_retry_and_closes_session(trial):
    trial.outcomes = [asyncio.CancelledError()]
    with pytest.raises(asyncio.CancelledError):
        await trial.run()
    assert trial.closed == trial.sessions and len(trial.sessions) == 1
    trial.sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_retries_can_be_disabled(trial, monkeypatch):
    monkeypatch.setenv('HARBOR_SWE_TRIAL_MAX_RETRIES', '0')
    trial.outcomes = [ConnectionError('network')]
    await trial.run()
    assert len(trial.sessions) == 1
    trial.sleep.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('limit', ['4', '-1', 'three'])
async def test_invalid_retry_limit_rejected_before_start(trial, monkeypatch, limit):
    monkeypatch.setenv('HARBOR_SWE_TRIAL_MAX_RETRIES', limit)
    with pytest.raises(ValueError):
        await trial.run()
    assert not trial.sessions


@pytest.mark.parametrize('error,category', [
    (ApiException(status=0, reason='container not found'), 'kubernetes_api'),
    (ApiException(status=503), 'kubernetes_api'),
    (ValueError("invalid literal for int(): OCI runtime exec failed"), 'sandbox_transport'),
    (RuntimeError('HTTP Error 500: Internal Server Error'), 'conversation_gateway'),
    (ValueError('Nonconsecutive conversation turn'), 'conversation_gateway'),
    (RuntimeError('User simulator unavailable: retries exhausted'), 'simulator_unavailable'),
])
def test_observed_transient_failure_classification(error, category):
    assert native._native_failure_category(error) == (category, True)
