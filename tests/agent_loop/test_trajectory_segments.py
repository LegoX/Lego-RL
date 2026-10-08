"""Regressions for context rewrites: trained tokens keep their actual prefixes."""

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from verl_patch.agent_loop.trajectory import TrajectoryRecorder, select_trajectories
from verl_patch.agent_loop.vllm_chat_completion_proxy import _VLLMChatCompletionsProxy


def record(recorder, prompt, answer, step=0, tail=()):
    recorder.record(
        prompt,
        answer,
        [-i / 10 for i in answer],
        [[[i]] for i in answer],
        context_tail=tail,
        min_global_steps=step,
        max_global_steps=step,
    )


def trained_pairs(segments):
    pairs = []
    for segment in segments:
        tokens = segment["prompt_ids"] + segment["response_ids"]
        for i, flag in enumerate(
            segment["response_mask"], start=len(segment["prompt_ids"])
        ):
            if flag:
                pairs.append((tokens[:i], tokens[i]))
    return pairs


def test_compaction_keeps_all_actions_under_original_conditioning():
    r = TrajectoryRecorder()
    # Q A1 O1 A2 O2 A3 -> Q Z A3 O3 A4
    requests = [
        ([1], [10, 11]),
        ([1, 10, 11, 2], [20]),
        ([1, 10, 11, 2, 20, 3], [30]),
        ([1, 50, 30, 4], [40, 41]),
    ]
    expected = []
    for step, (prompt, answer) in enumerate(requests):
        record(r, prompt, answer, step)
        expected.extend((prompt + answer[:i], token) for i, token in enumerate(answer))
    segments = r.export()
    assert len(segments) == 2
    assert trained_pairs(segments) == expected
    assert segments[1]["prompt_ids"] == [1, 50, 30, 4]
    assert segments[1]["response_mask"] == [1, 1]  # echoed A3 is only in the prompt
    assert segments[0]["min_global_steps"] == 0
    assert segments[0]["max_global_steps"] == 2
    assert segments[1]["min_global_steps"] == 3
    assert segments[0]["response_routing"][2] is None  # O1
    assert segments[1]["response_logprobs"] == [-4.0, -4.1]


def test_selection_counts_only_generated_tokens_and_archives_unselected():
    r = TrajectoryRecorder()
    record(r, [1], [10, 11])
    record(r, list(range(100, 200)), [20])
    assert select_trajectories(r.export(), "longest")[0][0] == 0
    record(r, [3], [30, 31])
    assert select_trajectories(r.export(), "longest")[0][0] == 2
    assert len(select_trajectories(r.export(), "all")) == 3
    assert len(r.export()) == 3
    with pytest.raises(ValueError):
        select_trajectories([], "last")


def test_eos_tail_is_context_and_retry_removes_only_latest_generation():
    r = TrajectoryRecorder()
    record(r, [1], [10, 99], tail=[88])
    record(r, [1, 10, 99, 88, 2], [20, 99], tail=[88])
    assert r.rollback_last_generation() == [1, 10, 99, 88, 2]
    record(r, [1, 10, 99, 88, 2], [21, 99], tail=[88])
    assert r.export()[0]["response_mask"] == [1, 1, 0, 0, 1, 1, 0]
    assert r.discarded_generations[0]["output_ids"] == [20, 99]
    assert [t for _, t in trained_pairs(r.export())] == [10, 99, 21, 99]


def test_missing_logprobs_are_never_fabricated():
    r = TrajectoryRecorder()
    r.record([1], [2], None, [None])
    assert r.export()[0]["response_logprobs"] is None
    with pytest.raises(ValueError, match="logprobs"):
        r.record([1, 2], [3], [], [None])
    assert trained_pairs(r.export()) == [([1], 2)]


@pytest.fixture
def proxy():
    p = object.__new__(_VLLMChatCompletionsProxy)
    p._sessions = {}
    p._sessions_lock = asyncio.Lock()
    p._session_state_locks = {}
    p._session_generation_locks = {}
    p._session_attempts = {}
    p._main_initial_messages = {}
    p._sub_session_keys = {}
    p._processor = None
    p._assistant_eos_token_id = 99
    p._assistant_eos_tail = [88]

    async def render(messages, tools=None, **kwargs):
        ids = [int(token) for msg in messages for token in msg["content"].split()]
        return ([500] if tools else []) + ids + [77]

    async def done(*args, **kwargs):
        pass

    p._agent_loop = SimpleNamespace(apply_chat_template=render)
    p._session_scheduler = SimpleNamespace(finish_session=done)
    return p


async def generate(proxy, messages, answer, tools=None):
    s = proxy._sessions["s"]
    prompt = await proxy._compute_prompt_ids(s, messages, tools, "s")
    output = SimpleNamespace(
        token_ids=answer,
        log_probs=[-0.2] * len(answer),
        routed_experts=None,
        num_preempted=0,
        stop_reason="stop",
        extra_fields={},
    )
    await proxy._finalize_generation("s", s, output, len(prompt))
    s["messages_snapshot"] += [
        {"role": "assistant", "content": " ".join(map(str, answer))}
    ]
    s["last_generation_snapshot"] = deepcopy(s["messages_snapshot"])
    return prompt


@pytest.mark.asyncio
async def test_actual_proxy_archive_keeps_old_prefix_and_masks_historical_assistants(
    proxy, tmp_path
):
    await proxy.open_session("s", tmp_path)
    q = [{"role": "user", "content": "1"}]
    p1 = await generate(proxy, q, [10, 99])
    next_messages = proxy._sessions["s"]["messages_snapshot"] + [
        {"role": "user", "content": "2"}
    ]
    p2 = await generate(proxy, next_messages, [20, 99])
    rewritten = q + [
        {"role": "user", "content": "50"},
        {"role": "assistant", "content": "20 99"},
        {"role": "user", "content": "3"},
    ]
    p3 = await generate(proxy, rewritten, [30, 99])
    meta = await proxy.pop_session("s")
    segments = meta["trajectory_segments"]
    assert len(segments) == 2
    assert trained_pairs(segments) == [
        (p1, 10),
        (p1 + [10], 99),
        (p2, 20),
        (p2 + [20], 99),
        (p3, 30),
        (p3 + [30], 99),
    ]
    assert (
        json.loads((tmp_path / "proxy_trajectory.json").read_text())[
            "trajectory_segments"
        ]
        == segments
    )


@pytest.mark.asyncio
async def test_failed_rebuild_does_not_replace_archive_and_retry_is_transactional(
    proxy,
):
    await proxy.open_session("s")
    q = [{"role": "user", "content": "1"}]
    p = await generate(proxy, q, [10, 99])
    s = proxy._sessions["s"]
    # A retry's first generation attempt fails after prompt preparation.
    await proxy._compute_prompt_ids(s, q, None, "s")
    assert trained_pairs(s["trajectory_recorder"].export()) == [(p, 10), (p + [10], 99)]
    retry_prompt = await generate(proxy, q, [11, 99])
    assert retry_prompt == p
    meta = await proxy.pop_session("s")
    assert trained_pairs(meta["trajectory_segments"]) == [(p, 11), (p + [11], 99)]
    assert len(meta["discarded_generations"]) == 1


@pytest.mark.asyncio
async def test_changed_tool_schema_is_a_new_context_not_a_retry(proxy):
    await proxy.open_session("s")
    q = [{"role": "user", "content": "1"}]
    p1 = await generate(proxy, q, [10])
    p2 = await generate(proxy, q, [20], tools=[{"name": "new_tool"}])
    meta = await proxy.pop_session("s")
    assert len(meta["trajectory_segments"]) == 2
    assert trained_pairs(meta["trajectory_segments"]) == [(p1, 10), (p2, 20)]


@pytest.mark.asyncio
async def test_replaced_assistant_retry_survives_a_failed_attempt(proxy):
    await proxy.open_session("s")
    q = [{"role": "user", "content": "1"}]
    await generate(proxy, q, [10, 99])
    replacement = q + [{"role": "assistant", "content": "11 99"}]
    s = proxy._sessions["s"]
    await proxy._compute_prompt_ids(s, replacement, None, "s")
    assert s["pending_retry"]
    # No finalize: the engine failed. Retrying the same replacement must still
    # discard the previous successful answer, exactly once.
    prompt = await generate(proxy, replacement, [20, 99])
    meta = await proxy.pop_session("s")
    assert trained_pairs(meta["trajectory_segments"]) == [
        (prompt, 20),
        (prompt + [20], 99),
    ]
    assert [g["output_ids"] for g in meta["discarded_generations"]] == [[10, 99]]


@pytest.mark.asyncio
async def test_issue_21_rewrite_stays_on_main_route_and_preserves_every_action(proxy):
    await proxy.open_session("s")
    q = [{"role": "user", "content": "1"}]
    messages = q
    expected = []
    for answer, observation in (([10, 99], "2"), ([20, 99], "3"), ([30, 99], "4")):
        assert await proxy._route_subagent_session("s", messages) == ("s", False)
        prompt = await generate(proxy, messages, answer)
        expected.extend((prompt + answer[:i], token) for i, token in enumerate(answer))
        messages = deepcopy(proxy._sessions["s"]["messages_snapshot"])
        messages.append({"role": "user", "content": observation})
    # Q A1 O1 A2 O2 A3 -> Q A1 Z A3 O3. Retaining Q A1 exercises the
    # real main/subagent identity check, as in the reported issue.
    rewritten = q + [
        {"role": "assistant", "content": "10 99"},
        {"role": "user", "content": "50"},
        {"role": "assistant", "content": "30 99"},
        {"role": "user", "content": "4"},
    ]
    assert await proxy._route_subagent_session("s", rewritten) == ("s", False)
    prompt = await generate(proxy, rewritten, [40, 99])
    expected.extend([(prompt, 40), (prompt + [40], 99)])
    meta = await proxy.pop_session("s")
    assert not meta["disable_proxy_trajectory"]
    assert len(meta["trajectory_segments"]) == 2
    assert trained_pairs(meta["trajectory_segments"]) == expected
    assert all(
        lp == -0.2
        for segment in meta["trajectory_segments"]
        for mask, lp in zip(segment["response_mask"], segment["response_logprobs"])
        if mask
    )
