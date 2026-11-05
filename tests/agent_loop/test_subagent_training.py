"""Opt-in child trajectories keep their own context and one parent outcome."""

import asyncio
from types import SimpleNamespace as NS

import pytest

from test_native_compaction import proxy, sample, SYSTEM, TASK, actions, expected
from verl_patch.agent_loop import vllm_chat_completion_proxy as pm
from verl_patch.agent_loop.builtin_swe_agent_loop import BuiltinSWEAgentLoop
from verl_patch.agent_loop.harness import HarnessDefinition, inject_harness_endpoint


async def start(p):
    await p.open_session("root", compaction={"mode": "replace", "summary": "on_thread"})
    await p._route_subagent_session("root", [SYSTEM, TASK])
    return await sample(p, "root", [SYSTEM, TASK], "parent action")


async def child(p, name="child"):
    messages = [SYSTEM, {"role": "user", "content": name + " task"}]
    sid, is_child = await p._route_subagent_session("root", messages)
    assert is_child
    return sid, messages


@pytest.mark.asyncio
async def test_default_keeps_children_untrained(proxy, monkeypatch):
    monkeypatch.delenv("HARBOR_TRAIN_SUBAGENTS", raising=False)
    parent = await start(proxy)
    sid, msg = await child(proxy)
    await sample(proxy, sid, msg, "untrained child")
    meta = await proxy.pop_session("root")
    assert actions(meta["trajectory_segments"]) == expected(parent)
    assert not any(s.get("kind") == "subagent" for s in meta["trajectory_segments"])


@pytest.mark.asyncio
async def test_children_share_one_parent_episode_and_keep_exact_context(
    proxy, monkeypatch, tmp_path
):
    monkeypatch.setenv("HARBOR_TRAIN_SUBAGENTS", "1")
    parent = await start(proxy)
    wanted = []
    for name in ["first", "second"]:
        sid, msg = await child(proxy, name)
        turn = await sample(proxy, sid, msg, name + " action")
        wanted.append((sid, turn))
    meta = await proxy.pop_session("root")
    segments = meta["trajectory_segments"]
    assert meta["episode_id"] == "root"
    assert actions([s for s in segments if s.get("kind") != "subagent"]) == expected(
        parent
    )
    for sid, turn in wanted:
        assert actions(
            [s for s in segments if s.get("subagent_session_id") == sid]
        ) == expected(turn)
    assert meta["gen_tokens_total"] == len(parent[1]) + sum(
        len(t[1]) for _, t in wanted
    )
    assert meta["subagent_num_calls"] == 2
    assert not meta["subagent_archive_incomplete"]
    # Exercise the actual main episode output adapter, not a parallel trial backend.
    loop = BuiltinSWEAgentLoop.__new__(BuiltinSWEAgentLoop)
    loop.prompt_length, loop.response_length = 10000, 10000
    loop.trajectory_selection = "all"
    loop._max_turns = 0
    loop._tool_parser_name = "hermes"
    loop.base_trials_dir = str(tmp_path)
    definition = NS(name="test", protocol="openai")
    loop.harnesses = {"test": definition}
    outputs = loop._outputs_from_segments(
        meta, 1.0, "agent_completed", {}, {}, NS(definition=definition, source="test")
    )
    assert len(outputs) == 3
    assert {o.extra_fields["episode_id"] for o in outputs} == {"root"}
    assert all(o.reward_score == 1.0 for o in outputs)
    assert sum(o.extra_fields["segment_kind"] == "subagent" for o in outputs) == 2


@pytest.mark.asyncio
async def test_child_rewrites_and_retries_keep_only_actual_actions(
    proxy, monkeypatch, tmp_path
):
    monkeypatch.setenv("HARBOR_TRAIN_SUBAGENTS", "1")
    await start(proxy)
    sid, msg = await child(proxy)
    proxy._sessions["root"]["trajectory_dir"] = str(tmp_path)
    first = await sample(proxy, sid, msg, "first")
    state = proxy._sessions[sid]
    messages = state["messages_snapshot"] + [{"role": "user", "content": "next"}]
    await sample(proxy, sid, messages, "superseded")
    replacement = await sample(proxy, sid, messages, "replacement")
    rewritten = [SYSTEM, {"role": "user", "content": "rewritten child context"}]
    state["pending_rebuild_cause"] = "compaction_replace"
    last = await sample(proxy, sid, rewritten, "after rewrite")
    meta = await proxy.pop_session("root")
    segments = [s for s in meta["trajectory_segments"] if s.get("kind") == "subagent"]
    assert actions(segments) == expected(first) + expected(replacement) + expected(last)
    assert len(meta["subagent_discarded_generations"]) == 1
    import json

    audit = json.loads((tmp_path / "proxy_trajectory.json").read_text())
    assert len(audit["subagent_discarded_generations"]) == 1
    assert not audit["subagent_archive_incomplete"]


@pytest.mark.asyncio
async def test_child_output_is_charged_to_parent_budget(proxy, monkeypatch):
    monkeypatch.setenv("HARBOR_TRAIN_SUBAGENTS", "1")
    parent = await start(proxy)
    sid, msg = await child(proxy)
    turn = await sample(proxy, sid, msg, "child action")
    total = len(parent[1]) + len(turn[1])
    assert proxy._sessions["root"]["gen_tokens_total"] == total
    monkeypatch.setattr(pm, "_GEN_TOKEN_BUDGET", total)
    before = actions(proxy._sessions[sid]["trajectory_recorder"].export())
    with pytest.raises(pm._CompactionBudgetExceeded):
        await proxy._compute_prompt_ids(proxy._sessions[sid], msg, None, sid)
    assert actions(proxy._sessions[sid]["trajectory_recorder"].export()) == before
    assert proxy._sessions["root"]["context_overflow"]


@pytest.mark.asyncio
@pytest.mark.parametrize("problem", ["multimodal", "missing_logprobs"])
async def test_incomplete_child_invalidates_whole_episode(
    proxy, monkeypatch, tmp_path, problem
):
    monkeypatch.setenv("HARBOR_TRAIN_SUBAGENTS", "1")
    await start(proxy)
    sid, msg = await child(proxy)
    await sample(proxy, sid, msg, "child")
    state = proxy._sessions[sid]
    if problem == "multimodal":
        state["disable_proxy_trajectory"] = True
    else:
        state["trajectory_recorder"].current.response_logprobs = None
    meta = await proxy.pop_session("root")
    assert meta["disable_proxy_trajectory"] and meta["subagent_archive_incomplete"]
    loop = BuiltinSWEAgentLoop.__new__(BuiltinSWEAgentLoop)
    loop.trajectory_selection = "all"
    loop.base_trials_dir = str(tmp_path)
    definition = NS(name="test", protocol="openai")
    loop.harnesses = {"test": definition}
    loop._tool_parser_name = "hermes"
    out = loop._outputs_from_segments(
        meta, 1.0, "agent_completed", {}, {}, NS(definition=definition, source="test")
    )
    assert not any(out.response_mask)
    assert out.reward_score == 1.0


@pytest.mark.asyncio
async def test_child_summary_quote_does_not_reanchor_parent(proxy, monkeypatch):
    monkeypatch.setenv("HARBOR_TRAIN_SUBAGENTS", "1")
    await start(proxy)
    sid, msg = await child(proxy)
    text = "child summary " * 30
    await sample(proxy, sid, msg, text)
    assert not proxy._is_compaction_continuation(
        "root", [SYSTEM, {"role": "user", "content": text}]
    )
    assert proxy._main_identity["root"]["first_user"] == "original task"


@pytest.mark.asyncio
async def test_parent_cleanup_closes_children_and_late_completions(proxy, monkeypatch):
    monkeypatch.setenv("HARBOR_TRAIN_SUBAGENTS", "1")
    await start(proxy)
    sid, msg = await child(proxy)
    state = proxy._sessions[sid]
    await proxy.pop_session("root")
    assert state["recorder_closed"] and not proxy._sessions
    out = NS(
        token_ids=[1],
        log_probs=[-0.1],
        routed_experts=None,
        extra_fields={},
        num_preempted=0,
        stop_reason="stop",
    )
    with pytest.raises(asyncio.CancelledError, match="closed"):
        await proxy._finalize_generation(sid, state, out, 1)
    assert not proxy._sessions


def test_sdk_spawn_switch_is_independent_of_training(monkeypatch):
    monkeypatch.setenv("HARBOR_OPENHANDS_ENABLE_SUBAGENT", "1")
    monkeypatch.delenv("HARBOR_TRAIN_SUBAGENTS", raising=False)
    cfg = {}
    definition = HarnessDefinition("openhands_sdk", "openai", {})
    inject_harness_endpoint(
        cfg,
        definition,
        openai_base="http://proxy/sess/s/v1",
        anthropic_base="http://proxy/sess/s",
        openai_api_key="test",
        anthropic_api_key="test",
    )
    assert cfg["agent"]["kwargs"]["enable_subagent"] is True


def test_training_children_requires_all_segments(monkeypatch, tmp_path):
    from verl_patch.agent_loop import builtin_swe_agent_loop as module

    monkeypatch.setenv("HARBOR_TRAIN_SUBAGENTS", "1")
    monkeypatch.setattr(module, "HARBOR_AVAILABLE", True)
    monkeypatch.setattr(
        BuiltinSWEAgentLoop.__mro__[1], "__init__", lambda *args, **kwargs: None
    )
    x = BuiltinSWEAgentLoop.__new__(BuiltinSWEAgentLoop)
    x.rollout_config = NS(prompt_length=100, response_length=100)
    with pytest.raises(ValueError, match="requires trajectory_selection=all"):
        BuiltinSWEAgentLoop.__init__(
            x,
            harbor_cfg={"agent": {"import_path": "example:Agent"}},
            tool_parser="hermes",
            max_consecutive_no_tool=0,
            trials_dir=str(tmp_path),
            max_retries=1,
            trajectory_selection="longest",
        )


@pytest.mark.asyncio
async def test_child_summary_is_budgeted_once_and_exported_under_parent(
    proxy, monkeypatch
):
    monkeypatch.setenv("HARBOR_TRAIN_SUBAGENTS", "1")
    parent = await start(proxy)
    sid, msg = await child(proxy)
    first = await sample(proxy, sid, msg, "child execution")
    summ, _ = await proxy._open_summarizer_session(sid)
    turn = await sample(
        proxy,
        summ,
        [SYSTEM, {"role": "user", "content": "summarize child"}],
        "child summary",
    )
    await proxy._archive_summary_segment(proxy._sessions[summ])
    await proxy._archive_summary_segment(proxy._sessions[summ])
    total = len(parent[1]) + len(first[1]) + len(turn[1])
    assert proxy._sessions["root"]["gen_tokens_total"] == total
    meta = await proxy.pop_session("root")
    parts = [s for s in meta["trajectory_segments"] if s.get("kind") == "subagent"]
    assert actions(parts) == expected(first) + expected(turn)
    assert parts[-1]["subagent_segment_kind"] == "summary"
    assert not proxy._sessions and not proxy._sub_session_keys


@pytest.mark.asyncio
async def test_hidden_helper_calls_stay_untrained(proxy, monkeypatch):
    monkeypatch.setenv("HARBOR_TRAIN_SUBAGENTS", "1")
    parent = await start(proxy)
    messages = [
        {
            "role": "system",
            "content": "You are a title generator. You output ONLY a thread title.",
        },
        TASK,
    ]
    sid, _ = await proxy._route_subagent_session("root", messages)
    assert proxy._sessions[sid]["disable_proxy_trajectory"]
    meta = await proxy.pop_session("root")
    assert actions(meta["trajectory_segments"]) == expected(parent)
