import asyncio
from types import SimpleNamespace

import pytest

from verl_patch.agent_loop.vllm_chat_completion_proxy import (
    _messages_before_first_assistant,
    _messages_fingerprint,
    _VLLMChatCompletionsProxy,
)


def _proxy_for_routing():
    proxy = object.__new__(_VLLMChatCompletionsProxy)
    proxy._sessions = {}
    proxy._sessions_lock = asyncio.Lock()
    proxy._session_state_locks = {}
    proxy._session_generation_locks = {}
    proxy._session_attempts = {}
    proxy._main_initial_messages = {}
    proxy._sub_session_keys = {}
    proxy._session_scheduler = SimpleNamespace(
        finish_session=lambda *args, **kwargs: _done(),
    )
    return proxy


async def _done():
    return None


@pytest.mark.asyncio
async def test_subagent_routing_requires_exact_initial_message_prefix():
    proxy = _proxy_for_routing()
    sid = "session"
    main = [
        {"role": "system", "content": "same system"},
        {"role": "user", "content": "main task"},
    ]

    assert await proxy._route_subagent_session(sid, main) == (sid, False)

    # OpenCode Task sub-agents can have the exact same system prompt. Their
    # distinct opening user message must still isolate them from the main turn.
    sub = [
        {"role": "system", "content": "same system"},
        {"role": "user", "content": "explore this separately"},
    ]
    sub_key, is_subagent = await proxy._route_subagent_session(sid, sub)
    assert is_subagent is True
    assert sub_key == f"{sid}::sub::{_messages_fingerprint(sub)}"
    assert proxy._sessions[sub_key]["disable_proxy_trajectory"] is True

    other_sub = [
        {"role": "system", "content": "same system"},
        {"role": "user", "content": "search this separately"},
    ]
    other_key, other_is_subagent = await proxy._route_subagent_session(sid, other_sub)
    assert other_is_subagent is True
    assert other_key != sub_key

    # A later main request has a longer history, but the opening prefix is the
    # same and must continue using the bare trainable session.
    main_later = main + [
        {"role": "assistant", "content": "working"},
        {"role": "user", "content": "continue"},
    ]
    assert await proxy._route_subagent_session(sid, main_later) == (sid, False)

    # The same sub-agent's later request keeps a stable derived key.
    sub_later = sub + [
        {"role": "assistant", "content": "findings"},
        {"role": "user", "content": "more"},
    ]
    assert await proxy._route_subagent_session(sid, sub_later) == (sub_key, True)


@pytest.mark.asyncio
async def test_hidden_opencode_prompt_isolated_and_pop_cleans_routing_state():
    proxy = _proxy_for_routing()
    sid = "session"
    hidden = [
        {
            "role": "system",
            "content": "You are a title generator. You output ONLY a thread title.",
        },
        {"role": "user", "content": "name this thread"},
    ]
    hidden_key, is_subagent = await proxy._route_subagent_session(sid, hidden)
    assert is_subagent is True
    assert hidden_key.startswith(f"{sid}::hidden::")
    assert proxy._sessions[hidden_key]["disable_proxy_trajectory"] is True

    main = [
        {"role": "system", "content": "main"},
        {"role": "user", "content": "task"},
    ]
    await proxy._route_subagent_session(sid, main)
    assert sid in proxy._main_initial_messages

    await proxy.pop_session(sid)
    assert sid not in proxy._main_initial_messages
    assert hidden_key not in proxy._sessions


def test_initial_prefix_stops_before_first_assistant():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "follow-up"},
    ]
    assert _messages_before_first_assistant(messages) == messages[:2]
