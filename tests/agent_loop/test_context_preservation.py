"""Long inputs must survive the actual shared AgentLoopBase render path."""
import asyncio
from types import SimpleNamespace
import pytest
import verl.experimental.agent_loop.agent_loop as base_agent_loop
from verl_patch.agent_loop.builtin_swe_agent_loop import BuiltinSWEAgentLoop

@pytest.mark.asyncio
@pytest.mark.parametrize("prompt_count", [55000, 90000])
async def test_rollout_template_preserves_initial_and_rebuilt_context(monkeypatch, prompt_count):
    """The shared render path must preserve long inputs even without compaction."""
    expected = list(range(prompt_count))
    seen_kwargs = []

    def render(tokenizer, messages, **kwargs):
        seen_kwargs.append(kwargs)
        return expected[:kwargs["max_length"]] if kwargs.get("truncation") else expected

    monkeypatch.setattr(base_agent_loop, "apply_chat_template", render)
    loop = BuiltinSWEAgentLoop.__new__(BuiltinSWEAgentLoop)
    loop.loop = asyncio.get_running_loop()
    loop.processor = None
    loop.tokenizer = object()
    loop.system_prompt = []
    loop.rollout_config = SimpleNamespace(prompt_length=40000)
    loop.apply_chat_template_kwargs = {"truncation": True, "max_length": 40000}
    actual = await loop.apply_chat_template([{"role": "user", "content": "long history"}])
    assert actual == expected
    assert seen_kwargs[0]["truncation"] is False
    assert loop.apply_chat_template_kwargs["truncation"] is True


@pytest.mark.asyncio
async def test_other_loops_keep_the_existing_prompt_column_cap(monkeypatch):
    expected = list(range(55000))
    monkeypatch.setattr(base_agent_loop, 'apply_chat_template', lambda *args, **kwargs: expected)
    loop = BuiltinSWEAgentLoop.__new__(BuiltinSWEAgentLoop)
    loop.preserve_chat_template_context = False
    loop.loop = asyncio.get_running_loop()
    loop.processor = None
    loop.tokenizer = object()
    loop.system_prompt = []
    loop.rollout_config = SimpleNamespace(prompt_length=40000)
    loop.apply_chat_template_kwargs = {}
    assert await loop.apply_chat_template([]) == expected[-40000:]
