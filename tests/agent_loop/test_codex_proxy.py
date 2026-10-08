import asyncio
import json
from types import SimpleNamespace

import pytest
from test_vllm_chat_completion_proxy import _proxy_for_routing
from verl_patch.agent_loop.vllm_chat_completion_proxy import _VLLMChatCompletionsProxy

def _proxy_for_generation():
    from unittest.mock import AsyncMock

    proxy = _proxy_for_routing()
    proxy._session_attempt_seq = 0
    proxy._max_consecutive_no_tool = 0
    proxy._vllm_tool_parser = None
    proxy._tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "OK")
    proxy._compute_prompt_ids = AsyncMock(return_value=[1, 2])
    proxy._finalize_generation = AsyncMock()
    proxy._server_manager = SimpleNamespace(generate=AsyncMock(
        return_value=SimpleNamespace(token_ids=[3], stop_reason="stop")))
    proxy._session_scheduler.acquire = AsyncMock(return_value=SimpleNamespace(
        release=AsyncMock(), wait_sec=0))
    return proxy


@pytest.mark.asyncio
@pytest.mark.parametrize("temperature", [1.0, 0.7])
async def test_generation_receives_trial_sampling_and_typed_tools(temperature):
    from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

    proxy = _proxy_for_generation()
    tokenizer = SimpleNamespace(get_vocab=lambda: {"<tool_call>": 1, "</tool_call>": 2})
    proxy._vllm_tool_parser = Qwen3CoderToolParser(tokenizer)
    output_text = (
        '<tool_call>\n<function=probe>\n'
        '<parameter=session_id>2763</parameter>\n'
        '<parameter=tty>false</parameter>\n'
        '<parameter=paths>["a", "b"]</parameter>\n'
        '<parameter=cmd>123</parameter>\n</function>\n</tool_call>'
    )
    proxy._tokenizer.decode = lambda ids, **kwargs: output_text
    tools = [{"type": "function", "function": {
        "name": "probe", "parameters": {"type": "object", "properties": {
            "session_id": {"type": "number"}, "tty": {"type": "boolean"},
            "paths": {"type": "array", "items": {"type": "string"}},
            "cmd": {"type": "string"},
        }}}}]
    await proxy.open_session("trial", harness_name="codex", sampling_params={"temperature": temperature, "top_p": 1.0, "top_k": -1})
    message, reason, _, _ = await proxy._generate_assistant_message(
        "trial", [{"role": "user", "content": "task"}], tools, {"model": "vllm_model"})
    assert reason == "tool_calls"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "session_id": 2763, "tty": False, "paths": ["a", "b"], "cmd": "123"}
    sp = proxy._server_manager.generate.await_args.kwargs["sampling_params"]
    assert sp == {"temperature": temperature, "top_p": 1.0, "top_k": -1, "logprobs": True}


@pytest.mark.parametrize("request_temperature, expected", [(None, 0.7), (0, 0.0), (0.3, 0.3)])
def test_explicit_sampling_overrides_defaults_without_losing_zero(request_temperature, expected):
    defaults = {"temperature": 0.7, "top_p": 0.9, "top_k": 5,
                "max_tokens": 100, "repetition_penalty": 1.1, "seed": 42}
    sp = _VLLMChatCompletionsProxy._translate_sampling_params(
        {"temperature": request_temperature, "max_completion_tokens": 10}, defaults=defaults)
    assert sp["temperature"] == expected
    assert sp["max_tokens"] == 10
    assert sp["repetition_penalty"] == 1.1
    assert sp["seed"] == 42
    assert sp["logprobs"] is True
    assert defaults["temperature"] == 0.7 and defaults["max_tokens"] == 100


@pytest.mark.asyncio
async def test_xml_parser_preserves_numeric_types_without_sampling_defaults():
    from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

    proxy = _proxy_for_generation()
    proxy._vllm_tool_parser = Qwen3CoderToolParser(SimpleNamespace(
        get_vocab=lambda: {"<tool_call>": 1, "</tool_call>": 2}))
    proxy._tokenizer.decode = lambda ids, **kwargs: (
        '<tool_call>\n<function=write_stdin>\n'
        '<parameter=session_id>2763</parameter>\n</function>\n</tool_call>')
    await proxy.open_session("trial", harness_name="codex")
    message, _, _, _ = await proxy._generate_assistant_message(
        "trial", [{"role": "user", "content": "task"}],
        [{"type": "function", "function": {"name": "write_stdin", "parameters": {
            "type": "object", "properties": {"session_id": {"type": "number"}}}}}], {})
    args = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert args["session_id"] == 2763
    assert not isinstance(args["session_id"], str)


@pytest.mark.asyncio
@pytest.mark.parametrize("harness", ["claude_code", "opencode", "openhands_sdk"])
async def test_codex_turn_cap_does_not_limit_other_harnesses(harness):
    proxy = _proxy_for_generation()
    await proxy.open_session("trial", harness_name=harness, max_turns=1)
    for _ in range(3):
        await proxy._generate_assistant_message("trial", [{"role": "user", "content": "task"}], None, {})
    assert proxy._server_manager.generate.await_count == 3
    assert proxy._sessions["trial"]["max_turns"] is None


@pytest.mark.asyncio
async def test_codex_turn_cap_does_not_charge_failed_generation_or_unlimited_sessions():
    proxy = _proxy_for_generation()
    await proxy.open_session("trial", harness_name="codex", max_turns=1)
    proxy._server_manager.generate.side_effect = RuntimeError("temporary backend failure")
    with pytest.raises(RuntimeError, match="temporary backend failure"):
        await proxy._generate_assistant_message("trial", [{"role": "user", "content": "task"}], None, {})
    assert proxy._sessions["trial"]["completed_assistant_turns"] == 0
    proxy._server_manager.generate.side_effect = None
    await proxy._generate_assistant_message("trial", [{"role": "user", "content": "task"}], None, {})
    assert proxy._sessions["trial"]["completed_assistant_turns"] == 1
    await proxy.open_session("unlimited", harness_name="codex", max_turns=0)
    for _ in range(3):
        await proxy._generate_assistant_message("unlimited", [{"role": "user", "content": "task"}], None, {})
    assert proxy._sessions["unlimited"]["turn_limit_reached"] is False


@pytest.mark.asyncio
async def test_sampling_defaults_are_scoped_to_codex_trials_and_subagents():
    proxy = _proxy_for_generation()
    defaults = {'temperature': .7, 'top_p': .9}
    await proxy.open_session('val', harness_name='codex', sampling_params=defaults)
    await proxy.open_session('train', harness_name='codex', sampling_params={'temperature': 1.0})
    await proxy.open_session('other', harness_name='opencode', sampling_params=defaults)
    defaults['temperature'] = 42
    for sid, task in [('val', 'task'), ('train', 'task'), ('val', 'subtask'), ('other', 'task')]:
        await proxy._generate_assistant_message(sid, [{'role': 'user', 'content': task}], None, {})
    calls = proxy._server_manager.generate.await_args_list
    assert [c.kwargs['sampling_params'].get('temperature') for c in calls] == [.7, 1.0, .7, None]
    assert '::sub::' in calls[2].kwargs['request_id']


@pytest.mark.asyncio
async def test_codex_turn_cap_is_shared_by_concurrent_subagents_and_terminal_is_not_sampled():
    proxy = _proxy_for_generation()
    await proxy.open_session('trial', harness_name='codex', max_turns=2)
    main = [{'role': 'user', 'content': 'task'}]
    await proxy._generate_assistant_message('trial', main, None, {})
    replies = await asyncio.gather(*[
        proxy._generate_assistant_message('trial', [{'role': 'user', 'content': task}], None, {})
        for task in ('subtask one', 'subtask two')
    ])
    assert proxy._server_manager.generate.await_count == 2
    assert proxy._finalize_generation.await_count == 2
    assert proxy._sessions['trial']['completed_assistant_turns'] == 2
    assert proxy._sessions['trial']['turn_limit_reached'] is True
    assert sum(reply[2] == reply[3] == 0 for reply in replies) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('harness', ['claude_code', 'opencode', 'openhands_sdk'])
async def test_other_harnesses_keep_existing_parser_request(harness):
    from unittest.mock import Mock
    proxy = _proxy_for_generation()
    parser = Mock()
    parser.extract_tool_calls.return_value = SimpleNamespace(tools_called=False, content=None)
    proxy._vllm_tool_parser = parser
    await proxy.open_session('trial', harness_name=harness)
    await proxy._generate_assistant_message('trial', [{'role': 'user', 'content': 'task'}],
        [{'type': 'function', 'function': {'name': 'probe', 'parameters': {'type': 'object'}}}], {})
    assert parser.extract_tool_calls.call_args.args[1].tools is None


@pytest.mark.asyncio
async def test_responses_http_json_sse_and_input_errors():
    from unittest.mock import AsyncMock
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    proxy = _proxy_for_generation()
    proxy._generate_assistant_message = AsyncMock(return_value=(
        {'role': 'assistant', 'content': 'done'}, 'stop', 12, 3))
    app = web.Application()
    app.router.add_post('/sess/{session_id}/v1/responses', proxy._handle_responses)
    async with TestClient(TestServer(app)) as client:
        url = '/sess/test/v1/responses'
        for stream in (False, True):
            response = await client.post(url, json={'input': 'hello', 'stream': stream, 'max_output_tokens': 42})
            assert response.status == 200
            if stream:
                text = await response.text()
                assert 'event: response.completed' in text and 'done' in text
            else:
                assert (await response.json())['output'][0]['content'][0]['text'] == 'done'
            assert proxy._generate_assistant_message.await_args.args[3]['max_tokens'] == 42
        before = proxy._generate_assistant_message.await_count
        for body in ([], {'input': 1}, {'input': 'hello', 'previous_response_id': 'old'},
                     {'input': [{'role': 'user', 'content': [{'type': 'input_image', 'image_url': 'data:'}]}]}):
            response = await client.post(url, json=body)
            assert response.status == 400
        assert proxy._generate_assistant_message.await_count == before
