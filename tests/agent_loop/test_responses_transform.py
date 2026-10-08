import pytest

from verl_patch.agent_loop.responses_transform import (
    decrypt_reasoning, encrypt_reasoning, openai_message_to_responses,
    responses_sse_events, responses_to_openai_messages,
)


def test_reasoning_text_parallel_tools_round_trip_as_one_assistant_turn():
    message = {'role': 'assistant', 'reasoning_content': 'think α', 'content': 'checking',
               'tool_calls': [{'id': f'c{i}', 'type': 'function',
                               'function': {'name': 'exec_command', 'arguments': '{"cmd":"ls"}'}}
                              for i in range(2)]}
    response = openai_message_to_responses(message, 'tool_calls', 'model', 20, 6, 'trial')
    items = [{'role': 'developer', 'content': 'rules'}, {'role': 'user', 'content': 'task'}]
    items += response['output'] + [{'type': 'function_call_output', 'call_id': f'c{i}', 'output': 'ok'} for i in range(2)]
    messages, tools = responses_to_openai_messages({'input': items})
    assert messages[0] == {'role': 'system', 'content': 'rules'}
    assert messages[2] == message
    assert [m['role'] for m in messages] == ['system', 'user', 'assistant', 'tool', 'tool']
    assert tools is None


def test_reasoning_of_next_turn_does_not_leak_into_previous_tool_block():
    items = [{'type': 'function_call', 'call_id': 'c1', 'name': 'exec', 'arguments': '{}'},
             {'type': 'function_call_output', 'call_id': 'c1', 'output': 'ok'},
             {'type': 'reasoning', 'encrypted_content': encrypt_reasoning('next reasoning')},
             {'type': 'message', 'role': 'assistant', 'content': 'next answer'}]
    messages, _ = responses_to_openai_messages({'input': items})
    assert 'reasoning_content' not in messages[0]
    assert messages[2] == {'role': 'assistant', 'content': 'next answer', 'reasoning_content': 'next reasoning'}


def test_function_schemas_preserved_and_server_side_tools_not_advertised():
    schema = {'type': 'object', 'properties': {'session_id': {'type': 'number'}}}
    messages, tools = responses_to_openai_messages({'input': 'task', 'instructions': 'rules', 'tools': [
        {'type': 'function', 'name': 'write_stdin', 'parameters': schema}, {'type': 'web_search'}, {'type': 'namespace'}]})
    assert messages == [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'task'}]
    assert len(tools) == 1 and tools[0]['function']['parameters'] == schema


@pytest.mark.parametrize('finish,event', [('stop', 'response.completed'), ('length', 'response.incomplete')])
def test_sse_terminal_status_and_usage(finish,event):
    response = openai_message_to_responses({'role': 'assistant', 'content': 'answer'}, finish, 'model', 9, 4, 's')
    events = list(responses_sse_events(response))
    assert events[-1][0] == event
    assert events[-1][1]['response']['usage']['total_tokens'] == 13
    assert [payload['sequence_number'] for _,payload in events] == list(range(len(events)))


def test_reasoning_round_trip_and_foreign_values():
    assert decrypt_reasoning(encrypt_reasoning('reason α')) == 'reason α'
    for value in (None, 'foreign opaque value', 'legorl:!'):
        assert decrypt_reasoning(value) == ''
