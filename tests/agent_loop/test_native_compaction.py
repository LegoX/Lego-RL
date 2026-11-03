"""Four native compaction shapes through the actual recorder and session router."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from verl_patch.agent_loop import vllm_chat_completion_proxy as pm
from verl_patch.agent_loop.harness import compaction_profile_for, inject_harness_endpoint, HarnessDefinition
from test_vllm_chat_completion_proxy import _proxy_for_routing

NAMES = ['claude_code', 'codex', 'opencode', 'openhands_sdk']
SYSTEM = {'role': 'system', 'content': 'shared system'}
TASK = {'role': 'user', 'content': 'original task'}

@pytest.fixture
def proxy(monkeypatch):
    monkeypatch.setattr(pm, '_COMPACTION_MAX', 3)
    monkeypatch.setattr(pm, '_GEN_TOKEN_BUDGET', 0)
    p = _proxy_for_routing()
    p._processor = None
    p._assistant_eos_token_id = None
    p._assistant_eos_tail = []
    async def render(messages, tools=None, **kwargs):
        return [ord(c) for m in messages for c in m['role']+':'+str(m['content'])]
    p._agent_loop = SimpleNamespace(apply_chat_template=render)
    return p

async def sample(p, sid, messages, text, tools=None):
    s = p._sessions[sid]
    prompt = await p._compute_prompt_ids(s, messages, tools, sid)
    ids = [ord(c) for c in text]
    out = SimpleNamespace(token_ids=ids, log_probs=[-.25]*len(ids), routed_experts=None,
                          extra_fields={}, num_preempted=0, stop_reason='stop')
    await p._finalize_generation(sid, s, out, len(prompt))
    s['messages_snapshot'] = deepcopy(messages)+[{'role':'assistant','content':text}]
    s['last_generation_snapshot'] = deepcopy(s['messages_snapshot'])
    s['last_assistant_text'], s['last_assistant_tokens'] = text, len(ids)
    return prompt, ids

def actions(segments):
    return [(s['prompt_ids']+s['response_ids'][:i], t, s['response_logprobs'][i])
            for s in segments for i,t in enumerate(s['response_ids']) if s['response_mask'][i]]

def expected(turn):
    prompt, ids = turn
    return [(prompt+ids[:i],t,-.25) for i,t in enumerate(ids)]

@pytest.mark.parametrize('name',NAMES)
def test_profile_configures_only_requested_harness(name, monkeypatch):
    monkeypatch.setenv('HARBOR_CONTEXT_BUDGET_TOKENS','96000')
    card=compaction_profile_for(name)
    d=HarnessDefinition(name,'anthropic' if name=='claude_code' else 'openai',{},compaction=card)
    cfg={'agent':{'kwargs':{},'env':{}}}
    inject_harness_endpoint(cfg,d,openai_base='http://proxy/sess/s/v1',anthropic_base='http://proxy/sess/s',
                            openai_api_key='test',anthropic_api_key='test',summarizer_base='http://proxy/sess/s/summ/v1')
    env,kw=cfg['agent']['env'],cfg['agent']['kwargs']
    if name=='claude_code':assert env['CLAUDE_CODE_AUTO_COMPACT_WINDOW']=='96000'
    if name=='codex':assert env['CODEX_AUTO_COMPACT_TOKEN_LIMIT']=='86400'
    if name=='openhands_sdk':
        assert kw['condenser']=='summarizer'
        assert kw['condenser_llm_base_url']=='http://proxy/sess/s/summ/v1'
    if name=='opencode':
        import json
        doc=json.loads(env['OPENCODE_CONFIG_CONTENT'])
        assert doc['agent']['compaction']['model'].endswith('_summ/vllm_model')

@pytest.mark.asyncio
@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('train_summary',[True,False])
async def test_two_compactions_keep_every_action_under_its_original_context(proxy,monkeypatch,name,train_summary):
    monkeypatch.setattr(pm,'_TRAIN_SUMMARY',train_summary)
    card=compaction_profile_for(name)
    await proxy.open_session('s',compaction=card,harness_name=name)
    messages=[SYSTEM,TASK]
    key,sub=await proxy._route_subagent_session('s',messages)
    assert (key,sub)==('s',False)
    turns=[await sample(proxy,'s',messages,'first execution')]
    want=expected(turns[0])
    generated=len(turns[0][1])
    for n in range(2):
        main=proxy._sessions['s']
        summary=f'summary {n}: '+('retained history '*25)
        summary_messages=main['messages_snapshot']+[{'role':'user','content':'summarize'}]
        if name=='claude_code':
            turn=await sample(proxy,'s',summary_messages,summary)
        else:
            if name=='codex':
                summary_messages=main['messages_snapshot']+[{'role':'user','content':card['summary_request_prefix']+' summarize'}]
                sid,_=await proxy._route_generation_session('s',summary_messages,None)
            else:
                sid,_=await proxy._route_generation_session('s',summary_messages,None,summarizer=True)
            turn=await sample(proxy,sid,summary_messages,summary)
            await proxy._archive_summary_segment(proxy._sessions[sid])
            await proxy._archive_summary_segment(proxy._sessions[sid])
        generated+=len(turn[1])
        if train_summary:want+=expected(turn)
        if name=='opencode':
            messages=[SYSTEM,TASK,{'role':'assistant','content':summary}]
        elif name in ('codex','openhands_sdk'):
            messages=[SYSTEM,TASK,{'role':'user','content':summary}]
        else:
            messages=[SYSTEM,{'role':'user','content':card['markers'][0]+' '+summary}]
        sid,sub=await proxy._route_subagent_session('s',messages)
        assert (sid,sub)==('s',False)
        turn=await sample(proxy,'s',messages,f'execution after compaction {n}')
        want+=expected(turn)
        generated+=len(turn[1])
    meta=await proxy.pop_session('s')
    assert meta['gen_tokens_total']==generated
    assert meta['compaction_count']==2
    assert actions(meta['trajectory_segments'])==want
    assert meta['compaction_shape_mismatch']==0
    assert meta['compaction_summary_missing']==0
    assert sum(s.get('kind')=='summary' for s in meta['trajectory_segments'])==(0 if name=='claude_code' else 2)

@pytest.mark.asyncio
async def test_budget_refusal_and_failed_preparation_do_not_destroy_archive(proxy,monkeypatch):
    await proxy.open_session('s',compaction=compaction_profile_for('claude_code'))
    turn=await sample(proxy,'s',[SYSTEM,TASK],'one')
    s=proxy._sessions['s']
    before=actions(s['trajectory_recorder'].export())
    s['compaction_count']=3
    s['pending_rebuild_cause']='compaction_replace'
    with pytest.raises(pm._CompactionBudgetExceeded):
        await proxy._compute_prompt_ids(s,[SYSTEM,TASK],None,'s')
    assert actions(s['trajectory_recorder'].export())==before==expected(turn)
    assert pm._proxy_error_from_exception(pm._CompactionBudgetExceeded('s','limit')).status==400
    s['compaction_count']=0
    monkeypatch.setattr(pm,'_GEN_TOKEN_BUDGET',s['gen_tokens_total'])
    with pytest.raises(pm._CompactionBudgetExceeded):
        await proxy._compute_prompt_ids(s,[SYSTEM,TASK],None,'s')

@pytest.mark.asyncio
@pytest.mark.parametrize('name',['opencode','openhands_sdk'])
async def test_summary_routes_cannot_bypass_parent_budget(proxy,monkeypatch,name):
    await proxy.open_session('s',compaction=compaction_profile_for(name))
    await sample(proxy,'s',[SYSTEM,TASK],'already generated')
    main=proxy._sessions['s']
    monkeypatch.setattr(pm,'_GEN_TOKEN_BUDGET',main['gen_tokens_total'])
    with pytest.raises(pm._CompactionBudgetExceeded):
        await proxy._route_generation_session('s',[SYSTEM,TASK],None,summarizer=True)
    assert proxy._summ_seq.get('s',0)==0
    assert len(proxy._sessions)==1

@pytest.mark.asyncio
@pytest.mark.parametrize('stream',[False,True])
async def test_summary_http_ingress_records_once_and_replays_sse(proxy,monkeypatch,stream):
    from unittest.mock import AsyncMock
    from aiohttp import web
    import json
    proxy._session_attempt_seq=0
    proxy._max_consecutive_no_tool=0
    proxy._vllm_tool_parser=None
    proxy._tokenizer=SimpleNamespace(decode=lambda ids,**kwargs:'summary')
    proxy._session_scheduler.acquire=AsyncMock(return_value=SimpleNamespace(wait_sec=0,release=AsyncMock()))
    proxy._server_manager=SimpleNamespace(generate=AsyncMock(return_value=SimpleNamespace(
        token_ids=[10,11],log_probs=[-.1,-.2],routed_experts=None,extra_fields={},
        num_preempted=0,stop_reason='stop')))
    await proxy.open_session('s',compaction=compaction_profile_for('openhands_sdk'))
    request=SimpleNamespace(match_info={'session_id':'s'},json=AsyncMock(return_value={
        'model':'policy','messages':[SYSTEM,TASK],'stream':stream}))
    chunks=[]
    class Stream:
        def __init__(self,**kwargs):self.headers=kwargs['headers'];self.status=200
        async def prepare(self,request):pass
        async def write(self,data):chunks.append(data)
        async def write_eof(self):pass
    monkeypatch.setattr(web,'StreamResponse',Stream)
    response=await proxy._handle_chat_summ(request)
    assert response.status==200
    assert proxy._server_manager.generate.await_count==1
    segments=proxy._sessions['s']['segments']
    assert len(segments)==1 and segments[0]['kind']=='summary'
    assert sum(segments[0]['response_mask'])==2
    if stream:
        assert chunks[-1]==b'data: [DONE]\n\n'
        payload=json.loads(chunks[0][6:])
        assert payload['choices'][0]['delta']['content']=='summary'
    else:
        assert json.loads(b''.join(chunks))['choices'][0]['message']['content']=='summary'

@pytest.mark.asyncio
async def test_late_summary_after_parent_export_cannot_attach_to_new_episode(proxy):
    await proxy.open_session('s',compaction=compaction_profile_for('openhands_sdk'))
    sid,_=await proxy._open_summarizer_session('s')
    child=proxy._sessions[sid]
    await sample(proxy,sid,[SYSTEM,TASK],'finished summary')
    exported=await proxy.pop_session('s')
    await proxy.open_session('s',compaction=compaction_profile_for('openhands_sdk'))
    await proxy._archive_summary_segment(child)
    assert not proxy._sessions['s']['segments']
    assert not exported['trajectory_segments']
    out=SimpleNamespace(token_ids=[1],log_probs=[-.1],routed_experts=None,
                        extra_fields={},num_preempted=0,stop_reason='stop')
    with pytest.raises(asyncio.CancelledError,match='closed'):
        await proxy._finalize_generation(sid,child,out,10)
    assert sid not in proxy._sessions
