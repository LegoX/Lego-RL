"""CPU integration tests using the real tokenizer, worker, filter and GSPO loss.

Uses synthetic boundary cases and the original failed trajectory; no remote
Harbor trial or distributed GPU training is launched.
"""
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer
from verl.experimental.agent_loop.agent_loop import AgentLoopWorker, AgentLoopOutput
from verl.experimental.agent_loop.sequence_layout import (
    normalize_training_output, training_tensor_widths, TrainingSequenceOverflow,
)
from verl_patch.agent_loop.builtin_swe_agent_loop import BuiltinSWEAgentLoop
from verl.trainer.ppo.trajectory_filter import apply_trajectory_filter, TerminationReason
from verl.trainer.ppo.core_algos import compute_policy_loss_gspo
from verl.workers.config import ActorConfig

torch.set_num_threads(4)
ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(autouse=True)
def enable(monkeypatch):
    monkeypatch.setenv('VERL_AGENT_FULL_SEQUENCE_TRAINING', '1')

@pytest.fixture(scope='session')
def tokenizer():
    path = os.environ.get('FULLSEQ_TOKENIZER_PATH')
    return AutoTokenizer.from_pretrained(path, local_files_only=True) if path else NS(pad_token_id=0)


def config():
    return NS(prompt_length=40000, response_length=120000, max_model_len=160000)


def worker(tokenizer):
    w = AgentLoopWorker.__new__(AgentLoopWorker)
    w.rollout_config = config()
    w.tokenizer = tokenizer
    w.processor = None
    w.mm_processor_kwargs = {}
    w.reward_loop_worker_handles = None
    w.distillation_enabled = False
    return w


def loop(tmp_path):
    x = BuiltinSWEAgentLoop.__new__(BuiltinSWEAgentLoop)
    x.rollout_config = config()
    x.prompt_length, x.response_length = 40000, 120000
    x.base_trials_dir = str(tmp_path)
    x.harnesses = {'openhands_sdk': NS()}
    x._tool_parser_name = 'qwen3_coder'
    x._max_turns = 200
    x.trajectory_selection = 'all'
    return x


def raw(plen, rlen, routes=True):
    mask = [int(i % 3 != 1) for i in range(rlen)]
    return dict(prompt_ids=[3 + i % 50 for i in range(plen)],
        response_ids=[4 + i % 50 for i in range(rlen)], response_mask=mask,
        response_logprobs=[-.1 - i % 7 * .01 if mask[i] else 0. for i in range(rlen)],
        response_routing=[[[i % 250 + 1, (i + 2) % 250 + 1]] if mask[i] else None for i in range(rlen)] if routes else [None]*rlen,
        num_turns=2)


def export(x, s):
    result = x._outputs_from_segments(
        {'episode_id': 'real', 'trajectory_segments': [s]},
        1., TerminationReason.AGENT_COMPLETED, {}, {'raw_prompt': []}, selection())
    return result[0] if isinstance(result, list) else result


def pad(w, output):
    return asyncio.run(w._agent_loop_postprocess(output, False, raw_prompt=[]))


def batch(w, outputs):
    return w._postprocess([pad(w, o) for o in outputs], input_non_tensor_batch={'uid': np.array(['same'] * len(outputs), dtype=object)})


def selection():
    return NS(definition=NS(name='openhands_sdk', protocol='openai'), source='test')

@pytest.mark.parametrize('plen,rlen', [(8114,135772),(1,159999),(159999,1),(60000,100000),(4,6)])
def test_boundaries_and_actual_worker_alignment(tokenizer,tmp_path,plen,rlen):
    s=raw(plen,rlen); out=export(loop(tmp_path),s)
    w=worker(tokenizer); padded=pad(w,out); total=plen+rlen
    assert tuple(padded.input_ids.shape)==(1,160000)
    assert tuple(padded.prompt_ids.shape)==(1,1)
    assert tuple(padded.response_ids.shape)==(1,159999)
    assert padded.input_ids[0,:total].tolist()==s['prompt_ids']+s['response_ids']
    assert padded.attention_mask.sum().item()==total
    assert padded.position_ids[0,:total].tolist()==list(range(total))
    assert padded.response_mask[0,:total-1].tolist()==[0]*(plen-1)+s['response_mask']
    torch.testing.assert_close(padded.response_logprobs[0,:total-1],torch.tensor([0.]*(plen-1)+s['response_logprobs']))
    np.testing.assert_array_equal(padded.routed_experts[0,:total].numpy(),out.routed_experts)
    # Router replay uses the forward at action_position-1, not the action position.
    mask=np.array(s['response_mask'],dtype=bool)
    action_positions=plen+np.flatnonzero(mask)
    assert np.all(out.routed_experts[action_positions-1].any(axis=(1,2)))
    assert len(out.prompt_ids)==plen  # normalization must not mutate archived output


def test_legacy_mode_keeps_its_existing_widths(monkeypatch):
    monkeypatch.delenv('VERL_AGENT_FULL_SEQUENCE_TRAINING')
    assert training_tensor_widths(config())==(40000,120000)
    s=raw(4,6);o=AgentLoopOutput(**{k:s[k] for k in ('prompt_ids','response_ids','response_mask','response_logprobs')},metrics={})
    assert normalize_training_output(o,config()) is o


def test_mixed_rows_and_reward_target(tokenizer,tmp_path):
    x=loop(tmp_path);w=worker(tokenizer)
    outputs=[export(x,raw(8114,135772)),export(x,raw(60000,30)),x._make_empty_output({'raw_prompt':[]}, reason='env_setup_failed')]
    b=batch(w,outputs)
    assert tuple(b.batch['input_ids'].shape)==(3,160000)
    assert tuple(b.batch['rollout_log_probs'].shape)==(3,159999)
    assert b.batch['rm_scores'].sum(dim=-1).tolist()==[1.,1.,0.]
    for i in range(2):
        j=b.batch['response_mask'][i].nonzero()[-1].item()
        assert b.batch['rm_scores'][i,j].item()==1.
    b.batch['token_level_rewards']=b.batch['rm_scores'].clone()
    b,metrics=apply_trajectory_filter(b,{'enable':True})
    assert b.non_tensor_batch['trajectory_filter_mask'].tolist()==[False,False,True]
    assert b.non_tensor_batch['uid'][2]!='same'


def test_true_overflow_is_zero_loss_trial_not_partial_reward(tokenizer,tmp_path):
    x=loop(tmp_path);s=raw(10,159991,routes=False)
    o = x._outputs_from_segments(
        {'episode_id': 'over-budget', 'trajectory_segments': [raw(3,5), s]},
        1., TerminationReason.AGENT_COMPLETED, {}, {'raw_prompt':[]}, selection())
    assert o.extra_fields['termination_reason']==TerminationReason.TRAINING_SEQUENCE_OVERFLOW
    assert not any(o.response_mask)
    assert o.reward_score==1.
    assert o.metrics.training_overflow_max_tokens==160001
    w=worker(tokenizer);b=batch(w,[export(x,raw(3,5)),o])
    b.batch['token_level_rewards']=b.batch['rm_scores'].clone()
    b,metrics=apply_trajectory_filter(b,{'enable':True})
    assert metrics['trajectory_filter/reason/training_sequence_overflow']==1
    assert b.non_tensor_batch['trajectory_filter_mask'].tolist()==[False,True]
    assert b.batch['token_level_rewards'][1].sum()==0
    assert list((tmp_path/'_dropped').glob('*.jsonl'))


def test_newer_selection_path_accepts_valid_and_rejects_whole_oversized_trial(tmp_path):
    x=loop(tmp_path);x.trajectory_selection='all'
    good=raw(8114,135772)
    meta={'trajectory_segments':[good,raw(5,9)],'episode_id':'e'}
    result=x._outputs_from_segments(meta,1.,TerminationReason.AGENT_COMPLETED,{}, {},selection())
    assert len(result)==2
    assert len(result[0].response_ids)==135772
    meta['trajectory_segments'].append(raw(2,159999,routes=False))
    out=x._outputs_from_segments(meta,1.,TerminationReason.AGENT_COMPLETED,{}, {},selection())
    assert isinstance(out,AgentLoopOutput)
    assert not any(out.response_mask)
    assert out.extra_fields['termination_reason']==TerminationReason.TRAINING_SEQUENCE_OVERFLOW


def test_validation_and_training_have_same_layout(tokenizer,tmp_path):
    w=worker(tokenizer);o=export(loop(tmp_path),raw(60000,9))
    train=pad(w,o)
    val=asyncio.run(w._agent_loop_postprocess(o,True,raw_prompt=[]))
    for key in ('input_ids','response_mask','position_ids','routed_experts'):
        assert torch.equal(getattr(train,key),getattr(val,key))


def test_gspo_forward_backward_matches_original_causal_sequence(tokenizer,tmp_path):
    """Actual GSPO loss + tiny causal model; not a 35B distributed GPU update."""
    torch.manual_seed(7)
    s=raw(7,19);out=export(loop(tmp_path),s)
    normalized=normalize_training_output(out,config())
    ids=torch.tensor([s['prompt_ids']+s['response_ids']])
    model=torch.nn.Sequential(torch.nn.Embedding(64,8),torch.nn.Linear(8,64)).double()
    def lp():
        # Context-dependent causal logits, cheap enough for a CPU regression.
        embeddings=model[0](ids)
        hidden=embeddings.cumsum(1)/torch.arange(1,ids.shape[1]+1).view(1,-1,1)
        logits=model[1](hidden[:,:-1])
        return logits.log_softmax(-1).gather(-1,ids[:,1:,None]).squeeze(-1)
    cfg=ActorConfig(strategy='megatron', rollout_n=8, use_dynamic_bsz=True, clip_ratio=.2, global_batch_info={'global_batch_size':1, 'dp_size':1})
    old=lp().detach()
    def train(mask,start):
        model.zero_grad();probs=lp()[:,start:]
        loss,_=compute_policy_loss_gspo(old[:,start:],probs,torch.ones_like(probs),
            torch.tensor([mask],dtype=probs.dtype),config=cfg)
        loss.backward();return loss.detach(),[p.grad.clone() for p in model.parameters()]
    ref_loss,ref_grads=train(s['response_mask'],len(s['prompt_ids'])-1)
    loss,grads=train(normalized.response_mask,0)
    torch.testing.assert_close(loss,ref_loss,rtol=1e-12,atol=1e-12)
    for a,b in zip(grads,ref_grads):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)


def test_real_failure_replay(tokenizer,tmp_path):
    path=Path(os.environ.get('FULLSEQ_FAILURE_PATH', str(ROOT/'failure-segment.npz')))
    if not path.exists():pytest.skip('Set FULLSEQ_FAILURE_PATH to the original trajectory fixture')
    z=np.load(path);s={k:z[k].tolist() for k in ('prompt_ids','response_ids','response_mask','response_logprobs')}
    # The archive's full-position routing is independently reconstructed during extraction.
    o=AgentLoopOutput(**s,routed_experts=z['routed_experts'],metrics={},reward_score=1.)
    assert len(s["prompt_ids"])+len(s["response_ids"]) <= 160000
    assert len(o.prompt_ids)==8114 and len(o.response_ids)==135772
    w=worker(tokenizer);p=pad(w,o);total=143886
    assert p.input_ids[0,:total].tolist()==s['prompt_ids']+s['response_ids']
    assert p.response_mask.sum()==30630
    assert p.response_mask[0,:8113].sum()==0
    torch.testing.assert_close(p.response_logprobs[0,8113:total-1],torch.tensor(s['response_logprobs']))
    np.testing.assert_array_equal(p.routed_experts[0,:total].numpy(),z['routed_experts'])
    b=w._postprocess([p],input_non_tensor_batch={'uid':np.array(['actual'],dtype=object)})
    assert b.batch['rm_scores'].sum()==1
    assert p.metrics.training_original_prompt_tokens==8114
    assert p.metrics.training_original_response_positions==135772
    assert p.metrics.training_sequence_tokens==143886


def test_metrics_report_logical_lengths_and_total_budget(tokenizer,tmp_path):
    from verl.trainer.ppo.metric_utils import compute_data_metrics
    w=worker(tokenizer);x=loop(tmp_path)
    b=batch(w,[export(x,raw(8114,135772)),export(x,raw(60000,100000))])
    for key in ('token_level_scores','token_level_rewards','advantages','returns'):
        b.batch[key]=b.batch['rm_scores'].clone()
    m=compute_data_metrics(b,use_critic=False)
    assert m['prompt_length/mean']==(8114+60000)/2
    assert m['response_length/mean']==(135772+100000)/2
    assert m['prompt_length/clip_ratio']==.5
    assert m['response_length/clip_ratio']==.5
    assert m['response_length_non_aborted/clip_ratio']==.5
    assert m['training_sequence_length/max']==160000
    assert m['training_sequence_length/at_capacity_ratio']==.5


def test_overflow_does_not_hide_alignment_errors(tmp_path):
    x = loop(tmp_path)
    s = raw(2, 159999, routes=False)
    s['response_logprobs'].pop()
    with pytest.raises(ValueError, match='alignment'):
        export(x, s)


def test_zero_loss_first_row_keeps_later_routing(tokenizer, tmp_path):
    x, w = loop(tmp_path), worker(tokenizer)
    empty = x._make_empty_output({'raw_prompt': []}, reason='env_setup_failed')
    good = export(x, raw(3, 5))
    b = batch(w, [empty, good])
    assert b.batch['routed_experts'][0].count_nonzero() == 0
    assert b.batch['routed_experts'][1].count_nonzero() > 0


def test_missing_trainable_routes_remain_an_error(tokenizer, tmp_path):
    x, w = loop(tmp_path), worker(tokenizer)
    a = export(x, raw(3, 5))
    b = export(x, raw(3, 5))
    b.routed_experts = None
    with pytest.raises(ValueError, match='missing routing'):
        batch(w, [a, b])
