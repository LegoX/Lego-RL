"""Exercise multi-round boundaries and episode GAE on the actual upstream path."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage
from verl.utils import tensordict_utils as tu
from verl.utils.episode_segments import (
    EPISODE_INDEX,
    PADDING_KEY,
    expand_episode_segments,
    validate_episode_training,
)
from verl_patch.agent_loop import vllm_chat_completion_proxy as pm
from test_episode_training import config, envelope, output, segment_batch
from test_native_compaction import sample
from test_vllm_chat_completion_proxy import _proxy_for_routing


def sao_config():
    cfg = config()
    cfg.algorithm.adv_estimator = "gae"
    cfg.critic = {
        "enable": True,
        "strategy": "veomni",
        "loss_agg_mode": "token-mean",
        "use_dynamic_bsz": True,
        "ppo_mini_batch_size": 2,
        "ppo_epochs": 2,
        "data_loader_seed": 42,
        "shuffle": False,
    }
    cfg.actor_rollout_ref.rollout.n = 1
    return cfg


def test_gae_envelope_keeps_queue_episode_count_and_training_segment_count():
    cfg = sao_config()
    validate_episode_training(cfg)
    batch = envelope(
        [[output("a", 1, 10), output("a", 1, 11)], [output("b", 0, 20)]],
        ["q1", "q2"],
    )
    assert len(batch) == 2
    batch.meta_info.pop("metrics")
    batch = expand_episode_segments(batch, cfg)
    assert batch.meta_info["num_rollout_episodes"] == 2
    assert batch.meta_info["num_training_segments"] == 3
    assert batch.batch[EPISODE_INDEX][:3].tolist() == [0, 0, 1]
    assert batch.batch[PADDING_KEY].tolist() == [False, False, False, True]
    assert batch.batch["response_mask"][-1].count_nonzero() == 0


def test_cross_segment_sao_matches_historical_policy_discount_and_preserves_critic_returns():
    data = DataProto.from_dict(
        tensors={
            "token_level_rewards": torch.tensor([[0., 0., 1.], [0., 1., 0.]]),
            "response_mask": torch.tensor([[1., 0., 1.], [1., 1., 0.]]),
            "values": torch.zeros(2, 3),
        },
        non_tensors={
            "tokens_after": np.array([2, 0]),
            "trial_gen_tokens": np.array([4, 4]),
        },
    )
    cfg = OmegaConf.create({
        "length_adaptive_lam_alpha": 1.5,
        "cross_segment_gae": True,
        "cross_segment_lam_mode": "trial",
        "lam_critic": 1.,
        "gae_whiten_advantages": False,
    })
    result = compute_advantage(data, "gae", gamma=1., lam=1., config=cfg)
    lam = 1 - 1 / (1.5 * 4)
    expected = torch.tensor([lam**3, lam**2, lam, 1.])
    active = result.batch["response_mask"].bool()
    torch.testing.assert_close(result.batch["advantages"][active], expected)
    torch.testing.assert_close(result.batch["returns"][active], torch.ones(4))
    # An observation is never counted as another generated token.
    assert result.batch["advantages"][0, 0] == pytest.approx(lam**3)


def test_critic_updates_whole_episodes_for_each_epoch_and_advances_scheduler_once(monkeypatch):
    cfg = sao_config()
    batch = segment_batch([3, 1, 2], cfg)
    calls = []
    monkeypatch.setattr("verl.trainer.ppo.ray_trainer.left_right_2_no_padding", lambda data: data)

    def train(data):
        real = ~data[PADDING_KEY]
        calls.append((sorted(data[EPISODE_INDEX][real].tolist()), tu.get(data, "update_lr_scheduler")))
        assert tu.get(data, "epochs") == 1
        assert tu.get(data, "dataloader_kwargs") == {"shuffle": False}
        result = tu.get_tensordict({}, {"metrics": {"mfu": 0., "vf_loss": .5}})
        return SimpleNamespace(get=lambda: result)

    trainer = object.__new__(RayPPOTrainer)
    trainer.config = cfg
    # TrainingWorker registers the train mesh, unlike ActorRolloutRefWorker's actor mesh.
    trainer.critic_wg = SimpleNamespace(train_mini_batch=train, _dispatch_info={"train": [0, 0, 1, 1]})
    result = trainer._update_critic(batch)
    assert calls == [([0, 0, 0, 1], False), ([2, 2], False),
                     ([0, 0, 0, 1], False), ([2, 2], True)]
    assert result.meta_info["metrics"]["critic/episode_optimizer_steps"] == 4
    assert result.meta_info["metrics"]["critic/vf_loss"] == [.5] * 4


@pytest.mark.asyncio
async def test_user_rounds_keep_original_actions_and_total_generation_budget(monkeypatch):
    monkeypatch.setattr(pm, "_GEN_TOKEN_BUDGET", 0)
    proxy = _proxy_for_routing()
    proxy._processor = None
    proxy._assistant_eos_token_id = None
    proxy._assistant_eos_tail = []

    async def render(messages, tools=None, **kwargs):
        return [ord(c) for row in messages for c in row["role"] + ":" + row["content"]]

    proxy._agent_loop = SimpleNamespace(apply_chat_template=render)
    await proxy.open_session("episode", harness_name="claude_code")
    await proxy.begin_user_round("episode", 1)
    messages = [{"role": "user", "content": "Fix the bug"}]
    first_prompt, first_tokens = await sample(proxy, "episode", messages, "first answer")
    before = proxy._sessions["episode"]["gen_tokens_total"]
    proxy._sessions["episode"]["compaction_count"] = 2
    await proxy.begin_user_round("episode", 2)
    assert proxy._sessions["episode"]["gen_tokens_total"] == before
    assert proxy._sessions["episode"]["compaction_count"] == 2
    resumed = deepcopy(messages) + [
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "Check the edge case"},
    ]
    second_prompt, second_tokens = await sample(proxy, "episode", resumed, "second answer")
    result = await proxy.pop_session("episode")
    segments = result["trajectory_segments"]
    assert len(segments) == 2
    for seg, prompt, tokens, number in zip(
        segments, [first_prompt, second_prompt], [first_tokens, second_tokens], [1, 2], strict=True
    ):
        assert seg["prompt_ids"] == prompt
        assert [token for token, active in zip(seg["response_ids"], seg["response_mask"], strict=True) if active] == tokens
        assert seg["response_logprobs"] == [-.25] * len(tokens)
        assert seg["user_round_id"] == number
    assert sum(sum(s["response_mask"]) for s in segments) == len(first_tokens + second_tokens)
    assert result["gen_tokens_total"] == len(first_tokens + second_tokens)


@pytest.mark.parametrize("override", [
    {"enable": False}, {"strategy": "megatron"}, {"loss_agg_mode": "seq-mean-token-mean"},
])
def test_unsupported_episode_critic_configuration_fails_before_training(override):
    cfg = sao_config()
    cfg.critic.update(override)
    with pytest.raises(ValueError, match="critic"):
        validate_episode_training(cfg)
