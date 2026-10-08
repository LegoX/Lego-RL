"""Exercise the patched verl worker -> training path without a GPU or server."""

import pickle
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopWorker,
    _InternalAgentLoopOutput,
)
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.trainer.ppo.trajectory_filter import apply_trajectory_filter
from verl.utils import tensordict_utils as tu
from verl.utils.episode_segments import (
    EPISODE_INDEX,
    PADDING_KEY,
    SEGMENTS_KEY,
    TOKEN_COUNT,
    expand_episode_segments,
    iter_episode_minibatches,
)
from verl.workers.utils.padding import left_right_2_no_padding

from verl_patch.agent_loop.builtin_swe_agent_loop import BuiltinSWEAgentLoop


def config():
    return OmegaConf.create(
        {
            "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
            "actor_rollout_ref": {
                "actor": {
                    "strategy": "fsdp2",
                    "ppo_mini_batch_size": 2,
                    "ppo_epochs": 1,
                    "data_loader_seed": 42,
                    "shuffle": False,
                    "use_dynamic_bsz": True,
                    "calculate_entropy": False,
                    "entropy_coeff": 0.0,
                },
                "rollout": {"n": 2, "multi_turn": {"enable": True}, "temperature": 1.0},
            },
            "trainer": {"nnodes": 1, "n_gpus_per_node": 2},
        }
    )


def output(episode, reward, token, *, selection="all", reason="agent_completed"):
    return _InternalAgentLoopOutput(
        prompt_ids=torch.tensor([[1, 2]]),
        response_ids=torch.tensor([[token, 0]]),
        input_ids=torch.tensor([[1, 2, token, 0]]),
        position_ids=torch.tensor([[0, 1, 2, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 0]]),
        response_mask=torch.tensor([[1, 0]]),
        response_logprobs=torch.tensor([[-0.2, 0]]),
        reward_score=reward,
        num_turns=1,
        metrics={},
        extra_fields={
            "episode_id": episode,
            "episode_reward": reward,
            "trajectory_selection": selection,
            "termination_reason": reason,
            "segment_index": token,
            "min_global_steps": 0,
            "max_global_steps": 0,
        },
    )


def envelope(groups, uids):
    worker = object.__new__(AgentLoopWorker)
    worker.reward_loop_worker_handles = None
    return worker._postprocess(
        groups, input_non_tensor_batch={"uid": np.array(uids, dtype=object)}
    )


def with_rewards(data):
    data.batch["token_level_rewards"] = data.batch["rm_scores"].clone()
    return data


def advantages(data):
    data = with_rewards(data)
    data, _ = apply_trajectory_filter(data, {"enable": True})
    return compute_advantage(data, "grpo", config=OmegaConf.create({}))


def test_envelope_preserves_episode_count_and_async_uid_until_training(monkeypatch):
    # Only the FlashAttention padding kernel is replaced on CPU; the actual
    # verl TensorDict conversion, worker batching and advantage code run below.
    def cpu_unpad(tensor, mask):
        indices = mask.flatten().nonzero().flatten()
        lengths = mask.sum(-1).to(torch.int32)
        offsets = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
        return tensor.flatten(0, 1)[indices], indices, offsets, int(lengths.max())

    monkeypatch.setattr("verl.workers.utils.padding.unpad_input", cpu_unpad)
    data = envelope(
        [[output("a", 1, 10), output("a", 1, 11)], [output("b", 0, 20)]], ["q", "q"]
    )
    assert len(data) == 2  # validation / queue accounting still sees two outcomes
    assert data.batch["rm_scores"].sum(-1).tolist() == [1, 0]
    assert len(data.meta_info["metrics"]) == 2
    # Ray can serialize nested DataProto; uid may be assigned after generation.
    data = pickle.loads(pickle.dumps(data))
    data.non_tensor_batch["uid"][:] = "async-q"
    data.meta_info.pop("metrics")
    expanded = expand_episode_segments(data, config())
    assert len(expanded) == 4  # three real segments and one inert padding row
    assert expanded.non_tensor_batch["uid"][:3].tolist() == ["async-q"] * 3
    assert expanded.non_tensor_batch["episode_id"][:3].tolist() == ["a", "a", "b"]
    assert expanded.batch["segment_padding"].tolist() == [False, False, False, True]
    assert expanded.batch["response_mask"][-1].sum() == 0
    assert SEGMENTS_KEY not in expanded.non_tensor_batch
    assert expanded.batch["responses"][:3, 0].tolist() == [10, 11, 20]
    # The padding flag survives the actual actor input conversion.
    td = left_right_2_no_padding(expanded.to_tensordict())
    assert td["segment_padding"].tolist() == [False, False, False, True]


def test_grpo_uses_unique_episode_rewards_not_duplicated_segment_rewards_or_other_queries():
    data = envelope(
        [
            [output("a", 1, 10), output("a", 1, 11), output("a", 1, 12)],
            [output("b", 0, 20)],
            [output("c", 100, 30)],
            [output("d", 98, 40)],
        ],
        ["q1", "q1", "q2", "q2"],
    )
    data.meta_info.pop("metrics")
    result = advantages(expand_episode_segments(data, config()))
    # Upstream GRPO uses sample std: [1,0] -> +/-1/sqrt(2).
    # Duplicating reward 1 three times would produce a different baseline/std.
    expected = torch.tensor([1, 1, 1, -1, 1, -1]) / 2**0.5
    torch.testing.assert_close(
        result.batch["advantages"][:6, 0], expected, atol=2e-6, rtol=1e-5
    )
    assert result.batch["advantages"][6:].count_nonzero() == 0
    assert result.batch["advantages"][:, 1].count_nonzero() == 0
    # Reordering rows for DP balancing cannot alter grouping.
    permutation = torch.tensor([4, 2, 5, 1, 0, 3])
    reordered = result.select_idxs(permutation)
    reordered = advantages(reordered)
    torch.testing.assert_close(
        reordered.batch["advantages"], result.batch["advantages"][permutation]
    )


def test_failed_episode_excluded_once_and_all_zero_groups_remain_finite():
    data = envelope(
        [
            [
                output("a", 1, 10, reason="timeout"),
                output("a", 1, 11, reason="timeout"),
            ],
            [output("b", 0, 20)],
            [output("c", 0, 30)],
        ],
        ["q"] * 3,
    )
    data.meta_info.pop("metrics")
    result = advantages(expand_episode_segments(data, config()))
    assert result.batch["advantages"].count_nonzero() == 0
    assert torch.isfinite(result.batch["advantages"]).all()


def test_longest_uses_unmodified_single_row_training_path():
    data = envelope(
        [
            output("a", 1, 10, selection="longest"),
            output("b", 0, 20, selection="longest"),
        ],
        ["q", "q"],
    )
    original = data.batch.clone()
    result = expand_episode_segments(data, config())
    assert len(result) == 2
    assert "episode_segments_expanded" not in result.meta_info
    assert "segment_padding" not in result.batch
    torch.testing.assert_close(result.batch["rm_scores"], original["rm_scores"])


@pytest.mark.parametrize(
    "override", [{"adv_estimator": "gae"}, {"use_kl_in_reward": True}]
)
def test_unsupported_episode_reward_semantics_fail_before_updates(override):
    cfg = config()
    cfg.algorithm.update(override)
    data = envelope([[output("a", 1, 10)]], ["q"])
    with pytest.raises(ValueError, match="GRPO"):
        expand_episode_segments(data, cfg)


def test_changed_context_can_overflow_prompt_region_without_losing_prefix(tmp_path):
    loop = object.__new__(BuiltinSWEAgentLoop)
    loop.prompt_length = 2
    loop.response_length = 10
    loop.trajectory_selection = "all"
    loop._max_turns = 0
    loop._tool_parser_name = "hermes"
    definition = SimpleNamespace(name="test", protocol="openai")
    loop.harnesses = {"test": definition}
    loop.base_trials_dir = str(tmp_path)
    segment = {
        "prompt_ids": [1, 50, 30, 4],
        "response_ids": [40, 41],
        "response_mask": [1, 1],
        "response_logprobs": [-0.3, -0.4],
        "response_routing": [None, None],
        "num_turns": 1,
    }
    outputs = loop._outputs_from_segments(
        {"episode_id": "e", "trajectory_segments": [segment]},
        1,
        "agent_completed",
        {},
        {},
        SimpleNamespace(definition=definition, source="default"),
    )
    out = outputs[0]
    assert out.prompt_ids + out.response_ids == [1, 50, 30, 4, 40, 41]
    assert out.response_mask == [0, 0, 1, 1]
    assert out.response_logprobs == [0, 0, -0.3, -0.4]
    assert out.extra_fields["episode_id"] == "e"


def test_gspo_loss_and_gradient_ignore_padding_but_ratio_remains_segment_local():
    from verl.trainer.ppo.core_algos import compute_policy_loss_gspo
    from verl.workers.config import ActorConfig

    cfg = ActorConfig(
        ppo_mini_batch_size=2, strategy="fsdp", rollout_n=1, use_dynamic_bsz=True
    )
    cfg.global_batch_info.update(global_batch_size=2)
    old = torch.tensor([[-1.0, -2.0], [-3.0, -4.0]])
    target = (old + torch.tensor([[0.01, 0.03], [-0.02, -0.04]])).requires_grad_()
    mask = torch.ones_like(old)
    advantage = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    loss, _ = compute_policy_loss_gspo(old, target, advantage, mask, config=cfg)
    grad = torch.autograd.grad(loss, target)[0]
    padded_target = torch.cat(
        [target.detach(), torch.tensor([[-5.0, -5.0]])]
    ).requires_grad_()
    padded_loss, _ = compute_policy_loss_gspo(
        torch.cat([old, torch.zeros(1, 2)]),
        padded_target,
        torch.cat([advantage, torch.zeros(1, 2)]),
        torch.cat([mask, torch.zeros(1, 2)]),
        config=cfg,
    )
    padded_grad = torch.autograd.grad(padded_loss, padded_target)[0]
    torch.testing.assert_close(loss, padded_loss)
    torch.testing.assert_close(grad, padded_grad[:2])
    assert padded_grad[2].count_nonzero() == 0


def test_fully_async_assembly_expands_after_episode_metrics_and_before_balancing():
    from verl.experimental.fully_async_policy.detach_utils import (
        assemble_batch_from_rollout_samples,
    )

    early, late = output("a", 1, 10), output("a", 1, 11)
    late.extra_fields.update(min_global_steps=2, max_global_steps=3)
    data = envelope([[early, late], [output("b", 0, 20)]], ["q", "q"])
    seen = []

    def balance(batch, metrics):
        seen.append(len(batch))
        # Simulate DP reordering, including moving the padding row.
        batch.reorder(torch.tensor([3, 2, 1, 0]))

    result = assemble_batch_from_rollout_samples(
        [SimpleNamespace(full_batch=data, rollout_status={})],
        None,
        config(),
        balance_batch=balance,
    )
    assert seen == [4]
    assert len(result.meta_info["trajectory_param_versions"]) == 2  # episode accounting
    assert result.meta_info["trajectory_param_versions"].tolist() == [3, 0]
    assert result.meta_info["fully_async/partial/max_partial_span"] == 3
    assert len(result.meta_info["global_token_num"]) == 4  # actual forward workload
    result = advantages(result)
    torch.testing.assert_close(
        result.batch["advantages"][:, 0],
        torch.tensor([0, -1, 1, 1]) / 2**0.5,
        atol=2e-6,
        rtol=1e-5,
    )


def segment_batch(sizes, cfg=None):
    cfg = cfg or config()
    data = envelope(
        [
            [output(str(e), e % 2, 10 + s) for s in range(size)]
            for e, size in enumerate(sizes)
        ],
        ["q"] * len(sizes),
    )
    data.meta_info.pop("metrics")
    return expand_episode_segments(data, cfg)


@pytest.mark.parametrize("dynamic", [True, False])
def test_eight_episodes_thirteen_segments_are_one_update_on_four_ranks(dynamic):
    cfg = config()
    actor = cfg.actor_rollout_ref.actor
    actor.ppo_mini_batch_size = 4  # rollout.n=2 -> 8 episodes
    actor.use_dynamic_bsz = dynamic
    actor.ppo_micro_batch_size_per_gpu = 2
    data = segment_batch([3, 2, 2, 2, 1, 1, 1, 1], cfg)
    updates = list(iter_episode_minibatches(data, actor, 2, 4))
    assert len(updates) == 1
    update = updates[0]
    assert len(update) == 16
    assert update.batch[PADDING_KEY].sum() == 3
    assert update.meta_info["num_update_episodes"] == 8
    assert update.meta_info["update_lr_scheduler"]
    assert update.batch["response_mask"].sum() == 13
    for e, n in enumerate([3, 2, 2, 2, 1, 1, 1, 1]):
        real = (update.batch[EPISODE_INDEX] == e) & ~update.batch[PADDING_KEY]
        assert real.sum() == n
        assert update.batch[TOKEN_COUNT][real].tolist() == [n] * n


def test_episode_shuffle_and_tail_keep_all_segments_in_one_step_per_epoch():
    cfg = config()
    actor = cfg.actor_rollout_ref.actor
    actor.ppo_epochs = 2
    actor.shuffle = True
    data = segment_batch([3, 1, 2, 1, 1, 2, 1, 1, 1], cfg)
    # Early logprob balancing can reorder rows; it must not change the plan.
    reordered = data.select_idxs(torch.arange(len(data) - 1, -1, -1))
    plans = list(iter_episode_minibatches(data, actor, 2, 2))
    other = list(iter_episode_minibatches(reordered, actor, 2, 2))
    assert len(plans) == 6  # 9 episodes / 4, including a 1-episode tail, twice
    assert [p.meta_info["num_update_episodes"] for p in plans] == [4, 4, 1] * 2
    assert [p.meta_info["update_lr_scheduler"] for p in plans] == [False] * 5 + [True]
    for epoch in range(2):
        seen = set()
        for plan, permuted in zip(
            plans[epoch * 3 : (epoch + 1) * 3], other[epoch * 3 : (epoch + 1) * 3]
        ):
            ids = plan.batch[EPISODE_INDEX][~plan.batch[PADDING_KEY]].tolist()
            assert not seen.intersection(ids)
            seen.update(ids)
            assert sorted(ids) == sorted(
                permuted.batch[EPISODE_INDEX][~permuted.batch[PADDING_KEY]].tolist()
            )
        assert seen == set(range(9))


def test_globally_empty_episode_update_is_skipped_and_last_active_steps_scheduler():
    cfg = config()
    data = segment_batch([1] * 8, cfg)
    data.batch["response_mask"][data.batch[EPISODE_INDEX] >= 4] = 0
    plans = list(iter_episode_minibatches(data, cfg.actor_rollout_ref.actor, 2, 2))
    assert len(plans) == 1
    assert plans[0].meta_info["update_lr_scheduler"]
    data.batch["response_mask"].zero_()
    assert list(iter_episode_minibatches(data, cfg.actor_rollout_ref.actor, 2, 2)) == []


@pytest.mark.parametrize(
    "mode", ["token-mean", "seq-mean-token-mean", "seq-mean-token-sum"]
)
def test_episode_loss_and_gradients_match_reference_across_microbatches(mode):
    from verl.trainer.ppo.core_algos import agg_loss

    # Episode 0 has three unequal segments, episode 1 has one. Context and
    # padding tokens must carry no weight in any reduction.
    mask = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0]])
    counts = torch.tensor([6, 6, 6, 2, 0])
    x = torch.linspace(0.1, 1.5, 15).reshape(5, 3).requires_grad_()
    terms = x.square() * mask
    if mode == "token-mean":
        reference = terms.sum() / 8
    elif mode == "seq-mean-token-mean":
        reference = (terms[:3].sum() / 6 + terms[3].sum() / 2) / 2
    else:
        reference = terms.sum() / 2
    expected_grad = torch.autograd.grad(reference, x)[0]
    # Simulate two ranks and differing microbatch groupings. The engine's
    # averaged DP reduction cancels agg_loss's dp_size multiplier.
    contributions = []
    for indices in ([3, 0], [4], [1, 2]):
        contributions.append(
            agg_loss(
                x[indices].square(),
                mask[indices],
                mode,
                dp_size=2,
                batch_num_tokens=8,
                global_batch_size=2,
                episode_token_count=counts[indices],
            )
        )
    actual = sum(contributions) / 2
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(torch.autograd.grad(actual, x)[0], expected_grad)


def test_gspo_episode_weighting_matches_segment_ratio_reference():
    from verl.trainer.ppo.core_algos import compute_policy_loss_gspo
    from verl.workers.config import ActorConfig

    cfg = ActorConfig(
        strategy="fsdp2", rollout_n=1, use_dynamic_bsz=True, clip_ratio=0.2
    )
    mask = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0]])
    counts = torch.tensor([6, 6, 6, 2, 0])
    advantage = torch.tensor([1, 1, 1, -1, 0.0])[:, None].expand_as(mask)
    old = torch.zeros_like(mask, dtype=torch.float32)
    current = (
        torch.tensor([0.0, 0.1, 0.4, -0.4, 0.0])[:, None]
        .expand_as(mask)
        .clone()
        .requires_grad_()
    )
    sizes = mask.sum(-1).clamp_min(1)
    ratios = ((current - old) * mask).sum(-1).div(sizes).exp()
    per_segment = torch.maximum(
        -advantage[:, 0] * ratios, -advantage[:, 0] * ratios.clamp(0.8, 1.2)
    )
    reference = (per_segment * mask.sum(-1) / counts.clamp_min(1)).sum() / 2
    expected = torch.autograd.grad(reference, current)[0]
    contributions = []
    for indices in ([2, 3], [0, 4], [1]):
        cfg.global_batch_info.update(
            global_batch_size=2, episode_token_count=counts[indices]
        )
        loss, _ = compute_policy_loss_gspo(
            old[indices],
            current[indices],
            advantage[indices],
            mask[indices],
            config=cfg,
        )
        contributions.append(loss)
    actual = sum(contributions)
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(torch.autograd.grad(actual, current)[0], expected)


def _distributed_episode_worker(rank, rendezvous, result_dir):
    """Real Gloo collectives, worker loop, microbatch packing and SGD updates."""
    import json
    from pathlib import Path

    import torch.distributed as dist
    from verl.trainer.ppo.core_algos import compute_policy_loss_gspo
    from verl.workers.config import ActorConfig
    from verl.workers.engine.utils import prepare_micro_batches
    from verl.workers.engine_workers import TrainingWorker

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        results = []
        for sizes in ([1], [3, 2, 2, 2, 1, 1, 1, 1]):
            cfg = config()
            cfg.actor_rollout_ref.actor.ppo_mini_batch_size = 4
            full = segment_batch(sizes, cfg)
            step = list(
                iter_episode_minibatches(full, cfg.actor_rollout_ref.actor, 2, 2)
            )[0]
            local = step.batch.chunk(2)[rank].clone()
            local["input_ids"] = torch.nested.as_nested_tensor(
                list(local["input_ids"].unbind()), layout=torch.jagged
            )
            tu.assign_non_tensor(
                local,
                mini_batch_size=len(step),
                epochs=1,
                dataloader_kwargs={},
                global_batch_size=len(sizes),
                update_lr_scheduler=True,
                use_dynamic_bsz=True,
                max_token_len_per_gpu=4,
            )
            worker = object.__new__(TrainingWorker)
            worker.engine = SimpleNamespace(
                get_data_parallel_size=lambda: 2,
                get_data_parallel_rank=lambda: rank,
                get_data_parallel_group=lambda: dist.group.WORLD,
                train_mode=lambda **kwargs: nullcontext(),
                is_mp_src_rank_with_outputs=lambda: True,
            )
            loss_config = ActorConfig(
                strategy="fsdp2", rollout_n=1, use_dynamic_bsz=True
            )
            theta = torch.nn.Parameter(torch.tensor(0.0))
            optimizer = torch.optim.SGD([theta], lr=0.1)
            seen = []

            def train_batch(data):
                microbatches, _ = prepare_micro_batches(data, dp_group=dist.group.WORLD)
                for micro in microbatches:
                    mask = micro["response_mask"]
                    loss_config.global_batch_info.update(
                        global_batch_size=tu.get(data, "global_batch_size"),
                        dp_size=2,
                        episode_token_count=micro[TOKEN_COUNT],
                    )
                    loss, _ = compute_policy_loss_gspo(
                        torch.zeros_like(mask),
                        theta.expand_as(mask),
                        torch.ones_like(mask),
                        mask,
                        config=loss_config,
                    )
                    loss.backward()
                dist.all_reduce(theta.grad)
                theta.grad.div_(2)
                optimizer.step()
                optimizer.zero_grad()
                seen.append(
                    [
                        tu.get(data, "global_batch_size"),
                        tu.get(data, "update_lr_scheduler"),
                    ]
                )
                return tu.get_tensordict({}, {"metrics": {"loss": 0.0}})

            worker.train_batch = train_batch
            worker.train_mini_batch(local)
            results.append({"seen": seen, "theta": theta.item()})
        (Path(result_dir) / f"rank{rank}.json").write_text(json.dumps(results))
    finally:
        dist.destroy_process_group()


def test_two_rank_episode_gradients_and_collectives_with_an_empty_rank(tmp_path):
    import json
    import torch.multiprocessing as mp

    mp.spawn(
        _distributed_episode_worker,
        args=(f"file://{tmp_path}/rendezvous", str(tmp_path)),
        nprocs=2,
        join=True,
    )
    for rank in range(2):
        results = json.loads((tmp_path / f"rank{rank}.json").read_text())
        for result, episode_count in zip(results, (1, 8)):
            assert result["seen"] == [[episode_count, True]]
            assert result["theta"] == pytest.approx(0.1)


def test_trainer_sends_whole_episodes_to_worker_and_steps_scheduler_once(monkeypatch):
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.workers.engine_workers import TrainingWorker

    def cpu_unpad(tensor, mask):
        indices = mask.flatten().nonzero().flatten()
        lengths = mask.sum(-1).to(torch.int32)
        offsets = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
        return tensor.flatten(0, 1)[indices], indices, offsets, int(lengths.max())

    monkeypatch.setattr("verl.workers.utils.padding.unpad_input", cpu_unpad)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda *args, **kwargs: 1)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda out, value, *args, **kwargs: out.__setitem__(0, value),
    )
    cfg = config()
    cfg.actor_rollout_ref.actor.ppo_epochs = 2
    worker = object.__new__(TrainingWorker)
    worker.engine = SimpleNamespace(
        get_data_parallel_size=lambda: 1,
        get_data_parallel_rank=lambda: 0,
        get_data_parallel_group=lambda: None,
        train_mode=lambda **kwargs: nullcontext(),
        is_mp_src_rank_with_outputs=lambda: True,
    )
    seen = []

    def train_batch(data):
        seen.append(
            (
                len(data),
                tu.get(data, "global_batch_size"),
                tu.get(data, "update_lr_scheduler"),
            )
        )
        assert (TOKEN_COUNT in data) == bool(
            tu.get(data, "episode_segments_expanded", False)
        )
        return tu.get_tensordict({}, {"metrics": {"mfu": 0.0, "loss": 1.0}})

    worker.train_batch = train_batch
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = cfg
    trainer.actor_rollout_wg = SimpleNamespace(update_actor=worker.train_mini_batch)
    trainer._get_dp_size = lambda *args: 1
    # Four episodes (eight segments), then one episode (one segment), twice.
    batch = segment_batch([3, 2, 2, 1, 1], cfg)
    result = trainer._update_actor(batch)
    assert seen == [(8, 4, False), (1, 1, False), (8, 4, False), (1, 1, True)]
    assert result.meta_info["metrics"]["actor/episode_optimizer_steps"] == 4
    assert result.meta_info["metrics"]["actor/loss"] == [1.0] * 4
    seen.clear()
    ordinary = envelope(
        [output(str(i), i % 2, 10, selection="longest") for i in range(4)],
        ["q"] * 4,
    )
    ordinary.meta_info.pop("metrics")
    ordinary = expand_episode_segments(ordinary, cfg)
    result = trainer._update_actor(ordinary)
    assert seen == [(4, 4, False), (4, 4, True)]
    assert "actor/episode_optimizer_steps" not in result.meta_info["metrics"]


def test_ppo_loss_passes_episode_denominators_to_policy_entropy_and_kl(monkeypatch):
    from verl.trainer.ppo.core_algos import kl_penalty
    from verl.workers.config import ActorConfig
    from verl.workers.config.actor import PolicyLossConfig
    from verl.workers.utils import losses

    # Focus on the real loss entry point and its field selection. Padding
    # conversion is covered separately by the trainer/worker integration test.
    monkeypatch.setattr(losses, "no_padding_2_padding", lambda tensor, data: tensor)
    cfg = ActorConfig(
        strategy="fsdp2",
        rollout_n=1,
        use_dynamic_bsz=True,
        loss_agg_mode="seq-mean-token-mean",
        entropy_coeff=0.2,
        use_kl_loss=True,
        kl_loss_coef=0.3,
        kl_loss_type="kl",
        policy_loss=PolicyLossConfig(loss_mode="gspo"),
    )
    mask = torch.tensor([[1, 0], [1, 1], [1, 1], [0, 0]])
    denominators = torch.tensor([3, 3, 2, 0])
    logp = torch.tensor(
        [[0.01, 0], [0.02, 0.03], [-0.01, -0.02], [0, 0.0]], requires_grad=True
    )
    entropy = logp + 2
    reference_logp = torch.full_like(logp, -0.1)
    data = tu.get_tensordict(
        {
            "response_mask": mask,
            "old_log_probs": torch.zeros_like(logp),
            "advantages": torch.ones_like(logp),
            "ref_log_prob": reference_logp,
            TOKEN_COUNT: denominators,
        },
        {"dp_size": 1, "batch_num_tokens": 5, "global_batch_size": 2},
    )
    actual, _ = losses.ppo_loss(cfg, {"log_probs": logp, "entropy": entropy}, data)
    ratio = (logp * mask).sum(-1).div(mask.sum(-1).clamp_min(1)).exp()
    weights = mask / denominators.clamp_min(1)[:, None] / 2
    reference = (
        weights
        * (
            -ratio[:, None]
            - 0.2 * entropy
            + 0.3 * kl_penalty(logp, reference_logp, "kl")
        )
    ).sum()
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(
        torch.autograd.grad(actual, logp, retain_graph=True)[0],
        torch.autograd.grad(reference, logp)[0],
    )
    # A later ordinary batch must not inherit a previous microbatch's counts.
    data.pop(TOKEN_COUNT)
    losses.ppo_loss(cfg, {"log_probs": logp, "entropy": entropy}, data)
    assert cfg.global_batch_info[TOKEN_COUNT] is None


@pytest.mark.parametrize("selection", ["longest", "all"])
def test_training_budget_cannot_hide_a_usable_segment(selection, tmp_path, monkeypatch):
    loop = object.__new__(BuiltinSWEAgentLoop)
    loop.prompt_length, loop.response_length = 2, 4
    loop.trajectory_selection = selection
    loop._max_turns, loop._tool_parser_name = 0, "hermes"
    definition = SimpleNamespace(name="test", protocol="openai")
    loop.harnesses, loop.base_trials_dir = {"test": definition}, str(tmp_path)
    usable = {
        "prompt_ids": [1, 2],
        "response_ids": [10],
        "response_mask": [1],
        "response_logprobs": [-0.2],
        "response_routing": [None],
        "num_turns": 1,
    }
    hidden = {
        **usable,
        "prompt_ids": list(range(10)),
        "response_ids": [20, 21],
        "response_mask": [1, 1],
        "response_logprobs": [-0.1, -0.2],
        "response_routing": [None, None],
    }
    result = loop._outputs_from_segments(
        {"episode_id": "e", "trajectory_segments": [usable, hidden]},
        1,
        "agent_completed",
        {},
        {},
        SimpleNamespace(definition=definition, source="default"),
    )
    result = result if isinstance(result, list) else [result]
    assert len(result) == 1
    assert result[0].response_ids == [10]
    assert result[0].reward_score == 1
    monkeypatch.setenv("HARBOR_EMPTY_RESPONSE_MASK_ONE", "1")
    empty = loop._outputs_from_segments(
        {"episode_id": "e", "trajectory_segments": [hidden]},
        1,
        "agent_completed",
        {},
        {},
        SimpleNamespace(definition=definition, source="default"),
    )
    assert empty.response_mask == [0]
    assert empty.reward_score == 1
    assert empty.extra_fields["termination_reason"] == "overlong"


def test_validation_reward_survives_an_empty_training_mask():
    worker = object.__new__(AgentLoopWorker)
    worker.reward_loop_worker_handles = None
    result = output("a", 1, 10)
    result.response_mask.zero_()
    assert worker._postprocess([result], validate=False).batch["rm_scores"].sum() == 0
    assert worker._postprocess([result], validate=True).batch["rm_scores"].sum() == 1
