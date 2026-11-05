import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from verl_patch.multiturn.metrics import aggregate_episodes, training_metrics


def summary(values, *, valid=True, profile="ccbench"):
    return dict(enabled=True, valid=valid, task_profile=profile, max_rounds=5,
                stop_reason="verifier_success" if values[-1] == 1 else "max_rounds", duration_sec=10,
                round_rewards=[dict(round_id=i+1, functional_reward=v,
                                    judge_score=0.5, training_reward=0.5*v+0.25) for i,v in enumerate(values)])


def test_cumulative_success_carries_early_success_without_inventing_rounds():
    result = aggregate_episodes([summary([1]), summary([0, 0, 1]), summary([0], valid=False)])
    p = "multiturn/ccbench/"
    assert result[p+"final_reward"] == 0.75  # NOT test reward 1 or sum of rounds
    assert result[p+"first_success_rate"] == 0.5
    assert result[p+"final_success_rate"] == 1
    assert result[p+"round/2/count"] == 1
    assert result[p+"round/2/reward"] == 0.25
    assert result[p+"round/2/cumulative_success_rate"] == 0.5
    assert result[p+"round/5/count"] == 0
    assert result[p+"round/5/cumulative_success_rate"] == 1
    assert p+"round/5/reward" not in result
    assert result[p+"invalid_ratio"] == pytest.approx(1/3)


def test_profiles_and_validation_prefix_do_not_mix_scores():
    result = aggregate_episodes([summary([0]), summary([1], profile="openswe")], prefix="val-multiturn")
    assert result["val-multiturn/ccbench/final_reward"] == 0.25
    assert result["val-multiturn/openswe/final_reward"] == 0.75


def test_training_deduplicates_round_fragments_and_padding_keeps_real_advantages():
    a, b = summary([0, 1]), summary([1])
    batch = SimpleNamespace(
        non_tensor_batch={"multiturn_summary": [json.dumps(a)]*3 + [json.dumps(b), json.dumps(a)],
                          "trial_id": ["a", "a", "a", "b", "__compaction_padding_0"],
                          "uid": ["task"]*5, "multiturn_round_id": [1,1,2,1,1],
                          "trial_gen_tokens": [50,50,50,10,50]},
        batch={"response_mask": torch.tensor([[1],[1],[1],[1],[0]]),
               "advantages": torch.tensor([[-1.],[-1.],[0.],[1.],[0.]])})
    result = training_metrics(batch, np.array([True]*4+[False]), torch.tensor([0.25,0.25,0.75,0.75]))
    p = "multiturn/ccbench/"
    assert result["reward_stats/trial_mean"] == 0.75
    assert result["reward_stats/trial_count"] == 2
    assert result["reward_stats/inconsistent_trials"] == 0
    assert result[p+"round/1/count"] == 2
    assert result[p+"round/1/trainable_count"] == 2
    assert result[p+"round/1/advantage_mean"] == 0
    assert result[p+"round/2/singleton_group_ratio"] == 1
    assert result[p+"generated_tokens"] == 30


def test_real_metric_entry_and_legacy_behavior():
    from verl import DataProto
    from verl.trainer.ppo.metric_utils import compute_data_metrics

    data = DataProto.from_dict(tensors={
        "prompts": torch.ones(2, 1, dtype=torch.long),
        "responses": torch.ones(2, 1, dtype=torch.long),
        "attention_mask": torch.ones(2, 2),
        "response_mask": torch.ones(2, 1),
        "token_level_scores": torch.tensor([[.25], [.75]]),
        "token_level_rewards": torch.tensor([[.25], [.75]]),
        "advantages": torch.zeros(2, 1), "returns": torch.zeros(2, 1),
    }, non_tensors={"trial_id": np.array(["a", "a"], dtype=object)})
    ordinary = compute_data_metrics(data, use_critic=False)
    assert "multiturn/ccbench/final_reward" not in ordinary
    data.non_tensor_batch["multiturn_summary"] = np.array([json.dumps(summary([0, 1]))] * 2)
    result = compute_data_metrics(data, use_critic=False)
    assert result["reward_stats/schema_version"] == 3
    assert result["reward_stats/trial_mean"] == .75
    assert result["reward_stats/trial_count"] == 1
    assert result["critic/score/mean"] == ordinary["critic/score/mean"]


def test_validation_entry_consumes_summaries_before_numeric_aggregation():
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    extras = {"reward": [0.75,0.75], "_multiturn_summary": [json.dumps(summary([1])), json.dumps(summary([0,1]))]}
    result = RayPPOTrainer._val_metrics_update(None, np.array(["test","test"]), ["a","b"], extras, [])
    assert result["val-multiturn/ccbench/first_success_rate"] == 0.5
    assert result["val-multiturn/ccbench/final_reward"] == 0.75
    assert "_multiturn_summary" in extras  # input reusable for merged validation


def test_filtered_episode_has_eval_score_but_no_trainable_count():
    batch = SimpleNamespace(non_tensor_batch={
        "multiturn_summary": [json.dumps(summary([1]))], "trial_id": ["a"],
        "trajectory_filter_mask": [True]},
        batch={"response_mask": torch.ones(1,1)})
    result = training_metrics(batch, np.array([True]), torch.tensor([0.75]))
    assert result["multiturn/ccbench/valid_episodes"] == 1
    assert result["multiturn/ccbench/trainable_episodes"] == 0
