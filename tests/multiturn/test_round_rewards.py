import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from harbor_patch.multiturn.config import MultiTurnConfig
from harbor_patch.multiturn.round_judge import RoundJudge, UXJudgeResult, build_round_context
from verl_patch.multiturn.advantages import compute_round_grpo_advantage

from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage


def judgment(score):
    return UXJudgeResult(
        reward=0.99,
        task_success_score=score,
        final_requirement_following_score=score,
        context_continuity_score=score,
        correction_handling_score=score,
        testing_and_verification_score=score,
        user_effort_score=score,
        rationale="evidence",
    )


def test_round_groups_ignore_compaction_duplication_padding_and_single_survivor():
    scores = torch.tensor([[0.2], [0.2], [0.8], [0.9], [0.3], [0.7], [100.0]])
    mask = torch.ones_like(scores)
    mask[-1] = 0
    adv, _ = compute_round_grpo_advantage(
        scores,
        mask,
        np.array(["task"] * 7),
        np.array(["A", "A", "B", "A", "B", "B", "padding"]),
        np.array([1, 1, 1, 2, 2, 3, 0]),
        norm_adv_by_std_in_grpo=False,
    )
    torch.testing.assert_close(adv[:, 0], torch.tensor([-0.3, -0.3, 0.3, 0.3, -0.3, 0.0, 0.0]))
    normalized, _ = compute_round_grpo_advantage(
        scores,
        mask,
        np.array(["task"] * 7),
        np.array(["A", "A", "B", "A", "B", "B", "padding"]),
        np.array([1, 1, 1, 2, 2, 3, 0]),
    )
    assert torch.isfinite(normalized).all()
    assert normalized[0] == normalized[1]


def test_real_verl_entry_uses_round_grouping_and_legacy_stays_unchanged():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"grpo_group_by_trial": True})

    def batch(rounds):
        fields = {"uid": np.array(["task"] * 4), "episode_id": np.array(["A", "B", "A", "B"])}
        if rounds:
            fields["multiturn_round_id"] = np.array([1, 1, 2, 2])
        return DataProto.from_dict(
            tensors={
                "token_level_rewards": torch.tensor([[0.2], [0.8], [0.9], [0.3]]),
                "response_mask": torch.ones(4, 1),
            },
            non_tensors=fields,
        )

    new = compute_advantage(batch(True), "grpo", norm_adv_by_std_in_grpo=False, config=cfg)
    torch.testing.assert_close(new.batch["advantages"][:, 0], torch.tensor([-0.3, 0.3, 0.3, -0.3]))
    old = compute_advantage(batch(False), "grpo", norm_adv_by_std_in_grpo=False, config=cfg)
    torch.testing.assert_close(old.batch["advantages"][:, 0], torch.tensor([-0.35, 0.25, 0.35, -0.25]))


def test_context_has_no_future_round_or_solution_and_preserves_rubric(tmp_path):
    current = tmp_path / "multiturn/round-001"
    current.mkdir(parents=True)
    (current / "claude-code.jsonl").write_text("tool ran the program")
    future = tmp_path / "multiturn/round-002"
    future.mkdir()
    (future / "claude-code.jsonl").write_text("FUTURE_SECRET")
    task = SimpleNamespace(name="task", instruction="Fix bug")
    context = build_round_context(
        task, [{"round_id": 1, "instruction": "fix", "functional_reward": 0.5}], tmp_path, 4000
    )
    prompt = RoundJudge._build_prompt(context)
    assert "Judge correctness as of round 1 only" in prompt
    assert "tool ran the program" in prompt
    assert "FUTURE_SECRET" not in prompt
    assert "task_success_score" in prompt


@pytest.mark.asyncio
async def test_judge_json_repair_and_invalid_score_retry(tmp_path):
    judge = object.__new__(RoundJudge)
    judge.options = MultiTurnConfig(judge_max_retries=2)
    calls = []

    async def call(*a, **kw):
        calls.append(kw)
        payload = judgment(40).model_dump()
        if len(calls) == 1:
            payload["task_success_score"] = float("nan")
        return SimpleNamespace(content="```json\n" + json.dumps(payload) + "\n```")

    judge._llm = SimpleNamespace(call=call)
    result = await judge.score({"rounds": [{"functional_reward": 1}], "evaluation_scope": "round_prefix"}, tmp_path)
    assert result.task_success_score == 40
    assert len(calls) == 2
    assert calls[0]["timeout"] == judge.options.judge_timeout_sec
    assert calls[0]["num_retries"] == 0
    assert "raw" not in calls[0]["response_format"].model_json_schema()["properties"]
    attempts = [json.loads(line) for line in (tmp_path / "judge-attempts.jsonl").read_text().splitlines()]
    assert attempts == [{"attempt": 1, "error_type": "ValidationError", "status_code": None}]
    assert (tmp_path / "judge-result.json").is_file()


@pytest.mark.parametrize(
    "values",
    [
        {"enabled": True, "max_rounds": 1, "reward_mode": "per_round"},
        {"enabled": True, "max_rounds": 1, "judge_enabled": True},
        {"enabled": True, "max_rounds": 1, "test_weight": 0.8},
    ],
)
def test_invalid_reward_config_fails_closed(values):
    with pytest.raises(ValueError):
        MultiTurnConfig(**values)


def test_judge_wire_schema_has_no_unconstrained_object():
    from harbor_patch.multiturn.round_judge import JudgeScores

    schema = JudgeScores.model_json_schema()
    assert schema["additionalProperties"] is False
    assert "raw" not in schema["properties"]
    assert set(schema["required"]) == set(schema["properties"])
    # Local archives can still parse the benchmark's optional raw metadata.
    result = judgment(80).model_dump()
    result["raw"] = {"provider_note": "kept for audit"}
    assert UXJudgeResult.model_validate(result).raw == result["raw"]
