"""GRPO baselines over distinct episodes at the same task/round checkpoint."""

from collections import defaultdict

import torch


def compute_round_grpo_advantage(
    token_level_rewards,
    response_mask,
    index,
    trial_index,
    round_ids,
    norm_adv_by_std_in_grpo=True,
    epsilon=1e-6,
):
    if trial_index is None:
        raise ValueError("Per-round GRPO requires trial identities")
    scores = token_level_rewards.sum(dim=-1)
    if not (len(scores) == len(index) == len(trial_index) == len(round_ids)):
        raise ValueError("Per-round GRPO metadata must align with batch rows")
    with torch.no_grad():
        episodes = defaultdict(list)
        for i, (uid, trial, round_id) in enumerate(zip(index, trial_index, round_ids, strict=True)):
            # Exclude failed/filtered samples and divisibility padding from statistics.
            if not response_mask[i].any():
                continue
            if not trial or round_id is None or int(round_id) < 1 or int(round_id) != round_id:
                raise ValueError("Trainable row requires a valid episode and user round")
            if not torch.isfinite(scores[i]):
                raise ValueError("Non-finite per-round training reward")
            episodes[(str(uid), int(round_id), str(trial))].append(i)
        groups = defaultdict(list)
        for (uid, round_id, trial), rows in episodes.items():
            # With reward KL disabled all segments have identical scores. If enabled,
            # average their adjusted scores once per episode, not once per segment.
            score = scores[rows].mean()
            groups[(uid, round_id)].append((rows, score))
        advantages = torch.zeros_like(scores)
        for members in groups.values():
            # Early stopping can leave a lone survivor at a later round. There is
            # no within-round comparison then: give zero rather than reward/std=0.
            if len(members) < 2:
                continue
            group_scores = torch.stack([score for _, score in members])
            mean = group_scores.mean()
            scale = group_scores.std() + epsilon if norm_adv_by_std_in_grpo else 1.0
            for rows, score in members:
                advantages[rows] = (score - mean) / scale
        advantages = advantages.unsqueeze(-1) * response_mask
        return advantages, advantages
