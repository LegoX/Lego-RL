"""Episode/round metrics shared by training and validation, independent of segments.

Training metrics describe the current optimizer batch; validation uses complete
sampled episodes. Scores always come from raw evaluator records, before KL shaping.
"""
import json
import math
from collections import defaultdict
from statistics import mean, pstdev


def decode_summary(value):
    try:
        value = json.loads(value) if isinstance(value, str) else value
        return value if isinstance(value, dict) and value.get("enabled") else None
    except (ValueError, TypeError):
        return None


def completed_rounds(summary):
    rounds = summary.get("round_rewards", [])
    return sorted((r for r in rounds if isinstance(r, dict)
                   and isinstance(r.get("round_id"), int)
                   and r["round_id"] > 0
                   and all(isinstance(r.get(k), (float, int)) and math.isfinite(r[k])
                           for k in ("training_reward", "functional_reward"))),
                  key=lambda r: r["round_id"])


def terminal_score(summary):
    if not summary or not summary.get("valid"):
        return None
    if summary.get("task_profile") == "swe_interact":
        value = summary.get("final_reward")
        return value if isinstance(value, (int, float)) and math.isfinite(value) else None
    rounds = completed_rounds(summary)
    return rounds[-1]["training_reward"] if rounds else None


def aggregate_episodes(summaries, prefix="multiturn"):
    groups = defaultdict(list)
    for value in summaries:
        summary = decode_summary(value)
        if summary:
            profile = summary.get("task_profile", "ccbench")
            if profile not in ("ccbench", "openswe", "swe_interact"):
                profile = "unknown"
            groups[profile].append(summary)
    result = {}
    for profile, attempts in groups.items():
        base = f"{prefix}/{profile}"
        if profile == "swe_interact":
            valid = [s for s in attempts if terminal_score(s) is not None]
            result[f"{base}/attempts"] = len(attempts)
            result[f"{base}/valid_episodes"] = len(valid)
            result[f"{base}/invalid_ratio"] = 1 - len(valid) / len(attempts)
            result[f"{base}/success_rate_all_attempts"] = mean(terminal_score(s) == 1 for s in attempts)
            for reason in ("agent_completed", "max_turns_reached", "overlong", "timeout"):
                result[f"{base}/stop/{reason}_ratio"] = mean(
                    s.get("termination_reason") == reason for s in attempts)
            result[f"{base}/first_round_overlong_ratio"] = mean(
                s.get("termination_reason") == "overlong" and s.get("num_user_rounds") == 1
                for s in attempts)
            if valid:
                result[f"{base}/final_reward"] = mean(terminal_score(s) for s in valid)
                result[f"{base}/final_success_rate"] = mean(terminal_score(s) == 1 for s in valid)
                reviewed = [s for s in valid if "approval_reward" in s]
                if reviewed:
                    result[f"{base}/approval_rate"] = mean(s["approval_reward"] == 1 for s in reviewed)
                result[f"{base}/feedback_count"] = mean(max(0, s.get("num_user_rounds", 0) - 1) for s in valid)
                result[f"{base}/duration_sec"] = mean(s.get("duration_sec", 0) for s in valid)
                if any("judge_status" in s for s in valid):
                    judged = [s for s in valid if s.get("judge_status") == "ok"
                              and isinstance(s.get("judge_score"), (int, float))
                              and math.isfinite(s["judge_score"])]
                    result[f"{base}/judge_coverage"] = len(judged) / len(valid)
                    result[f"{base}/judge_error_ratio"] = mean(s.get("judge_status") == "error" for s in valid)
                    result[f"{base}/judge_abstain_ratio"] = mean(s.get("judge_status") == "abstain" for s in valid)
                    rewards = [s["training_reward"] for s in valid
                               if isinstance(s.get("training_reward"), (int, float))
                               and math.isfinite(s["training_reward"])]
                    if rewards:
                        result[f"{base}/training_reward"] = mean(rewards)
                    if judged:
                        result[f"{base}/judge_score"] = mean(s["judge_score"] for s in judged)
                        for source, label in (("prompt_tokens", "judge_prompt_tokens"),
                                              ("completion_tokens", "judge_completion_tokens")):
                            usage = [(s.get("judge_usage") or {}).get(source) for s in judged]
                            usage = [value for value in usage if isinstance(value, (int, float))]
                            if usage:
                                result[f"{base}/{label}"] = mean(usage)
            continue
        valid = [(s, completed_rounds(s)) for s in attempts if s.get("valid", False)]
        valid = [(s, rs) for s, rs in valid if rs]
        result[f"{base}/attempts"] = len(attempts)
        result[f"{base}/valid_episodes"] = len(valid)
        result[f"{base}/invalid_ratio"] = 1 - len(valid) / len(attempts)
        for reason in ("verifier_success", "max_rounds", "simulator_no_op", "error", "cancelled"):
            result[f"{base}/stop/{reason}_ratio"] = mean(s.get("stop_reason", "error") == reason for s in attempts)
        for stage in ("environment", "agent", "verifier", "judge", "simulator"):
            result[f"{base}/failure/{stage}_ratio"] = mean(
                not s.get("valid", False) and s.get("failure_stage") == stage for s in attempts)
        if not valid:
            continue
        first = [rs[0] for _, rs in valid]
        final = [rs[-1] for _, rs in valid]
        result[f"{base}/final_reward"] = mean(r["training_reward"] for r in final)
        result[f"{base}/first_success_rate"] = mean(r["functional_reward"] == 1 for r in first)
        result[f"{base}/final_success_rate"] = mean(r["functional_reward"] == 1 for r in final)
        result[f"{base}/reward_gain"] = mean(b["training_reward"] - a["training_reward"] for a, b in zip(first, final))
        result[f"{base}/feedback_count"] = mean(max(0, len(rs) - 1) for _, rs in valid)
        result[f"{base}/duration_sec"] = mean(s.get("duration_sec", 0) for s, _ in valid)
        initially_failed = [(a, b) for a, b in zip(first, final) if a["functional_reward"] != 1]
        if initially_failed:
            result[f"{base}/recovery_rate"] = mean(b["functional_reward"] == 1 for _, b in initially_failed)
        max_rounds = max(int(s.get("max_rounds", len(rs))) for s, rs in valid)
        for number in range(1, max_rounds + 1):
            rb = f"{base}/round/{number}"
            observed = [r for _, rs in valid for r in rs if r["round_id"] == number]
            result[f"{rb}/count"] = len(observed)
            # Denominator is all valid attempts, including early successes.
            result[f"{rb}/cumulative_success_rate"] = mean(
                any(r["functional_reward"] == 1 and r["round_id"] <= number for r in rs)
                for _, rs in valid)
            if not observed:
                continue
            for source, label in (("training_reward", "reward"), ("functional_reward", "test"), ("judge_score", "judge")):
                values = [r[source] for r in observed if isinstance(r.get(source), (int, float)) and math.isfinite(r[source])]
                if values:
                    result[f"{rb}/{label}"] = mean(values)
                    result[f"{rb}/{label}_std"] = pstdev(values)
    return result


def training_metrics(batch, real, scores):
    """Called only for a multi-turn batch; leave legacy statistics untouched."""
    summaries = batch.non_tensor_batch["multiturn_summary"]
    identities = batch.non_tensor_batch.get("episode_id", batch.non_tensor_batch.get("trial_id"))
    row_scores = scores.detach().cpu().tolist()  # already selected by real mask
    episodes = {}
    missing = 0
    for i, is_real in enumerate(real):
        if not is_real:
            continue
        identity = identities[i] if identities is not None else None
        if not identity:
            missing += 1
            identity = f"missing-row-{i}"
        episodes.setdefault(str(identity), decode_summary(summaries[i]))
    valid = [s for s in episodes.values() if terminal_score(s) is not None]
    result = {
        "reward_stats/schema_version": 3,
        "reward_stats/trial_mean": mean(terminal_score(s) for s in valid) if valid else float("nan"),
        "reward_stats/trial_count": len(valid),
        "reward_stats/inconsistent_trials": 0,
        "reward_stats/missing_trial_id_rows": missing,
        "reward_stats/segment_mean": mean(row_scores) if row_scores else float("nan"),
        "reward_stats/segment_count": int(sum(real)),
        "reward_stats/padding_count": len(real) - int(sum(real)),
        "reward_stats/padding_ratio": 1 - sum(real) / len(real) if len(real) else 0,
        **aggregate_episodes(episodes.values()),
    }
    # Actual optimization eligibility, distinct from valid evaluation records.
    mask = batch.batch["response_mask"].bool()
    active = mask.any(-1).detach().cpu().tolist()
    round_ids = batch.non_tensor_batch.get("multiturn_round_id")
    uids = batch.non_tensor_batch.get("uid")
    filtered = batch.non_tensor_batch.get("trajectory_filter_mask", [False] * len(real))
    round_members = defaultdict(dict)
    by_profile = defaultdict(set)
    tokens = defaultdict(dict)
    advantages = batch.batch.get("advantages")
    for i, is_real in enumerate(real):
        summary = decode_summary(summaries[i])
        if not is_real or not active[i] or filtered[i] or not summary or not summary.get("valid"):
            continue
        profile = summary.get("task_profile", "ccbench")
        trial = str(identities[i]) if identities is not None else f"missing-row-{i}"
        by_profile[profile].add(trial)
        tokens[profile][trial] = batch.non_tensor_batch.get("trial_gen_tokens", [0] * len(real))[i]
        if round_ids is not None and int(round_ids[i]) > 0:
            key = (profile, int(round_ids[i]), str(uids[i]) if uids is not None else trial)
            value = advantages[i][mask[i]].float().mean().item() if advantages is not None else None
            round_members[key].setdefault(trial, []).append(value)
    for summary in episodes.values():
        if summary:
            result.setdefault(f"multiturn/{summary.get('task_profile', 'ccbench')}/trainable_episodes", 0)
    for profile, trials in by_profile.items():
        result[f"multiturn/{profile}/trainable_episodes"] = len(trials)
        result[f"multiturn/{profile}/generated_tokens"] = mean(float(v) for v in tokens[profile].values())
    round_stats = defaultdict(list)
    for (profile, number, _), members in round_members.items():
        round_stats[(profile, number)].append(members)
    for (profile, number), groups in round_stats.items():
        base = f"multiturn/{profile}/round/{number}"
        result[f"{base}/trainable_count"] = sum(len(g) for g in groups)
        result[f"{base}/singleton_group_ratio"] = mean(len(g) < 2 for g in groups)
        values = [mean(v for v in vs if v is not None) for g in groups for vs in g.values() if any(v is not None for v in vs)]
        if values:
            result[f"{base}/advantage_mean"] = mean(values)
            result[f"{base}/advantage_std"] = pstdev(values)
            result[f"{base}/nonzero_advantage_ratio"] = mean(abs(v) > 1e-8 for v in values)
    return result
