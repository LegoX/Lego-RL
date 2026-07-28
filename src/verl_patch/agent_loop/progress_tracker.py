"""Per-step Harbor trial progress aggregator.

Each worker reports a one-line trial summary (task name + termination reason
+ reward + turns + wall-clock) to a Ray named actor. The actor prints the
line immediately so the wandb console-capture sees one entry per trial, and
periodically renders a progress bar plus per-outcome counts.

Why a named actor instead of per-worker prints alone: with NUM_WORKERS=96 each
worker only sees ~5–6 trials per step, so a worker-local counter cannot say
"234/512". The actor centralises the count without touching verl's
AgentLoopManager. Lazy step start: workers don't need synchronisation — the
first report whose ``step`` differs from the actor's tracked step rolls the
counters forward and prints a step header.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import ray


_ACTOR_NAMESPACE = "harbor_progress"
_DEFAULT_BAR_LEN = 30
_DEFAULT_SUMMARY_INTERVAL = 16  # print aggregate bar every N reports


def _actor_name() -> str:
    exp = os.environ.get("HARBOR_EXP_NAME") or "default"
    return f"harbor_progress_tracker_{exp}"


@ray.remote(num_cpus=0)
class HarborProgressTracker:
    def __init__(self, summary_interval: int = _DEFAULT_SUMMARY_INTERVAL) -> None:
        self.step: Optional[int] = None
        self.target_total: int = 0
        self.done: int = 0
        self.outcomes: dict[str, int] = {}
        self.t0: float = 0.0
        self.last_summary_at: int = 0
        self.summary_interval = summary_interval

    def report(
        self,
        step: int,
        target_total: int,
        task_name: str,
        outcome: str,
        reward: float,
        num_turns: int,
        elapsed_s: float,
        prompt_len: int,
        response_len: int,
    ) -> None:
        if self.step != step:
            self._begin_step(step, target_total)

        self.done += 1
        self.outcomes[outcome] = self.outcomes.get(outcome, 0) + 1

        # Per-trial line — captured by wandb's stdout
        print(
            f"[Trial] step={step} {self.done}/{self.target_total} "
            f"task={task_name[:55]} outcome={outcome} reward={reward:.2f} "
            f"turns={num_turns} elapsed={elapsed_s:.0f}s "
            f"prompt={prompt_len} resp={response_len}",
            flush=True,
        )

        if (
            self.done - self.last_summary_at >= self.summary_interval
            or self.done >= self.target_total
        ):
            self._print_summary()
            self.last_summary_at = self.done

    def _begin_step(self, step: int, target_total: int) -> None:
        if self.step is not None and self.target_total > 0 and self.done < self.target_total:
            print(
                f"[Progress] step {self.step} cut short at {self.done}/{self.target_total} "
                f"(new step {step} starting)",
                flush=True,
            )
        self.step = step
        self.target_total = max(int(target_total), 1)
        self.done = 0
        self.outcomes = {}
        self.t0 = time.time()
        self.last_summary_at = 0
        print(
            f"[Progress] === step {step} starting: target {self.target_total} trials ===",
            flush=True,
        )

    def _print_summary(self) -> None:
        wall = time.time() - self.t0
        filled = int(_DEFAULT_BAR_LEN * self.done / self.target_total)
        bar = "█" * filled + "░" * (_DEFAULT_BAR_LEN - filled)
        eta = wall / max(self.done, 1) * max(self.target_total - self.done, 0)
        outcomes_str = " ".join(
            f"{k}={v}" for k, v in sorted(self.outcomes.items(), key=lambda kv: -kv[1])
        )
        pct = 100 * self.done / self.target_total
        print(
            f"[Progress] step={self.step} [{bar}] {self.done}/{self.target_total} "
            f"({pct:.0f}%) wall={wall:.0f}s eta={eta:.0f}s | {outcomes_str}",
            flush=True,
        )


def get_or_create_tracker():
    """Return a handle to the named tracker actor; create if missing.

    Returns None on any error (e.g. Ray not initialised in unit-test paths).
    Workers should call ``report.remote(...)`` fire-and-forget; we never block
    the rollout on tracker availability.
    """
    if not ray.is_initialized():
        return None
    name = _actor_name()
    try:
        return HarborProgressTracker.options(
            name=name,
            namespace=_ACTOR_NAMESPACE,
            get_if_exists=True,
            lifetime="detached",
        ).remote()
    except Exception as exc:  # pragma: no cover
        print(f"[ProgressTracker] failed to obtain actor: {exc}", flush=True)
        return None


def classify_outcome(
    *,
    is_timeout: bool,
    is_context_error: bool,
    has_chat: bool,
    has_tokens: bool,
    reward: float,
) -> str:
    """One label per trial. Mutually exclusive."""
    if is_timeout:
        return "timeout"
    if is_context_error:
        return "context_error"
    if not has_chat:
        return "no_chat_history"
    if not has_tokens:
        return "no_tokens"
    if reward and reward > 0:
        return "success"
    return "zero_reward"
