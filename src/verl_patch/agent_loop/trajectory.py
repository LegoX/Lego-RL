"""Capture generated tokens independently of the harness's mutable transcript.

A segment is a linear sequence whose trained tokens retain their original left
context. Rewrites start a new segment; historical assistant messages in its
prompt are conditioning, never newly sampled actions.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrajectorySegment:
    prompt_ids: list[int]
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    response_logprobs: list[float] | None = field(default_factory=list)
    response_routing: list[Any] = field(default_factory=list)
    # Response offsets and policy versions for each actual generation. These
    # boundaries also let a last-assistant retry undo precisely one generation.
    generations: list[tuple[int, int, int | None, int | None]] = field(
        default_factory=list
    )

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_ids + self.response_ids

    @property
    def num_model_tokens(self) -> int:
        return sum(self.response_mask)

    def append_context(self, ids: list[int]) -> None:
        self.response_ids.extend(ids)
        self.response_mask.extend([0] * len(ids))
        if self.response_logprobs is not None:
            self.response_logprobs.extend([0.0] * len(ids))
        self.response_routing.extend([None] * len(ids))

    def as_dict(self) -> dict[str, Any]:
        versions = [
            (lo, hi)
            for _, _, lo, hi in self.generations
            if lo is not None and hi is not None
        ]
        return {
            "prompt_ids": list(self.prompt_ids),
            "response_ids": list(self.response_ids),
            "response_mask": list(self.response_mask),
            "response_logprobs": None
            if self.response_logprobs is None
            else list(self.response_logprobs),
            "response_routing": deepcopy(self.response_routing),
            "num_turns": len(self.generations),
            "min_global_steps": min((lo for lo, _ in versions), default=None),
            "max_global_steps": max((hi for _, hi in versions), default=None),
        }


class TrajectoryRecorder:
    """Seal a segment only when a successful generation uses a new prefix.

    Preparing or failing a request does not destroy the last usable trajectory.
    Only the proxy's explicit last-assistant retry path calls rollback; arbitrary
    context rewrites never infer that earlier model actions were abandoned.
    """

    def __init__(self) -> None:
        self.segments: list[TrajectorySegment] = []
        self.current: TrajectorySegment | None = None
        self.discarded_generations: list[dict[str, Any]] = []

    def record(
        self,
        prompt_ids: list[int],
        output_ids: list[int],
        logprobs: list[float] | None,
        routing: list[Any],
        *,
        context_tail: list[int] | tuple[int, ...] = (),
        min_global_steps: int | None = None,
        max_global_steps: int | None = None,
    ) -> None:
        if logprobs is not None and len(logprobs) != len(output_ids):
            raise ValueError(
                "Generated token ids and logprobs must have identical lengths"
            )
        if len(routing) != len(output_ids):
            raise ValueError(
                "Generated token ids and routing must have identical lengths"
            )
        if not output_ids:
            return
        previous = self.current.token_ids if self.current is not None else []
        if self.current is None or prompt_ids[: len(previous)] != previous:
            if self.current is not None and self.current.num_model_tokens:
                self.segments.append(self.current)
            self.current = TrajectorySegment(prompt_ids=list(prompt_ids))
        else:
            self.current.append_context(prompt_ids[len(previous) :])

        segment = self.current
        start = len(segment.response_ids)
        segment.response_ids.extend(output_ids)
        segment.response_mask.extend([1] * len(output_ids))
        if logprobs is None:
            segment.response_logprobs = None
        elif segment.response_logprobs is not None:
            segment.response_logprobs.extend(logprobs)
        segment.response_routing.extend(deepcopy(routing))
        segment.generations.append(
            (start, len(segment.response_ids), min_global_steps, max_global_steps)
        )
        segment.append_context(list(context_tail))

    def rollback_last_generation(self) -> list[int] | None:
        """Remove a superseded latest assistant; retain an audit record of it."""
        segment = self.current
        if segment is None or not segment.generations:
            return None
        start, end, lo, hi = segment.generations.pop()
        prompt_ids = segment.prompt_ids + segment.response_ids[:start]
        self.discarded_generations.append(
            {
                "reason": "last_assistant_replaced",
                "prompt_ids": prompt_ids,
                "output_ids": segment.response_ids[start:end],
                "output_logprobs": None
                if segment.response_logprobs is None
                else segment.response_logprobs[start:end],
                "min_global_steps": lo,
                "max_global_steps": hi,
            }
        )
        del segment.response_ids[start:]
        del segment.response_mask[start:]
        del segment.response_routing[start:]
        if segment.response_logprobs is not None:
            del segment.response_logprobs[start:]
        return prompt_ids

    def export(self) -> list[dict[str, Any]]:
        segments = self.segments + ([self.current] if self.current is not None else [])
        return [segment.as_dict() for segment in segments if segment.num_model_tokens]


def select_trajectories(
    segments: list[dict[str, Any]], selection: str
) -> list[tuple[int, dict[str, Any]]]:
    """Select by sampled-token count; ties prefer the later segment."""
    if selection not in {"longest", "all"}:
        raise ValueError(
            f"Unknown trajectory_selection={selection!r}; expected 'longest' or 'all'"
        )
    eligible = [
        (i, segment)
        for i, segment in enumerate(segments)
        if any(segment["response_mask"])
    ]
    if selection == "all" or not eligible:
        return eligible
    return [max(eligible, key=lambda item: (sum(item[1]["response_mask"]), item[0]))]
