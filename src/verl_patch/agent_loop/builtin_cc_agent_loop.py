# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
BuiltinCCAgentLoop - Harbor SWE-bench agent loop for verl AgentLoopManager.

This agent loop integrates Harbor's trial module with verl's latest AgentLoopManager
architecture. It runs Harbor's trial.run() to execute complete agent trajectories
for SWE-bench tasks and converts the results to verl's AgentLoopOutput format.

Architecture (Design A):
    Harbor manages the entire agent loop:
    - Calls the in-process vLLM chat-completion proxy (see
      ``_VLLMChatCompletionsProxy`` below) directly — claude-code hits the
      proxy's Anthropic ``/v1/messages`` endpoint, no standalone LiteLLM. The
      proxy forwards every chat completion to ``server_manager.generate(...)``
      so partial-rollout abort/retry from ``FullyAsyncLLMServerManager`` becomes
      transparent to Harbor.
    - Executes tools (bash, file operations, docker interactions)
    - Generates multi-turn chat history
    - Returns ATIF trajectory (chat history + metadata)

    This loop bridges Harbor's trial output to verl's AgentLoopOutput:
    - Input: task data (harbor_task_path, extra_info) from batch
    - Execution: Harbor Trial.run() with configured agent
    - Output: AgentLoopOutput with chat history as prompt/response tokens

Key features:
    - Multi-turn conversation support via Harbor's agent loop
    - Tool calling (bash, file edit, docker) via Harbor's tool suite
    - SWE-bench task verification via Harbor's verifier
    - Automatic reward extraction (0.0 or 1.0 based on test pass rate)
    - Support for Claude Code and Terminus-2 agents
    - Partial-rollout transparent recovery via in-process vLLM proxy

Integration points:
    - Acquires verl's ``AsyncLLMServerManager`` from AgentLoopManager
    - Spawns an aiohttp proxy on ``0.0.0.0`` (all interfaces) and advertises
      this host's primary IPv4 in ``api_base`` so remote Harbor sandboxes can
      reach it like a normal vLLM HTTP endpoint
      requests into ``server_manager.generate(...)``
    - Passes proxy URL to Harbor agent via agent.kwargs.api_base
    - Converts Harbor's ATIF chat history to OpenAI-format messages
    - Tokenizes messages using verl's tokenizer/processor
    - Returns reward score in AgentLoopOutput.reward_score
"""

import json
import logging
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

# Harbor phase timing keys reported via AgentLoopOutput.metrics. They surface as
# ``timing_s/agent_loop/{key}/{min,max,mean}`` in the trainer's timing metrics.
# - harbor_env_setup      : environment build + start
# - harbor_agent_setup    : agent install / setup inside environment
# - harbor_agent_execute  : LLM + tool-call loop execution
# - harbor_verifier       : verifier run time
# - harbor_total          : trial.run() total wall time (includes retries)
_HARBOR_TIMING_KEYS = (
    "harbor_env_setup",
    "harbor_agent_setup",
    "harbor_agent_execute",
    "harbor_verifier",
    "harbor_total",
    "harbor_agent_tool_call",
    "harbor_agent_model_inference",
)

# Proxy-side counters surfaced as `timing_s/agent_loop/{key}/{min,max,mean}` and
# in extra_fields. Pre-populated alongside ``_HARBOR_TIMING_KEYS`` so empty
# trials still aggregate cleanly.
_PROXY_METRIC_KEYS = (
    "proxy_num_calls",            # number of chat completion calls handled
    "proxy_num_aborts",           # number of calls whose underlying vLLM request was aborted
    "proxy_num_preempted",        # sum of per-call ``num_preempted`` reported by vLLM
    "proxy_total_prompt_tokens",  # sum of prompt token counts across calls
    "proxy_total_completion_tokens",  # sum of completion token counts across calls
    # Per-turn tool/agent gap (sec between one completion and the next request ≈ one round
    # of in-pod tool execution + reasoning), summarized per trajectory. verl reports each as
    # timing_s/agent_loop/<key>/{min,max,mean} across trajectories, so e.g.
    # proxy_tool_gap_max/max is the single slowest tool round in the whole step (catches hangs).
    "proxy_tool_gap_mean",        # mean inter-turn gap within this trajectory
    "proxy_tool_gap_max",         # slowest single inter-turn gap within this trajectory
    "proxy_tool_gap_min",         # fastest single inter-turn gap within this trajectory
    "proxy_tool_gap_p90",         # p90 inter-turn gap within this trajectory
)

# Canonical trajectory-termination taxonomy (single source of truth in verl). The string
# values are the contract that flows through extra_fields -> non_tensor_batch -> the
# trajectory filter, so the fallback mirrors them verbatim if verl can't be imported
# (e.g. the old verl checkout without this module).
try:
    from verl.trainer.ppo.trajectory_filter import TerminationReason
except Exception:  # pragma: no cover - keeps rollout alive on old verl
    class TerminationReason:
        AGENT_COMPLETED = "agent_completed"
        OVERLONG = "overlong"
        MAX_TURNS_REACHED = "max_turns_reached"
        TIMEOUT = "timeout"
        ENV_SETUP_FAILED = "env_setup_failed"

# Strict (RL-training) behaviour gives up immediately on an agent crash
# (NonZeroAgentExitCodeError) or context-length error — retrying greedy rollouts
# is pointless because they're deterministic. With temperature sampling (eval),
# a fresh attempt explores a different trajectory and can recover a task that
# crashed/looped the first time. Set HARBOR_RETRY_ON_CRASH=1 (eval) to retry
# crashed/context-error attempts up to max_retries (only while reward<=0 and
# attempts remain; a partial patch that already passed is kept, not retried).
# Default 0 keeps the strict no-retry-on-crash training behaviour.
_RETRY_ON_CRASH = os.environ.get(
    "HARBOR_RETRY_ON_CRASH", "0"
).lower() in ("1", "true", "yes")

import numpy as np
import torch

from omegaconf import OmegaConf
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

# Harbor imports - wrapped in try/except for graceful degradation
try:
    from harbor.trial.trial import Trial
    from harbor.models.trial.config import TrialConfig
    HARBOR_AVAILABLE = True
except ImportError:
    HARBOR_AVAILABLE = False
    Trial = None
    TrialConfig = None

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
logger.setLevel("DEBUG")

# R3 diagnostic: emit a few WARN lines per worker process to confirm routing capture,
# then go quiet (avoid per-trajectory spam in real runs).
_R3_DIAG_LOGGED = 0
_R3_CMP_LOGGED = 0

# Suppress urllib3 warnings for Harbor's HTTPS connections with verify=False
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    pass

# Maximum retries for Harbor trial execution
MAX_RETRIES = 2

# Agent names that speak the Anthropic API format. The in-process proxy serves
# these via its /v1/messages endpoint (no standalone LiteLLM proxy involved).
_ANTHROPIC_FORMAT_AGENTS = {"claude-code"}

from .vllm_chat_completion_proxy import (  # pyright: ignore[reportMissingImports]
    _ensure_proxy_started,
)


def _safe_session_slug(task_path: str) -> str:
    """Build a short, filesystem-safe-ish slug from a task_path for tracing."""
    name = Path(task_path).name if task_path else "task"
    # keep only [a-zA-Z0-9._-]
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)
    return safe[:48] or "task"


class BuiltinCCAgentLoop(AgentLoopBase):
    """
    Harbor SWE-bench agent loop for verl AgentLoopManager.

    This agent loop runs complete Harbor trials for SWE-bench tasks and
    converts the results to verl's AgentLoopOutput format for training.

    The loop manages:
    1. Harbor trial initialization with task path and agent config
    2. Trial execution (multi-turn agent + tool calls + verification)
    3. Chat history extraction and conversion to OpenAI format
    4. Reward extraction (verifier result)
    5. Tokenization and output formatting

    Key config (via rollout_config.agent):
        harbor_cfg (dict): Harbor TrialConfig dict (default.yaml + overrides)
        trials_dir (str): Base directory for Harbor trial outputs
        max_retries (int): Max retry attempts for failed trials
        agent_type (str): "claude-code" or other agent name
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not HARBOR_AVAILABLE:
            raise ImportError(
                "Harbor is not installed. Please install harbor package to use BuiltinCCAgentLoop. "
                "pip install harbor"
            )

        # Harbor configuration
        # self.harbor_cfg = OmegaConf.to_container(self.rollout_config.agent.get("harbor_cfg", {}), resolve=True)
        self.harbor_cfg = OmegaConf.to_container(kwargs.get("harbor_cfg", {}), resolve=True)
        self.base_trials_dir = self.harbor_cfg.get("agent", {}).get("trials_dir", "./harbor_trials")
        self.max_retries = self.harbor_cfg.get("agent", {}).get("max_retries", MAX_RETRIES)

        # Prompt/response length limits
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        # Turn cap (HARBOR_AGENT_MAX_ITERATIONS) — used only to label a trajectory that
        # exhausted it as termination_reason="max_turns_reached". 0 => unknown/disabled.
        self._max_turns = int(os.environ.get("HARBOR_AGENT_MAX_ITERATIONS", "0") or "0")

        # Detect agent type
        agent_name = self.harbor_cfg.get("agent", {}).get("name", "")
        self.is_claude_code = agent_name in _ANTHROPIC_FORMAT_AGENTS

        self._tool_parser_name = self.harbor_cfg.get("agent", {}).pop("tool_parser", "hermes")

        # Early termination: stop a session after N consecutive assistant turns
        # without any tool_calls. 0 = disabled (default).
        self._max_consecutive_no_tool = int(
            self.harbor_cfg.get("agent", {}).get(
                "max_consecutive_no_tool",
                os.getenv("HARBOR_MAX_CONSECUTIVE_NO_TOOL", "0"),
            )
        )

        # When True (default), AgentLoopOutput token fields come from the in-process
        # proxy trajectory (exact rollout tokens + masks + optional logprobs). When
        # False or when using a VLM processor (proxy disables trajectory), fall back
        # to ``_tokenize_chat_history`` on Harbor chat transcripts.
        self._prefer_proxy_trajectory = bool(
            self.harbor_cfg.get("agent", {}).get("prefer_proxy_trajectory", True)
        )

        # vLLM endpoint will be injected via server_manager at runtime
        logger.info(
            "BuiltinCCAgentLoop initialized. agent=%s  trials_dir=%s  max_retries=%d "
            "tool_parser=%s prefer_proxy_trajectory=%s max_consecutive_no_tool=%d",
            agent_name,
            self.base_trials_dir,
            self.max_retries,
            self._tool_parser_name,
            self._prefer_proxy_trajectory,
            self._max_consecutive_no_tool,
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """
        Run Harbor trial for SWE-bench task and return AgentLoopOutput.

        Args:
            sampling_params: LLM sampling parameters (temperature, top_p, etc.)
            **kwargs: Batch fields from dataset (must include 'raw_prompt', 'extra_info')

        Returns:
            AgentLoopOutput with:
                - prompt_ids: Tokenized chat history (prompt part)
                - response_ids: Tokenized assistant response
                - response_mask: 1s for LLM tokens, 0s for tool responses
                - reward_score: Verifier reward (0.0 or 1.0)
                - num_turns: Number of chat turns
                - metrics: Performance metrics
                - extra_fields: Additional metadata (turn_scores, tool_rewards)
        """
        # Extract task data from kwargs
        raw_prompt = kwargs.get("raw_prompt", [])
        extra_info = kwargs.get("extra_info", {})

        # Get Harbor task path
        task_path = (
            extra_info.get("harbor_task_path")
            or extra_info.get("task_path")
            or (raw_prompt[0].get("content") if raw_prompt else "")
        )

        if not task_path:
            logger.error("No task_path found in kwargs")
            return self._make_empty_output(kwargs)

        # Get global step for trial directory organization
        global_steps = kwargs.get("global_steps", 0)

        # Timing metrics - pre-populate with harbor phase keys so aggregation is
        # stable across samples even when a trial ends early (e.g. empty output
        # before verification). Harbor phase timings are filled in by
        # ``_run_harbor_trial`` based on ``TrialResult.{environment_setup,
        # agent_setup, agent_execution, verifier, started_at/finished_at}``.
        metrics: dict[str, float] = {k: 0.0 for k in _HARBOR_TIMING_KEYS}
        for k in _PROXY_METRIC_KEYS:
            metrics.setdefault(k, 0.0)

        # Spin up the in-process vLLM proxy (singleton per Ray actor).
        # Per-attempt sessions are created inside ``_run_harbor_trial`` so each
        # retry gets a clean session (no cross-attempt trajectory contamination).
        proxy = await _ensure_proxy_started(self)

        # print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [DEBUG] in run, start run harbor trial, task_path: {task_path}, global_steps: {global_steps}, sampling_params: {sampling_params}", flush=True)
        with simple_timer("harbor_total", metrics):
            reward, trial_reason, session_meta = await self._run_harbor_trial(
                task_path=task_path,
                global_steps=global_steps,
                sampling_params=sampling_params,
                proxy=proxy,
                metrics=metrics,
            )

        # Alias harbor_total into the standard ``generate_sequences`` key so verl's
        # ``AgentLoopManager._performance_metrics`` slowest-sample logic -- which
        # uses ``argmax(generate_sequences + tool_calls + compute_score)`` -- can
        # pick the slowest trial based on actual wall-clock trial duration.
        metrics["generate_sequences"] = metrics.get("harbor_total", 0.0)

        # Token source of truth is the in-process proxy trajectory: the exact
        # rollout token_ids + masks + per-token logprobs that vLLM sampled,
        # archived turn-by-turn in ``_finalize_generation``. Claude Code never
        # populates ``agent_result.metadata['all_messages']`` (it talks to the
        # proxy's /v1/messages endpoint directly under Design A; the LiteLLM
        # trajectory log harbor_patch.ClaudeCode used to read no longer exists),
        # so -- exactly like ``BuiltinSWEAgentLoop`` -- we build the output
        # straight from ``session_meta`` and NEVER re-tokenize a chat transcript.
        # This keeps token fidelity by construction (no detokenize->retokenize
        # drift, real logprobs) and is robust to crash/timeout/context-error
        # trials: their pre-failure tokens are already in ``traj_acc_ids``.
        traj_acc = session_meta.get("traj_acc_ids") or []
        prompt_token_len = int(session_meta.get("initial_prompt_token_len") or 0)
        tail_mask = session_meta.get("traj_response_mask") or []
        tail_lp = session_meta.get("traj_response_logprobs") or []
        tail_routing = session_meta.get("traj_response_routing") or []
        # R3-cmp probe: is the archive itself asymmetric (routing lost but logprob kept)?
        # Compare, over the SAME response region, routing-non-None vs logprob-non-zero.
        global _R3_CMP_LOGGED
        if _R3_CMP_LOGGED < 0:  # [R3-cmp] probe disabled (set >0 to re-enable)
            _R3_CMP_LOGGED += 1
            _rt_nn = sum(1 for r in tail_routing if r is not None)
            _lp_nz = sum(1 for x in tail_lp if x not in (0.0, 0))
            _m1 = sum(1 for m in tail_mask if m)
            print(
                f"[R3-cmp] len(mask)={len(tail_mask)} len(lp)={len(tail_lp)} len(routing)={len(tail_routing)} "
                f"mask1={_m1} routing_nonNone={_rt_nn} lp_nonzero={_lp_nz}",
                flush=True,
            )
        traj_ok = (
            not session_meta.get("disable_proxy_trajectory")
            and traj_acc
            and prompt_token_len <= len(traj_acc)
            and len(tail_mask) == (len(traj_acc) - prompt_token_len)
            and len(tail_lp) == len(tail_mask)
        )

        if not traj_ok:
            logger.warning(
                "Invalid proxy trajectory (task=%s): traj_len=%d prompt_len=%d "
                "mask_len=%d logprob_len=%d disable=%s",
                task_path,
                len(traj_acc),
                prompt_token_len,
                len(tail_mask),
                len(tail_lp),
                session_meta.get("disable_proxy_trajectory"),
            )
            return self._make_empty_output(kwargs, metrics=metrics, session_meta=session_meta)

        prompt_ids = traj_acc[:prompt_token_len]
        response_ids = traj_acc[prompt_token_len:]
        response_mask = tail_mask
        response_logprobs = tail_lp
        # R3: per-token routing aligned to the response region (same length as the
        # mask/logprobs). Empty when routing was not captured (R3 off / non-MoE).
        response_routing = tail_routing if len(tail_routing) == len(tail_mask) else []

        if not prompt_ids or not response_ids:
            logger.warning(
                "Empty proxy trajectory token ids (prompt=%d, response=%d), task=%s",
                len(prompt_ids), len(response_ids), task_path,
            )
            return self._make_empty_output(kwargs, metrics=metrics, session_meta=session_meta)

        # Real verifier reward for every kept trajectory (agent_completed / overlong /
        # max_turns_reached). Categories the trajectory filter drops (timeout,
        # env_setup_failed) get their reward+advantage zeroed there, so no force-0 here.
        reward_score = reward

        # Resolve global_steps span. ``min/max_global_steps`` from the proxy's
        # session metadata reflects all weight versions actually observed
        # during this trial's vLLM calls (across partial-rollout retries).
        # When the proxy never received a call (extremely unlikely here but
        # guard anyway), fall back to the step at trial dispatch.
        cur_step = kwargs.get("global_steps", 0)
        min_gs = session_meta.get("min_global_steps")
        max_gs = session_meta.get("max_global_steps")
        min_gs = min_gs if min_gs is not None else cur_step
        max_gs = max_gs if max_gs is not None else cur_step

        # num_turns from the proxy's messages snapshot (Claude Code chat history
        # is not available verl-side; the snapshot mirrors what the model saw).
        messages_snapshot = session_meta.get("messages_snapshot") or []

        # R3: full-length [prompt+response, num_layers, topk] routing (None when R3 off).
        routed_experts = self._build_routed_experts(prompt_ids, response_routing[:self.response_length])
        global _R3_DIAG_LOGGED
        if _R3_DIAG_LOGGED < 0:  # [R3] CAPTURED probe disabled (set >0 to log first N)
            _R3_DIAG_LOGGED += 1
            if routed_experts is not None:
                _resp = routed_experts[prompt_token_len:]
                _nz = int((_resp.reshape(_resp.shape[0], -1) != 0).any(axis=1).sum())
                print(
                    f"[R3] CAPTURED task={Path(task_path).name} routed_experts={tuple(routed_experts.shape)} "
                    f"nonzero_resp_rows={_nz} response_mask_sum={int(sum(response_mask[:self.response_length]))}",
                    flush=True,
                )
            else:
                # routing was lost between proxy and here — show why
                print(
                    f"[R3] NONE task={Path(task_path).name} tail_routing_len={len(tail_routing)} "
                    f"tail_mask_len={len(tail_mask)} resp_routing_len={len(response_routing)}",
                    flush=True,
                )

        # Single termination classification (consumed by algorithm.trajectory_filter).
        # _run_harbor_trial returns the trial-level reason (agent_completed / timeout /
        # overlong-from-context-overflow). Refine a plain completion here with the two
        # signals only known after assembling the trajectory: exhausting the turn cap
        # (max_turns_reached) and a response longer than the training window — truncated by
        # the slicing below, its verifier verdict reflecting a budget limit (DAPO "overlong").
        # Priority: context overflow (upstream) > max_turns_reached > response-window overlong.
        num_turns = len([m for m in messages_snapshot if m.get("role") == "assistant"])
        termination_reason = trial_reason
        if termination_reason == TerminationReason.AGENT_COMPLETED:
            if self._max_turns and num_turns >= self._max_turns:
                termination_reason = TerminationReason.MAX_TURNS_REACHED
            elif len(response_ids) > self.response_length:
                termination_reason = TerminationReason.OVERLONG

        # Build output
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:self.response_length],
            response_mask=response_mask[:self.response_length],
            response_logprobs=response_logprobs[:self.response_length],
            routed_experts=routed_experts,
            multi_modal_data=None,
            reward_score=reward_score,
            num_turns=num_turns,
            metrics=metrics,
            extra_fields={
                "turn_scores": [],
                "tool_rewards": [],
                "termination_reason": termination_reason,
                "raw_prompt": raw_prompt,
                "min_global_steps": min_gs,
                "max_global_steps": max_gs,
                "proxy_num_calls": int(metrics.get("proxy_num_calls", 0)),
                "proxy_num_aborts": int(metrics.get("proxy_num_aborts", 0)),
                "proxy_num_preempted": int(metrics.get("proxy_num_preempted", 0)),
                "trajectory_token_source": "proxy_tokens",
                "tool_parser": self._tool_parser_name,
            },
        )

        return output

    @staticmethod
    def _build_routed_experts(prompt_ids: list, response_routing: list):
        """Assemble the full-length ``[prompt+response, num_layers, topk]`` routing
        tensor expected by verl's agent-loop ``_pad`` (which places routed_experts at
        the real prompt start, unlike the response-only logprobs).

        ALIGNMENT CONVENTION: ``routed_experts[j]`` is the routing of the forward
        pass AT position j. vLLM's capturer reports, for each sampled token, the
        routing of the forward that GENERATED it — and that forward ran at the
        token's PREVIOUS position — so each captured row is stored at
        ``position(token) - 1``. Training engines must therefore consume rows
        as-is, with NO shift of their own (the FSDP engine's old ``roll -1`` and
        any equivalent are superseded by this placement). One row per turn lands
        on the last context token of the preceding prompt/tool chunk (for the
        first turn that is inside the prompt region), which is where that forward
        really ran.

        The prompt region and any non-sampled / missing positions are zero
        placeholders ("not recorded — let the model route natively"); only
        forwards that produced sampled tokens carry real expert ids. Returns
        ``None`` when no routing was captured (R3 off / non-MoE), so the field
        stays absent and training is unaffected.
        """
        dims = None
        for r in response_routing:
            if r is not None:
                dims = (len(r), len(r[0]))
                break
        if dims is None:
            return None
        num_layers, topk = dims
        plen = len(prompt_ids)
        # uint8: expert ids are 0..255 (num_experts=256), so this fits and is 8x
        # smaller than int64 (~335MB->42MB per 128K trajectory); verl's padding /
        # the router cast routed ids to long at use, so dtype here is just storage.
        arr = np.zeros((plen + len(response_routing), num_layers, topk), dtype=np.uint8)
        for i, r in enumerate(response_routing):
            if r is not None and plen + i - 1 >= 0:
                arr[plen + i - 1] = np.asarray(r, dtype=np.uint8)
        return arr

    @staticmethod
    def _timing_duration(timing_info) -> float:
        """Convert a harbor ``TimingInfo`` (or any obj with started_at/finished_at)
        to a duration in seconds. Returns 0.0 when information is missing.
        """
        if timing_info is None:
            return 0.0
        started_at = getattr(timing_info, "started_at", None)
        finished_at = getattr(timing_info, "finished_at", None)
        if started_at is None or finished_at is None:
            return 0.0
        try:
            return max(0.0, (finished_at - started_at).total_seconds())
        except Exception:  # pragma: no cover - defensive
            return 0.0

    def _collect_harbor_timings(self, results, metrics: dict[str, float]) -> None:
        """Populate per-phase harbor timings into ``metrics`` from a
        ``TrialResult`` instance. Keys are additive so retries accumulate.
        """
        if results is None:
            return
        metrics["harbor_env_setup"] = (
            metrics.get("harbor_env_setup", 0.0)
            + self._timing_duration(getattr(results, "environment_setup", None))
        )
        metrics["harbor_agent_setup"] = (
            metrics.get("harbor_agent_setup", 0.0)
            + self._timing_duration(getattr(results, "agent_setup", None))
        )
        metrics["harbor_agent_execute"] = (
            metrics.get("harbor_agent_execute", 0.0)
            + self._timing_duration(getattr(results, "agent_execution", None))
        )
        metrics["harbor_verifier"] = (
            metrics.get("harbor_verifier", 0.0)
            + self._timing_duration(getattr(results, "verifier", None))
        )
        agent_metadata = (
            getattr(results, "agent_result", None).metadata
            if getattr(results, "agent_result", None) is not None
            else None
        )
        if isinstance(agent_metadata, dict):
            metrics["harbor_agent_tool_call"] = (
                metrics.get("harbor_agent_tool_call", 0.0)
                + float(agent_metadata.get("harbor_agent_tool_call") or 0.0)
            )
            metrics["harbor_agent_model_inference"] = (
                metrics.get("harbor_agent_model_inference", 0.0)
                + float(agent_metadata.get("harbor_agent_model_inference") or 0.0)
            )

    @staticmethod
    def _assign_proxy_metrics_from_session(metrics: dict[str, float], meta: dict) -> None:
        """Overwrite proxy counters from a single completed proxy session (one Harbor attempt)."""
        metrics["proxy_num_calls"] = float(meta.get("num_calls") or 0)
        metrics["proxy_num_aborts"] = float(meta.get("num_aborts") or 0)
        metrics["proxy_num_preempted"] = float(meta.get("num_preempted") or 0)
        metrics["proxy_total_prompt_tokens"] = float(meta.get("total_prompt_tokens") or 0)
        metrics["proxy_total_completion_tokens"] = float(
            meta.get("total_completion_tokens") or 0
        )
        # Per-turn tool/agent gap stats for this trajectory (sec). One gap per round of
        # in-pod tool execution + reasoning between LLM calls; CC's individual tools run in
        # the pod and aren't visible here, so this is the closest verl-side signal.
        gaps = [float(g) for g in (meta.get("tool_gap_secs") or [])]
        if gaps:
            arr = np.asarray(gaps, dtype=np.float64)
            metrics["proxy_tool_gap_mean"] = float(arr.mean())
            metrics["proxy_tool_gap_max"] = float(arr.max())
            metrics["proxy_tool_gap_min"] = float(arr.min())
            metrics["proxy_tool_gap_p90"] = float(np.percentile(arr, 90))

    async def _run_harbor_trial(
        self,
        task_path: str,
        global_steps: int,
        sampling_params: dict[str, Any],
        proxy: Any,
        metrics: dict[str, float] | None = None,
    ) -> tuple[float, str, dict]:
        """
        Execute Harbor trial with retries. Each retry uses a new proxy ``session_id``,
        ``api_base``, and Harbor ``trial_name`` so KV / logs do not leak across attempts.

        Returns:
            reward, termination_reason (trial-level: agent_completed / timeout / overlong),
            session_meta from the **terminal** attempt (any ``break`` path). The caller
            refines agent_completed into max_turns_reached / overlong / env_setup_failed
            using response length, turn count and trajectory validity. Retries that
            ``continue`` do not contribute
            to ``metrics`` or ``session_meta``. The trajectory token_ids/masks/logprobs
            live in ``session_meta`` (proxy-archived); chat transcripts are not used.
        """
        if metrics is None:
            metrics = {}

        cfg = deepcopy(self.harbor_cfg)
        cfg["trials_dir"] = f"{self.base_trials_dir}/step_{global_steps:04d}"
        cfg["task"] = {"path": task_path}

        reward = 0.0
        trial_reason = TerminationReason.AGENT_COMPLETED
        session_meta: dict = {}

        # Claude Code (Anthropic format) now connects to the in-process proxy
        # directly via its /v1/messages endpoint (Design A) — no LiteLLM. Each
        # attempt's ANTHROPIC_BASE_URL is set to the proxy's per-session URL
        # below so the sampled token_ids/log_probs are archived with fidelity.

        for attempt in range(self.max_retries):
            terminal_commit = False
            prefix = f"[task={Path(task_path).name}] attempt {attempt + 1}/{self.max_retries}"
            session_id = f"{_safe_session_slug(task_path)}-{uuid4().hex[:8]}"
            api_base = proxy.session_url(session_id)
            anthropic_base = proxy.session_anthropic_base(session_id)
            trajectory_dir = Path(self.base_trials_dir) / f"step_{global_steps:04d}" / session_id
            await proxy.open_session(session_id, trajectory_dir=trajectory_dir)

            cfg.setdefault("agent", {}).setdefault("kwargs", {})["api_base"] = api_base
            # Point Claude Code's Anthropic client at this session's proxy URL.
            cfg.setdefault("agent", {}).setdefault("env", {})["ANTHROPIC_BASE_URL"] = anthropic_base
            cfg["trial_name"] = session_id

            if "temperature" in sampling_params:
                cfg.setdefault("agent", {}).setdefault("kwargs", {})["temperature"] = sampling_params["temperature"]
            if "top_p" in sampling_params:
                cfg.setdefault("agent", {}).setdefault("kwargs", {})["top_p"] = sampling_params["top_p"]

            results = None
            try:
                # Validate and create trial
                trial_config = TrialConfig.model_validate(cfg)
                trial = await Trial.create(trial_config)

                # Run trial (async)
                results = await trial.run()

                # Check for errors
                exc_type = (
                    results.exception_info.exception_type
                    if results.exception_info
                    else None
                )
                exc_msg = (
                    str(results.exception_info.exception_message or "")
                    if results.exception_info
                    else ""
                )

                is_context_overflow = (
                    exc_type == "ContextLengthExceededError"
                    or "maximum context length" in exc_msg
                    or "leaves no room to generate" in exc_msg
                    or "context_length_exceeded" in exc_msg
                )
                is_timeout = (
                    exc_type == "AgentTimeoutError"
                    or exc_type == "VerifierTimeoutError"
                    or "Verifier execution timed out" in exc_msg
                )
                agent_exited_nonzero = (
                    exc_type == "NonZeroAgentExitCodeError"
                    or "Command failed (exit" in exc_msg
                )

                # Real verifier reward whenever harbor produced a verdict — used for ALL
                # kept categories (completed / overlong / max_turns_reached). Only
                # env-setup failures (no usable trajectory) and the filter's drop list
                # end up at reward 0.
                verifier_reward = (
                    float(results.verifier_result.rewards.get("reward", 0.0))
                    if results.verifier_result
                    else 0.0
                )

                if is_timeout:
                    logger.warning("%s timeout (no retry). results=%s", prefix, results)
                    reward = verifier_reward
                    trial_reason = TerminationReason.TIMEOUT
                    terminal_commit = True
                    break

                if is_context_overflow:
                    # Prompt overflowed the model context window mid-rollout. The proxy
                    # already archived the pre-overflow tokens, so the (truncated)
                    # trajectory is usable and KEPT with its real verifier reward.
                    reward = verifier_reward
                    trial_reason = TerminationReason.OVERLONG
                    logger.debug("%s context overflow — committing reward=%.3f", prefix, reward)
                    terminal_commit = True
                    break

                if agent_exited_nonzero:
                    # Non-zero agent exit with a real (partial) trajectory + verifier
                    # verdict: treat as completed; the proxy's context_overflow signal
                    # (consulted after the loop) upgrades it to overlong when applicable.
                    logger.warning("%s agent exited non-zero (no retry). results=%s", prefix, results)
                    reward = verifier_reward
                    trial_reason = TerminationReason.AGENT_COMPLETED
                    terminal_commit = True
                    break

                elif not results.verifier_result:
                    logger.warning(
                        "%s no verifier result (exception=%s), retrying.",
                        prefix,
                        results.exception_info,
                    )
                    continue

                else:
                    reward = verifier_reward
                    trial_reason = TerminationReason.AGENT_COMPLETED
                    logger.debug("%s success  reward=%.3f", prefix, reward)
                    terminal_commit = True
                    break

            except Exception as exc:
                print(f"[DEBUG] {prefix} exception: {exc}, results={results}")
                logger.warning("%s exception: %s  results=%s", prefix, exc, results)
                continue
            finally:
                attempt_meta = await proxy.pop_session(session_id)
                if terminal_commit:
                    self._collect_harbor_timings(results, metrics)
                    self._assign_proxy_metrics_from_session(metrics, attempt_meta)
                    session_meta = attempt_meta

        # Authoritative overflow signal: the proxy sets ``context_overflow`` the moment a
        # generate call is refused for exceeding the model context, even when the error is
        # later swallowed/relabelled crossing the SDK boundary. Upgrade a plain completion.
        if session_meta.get("context_overflow") and trial_reason == TerminationReason.AGENT_COMPLETED:
            trial_reason = TerminationReason.OVERLONG

        return reward, trial_reason, session_meta

    def _make_empty_output(
        self,
        kwargs: dict,
        metrics: dict[str, float] | None = None,
        session_meta: dict | None = None,
    ) -> AgentLoopOutput:
        """
        Create empty AgentLoopOutput for failed trials.

        Args:
            kwargs: Original kwargs for extracting raw_prompt
            metrics: Optional timing metrics already collected (harbor phase
                timings, total, etc.). Missing keys are filled with defaults.
            session_meta: Optional proxy session metadata (num_calls,
                num_aborts, min/max_global_steps). Used to keep
                ``min/max_global_steps`` accurate even when the trial failed
                mid-flight after observing several weight versions.

        Returns:
            AgentLoopOutput with empty trajectory
        """
        base_metrics: dict[str, float] = {
            "generate_sequences": 0.0,
            "tool_calls": 0.0,
            "num_preempted": -1,
        }
        for key in _HARBOR_TIMING_KEYS:
            base_metrics.setdefault(key, 0.0)
        for key in _PROXY_METRIC_KEYS:
            base_metrics.setdefault(key, 0.0)
        if metrics:
            base_metrics.update(metrics)
        # Keep ``generate_sequences`` aligned with ``harbor_total`` so verl's
        # slowest-sample selection reflects actual trial wall-clock time.
        base_metrics["generate_sequences"] = base_metrics.get("harbor_total", 0.0)

        session_meta = session_meta or {}
        cur_step = kwargs.get("global_steps", 0)
        min_gs = session_meta.get("min_global_steps")
        max_gs = session_meta.get("max_global_steps")
        min_gs = min_gs if min_gs is not None else cur_step
        max_gs = max_gs if max_gs is not None else cur_step
        return AgentLoopOutput(
            prompt_ids=[0],  # dummy token
            response_ids=[0],
            # mask=1 (not 0) so a batch made entirely of failed trials still has
            # >=1 valid token; otherwise verl's compute_rollout_correction asserts
            # "response_mask must contain at least one valid token" and crashes the
            # whole fit step. reward=0 + single token => ~zero gradient, so this
            # failed sample contributes no learning signal but keeps the run alive.
            response_mask=[1],
            response_logprobs=[0],
            routed_experts=None,
            multi_modal_data=None,
            reward_score=0.0,
            num_turns=0,
            metrics=base_metrics,
            extra_fields={
                "turn_scores": [],
                "tool_rewards": [],
                # Env-setup failure: the trial never produced a usable trajectory (env setup
                # failed, invalid/empty proxy trajectory). The dummy 1-token sample only
                # keeps the batch alive; algorithm.trajectory_filter drops it (it is in the
                # default drop_reasons) from the loss and GRPO group statistics.
                "termination_reason": TerminationReason.ENV_SETUP_FAILED,
                "raw_prompt": kwargs.get("raw_prompt", []),
                "min_global_steps": min_gs,
                "max_global_steps": max_gs,
                "proxy_num_calls": int(base_metrics.get("proxy_num_calls", 0)),
                "proxy_num_aborts": int(base_metrics.get("proxy_num_aborts", 0)),
                "proxy_num_preempted": int(base_metrics.get("proxy_num_preempted", 0)),
                "trajectory_token_source": 'dummy',
                "tool_parser": self._tool_parser_name,
            },
        )