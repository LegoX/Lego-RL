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
BuiltinSWEAgentLoop - Harbor SWE-bench agent loop for verl AgentLoopManager.

This agent loop integrates Harbor's trial module with verl's latest AgentLoopManager
architecture. It runs Harbor's trial.run() to execute complete agent trajectories
for SWE-bench tasks and converts the results to verl's AgentLoopOutput format.

Architecture:
    Harbor manages the entire agent loop:
    - Calls vLLM via LiteLLM (OpenAI format) — but routed through an in-process
      proxy (see ``_VLLMChatCompletionsProxy`` below) instead of hitting vLLM
      directly. The proxy forwards every chat completion to
      ``server_manager.generate(...)`` so partial-rollout abort/retry from
      ``FullyAsyncLLMServerManager`` becomes transparent to Harbor.
    - Executes tools (bash, file operations, docker interactions)
    - Generates multi-turn chat history
    - Returns ATIF trajectory (chat history + metadata)

    This loop bridges Harbor's trial output to verl's AgentLoopOutput:
    - Input: task data (harbor_task_path, extra_info) from batch
    - Execution: Harbor Trial.run() with configured agent
    - Output: AgentLoopOutput with proxy-captured prompt/response tokens

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
    - Returns reward score in AgentLoopOutput.reward_score
    - Rollout tokens (prompt/response ids, mask, logprobs) come from the proxy session
      trajectory built during chat completions (see ``vllm_chat_completion_proxy``)
"""

import logging
import os
import json
import time
from copy import deepcopy
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
    "proxy_model_inference",      # cumulative ``server_manager.generate`` wall time (seconds)
    "proxy_tool_call",            # cumulative gap between completions and next request (seconds)
    # Per-turn version of proxy_tool_call: one gap per round of in-pod tool execution + reasoning,
    # summarized per trajectory. verl reports each as timing_s/agent_loop/<key>/{min,max,mean}, so
    # proxy_tool_gap_max/max is the slowest single tool round in the whole step (catches hangs).
    "proxy_tool_gap_mean",
    "proxy_tool_gap_max",
    "proxy_tool_gap_min",
    "proxy_tool_gap_p90",
)

from omegaconf import OmegaConf
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.utils.profiler import simple_timer

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

# Suppress urllib3 warnings for Harbor's HTTPS connections with verify=False
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    pass

# Maximum retries for Harbor trial execution
MAX_RETRIES = 2

import numpy as np

from .vllm_chat_completion_proxy import (  # pyright: ignore[reportMissingImports]
    _ensure_proxy_started,
)

# Canonical trajectory-termination taxonomy (single source of truth in verl). The string
# values are the contract that flows through extra_fields -> non_tensor_batch -> the trajectory
# filter, so the fallback mirrors them verbatim if verl can't be imported (old verl checkout).
try:
    from verl.trainer.ppo.trajectory_filter import TerminationReason
except Exception:  # pragma: no cover - keeps rollout alive on old verl
    class TerminationReason:
        AGENT_COMPLETED = "agent_completed"
        OVERLONG = "overlong"
        MAX_TURNS_REACHED = "max_turns_reached"
        TIMEOUT = "timeout"
        ENV_SETUP_FAILED = "env_setup_failed"


def _termination_reason_flags(termination_reason: str) -> dict[str, bool]:
    """Map the single termination taxonomy to trajectory_filter legacy flags."""
    return {
        "is_context_error": termination_reason in {"context_error", "context_length_exceeded"},
        "is_env_failure": termination_reason == TerminationReason.ENV_SETUP_FAILED,
        "is_timeout": termination_reason == TerminationReason.TIMEOUT,
        "is_overlong": termination_reason == TerminationReason.OVERLONG,
    }


def _safe_session_slug(task_path: str) -> str:
    """Build a short, filesystem-safe-ish slug from a task_path for tracing."""
    name = Path(task_path).name if task_path else "task"
    # keep only [a-zA-Z0-9._-]
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)
    return safe[:48] or "task"


class BuiltinSWEAgentLoop(AgentLoopBase):
    """
    Harbor SWE-bench agent loop for verl AgentLoopManager.

    This agent loop runs complete Harbor trials for SWE-bench tasks and
    converts the results to verl's AgentLoopOutput format for training.

    The loop manages:
    1. Harbor trial initialization with task path and agent config
    2. Trial execution (multi-turn agent + tool calls + verification)
    3. Reward extraction (verifier result)
    4. Proxy trajectory → AgentLoopOutput token fields

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
                "Harbor is not installed. Please install harbor package to use BuiltinSWEAgentLoop. "
                "pip install harbor"
            )

        # Harbor configuration
        # self.harbor_cfg = OmegaConf.to_container(self.rollout_config.agent.get("harbor_cfg", {}), resolve=True)
        self.harbor_cfg = OmegaConf.to_container(kwargs.get("harbor_cfg", {}), resolve=True)
        self.base_trials_dir = self.harbor_cfg.get("agent", {}).get("trials_dir", "./harbor_trials")
        self.max_retries = self.harbor_cfg.get("agent", {}).get("max_retries", MAX_RETRIES)

        # Validation-specific overrides. Each defaults to the training value (or None =
        # "leave harbor_cfg as-is") so behavior is UNCHANGED unless the env var is set.
        # Lets validation use more retries / longer pod-startup / longer agent timeout
        # WITHOUT affecting per-step training latency.
        self.val_max_retries = int(os.environ.get("HARBOR_VAL_MAX_RETRIES", self.max_retries))
        _val_pod_startup = os.environ.get("K8S_VAL_POD_STARTUP_TIMEOUT")
        self.val_pod_startup_timeout = int(_val_pod_startup) if _val_pod_startup else None
        _val_agent_timeout = os.environ.get("HARBOR_VAL_AGENT_MAX_TIMEOUT_SEC", 4800)
        self.val_agent_max_timeout = int(_val_agent_timeout) if _val_agent_timeout else None
        _val_pod_deadline = os.environ.get("K8S_VAL_POD_ACTIVE_DEADLINE_SECONDS")
        self.val_pod_active_deadline = int(_val_pod_deadline) if _val_pod_deadline else None

        # Prompt/response length limits
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        # Turn cap (HARBOR_AGENT_MAX_ITERATIONS) — used only to label a trajectory that
        # exhausted it as termination_reason="max_turns_reached". 0 => unknown/disabled.
        self._max_turns = int(os.environ.get("HARBOR_AGENT_MAX_ITERATIONS", "0") or "0")

        agent_name = self.harbor_cfg.get("agent", {}).get("name", "")
        self._tool_parser_name = self.harbor_cfg.get("agent", {}).pop("tool_parser", "hermes")

        # Early termination: stop a session after N consecutive assistant turns
        # without any tool_calls. 0 = disabled (default).
        self._max_consecutive_no_tool = int(
            self.harbor_cfg.get("agent", {}).get(
                "max_consecutive_no_tool",
                os.getenv("HARBOR_MAX_CONSECUTIVE_NO_TOOL", "0"),
            )
        )

        # Lazily started on first ``run()``; singleton per ``server_manager`` (Ray actor).
        self._vllm_chat_proxy = None

        # Lazily started on first ``run()``; singleton per ``server_manager`` (Ray actor).
        self._vllm_chat_proxy = None

        # vLLM endpoint will be injected via server_manager at runtime
        logger.info(
            "BuiltinSWEAgentLoop initialized. agent=%s  trials_dir=%s  max_retries=%d "
            "tool_parser=%s",
            agent_name,
            self.base_trials_dir,
            self.max_retries,
            self._tool_parser_name,
        )

    async def _get_vllm_chat_proxy(self):
        """Ensure the in-process vLLM Chat Completions proxy is started once per worker."""
        if self._vllm_chat_proxy is None:
            self._vllm_chat_proxy = await _ensure_proxy_started(self)
        return self._vllm_chat_proxy

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """
        Run Harbor trial for SWE-bench task and return AgentLoopOutput.

        Args:
            sampling_params: LLM sampling parameters (temperature, top_p, etc.)
            **kwargs: Batch fields from dataset (must include 'raw_prompt', 'extra_info')

        Returns:
            AgentLoopOutput with:
                - prompt_ids / response_ids: from proxy session trajectory
                - response_mask: 1s for LLM tokens, 0s for tool/user tokens
                - response_logprobs: per-response-token logprobs from vLLM
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
            return self._make_empty_output(kwargs, reason="no_task_path")

        # Get global step for trial directory organization
        global_steps = kwargs.get("global_steps", 0)
        # True during validation rollouts; selects the val-specific retry/timeout knobs.
        is_val = bool(kwargs.get("validate", False))

        # Timing metrics - pre-populate with harbor phase keys so aggregation is
        # stable across samples even when a trial ends early (e.g. empty output
        # before verification). Harbor phase timings are filled in by
        # ``_run_harbor_trial`` based on ``TrialResult.{environment_setup,
        # agent_setup, agent_execution, verifier, started_at/finished_at}``.
        metrics: dict[str, float] = {k: 0.0 for k in _HARBOR_TIMING_KEYS}
        for k in _PROXY_METRIC_KEYS:
            metrics.setdefault(k, 0.0)

        # Proxy process binds once per worker; each Harbor retry uses a fresh session_id,
        # api_base path, trial_name, and trajectory dir (see ``_run_harbor_trial``).
        proxy = await self._get_vllm_chat_proxy()

        with simple_timer("harbor_total", metrics):
            reward, trial_reason, session_meta = await self._run_harbor_trial(
                task_path=task_path,
                global_steps=global_steps,
                sampling_params=sampling_params,
                proxy=proxy,
                metrics=metrics,
                is_val=is_val,
            )

        # Alias harbor_total into the standard ``generate_sequences`` key so verl's
        # ``AgentLoopManager._performance_metrics`` slowest-sample logic -- which
        # uses ``argmax(generate_sequences + tool_calls + compute_score)`` -- can
        # pick the slowest trial based on actual wall-clock trial duration.
        metrics["generate_sequences"] = metrics.get("harbor_total", 0.0)

        traj_acc = session_meta.get("traj_acc_ids") or []
        prompt_token_len = int(session_meta.get("initial_prompt_token_len") or 0)
        tail_mask = session_meta.get("traj_response_mask") or []
        tail_lp = session_meta.get("traj_response_logprobs") or []
        # R3: per-token routing captured by the shared proxy (loop-agnostic).
        tail_routing = session_meta.get("traj_response_routing") or []
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
            return self._make_empty_output(kwargs, metrics=metrics, session_meta=session_meta, reason="invalid_trajectory")

        prompt_ids = traj_acc[:prompt_token_len]
        response_ids = traj_acc[prompt_token_len:]
        response_mask = tail_mask
        response_logprobs = tail_lp
        # R3: routing aligned to the response region (same length as mask/logprobs).
        # Empty when routing was not captured (R3 off / non-MoE).
        response_routing = tail_routing if len(tail_routing) == len(tail_mask) else []

        if not prompt_ids or not response_ids:
            logger.warning(
                "Empty proxy trajectory token ids (prompt=%d, response=%d), task=%s",
                len(prompt_ids), len(response_ids), task_path,
            )
            return self._make_empty_output(kwargs, metrics=metrics, session_meta=session_meta, reason="empty_token_ids")

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

        messages_snapshot = session_meta.get("messages_snapshot") or []

        # Single termination classification (consumed by algorithm.trajectory_filter).
        # _run_harbor_trial returns the trial-level reason (agent_completed / timeout /
        # overlong-from-context-overflow). Refine a plain completion here with the two signals
        # only known after assembling the trajectory: exhausting the turn cap
        # (max_turns_reached) and a response longer than the training window — truncated by the
        # slicing below, its verifier verdict reflecting a budget limit (DAPO "overlong").
        # Priority: context overflow (upstream) > max_turns_reached > response-window overlong.
        num_turns = len([m for m in messages_snapshot if m.get("role") == "assistant"])
        termination_reason = trial_reason
        if termination_reason == TerminationReason.AGENT_COMPLETED:
            if self._max_turns and num_turns >= self._max_turns:
                termination_reason = TerminationReason.MAX_TURNS_REACHED
            elif len(response_ids) > self.response_length:
                termination_reason = TerminationReason.OVERLONG

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:self.response_length],
            response_mask=response_mask[:self.response_length],
            response_logprobs=response_logprobs[:self.response_length],
            # R3: full-length [prompt+response, layers, topk] routing tensor;
            # None when no routing was captured, so non-R3 runs are unaffected.
            routed_experts=self._build_routed_experts(
                prompt_ids, response_routing[: self.response_length]
            ),
            multi_modal_data=None,
            reward_score=reward_score,
            num_turns=num_turns,
            metrics=metrics,
            extra_fields={
                "turn_scores": [],
                "tool_rewards": [],
                "termination_reason": termination_reason,
                **_termination_reason_flags(termination_reason),
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
        metrics["proxy_model_inference"] = float(meta.get("proxy_model_inference_sec") or 0.0)
        metrics["proxy_tool_call"] = float(meta.get("proxy_tool_call_sec") or 0.0)
        # Per-turn tool/agent gap stats (sec) for this trajectory; in-pod tools are invisible,
        # so the inter-LLM-call gap is the closest verl-side signal (a hung tool round -> big max).
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
        is_val: bool = False,
    ) -> tuple[float, str, dict]:
        """
        Execute Harbor trial with retries. Each retry uses a new proxy ``session_id``,
        ``api_base``, and Harbor ``trial_name`` so KV / logs do not leak across attempts.

        Returns:
            reward, termination_reason (trial-level: agent_completed / timeout / overlong),
            session_meta from the **terminal** attempt (any ``break`` path). The caller refines
            agent_completed into max_turns_reached / overlong / env_setup_failed. Retries that
            ``continue`` do not contribute to ``metrics`` or ``session_meta``.
        """
        if metrics is None:
            metrics = {}

        cfg = deepcopy(self.harbor_cfg)
        cfg["trials_dir"] = f"{self.base_trials_dir}/step_{global_steps:04d}"
        cfg["task"] = {"path": task_path}

        # Validation-only overrides. Training trials keep the shared harbor_cfg values,
        # so per-step training latency is unaffected. When the val env vars are unset
        # these all fall back to the training values (no behavioral change).
        max_retries = self.max_retries
        if is_val:
            max_retries = self.val_max_retries
            if self.val_pod_startup_timeout is not None:
                cfg.setdefault("environment", {}).setdefault("kwargs", {})[
                    "pod_startup_timeout_sec"
                ] = self.val_pod_startup_timeout
            if self.val_pod_active_deadline is not None:
                cfg.setdefault("environment", {}).setdefault("kwargs", {})[
                    "pod_active_deadline_seconds"
                ] = self.val_pod_active_deadline
            if self.val_agent_max_timeout is not None:
                cfg.setdefault("agent", {})["max_timeout_sec"] = self.val_agent_max_timeout

        reward = 0.0
        trial_reason = TerminationReason.AGENT_COMPLETED
        session_meta: dict = {}

        # NOTE: For Claude Code (Anthropic format) we still need a separate
        # LiteLLM proxy that bridges Anthropic to OpenAI. The in-process proxy
        # only speaks OpenAI Chat Completions for now; supporting Anthropic
        # is tracked in TODO_http_proxy.md.

        for attempt in range(max_retries):
            terminal_commit = False
            prefix = f"[task={Path(task_path).name}] attempt {attempt + 1}/{max_retries}"
            session_id = f"{_safe_session_slug(task_path)}-{uuid4().hex[:8]}"
            api_base = proxy.session_url(session_id)
            trajectory_dir = Path(self.base_trials_dir) / f"step_{global_steps:04d}" / session_id
            await proxy.open_session(session_id, trajectory_dir=trajectory_dir)

            cfg.setdefault("agent", {}).setdefault("kwargs", {})["api_base"] = api_base
            agent_env = cfg.setdefault("agent", {}).setdefault("env", {})
            agent_env["LLM_BASE_URL"] = api_base
            agent_env.setdefault(
                "LLM_API_KEY",
                os.environ.get("LLM_API_KEY", "dummy-key-for-local-vllm"),
            )
            agent_name = cfg.get("agent", {}).get("name", "")
            agent_import_path = cfg.get("agent", {}).get("import_path", "")
            if (agent_name and "claude-code" in agent_name) or (agent_import_path and "ClaudeCode" in agent_import_path):
                cfg.setdefault("agent", {})["model_name"] = cfg.get("agent", {}).get("model_name", "").split("/")[-1]
                agent_env["ANTHROPIC_BASE_URL"] = api_base
                agent_env.setdefault(
                    "ANTHROPIC_API_KEY",
                    os.environ.get("ANTHROPIC_API_KEY", "dummy-key-for-local-vllm"),
                )
            if (agent_name and "opencode" in agent_name) or (agent_import_path and "OpenCode" in agent_import_path):
                agent_env["HOSTED_VLLM_BASE_URL"] = api_base
                agent_env.setdefault(
                    "HOSTED_VLLM_API_KEY",
                    os.environ.get("HOSTED_VLLM_API_KEY", "dummy-key-for-local-vllm"),
                )
                agent_env["OPENCODE_TEMPERATURE"] = os.environ.get("OPENCODE_TEMPERATURE", "1.0")
                agent_env["OPENCODE_CONFIG_CONTENT"] = os.environ.get("HARBOR_OPENCODE_CONFIG_CONTENT", "dummy-config-for-local-vllm")
            cfg["trial_name"] = session_id
            # TODO: remove this later
            # cfg.setdefault("agent", {}).setdefault("kwargs", {})["session_id"] = session_id

            if "temperature" in sampling_params:
                cfg.setdefault("agent", {}).setdefault("kwargs", {})["temperature"] = sampling_params[
                    "temperature"
                ]
            if "top_p" in sampling_params:
                cfg.setdefault("agent", {}).setdefault("kwargs", {})["top_p"] = sampling_params["top_p"]

            results = None
            try:
                # Validate and create trial
                trial_config = TrialConfig.model_validate(cfg)
                trial = await Trial.create(trial_config)

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
                # kept categories (completed / overlong / max_turns_reached). Only env-setup
                # failures (no usable trajectory) and the filter's drop list end up at 0.
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

    @staticmethod
    def _build_routed_experts(prompt_ids: list, response_routing: list):
        """Assemble the full-length ``[prompt+response, num_layers, topk]`` routing
        tensor expected by verl's agent-loop ``_pad`` (which places routed_experts at
        the real prompt start, unlike the response-only logprobs).

        Local copy of ``BuiltinCCAgentLoop._build_routed_experts`` — kept
        independent per loop on purpose, but the two MUST stay byte-identical:
        the placement convention below is paired with the verl-side row-content
        replay mask (verl commit 75b9ea70 / harbor commit 1439084); diverging
        copies reintroduce the off-by-one this convention eliminated.

        ALIGNMENT CONVENTION: ``routed_experts[j]`` is the routing of the forward
        pass AT position j. vLLM's capturer reports, for each sampled token, the
        routing of the forward that GENERATED it — and that forward ran at the
        token's PREVIOUS position — so each captured row is stored at
        ``position(token) - 1``. Training engines must therefore consume rows
        as-is, with NO shift of their own. The prompt region and any non-sampled
        / missing positions are zero placeholders ("not recorded — let the model
        route natively"); only forwards that produced sampled tokens carry real
        expert ids. Returns ``None`` when no routing was captured (R3 off /
        non-MoE), so the field stays absent and training is unaffected.
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
        # smaller than int64; verl's padding / the router cast routed ids to long
        # at use, so dtype here is just storage.
        arr = np.zeros((plen + len(response_routing), num_layers, topk), dtype=np.uint8)
        for i, r in enumerate(response_routing):
            if r is not None and plen + i - 1 >= 0:
                arr[plen + i - 1] = np.asarray(r, dtype=np.uint8)
        return arr

    def _make_empty_output(
        self,
        kwargs: dict,
        metrics: dict[str, float] | None = None,
        session_meta: dict | None = None,
        reason: str = "empty",
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
        # Audit every dropped/empty trial to a per-worker jsonl so silent drops
        # (e.g. val tasks that never persist a trial dir) are traceable afterward.
        try:
            _rp = kwargs.get("raw_prompt", []) or []
            _ei = kwargs.get("extra_info", {}) or {}
            _tp = _ei.get("harbor_task_path") or _ei.get("task_path") or (_rp[0].get("content") if _rp else "")
            _rec = {
                "ts": round(time.time(), 1),
                "task": Path(_tp).name if _tp else "",
                "reason": reason,
                "validate": bool(kwargs.get("validate", False)),
                "global_steps": int(kwargs.get("global_steps", 0) or 0),
            }
            _dd = Path(self.base_trials_dir) / "_dropped"
            _dd.mkdir(parents=True, exist_ok=True)
            with open(_dd / f"dropped.{os.getpid()}.jsonl", "a") as _f:
                _f.write(json.dumps(_rec) + "\n")
        except Exception as _e:
            logger.warning("could not record dropped trial: %s", _e)

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
            response_mask=[0],
            response_logprobs=[0],
            routed_experts=None,
            multi_modal_data=None,
            reward_score=0.0,
            num_turns=0,
            metrics=base_metrics,
            extra_fields={
                "turn_scores": [],
                "tool_rewards": [],
                # Env-setup failure: the trial never produced a usable trajectory. The dummy
                # sample only keeps the batch alive; algorithm.trajectory_filter drops it (it
                # is in the default drop_reasons) from the loss and GRPO group statistics.
                "termination_reason": TerminationReason.ENV_SETUP_FAILED,
                **_termination_reason_flags(TerminationReason.ENV_SETUP_FAILED),
                "raw_prompt": kwargs.get("raw_prompt", []),
                "min_global_steps": min_gs,
                "max_global_steps": max_gs,
                "proxy_num_calls": int(base_metrics.get("proxy_num_calls", 0)),
                "proxy_num_aborts": int(base_metrics.get("proxy_num_aborts", 0)),
                "proxy_num_preempted": int(base_metrics.get("proxy_num_preempted", 0)),
                "trajectory_token_source": "empty",
                "tool_parser": self._tool_parser_name,
            },
        )