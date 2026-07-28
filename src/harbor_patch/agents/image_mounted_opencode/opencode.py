"""harbor_patch image-mounted OpenCode — exposes chat history to verl via
the LiteLLM proxy trajectory log.

Naming mirrors ``image_mounted_claude_code``: the dominant configuration mounts
a pre-built opencode runtime image at ``/opt/custom-agent-runtime/opencode``.
The parent ``CustomOpenCode`` auto-detects the mount and falls back to an
in-container ``npm install``-based path when no image is mounted, so this
single class covers both modes.

Architecture (matches the CC pattern in this repo):

    opencode (in pod, hosted_vllm provider via @ai-sdk/openai-compatible)
      -- OpenAI Chat Completions -->
        LiteLLM proxy + trajectory_logger callback
          -- writes one JSON line per LLM call to logs_dir/litellm-trajectory.jsonl
          -- forwards as openai/{served} to vLLM

Upstream ``CustomOpenCode`` already injects ``x-trajectory-output-path`` into
the provider's ``options.headers`` (see ``_inject_litellm_debug_headers``),
but does not populate ``context.metadata["all_messages"]`` itself — verl's
``BuiltinCCAgentLoop`` would otherwise see an empty trial. We subclass
``CustomOpenCode`` and override ``populate_context_post_run`` to read the
JSONL and assemble the final OpenAI-format chat history. The last successful
record's ``request_body.messages`` is the full prompt the LLM saw on the last
turn; we append the corresponding assistant response to produce the trajectory.
"""

import asyncio
import json
import os
from typing import Any

from harbor.agents.custom.opencode import CustomOpenCode as _BaseCustomOpenCode
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


# K8s ImageVolume (beta in 1.31+) can report Pod=Ready before the volume's
# contents are fully extracted on disk. When many trial pods spawn concurrently
# (~96 AgentLoopWorkers), parents `_detect_mounted_runtime` then sees the
# dangling symlink, returns None, and the install path falls back to the
# slow npm route — every trial hits the 360s agent setup timeout. Retry a
# few times to ride out the extraction window.
_DETECT_RETRY_ATTEMPTS = int(os.environ.get("HARBOR_OC_DETECT_RETRY_ATTEMPTS", 30))
_DETECT_RETRY_SLEEP_SEC = float(os.environ.get("HARBOR_OC_DETECT_RETRY_SLEEP_SEC", 2.0))


# Trial is dropped (all_messages left empty -> verl reward=0) when any LLM
# call's prompt+completion exceeds this many tokens. Tuned to stay safely
# below the 100K vLLM max_model_len so a trial never reaches the FA
# scheduler_metadata overflow regime. Override via env.
_OVERLONG_TOTAL_TOKENS = int(os.environ.get("HARBOR_TRIAL_OVERLONG_TOTAL_TOKENS", 90000))


class OpenCode(_BaseCustomOpenCode):
    """CustomOpenCode + ``metadata['all_messages']`` / ``['tools']``."""

    async def _detect_mounted_runtime(
        self, environment: BaseEnvironment
    ) -> tuple[str, str] | None:
        # Retry parent's detection a few times to ride out ImageVolume
        # extraction races when many pods spawn concurrently. See module-level
        # comment on _DETECT_RETRY_ATTEMPTS for context.
        runtime_opencode = self._mounted_runtime_opencode()
        runtime_env_script = self._mounted_runtime_env_script()
        for attempt in range(1, _DETECT_RETRY_ATTEMPTS + 1):
            result = await super()._detect_mounted_runtime(environment)
            if result is not None:
                self.logger.warning(
                    "[OC-DETECT] success attempt=%d/%d pod=%s",
                    attempt, _DETECT_RETRY_ATTEMPTS, getattr(environment, "pod_name", "?"),
                )
                return result
            # Re-run the exact same probe but capture stdout/stderr/rc so we
            # can see WHY parent returned None. (parent only logs at .debug
            # which trial.log filters out.) Cheap diagnostic — single exec.
            try:
                probe = await environment.exec(
                    command=(
                        f"ls -la {runtime_opencode} 2>&1; echo '--rc1='$?; "
                        f"ls -la {runtime_env_script} 2>&1; echo '--rc2='$?; "
                        f"readlink -f {runtime_opencode} 2>&1"
                    ),
                )
                self.logger.warning(
                    "[OC-DETECT] fail attempt=%d/%d pod=%s probe_rc=%s probe_out=%r",
                    attempt, _DETECT_RETRY_ATTEMPTS,
                    getattr(environment, "pod_name", "?"),
                    probe.return_code,
                    (probe.stdout or "")[:500] + " | err=" + (probe.stderr or "")[:200],
                )
            except Exception as e:
                self.logger.warning(
                    "[OC-DETECT] fail attempt=%d/%d pod=%s probe_exc=%r",
                    attempt, _DETECT_RETRY_ATTEMPTS,
                    getattr(environment, "pod_name", "?"), e,
                )
            if attempt < _DETECT_RETRY_ATTEMPTS:
                await asyncio.sleep(_DETECT_RETRY_SLEEP_SEC)
        return None

    def populate_context_post_run(self, context: AgentContext) -> None:
        super().populate_context_post_run(context)

        if context.metadata is None:
            context.metadata = {}
        context.metadata.setdefault("all_messages", [])
        context.metadata.setdefault("tools", [])

        if context.metadata["all_messages"]:
            return

        records = self._read_all_litellm_records()
        if not records:
            return

        # Drop the trial early if any single LLM call reached the overlong
        # regime that drives vLLM into FA scheduler_metadata overflow. Leaving
        # all_messages empty makes verl's _run_harbor_trial produce reward=0
        # without retry, isolating the failure from the rest of the batch.
        if self._has_overlong_call(records):
            self.logger.warning(
                "trial in %s reached >=%d tokens on a single call; marking trial empty",
                self._litellm_trajectory_path,
                _OVERLONG_TOTAL_TOKENS,
            )
            return

        record = records[-1]
        for r in reversed(records):
            if r.get("success", True):
                record = r
                break

        messages = self._extract_final_messages(record)
        if messages:
            context.metadata["all_messages"] = messages

        tools = (record.get("request_body") or {}).get("tools") or []
        if tools:
            context.metadata["tools"] = tools

    def _read_all_litellm_records(self) -> list[dict[str, Any]]:
        """Return every JSONL record from the LiteLLM trajectory file."""
        path = self._litellm_trajectory_path
        if not path.exists():
            self.logger.debug(f"No LiteLLM trajectory at {path}")
            return []

        records: list[dict[str, Any]] = []
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError as exc:
            self.logger.debug(f"Failed to read {path}: {exc}")
            return []

        return records

    @staticmethod
    def _has_overlong_call(
        records: list[dict[str, Any]],
        threshold: int = _OVERLONG_TOTAL_TOKENS,
    ) -> bool:
        """True if any single LLM call had prompt+completion >= threshold."""
        for record in records:
            usage = (record.get("response_body") or {}).get("usage") or record.get("usage") or {}
            prompt = usage.get("prompt_tokens") or 0
            completion = usage.get("completion_tokens") or 0
            if prompt + completion >= threshold:
                return True
        return False

    @staticmethod
    def _extract_final_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
        """Build the final OpenAI-format chat history from one JSONL record.

        ``request_body.messages`` is the full prompt at the time of the call;
        appending the assistant response from ``response_body.choices[0]``
        gives the complete trajectory.
        """
        request_body = record.get("request_body") or {}
        messages = list(request_body.get("messages") or [])

        response_body = record.get("response_body") or {}
        choices = response_body.get("choices") or []
        if not choices:
            return messages

        assistant = (choices[0] or {}).get("message")
        if not assistant:
            return messages

        # Drop empty content when only tool_calls are present (OpenAI rejects "")
        message = dict(assistant)
        if message.get("content") == "" and message.get("tool_calls"):
            message["content"] = None
        messages.append(message)
        return messages
