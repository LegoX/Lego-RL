"""harbor_patch image-mounted ClaudeCode — exposes chat history to verl via
the LiteLLM proxy trajectory log.

Naming mirrors ``image_mounted_openhands_ai``: the dominant configuration
mounts a pre-built claude-code runtime image at
``/opt/custom-agent-runtime/claude-code``. The parent ``CustomClaudeCode``
auto-detects the mount and falls back to an in-container ``npm install`` when
no image is mounted, so this single class covers both modes.

Architecture (matches OpenHands SDK pattern in this repo):

    claude-code (in pod)
      -- Anthropic format -->
        LiteLLM proxy + trajectory_logger callback
          -- writes one JSON line per LLM call to logs_dir/litellm-trajectory.jsonl
          -- forwards as OpenAI to vLLM

The upstream ClaudeCode does not populate ``context.metadata["all_messages"]``,
so the verl ``BuiltinSWEAgentLoop`` sees an empty trial. We subclass
``CustomClaudeCode`` (which already wires the ``x-trajectory-output-path``
header and supports the mounted runtime image) and override
``populate_context_post_run`` to read that JSONL and assemble the final
OpenAI-format chat history. The last record's ``request_body.messages`` is
the full prompt the LLM saw on the last turn; we append the corresponding
assistant response to that to produce the trajectory.
"""

import json
import os
from typing import Any

from harbor.agents.custom.claude_code import CustomClaudeCode as _BaseCustomClaudeCode
from harbor.models.agent.context import AgentContext


# Trial is dropped (all_messages left empty -> verl reward=0) when any LLM
# call's prompt+completion exceeds this many tokens. Tuned to stay safely
# below the 100K vLLM max_model_len so a trial never reaches the FA
# scheduler_metadata overflow regime during RL-training log_prob recompute.
# Override via env. Set to 0 (or any value <= 0) to DISABLE the cutoff entirely
# — appropriate for pure evaluation (val_only), where no training step recomputes
# log_prob over the trajectory, so a long-but-solved trial should score its real
# verifier reward instead of being zeroed.
_OVERLONG_TOTAL_TOKENS = int(os.environ.get("HARBOR_TRIAL_OVERLONG_TOTAL_TOKENS", 90000))


class ClaudeCode(_BaseCustomClaudeCode):
    """CustomClaudeCode + ``metadata['all_messages']`` / ``['tools']``."""

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

        # Drop the trial early if it shows either of the two patterns that
        # otherwise drive vLLM into the FA scheduler_metadata overflow
        # regime. Leaving all_messages empty makes verl's _run_harbor_trial
        # produce reward=0 without retry, isolating the failure from the
        # rest of the batch.
        if self._has_hermes_parse_failure(records):
            self.logger.warning(
                "hermes_tool_parser failure detected in %s; marking trial empty",
                self._litellm_trajectory_path,
            )
            return

        if _OVERLONG_TOTAL_TOKENS > 0 and self._has_overlong_call(records):
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
    def _has_hermes_parse_failure(records: list[dict[str, Any]]) -> bool:
        """True if any response had vLLM's hermes_tool_parser fall back to
        returning the raw output as content (tool_calls empty AND content
        still contains a <tool_call> token)."""
        for record in records:
            response_body = record.get("response_body") or {}
            choices = response_body.get("choices") or []
            if not choices:
                continue
            msg = (choices[0] or {}).get("message") or {}
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                continue
            content = msg.get("content")
            if isinstance(content, list):
                content = "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            if isinstance(content, str) and "<tool_call>" in content:
                return True
        return False

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
