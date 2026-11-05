"""Native CC resume without duplicating installation or request configuration."""

import json
import shlex
from pathlib import Path
from uuid import uuid4

from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor_patch.agents.image_mounted_claude_code import ClaudeCode


class MultiTurnAgentError(RuntimeError):
    pass


class MultiTurnClaudeCode(ClaudeCode):
    _multiturn_session_id: str | None = None
    _multiturn_round: int = 0

    def _current_claude_command(self) -> str:
        command = super()._current_claude_command()
        if self._multiturn_session_id is not None:
            flag = "--session-id" if self._multiturn_round == 1 else "--resume"
            command += f" {flag} {shlex.quote(self._multiturn_session_id)}"
        return command

    async def run_user_round(
        self, instruction, environment, context, round_id, log_path: Path
    ):
        if round_id == 1:
            self._multiturn_session_id = str(uuid4())
        elif not self._multiturn_session_id or round_id != self._multiturn_round + 1:
            raise MultiTurnAgentError("Missing or out-of-order Claude resume session")
        self._multiturn_round = round_id
        # Parent run preserves runtime mounts, endpoint injection, skills, MCP,
        # sampling headers and prompt templates. Only the native CLI flag varies.
        try:
            await super().run(instruction, environment, context)
        except NonZeroAgentExitCodeError:
            # CC exits 1 on a context limit, before the caller can inspect its
            # structured result. Archive the round before classifying the stop.
            await environment.download_file("/logs/agent/claude-code.txt", log_path)
            result = self._validate_round_log(log_path)
            if result.get("stop_reason") == "context_limit":
                return result
            raise
        await environment.download_file("/logs/agent/claude-code.txt", log_path)
        return self._validate_round_log(log_path)

    def _validate_round_log(self, log_path: Path) -> dict:
        session_ids = set()
        result = None
        with log_path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                # Nested agent events may carry their own session identifiers.
                # Only the top-level native conversation must match --resume.
                if record.get("parent_tool_use_id") is not None:
                    continue
                if record.get("session_id"):
                    session_ids.add(record["session_id"])
                if record.get("type") == "result":
                    result = record
        # The parent's shell uses tee: its exit status alone does not prove the
        # agent completed. Also reject a silently restarted native conversation.
        if session_ids != {self._multiturn_session_id} or result is None:
            raise MultiTurnAgentError(
                "Claude did not complete in the expected native session"
            )
        if result.get("is_error"):
            message = str(result.get("result", "")).strip().lower()
            # Only the terminal result in the expected native session counts;
            # never classify quoted tool output or unrelated blocking limits.
            if any(marker in message for marker in (
                "prompt is too long", "maximum context length",
                "context_length_exceeded", "generation budget exhausted",
                "leaves no room to generate",
            )):
                return {
                    "session_id": self._multiturn_session_id,
                    "final_message": "",
                    "stop_reason": "context_limit",
                }
            raise MultiTurnAgentError(
                "Claude returned an error result; see the round log"
            )
        return {
            "session_id": self._multiturn_session_id,
            "final_message": result.get("result", ""),
        }
