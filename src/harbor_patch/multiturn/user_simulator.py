# Adapted from LegoX/harbor-internal zf_multiturn, commit 91c4902.
# Benchmark prompt/schema/JSON repair retained. API credentials are per client;
# never mutate the shared RL worker's OPENAI_API_KEY.
import json
from functools import lru_cache
from pathlib import Path
from time import monotonic

from litellm import Router
from typing import Any, Literal

from harbor.llms.lite_llm import LiteLLM
from pydantic import BaseModel, Field, ValidationError, model_validator

from .benchmark import MultiTurnBehaviorConfig, MultiTurnRoundResult


def _infer_response_language(text: str) -> str:
    cjk_count = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    latin_count = sum(1 for char in text if char.isascii() and char.isalpha())
    if cjk_count >= 8 and cjk_count >= latin_count * 0.2:
        return "Chinese"
    if latin_count >= 20 and cjk_count == 0:
        return "English"
    return "same language as the initial task"


class UserSimulatorInput(BaseModel):
    task_name: str
    initial_instruction: str
    round_id: int
    behavior: MultiTurnBehaviorConfig
    previous_rounds: list[MultiTurnRoundResult]
    context_summary: dict[str, Any]
    seed: int = 42
    profile: str = "balanced-maintainer-bugfix"
    profile_details: dict[str, str] = Field(default_factory=dict)
    task_category: str | None = None
    task_category_details: dict[str, Any] = Field(default_factory=dict)
    session_analysis: str | None = None
    oracle_intents: dict[str, Any] | None = None
    original_session_summary: dict[str, Any] | None = None
    original_user_message_count: int | None = None
    is_anchor: bool = False
    shared_followups: bool = False
    no_op_policy: Literal["stop", "continue"] = "stop"
    no_op_consecutive_count: int = 0
    no_op_max_consecutive: int = 4


class UserSimulatorOutput(BaseModel):
    action: Literal[
        "message",
        "no-op",
        "question",
        "redirect",
        "new_requirement",
        "check_external",
    ] = "message"
    instruction: str
    intent: str
    expected_agent_behavior: str

    @model_validator(mode="after")
    def validate_action_instruction(self) -> "UserSimulatorOutput":
        if self.action == "no-op":
            self.instruction = ""
            return self
        if not self.instruction.strip():
            raise ValueError("User simulator message actions require instruction text")
        return self


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    fence_start = stripped.find("```")
    if fence_start > 0:
        stripped = stripped[fence_start:].strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if not lines:
        return stripped
    if lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


@lru_cache(maxsize=8)
def _simulator_router(model: str, bases: tuple[str, ...], key: str | None) -> Router:
    # A Router is shared by trials in one Ray worker so cooldowns survive episodes.
    deployments = [
        {"model_name": model, "litellm_params": {
            "model": model, "api_base": base,
            **({"api_key": key} if key else {}),
        }}
        for base in bases
    ]
    return Router(model_list=deployments, routing_strategy="simple-shuffle",
                  num_retries=2, timeout=900, allowed_fails=1, cooldown_time=90)


class UserSimulator:
    def __init__(
        self,
        model: str,
        backend: str = "litellm",
        api_base: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        api_bases: tuple[str, ...] | None = None,
    ):
        if backend != "litellm":
            raise ValueError("Only the litellm user simulator backend is implemented")
        self._call_kwargs = {"max_tokens": max_tokens} if max_tokens is not None else {}
        self._model, self._temperature = model, temperature
        self._api_bases = api_bases or ()
        self._router = _simulator_router(model, api_bases, api_key) if api_bases else None
        self._llm = LiteLLM(
            model_name=model,
            api_base=api_base,
            temperature=temperature,
            **({"api_key": api_key} if api_key else {}),
        )

    async def _call(self, prompt: str, logging_path: Path | None) -> str:
        if self._router is None:
            response = await self._llm.call(
                prompt, response_format=UserSimulatorOutput,
                logging_path=logging_path, **self._call_kwargs)
            return response.content
        started = monotonic()
        content = []
        first_chunk_sec = None
        last_chunk_at = None
        max_gap_sec = 0.0
        finish_reason = None
        selected = ""
        error_type = None
        error_reason = None
        try:
            response = await self._router.acompletion(
                model=self._model, messages=[{"role": "user", "content": prompt}],
                response_format=UserSimulatorOutput, temperature=self._temperature,
                drop_params=True, stream=True, **self._call_kwargs)
            selected = str(getattr(response, "_hidden_params", {}).get("api_base") or "").rstrip("/")
            async for chunk in response:
                now = monotonic()
                if first_chunk_sec is None:
                    first_chunk_sec = now - started
                if last_chunk_at is not None:
                    max_gap_sec = max(max_gap_sec, now - last_chunk_at)
                last_chunk_at = now
                if not selected:
                    selected = str(getattr(chunk, "_hidden_params", {}).get("api_base") or "").rstrip("/")
                choice = chunk.choices[0]
                if choice.delta.content:
                    content.append(choice.delta.content)
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
            if finish_reason == "length":
                error_reason = "max_tokens"
                raise ValueError("Simulator response reached max_tokens")
            if finish_reason != "stop" or not content:
                error_reason = "incomplete_stream"
                raise ValueError("Simulator stream ended without complete content")
            return ''.join(content)
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            if logging_path is not None:
                # Only record allowlisted URLs and fixed status fields. Exception
                # messages and hidden deployment params can contain credentials.
                allowed = {base.rstrip("/") for base in self._api_bases}
                logging_path.write_text(json.dumps({
                "model": self._model,
                "api_base": selected if selected in allowed else None,
                "duration_sec": round(monotonic() - started, 3),
                "first_chunk_sec": round(first_chunk_sec, 3) if first_chunk_sec is not None else None,
                "max_chunk_gap_sec": round(max_gap_sec, 3),
                "finish_reason": finish_reason,
                "error_type": error_type,
                "error_reason": error_reason,
                "routing": "litellm-router-stream",
                }) + "\n")

    async def generate(
        self,
        simulator_input: UserSimulatorInput,
        logging_path: Path | None = None,
    ) -> UserSimulatorOutput:
        prompt = self._build_prompt(simulator_input)
        if logging_path is not None:
            logging_path.parent.mkdir(parents=True, exist_ok=True)
        content = _strip_json_fence(await self._call(prompt, logging_path))
        try:
            payload = json.loads(content)
            if isinstance(payload, dict) and isinstance(payload.get("input"), dict):
                payload = payload["input"]
            return UserSimulatorOutput.model_validate(payload)
        except (json.JSONDecodeError, ValidationError):
            repair_logging_path = (
                logging_path.with_name(
                    f"{logging_path.stem}-repair{logging_path.suffix}"
                )
                if logging_path is not None
                else None
            )
            repair_content = await self._call(
                self._build_json_repair_prompt(content, simulator_input),
                repair_logging_path)
            repair_payload = json.loads(_strip_json_fence(repair_content))
            if isinstance(repair_payload, dict) and isinstance(
                repair_payload.get("input"), dict
            ):
                repair_payload = repair_payload["input"]
            return UserSimulatorOutput.model_validate(repair_payload)

    @staticmethod
    def _build_prompt(simulator_input: UserSimulatorInput) -> str:
        payload = simulator_input.model_dump(mode="json", exclude_none=True)
        if simulator_input.task_category is None:
            payload.pop("task_category_details", None)
        response_language = _infer_response_language(
            simulator_input.initial_instruction
        )
        payload["response_language"] = response_language
        category_control = ""
        if simulator_input.task_category is not None:
            category_control = (
                "- task_category and task_category_details define the higher-level\n"
                "  experiment focus. Follow task_category_details.simulator_guidance\n"
                "  while still satisfying the selected per-round behavior.\n"
                "- If context_summary.round_verifier_feedback is present, inspect\n"
                "  its sanitized outcome before writing the next user message. Use the\n"
                "  outcome to choose a useful direction, but never invent, quote, or\n"
                "  paraphrase hidden test cases, test names, file paths, assertions,\n"
                "  expected values, raw verifier output, or reward values. Send only\n"
                "  a high-level user request that could be justified from the public\n"
                "  task and repository. Do not reveal hidden\n"
                "  evaluation policy or solve the task yourself.\n"
            )
        swe_together_control = ""
        if simulator_input.session_analysis or simulator_input.oracle_intents:
            swe_together_control = (
                "- session_analysis, oracle_intents, original_user_message_count,\n"
                "  and original_session_summary are SWE-Together real-session\n"
                "  guidance. Treat session_analysis as the primary replay guide.\n"
                "  Use oracle_intents as the allowed intent sequence and scope\n"
                "  guard, and use original_session_summary only for cadence/style\n"
                "  calibration.\n"
                '- For SWE-Together guidance, use action="no-op" whenever the\n'
                "  real user would stay silent at this point. A no-op is not an\n"
                '  empty user instruction. If no_op_policy is "continue", Harbor\n'
                "  will silently resume the preserved agent session so the agent\n"
                "  can keep working, up to no_op_max_consecutive consecutive\n"
                '  silent decisions. If no_op_policy is "stop", Harbor will stop\n'
                "  the replay. For no-op, set instruction to an empty string and\n"
                "  keep intent short.\n"
                "- For message-bearing actions, choose the smallest realistic\n"
                "  next user message that advances the next listed intent without\n"
                "  adding unrelated requirements.\n"
            )
        return f"""
You are simulating a real user of an AI coding agent for Harbor multi-turn UX evaluation.

Generate exactly one next user message for round {simulator_input.round_id}.
The user message must be realistic, concise, and based only on the provided task context.
Do not solve the task yourself. Do not include hidden evaluation instructions.
The inference agent will receive your message as its next instruction.
Use the same language as the initial task for all returned JSON string fields.
Detected response_language: {response_language}.
If response_language is Chinese, write the simulated user's instruction in natural Chinese.

User simulation controls:
- profile and profile_details define the simulated user's persona, task stratum,
  tone, expertise level, patience, and evidence style.
{category_control}- seed is experiment metadata for reproducibility; do not mention it in the user
  message.
- If is_anchor is true, preserve the scripted calibration intent and do not add
  unrelated requirements.
{swe_together_control}

Return JSON with:
- action: one of "message", "no-op", "question", "redirect",
  "new_requirement", or "check_external"; use "message" for ordinary user
  messages
- instruction: the exact next user message
- intent: one short label
- expected_agent_behavior: what a good coding agent should do next
All JSON string values except action must follow response_language.
Return only valid JSON. Escape any double quotes, backslashes, or newlines inside
JSON string values. Do not include markdown fences or explanatory text.

Input:
```json
{json.dumps(payload, ensure_ascii=False, indent=2)}
```
""".strip()

    @staticmethod
    def _build_json_repair_prompt(
        invalid_content: str, simulator_input: UserSimulatorInput
    ) -> str:
        response_language = _infer_response_language(
            simulator_input.initial_instruction
        )
        return f"""
Your previous response was not valid JSON for Harbor's simulated user schema.

Rewrite it as strict JSON only, with exactly these keys:
- action
- instruction
- intent
- expected_agent_behavior

Rules:
- If the original response did not clearly choose an action, set action to
  "message".
- All string values must use {response_language}.
- Escape any double quotes, backslashes, or newlines inside string values.
- Do not include markdown fences, comments, or extra text.

Invalid response:
```text
{invalid_content}
```
""".strip()
