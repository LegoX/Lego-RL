"""OpenAI Responses API <-> OpenAI Chat Completions transforms.

Codex CLI speaks only the Responses API (``wire_api = "chat"`` was removed in
codex-cli 0.122.0), so the in-process proxy needs a third wire format beside
``/v1/chat/completions`` (OpenCode / OpenHands) and ``/v1/messages`` (Claude
Code). The proxy's canonical internal representation is OpenAI Chat
Completions ``messages`` + ``tools``; everything downstream of that -- the
``apply_chat_template`` tokenization, R3 routed-expert capture, trajectory
archival -- is wire-format agnostic. This module is the adapter, and nothing
here touches proxy state.

Shapes below were captured off the wire from codex-cli 0.153.4 and 0.125.0
pointed at a logging mock, not read off a docs page.

Request (``input[]`` is a flat item stream, NOT a message list)::

    {"type":"message","role":"developer","content":[{"type":"input_text",...}]}
    {"type":"message","role":"user",     "content":[{"type":"input_text",...}]}
    {"type":"function_call","id":..,"call_id":..,"name":..,"arguments":"<json str>"}
    {"type":"function_call_output","id":..,"call_id":..,"output":"<str>"}
    {"type":"reasoning","summary":[..],"content":[..],"encrypted_content":".."}

Three things bite, none of which exist on the Anthropic path:

1. ``role: "developer"`` -- Qwen chat templates know ``system``, not
   ``developer``. Left unmapped it is dropped or rendered wrong and the model
   trains on a prompt it never saw at rollout.
2. Item grouping is not message grouping. A turn with three parallel tool
   calls arrives as three sibling ``function_call`` items; emitting one
   assistant message each produces N turns where the model generated one and
   breaks prefix matching on every later turn. A ``reasoning`` item marks the
   start of the NEXT block, so it must flush a block that already has its
   outputs.
3. ``web_search`` and ``namespace`` tools have no chat equivalent and the
   model can never satisfy them. ``[tools] web_search = false`` in config.toml
   does NOT suppress web_search (tested on 0.153.4) -- it has to be filtered
   here.

``encrypted_content`` needs no real cryptography: the proxy is the API server
on both ends, so the value only has to round-trip opaquely through the
harness. Same approach as NVIDIA's Polar (Apache-2.0), whose
``gateway/transform/openai_responses.py`` solves this problem against the same
harbor harness.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any
from uuid import uuid4

__all__ = [
    "responses_to_openai_messages",
    "openai_message_to_responses",
    "responses_sse_events",
    "encrypt_reasoning",
    "decrypt_reasoning",
]

# Tool types the served model cannot possibly satisfy. ``web_search`` is
# executed server-side by OpenAI; ``namespace`` (multi_agent_v1) has no chat
# encoding at all. Dropping multi-agent is also what we want on RL grounds:
# subagent spawning multiplies LLM calls per session and explodes num_turns,
# the same reason agent_loop_config_cc.yaml documents for Claude Code's Task
# tool.
_DROPPED_TOOL_TYPES = frozenset({"web_search", "namespace", "web_search_preview"})

_REASONING_PREFIX = "legorl:"


# --------------------------------------------------------------------------
# reasoning round-trip
# --------------------------------------------------------------------------
def encrypt_reasoning(text: str) -> str:
    """Pack reasoning text into a Responses-style ``encrypted_content``.

    Opaque round-trip only -- base64 so it survives transport unmangled.
    """
    if not text:
        return ""
    return _REASONING_PREFIX + base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")


def decrypt_reasoning(encrypted: Any) -> str:
    """Reverse of :func:`encrypt_reasoning`; "" on anything unexpected."""
    if not isinstance(encrypted, str) or not encrypted.startswith(_REASONING_PREFIX):
        return ""
    try:
        return base64.urlsafe_b64decode(
            encrypted[len(_REASONING_PREFIX):].encode("ascii")
        ).decode("utf-8")
    except Exception:
        return ""


# --------------------------------------------------------------------------
# request: Responses -> OpenAI chat
# --------------------------------------------------------------------------
def _content_to_text(content: Any) -> str:
    """Flatten a Responses content field to plain text.

    Handles the str form and the block-list form, whose text blocks are typed
    ``input_text`` on the way in and ``output_text`` on the way back out.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            if block.get("type") in ("input_text", "output_text", "text", "summary_text"):
                text = block.get("text") or ""
                if not isinstance(text, str):
                    raise ValueError("Text content must be a string")
                parts.append(text)
            else:
                raise ValueError("This Responses adapter supports text content only")
    return "".join(parts)


def _reasoning_item_to_text(item: dict) -> str:
    """Pull reasoning text back out of a replayed ``reasoning`` item.

    ``encrypted_content`` is what we put there ourselves and is authoritative;
    ``summary`` / ``content`` are the readable fallbacks.
    """
    decrypted = decrypt_reasoning(item.get("encrypted_content"))
    if decrypted:
        return decrypted
    for key in ("content", "summary"):
        text = _content_to_text(item.get(key))
        if text:
            return text
    return ""


def _flush_tool_block(
    tool_calls: list[dict],
    tool_outputs: list[dict],
    reasoning: str,
    assistant_text: str | None = None,
) -> list[dict]:
    """Collapse one accumulated turn into chat messages.

    All ``function_call`` items seen since the last flush belong to a SINGLE
    assistant message (the model emitted them in one generation); their
    outputs follow as ``role:"tool"`` messages.

    The Responses message and function-call items for one sampled assistant
    turn must remain one chat message when Codex echoes them back.
    """
    messages: list[dict] = []
    if tool_calls:
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": assistant_text if assistant_text else None,
            "tool_calls": tool_calls,
        }
        if reasoning:
            assistant["reasoning_content"] = reasoning
        messages.append(assistant)
    elif assistant_text is not None:
        assistant = {"role": "assistant", "content": assistant_text}
        if reasoning:
            assistant["reasoning_content"] = reasoning
        messages.append(assistant)
    elif reasoning:
        messages.append({"role": "assistant", "content": "", "reasoning_content": reasoning})
    messages.extend(tool_outputs)
    return messages


def responses_to_openai_messages(body: dict) -> tuple[list[dict], list[dict] | None]:
    """Responses request body -> ``(openai_messages, openai_tools)``.

    ``instructions`` and every ``developer``/``system`` message are merged into
    one leading system message, matching what the chat template expects.
    """
    if not isinstance(body, dict):
        raise ValueError("Responses request must be a JSON object")
    if body.get("previous_response_id"):
        raise ValueError("Send the complete input history; previous_response_id is not supported")
    if not isinstance(body.get("input"), (str, list)):
        raise ValueError("Responses input must be a string or a list of items")
    system_parts: list[str] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions:
        system_parts.append(instructions)
    elif instructions:
        text = _content_to_text(instructions)
        if text:
            system_parts.append(text)

    body_messages: list[dict] = []
    pending_tool_calls: list[dict] = []
    pending_tool_outputs: list[dict] = []
    pending_reasoning = ""
    pending_assistant_text: str | None = None

    def flush() -> None:
        nonlocal pending_tool_calls, pending_tool_outputs, pending_reasoning
        nonlocal pending_assistant_text
        if (
            pending_tool_calls
            or pending_tool_outputs
            or pending_reasoning
            or pending_assistant_text is not None
        ):
            body_messages.extend(
                _flush_tool_block(
                    pending_tool_calls,
                    pending_tool_outputs,
                    pending_reasoning,
                    pending_assistant_text,
                )
            )
            pending_tool_calls = []
            pending_tool_outputs = []
            pending_reasoning = ""
            pending_assistant_text = None

    raw_input = body.get("input")
    if isinstance(raw_input, str):
        items: list[Any] = [{"type": "message", "role": "user", "content": raw_input}]
    else:
        items = list(raw_input or [])

    for item in items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type") or "message"

        if item_type == "reasoning":
            # A reasoning item opens the NEXT turn block. If the current block
            # already has its outputs, close it first, or this reasoning is
            # attributed to the previous assistant message.
            if pending_tool_outputs:
                flush()
            text = _reasoning_item_to_text(item)
            if text:
                pending_reasoning = f"{pending_reasoning}\n{text}" if pending_reasoning else text
            continue

        if item_type == "function_call":
            if pending_tool_outputs:
                flush()
            arguments = item.get("arguments")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments or {}, ensure_ascii=False)
            pending_tool_calls.append(
                {
                    "id": item.get("call_id") or item.get("id") or f"call_{uuid4().hex[:24]}",
                    "type": "function",
                    "function": {"name": item.get("name") or "", "arguments": arguments},
                }
            )
            continue

        if item_type == "function_call_output":
            output = item.get("output")
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False) if output is not None else ""
            pending_tool_outputs.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id") or item.get("id") or "",
                    "content": output,
                }
            )
            continue

        if item_type == "message":
            role = item.get("role") or "user"
            text = _content_to_text(item.get("content"))
            if role == "assistant":
                # Reasoning immediately followed by text belongs to this same
                # generated turn, including any subsequent function calls.
                if pending_tool_outputs or pending_tool_calls or pending_assistant_text is not None:
                    flush()
                pending_assistant_text = text
                continue
            flush()
            if role in ("developer", "system"):
                # H1: the chat template has no `developer` role.
                if text:
                    system_parts.append(text)
                continue
            body_messages.append({"role": role, "content": text})
            continue

        # Unknown item type: keep its text rather than silently dropping the
        # turn, so a prompt change shows up as content instead of a hole.
        text = _content_to_text(item.get("content"))
        if text:
            flush()
            body_messages.append({"role": item.get("role") or "user", "content": text})

    flush()

    messages: list[dict] = []
    if system_parts:
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})
    messages.extend(body_messages)

    return messages, _convert_tools(body.get("tools"))


def _convert_tools(tools: Any) -> list[dict] | None:
    """Responses tools -> chat tools.

    A Responses function tool is FLAT (``{type,name,description,parameters}``);
    chat nests everything under ``function``. Non-function types are dropped --
    see ``_DROPPED_TOOL_TYPES``.
    """
    if not tools:
        return None
    converted: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type")
        if tool_type in _DROPPED_TOOL_TYPES:
            continue
        if tool_type not in ("function", None):
            continue
        if "function" in tool and isinstance(tool["function"], dict):
            converted.append({"type": "function", "function": dict(tool["function"])})
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name") or "",
                    "description": tool.get("description") or "",
                    "parameters": tool.get("parameters") or {},
                },
            }
        )
    return converted or None


def convert_tool_choice(tool_choice: Any) -> Any:
    """Responses ``tool_choice`` -> chat ``tool_choice``."""
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        name = tool_choice.get("name") or (tool_choice.get("function") or {}).get("name")
        if name:
            return {"type": "function", "function": {"name": name}}
    return tool_choice


# --------------------------------------------------------------------------
# response: OpenAI chat -> Responses
# --------------------------------------------------------------------------
def openai_message_to_responses(
    message: dict,
    finish_reason: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    session_id: str,
    *,
    include_reasoning: bool = True,
) -> dict:
    """Assembled assistant message -> a completed Responses response body."""
    response_id = f"resp_{uuid4().hex[:24]}"
    output: list[dict] = []

    reasoning_text = message.get("reasoning_content") or ""
    if reasoning_text and include_reasoning:
        output.append(
            {
                "type": "reasoning",
                "id": f"rs_{uuid4().hex[:24]}",
                "summary": [{"type": "summary_text", "text": reasoning_text}],
                "content": [],
                "encrypted_content": encrypt_reasoning(reasoning_text),
            }
        )

    content = message.get("content")
    if content:
        output.append(
            {
                "type": "message",
                "id": f"msg_{uuid4().hex[:24]}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content, "annotations": []}],
            }
        )

    for call in message.get("tool_calls") or []:
        func = call.get("function") or {}
        arguments = func.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {}, ensure_ascii=False)
        output.append(
            {
                "type": "function_call",
                "id": f"fc_{uuid4().hex[:24]}",
                "call_id": call.get("id") or f"call_{uuid4().hex[:24]}",
                "name": func.get("name") or "",
                "arguments": arguments,
                "status": "completed",
            }
        )

    incomplete = None
    if finish_reason == "length":
        incomplete = {"reason": "max_output_tokens"}

    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "incomplete" if incomplete else "completed",
        "model": model,
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": prompt_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": completion_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "incomplete_details": incomplete,
        "error": None,
        "metadata": {},
    }


# --------------------------------------------------------------------------
# response: Responses -> SSE event sequence
# --------------------------------------------------------------------------
def responses_sse_events(response: dict) -> list[tuple[str, dict]]:
    """Replay a finished response as the SSE sequence codex expects.

    Generation has already completed, so token fidelity is unaffected -- this
    only re-shapes the result into the wire form the client reads. The four
    lifecycle events plus per-item added/done are the set codex 0.153.4 was
    observed to accept; the text/argument delta events are emitted too so a
    client that renders incrementally still sees content.
    """
    events: list[tuple[str, dict]] = []
    seq = 0

    def emit(name: str, payload: dict) -> None:
        nonlocal seq
        payload = {"type": name, "sequence_number": seq, **payload}
        events.append((name, payload))
        seq += 1

    skeleton = {k: v for k, v in response.items() if k != "output"}
    skeleton["output"] = []
    skeleton["status"] = "in_progress"
    emit("response.created", {"response": skeleton})
    emit("response.in_progress", {"response": skeleton})

    for index, item in enumerate(response.get("output") or []):
        item_type = item.get("type")
        item_id = item.get("id")

        opening = dict(item)
        if item_type == "message":
            opening["content"] = []
            opening["status"] = "in_progress"
        elif item_type == "function_call":
            opening = {**item, "arguments": "", "status": "in_progress"}
        emit("response.output_item.added", {"output_index": index, "item": opening})

        if item_type == "message":
            for content_index, part in enumerate(item.get("content") or []):
                text = part.get("text") or ""
                emit(
                    "response.content_part.added",
                    {
                        "item_id": item_id,
                        "output_index": index,
                        "content_index": content_index,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    },
                )
                if text:
                    emit(
                        "response.output_text.delta",
                        {
                            "item_id": item_id,
                            "output_index": index,
                            "content_index": content_index,
                            "delta": text,
                        },
                    )
                emit(
                    "response.output_text.done",
                    {
                        "item_id": item_id,
                        "output_index": index,
                        "content_index": content_index,
                        "text": text,
                    },
                )
                emit(
                    "response.content_part.done",
                    {
                        "item_id": item_id,
                        "output_index": index,
                        "content_index": content_index,
                        "part": part,
                    },
                )

        elif item_type == "function_call":
            arguments = item.get("arguments") or ""
            if arguments:
                emit(
                    "response.function_call_arguments.delta",
                    {"item_id": item_id, "output_index": index, "delta": arguments},
                )
            emit(
                "response.function_call_arguments.done",
                {"item_id": item_id, "output_index": index, "arguments": arguments},
            )

        elif item_type == "reasoning":
            for summary_index, part in enumerate(item.get("summary") or []):
                text = part.get("text") or ""
                emit(
                    "response.reasoning_summary_part.added",
                    {
                        "item_id": item_id,
                        "output_index": index,
                        "summary_index": summary_index,
                        "part": {"type": "summary_text", "text": ""},
                    },
                )
                if text:
                    emit(
                        "response.reasoning_summary_text.delta",
                        {
                            "item_id": item_id,
                            "output_index": index,
                            "summary_index": summary_index,
                            "delta": text,
                        },
                    )
                emit(
                    "response.reasoning_summary_text.done",
                    {
                        "item_id": item_id,
                        "output_index": index,
                        "summary_index": summary_index,
                        "text": text,
                    },
                )
                emit(
                    "response.reasoning_summary_part.done",
                    {
                        "item_id": item_id,
                        "output_index": index,
                        "summary_index": summary_index,
                        "part": part,
                    },
                )

        emit("response.output_item.done", {"output_index": index, "item": item})

    terminal = "response.incomplete" if response["status"] == "incomplete" else "response.completed"
    emit(terminal, {"response": response})
    return events
