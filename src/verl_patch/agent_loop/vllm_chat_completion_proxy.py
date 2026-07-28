"""
In-process vLLM Chat Completions proxy for Harbor.

This module hosts the aiohttp server that mimics the OpenAI
`/v1/chat/completions` + `/v1/models` endpoints. Harbor (via LiteLLM) sends HTTP
requests to this proxy; the proxy tokenizes messages using the same
`apply_chat_template` path as verl agent loops and forwards the request to

Kept as a standalone module to avoid bloating `builtin_swe_agent_loop.py`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import aiohttp
import aiohttp.web

from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# R3 diagnostic counter: a few VISIBLE (AgentLoopWorker process) lines confirming
# whether the TokenOutput the proxy receives actually carries routed_experts.
_R3_PROXY_DIAG = 0
# R3 probe: per-turn, how many of the generated (last n) rows are actually non-zero —
# splits "vLLM read returns zeros for gen tokens" from "we lose it in archiving".
_R3_TURN_PROBE = 0

# When a genuine sequence divergence forces the lossy rebuild path (CC mutated
# mid-history — e.g. a dynamic <system-reminder> insertion — which id-anchoring
# cannot absorb), recompute the rebuilt prefix's per-token logprobs via vLLM
# prompt_logprobs instead of training on zeros. Kill-switch: set to 0 to revert
# to the old zero-fill behavior (e.g. if prompt_logprobs + prefix-cache misbehave).
_RECOMPUTE_MISMATCH_LP = os.environ.get(
    "HARBOR_RECOMPUTE_MISMATCH_LOGPROBS", "1"
).strip().lower() not in ("0", "false", "no", "")

# Claude Code Task-tool sub-agents (Explore, web-search, ...) inherit
# ANTHROPIC_BASE_URL → the same proxy session_id as the main agent, but are
# *distinct conversations with distinct system prompts*. A same-agent variation
# (a dynamic <system-reminder>/<env> edit) keeps the whole identity preamble
# identical → long common prefix; a different sub-agent diverges within the
# first sentence. This threshold separates "reminder change on the main thread"
# (kept, rebuild+recompute) from "different sub-agent" (routed to isolated,
# non-archived state). Override via env.
_MAIN_AGENT_PREFIX_MATCH = int(
    os.environ.get("HARBOR_MAIN_AGENT_PREFIX_MATCH", "256")
)

_OPENCODE_HIDDEN_SYSTEM_PROMPT_PREFIXES = (
    "You are a title generator. You output ONLY a thread title.",
)


def _system_text(messages: list[dict]) -> str:
    """Leading system-message content (OpenAI-format), or '' if none."""
    if messages and messages[0].get("role") == "system":
        c = messages[0].get("content")
        return c if isinstance(c, str) else ""
    return ""


def _is_opencode_hidden_system_prompt(sys_text: str) -> bool:
    """OpenCode internal agents must not claim the main trainable thread."""
    return any(
        sys_text.startswith(prefix)
        for prefix in _OPENCODE_HIDDEN_SYSTEM_PROMPT_PREFIXES
    )


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


try:
    from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser
except Exception:
    Hermes2ProToolParser = None

try:
    from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
except Exception:
    Qwen3CoderToolParser = None

_VLLM_TOOL_PARSER_CLASSES: dict[str, type | None] = {
    "hermes": Hermes2ProToolParser,
    "qwen3_coder": Qwen3CoderToolParser,
}

import re as _re

_HERMES_TOOL_CALL_RE = _re.compile(
    r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", _re.DOTALL
)


def _lenient_hermes_extract(decoded_out: str) -> tuple[str | None, list[dict]] | None:
    """Fallback tool-call parser using json.loads(strict=False).

    Handles the most common vllm hermes parser failures: unescaped control
    characters inside JSON strings (e.g. literal newlines/tabs in argument
    values).  Returns (content, tool_calls_list) on success, None on failure.
    """
    if "<tool_call>" not in decoded_out:
        return None
    try:
        matches = _HERMES_TOOL_CALL_RE.findall(decoded_out)
        if not matches:
            return None
        raw_calls = []
        for m in matches:
            raw = m[0] if m[0] else m[1]
            raw = raw.strip()
            if not raw:
                continue
            parsed = json.loads(raw, strict=False)
            raw_calls.append(parsed)
        if not raw_calls:
            return None
        tool_calls = []
        for fc in raw_calls:
            tool_calls.append({
                "id": f"call_{uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": fc["name"],
                    "arguments": json.dumps(fc.get("arguments", {}), ensure_ascii=True),
                },
            })
        content = decoded_out[: decoded_out.find("<tool_call>")]
        return (content if content else None, tool_calls)
    except Exception:
        return None


def _compute_gen_prompt_len(tokenizer: Any, apply_chat_template_kwargs: dict | None) -> int:
    """Length (in tokens) of the trailing ``add_generation_prompt`` block that
    ``apply_chat_template`` always appends. For chat templates that emit
    ``<|im_start|>assistant\\n`` (Qwen-style) this is typically 3 tokens.

    Computed once at proxy construction time by diffing the tokenized form of a
    single empty-user message with and without ``add_generation_prompt``. We
    intentionally use a dummy user turn (instead of an empty message list)
    because some templates raise on an empty list or inject a default system
    block whose length we do not want to entangle into the diff.
    """
    kwargs = dict(apply_chat_template_kwargs or {})
    try:
        with_gp = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            tools=None,
            add_generation_prompt=True,
            tokenize=True,
            **kwargs,
        )
        without_gp = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            tools=None,
            add_generation_prompt=False,
            tokenize=True,
            **kwargs,
        )
        with_gp_ids = normalize_token_ids(with_gp)
        without_gp_ids = normalize_token_ids(without_gp)
        logger.info(
            "[Proxy] computed gen_prompt length: with_gp=%d, without_gp=%d",
            len(with_gp_ids),
            len(without_gp_ids),
        )
        return max(0, len(with_gp_ids) - len(without_gp_ids))
    except Exception as exc:
        logger.exception(
            "[Proxy] failed to compute gen_prompt length (err=%s); falling back to 0.",
            exc,
        )
        return 0


def _compute_assistant_eos_tail(
    tokenizer: Any, apply_chat_template_kwargs: dict | None
) -> tuple[int | None, list[int]]:
    """Return ``(eos_token_id, tail_after_eos)`` for the assistant turn format.

    ``tail_after_eos`` is the (typically single ``\\n``) tokens that the chat
    template emits *after* the assistant turn's EOS marker. vLLM stops at
    EOS without emitting these trailing tokens, so the proxy needs to
    backfill them whenever it appends a vLLM completion that ended at EOS,
    in order to keep ``rolling_acc_ids`` and ``traj_acc_ids`` byte-equivalent
    to a one-shot ``apply_chat_template`` result.

    Computed once at proxy construction time by encoding a trivial assistant
    message and slicing the tokens after the last occurrence of
    ``tokenizer.eos_token_id``. When the tokenizer has no EOS or the chat
    template emits no trailing tokens, both return values are empty/None
    and the caller treats the backfill as a no-op.
    """
    kwargs = dict(apply_chat_template_kwargs or {})
    try:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            logger.info(f"[Proxy] computed assistant EOS tail: eos_token_id=None, tail_after_eos=[]")
            return None, []
        sample = tokenizer.apply_chat_template(
                    [{"role": "user", "content": "dummy_input"},
                    {"role": "assistant", "content": "dummy_output"}],
                    tools=None,
                    add_generation_prompt=False,
                    tokenize=True,
                    **kwargs,
                )
        sample = normalize_token_ids(sample)
        for i in range(len(sample) - 1, -1, -1):
            if sample[i] == eos_token_id:
                logger.info(f"[Proxy] computed assistant EOS tail: eos_token_id={eos_token_id}, tail_after_eos={sample[i + 1:]}")
                return int(eos_token_id), list(sample[i + 1 :])
        logger.info(f"[Proxy] computed assistant EOS tail: eos_token_id={eos_token_id}, tail_after_eos=None")
        return int(eos_token_id), []
    except Exception as exc:
        logger.exception(
            "[Proxy] failed to compute assistant EOS tail (err=%s); "
            "backfill disabled.",
            exc,
        )
        return None, []


def _build_vllm_tool_parser(tool_parser_name: str, tokenizer: Any):
    parser_cls = _VLLM_TOOL_PARSER_CLASSES.get(tool_parser_name)
    if parser_cls is None:
        logger.warning(
            "Unsupported or unavailable tool_parser=%s; supported=%s. Tool parsing disabled.",
            tool_parser_name,
            ",".join(sorted(_VLLM_TOOL_PARSER_CLASSES)),
        )
        return None
    try:
        return parser_cls(tokenizer)
    except Exception as exc:
        logger.warning("vLLM tool parser init failed (name=%s, err=%s); tool parsing disabled.", tool_parser_name, exc)
        return None


def _copy_messages_for_snapshot(messages: list[dict]) -> list[dict]:
    """Shallow copy messages for prefix comparison across HTTP requests."""
    return [dict(m) for m in messages]


def _utc_debug_timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def _message_snap_debug_entry(index: int, message: dict, source: str) -> dict:
    return {
        "index": index,
        "role": message.get("role"),
        "source": source,
        "timestamp": _utc_debug_timestamp(),
    }


def _message_snap_debug_for(
    messages: list[dict], source: str, start_index: int = 0
) -> list[dict]:
    return [
        _message_snap_debug_entry(start_index + i, m, source)
        for i, m in enumerate(messages)
    ]


def _write_proxy_trajectory_file(trajectory_dir: Path, payload: dict) -> None:
    """Atomically write proxy trajectory debug JSON under Harbor's trial folder."""
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    path = trajectory_dir / "proxy_trajectory.json"
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def _tool_calls_equiv(snap_tcs, inc_tcs) -> bool:
    """Anchor assistant turns on tool_call identity (id alone), NOT name/arguments.

    Claude Code re-serializes the *arguments* of echoed tool_calls in open-ended
    ways — JSON-string <-> native list/dict, key reorder, schema-default fill,
    blank-line whitespace, and more — but round-trips the call ``id`` and
    ``name`` verbatim (empirically 151/151 prefix-mismatches had identical
    id+name with args-only diffs). Crucially, the real sampled token_ids +
    logprobs for this turn already live in ``traj_acc_ids`` (appended verbatim
    from ``output.token_ids`` in ``_finalize_generation``, never re-tokenized
    from this echoed history), so however CC re-renders the arguments is
    irrelevant to *alignment*. Matching on the verbatim ``id`` is therefore both
    sufficient and robust to any future re-serialization quirk — no per-transform
    allowlist to maintain, no lossy rebuild triggered by cosmetic arg diffs.
    """
    snap_tcs = snap_tcs or []
    inc_tcs = inc_tcs or []
    if len(snap_tcs) != len(inc_tcs):
        return False
    for s, i in zip(snap_tcs, inc_tcs):
        s, i = s or {}, i or {}
        sid, iid = s.get("id"), i.get("id")
        if sid or iid:
            # Anchor on the verbatim id ALONE. CC renames some tools between the
            # advertised name (what the model generated, archived in token form)
            # and its internal registry name when echoing history — empirically
            # ``Task``->``Agent`` (sub-agent spawn) and ``KillShell``->``TaskStop``
            # — while round-tripping the call ``id`` verbatim. So the name is NOT a
            # reliable identity signal; the id is. Comparing name here used to force
            # a needless rebuild on every sub-agent spawn (~57% of the C-class
            # mismatches), and that rebuild would also retokenize the turn under
            # CC's renamed tool — corrupting the very tokens the model sampled.
            if sid != iid:
                return False
        else:
            # No id on either side: fall back to name as the only identity signal.
            if s.get("function", {}).get("name") != i.get("function", {}).get("name"):
                return False
    return True


def _message_equiv(snap_m: dict, inc_m: dict) -> bool:
    """Whether two messages are the same conversation turn for prefix alignment.

    Exact equality is the fast path. Otherwise, only an *assistant* turn may
    still match, and only via its tool_call identity (see ``_tool_calls_equiv``):
    the turn's sampled tokens are archived, so a re-serialized-arguments echo is
    the same turn. A content-only assistant turn (no tool_calls) that is not
    byte-equal is treated as a genuine divergence → caller falls back to
    retokenize. user/tool/system turns must match exactly (CC round-trips those
    faithfully), so any difference there is real.
    """
    if snap_m == inc_m:
        return True
    if snap_m.get("role") != "assistant" or inc_m.get("role") != "assistant":
        return False
    s_tcs = snap_m.get("tool_calls") or []
    i_tcs = inc_m.get("tool_calls") or []
    if s_tcs or i_tcs:
        return _tool_calls_equiv(s_tcs, i_tcs)
    return False


def _messages_is_prefix(prefix: list[dict], full: list[dict]) -> bool:
    if len(prefix) > len(full):
        return False
    return all(_message_equiv(p, f) for p, f in zip(prefix, full[: len(prefix)]))


def _messages_have_multimodal(messages: list[dict]) -> bool:
    """True iff any message carries real multimodal content (image/video/audio).

    The proxy only needs the HF ``processor`` (and must disable token-level
    trajectory archiving) for genuinely multimodal requests, where image patches
    are not token-in/token-out. A text-only model that merely ships a
    ``processor_config.json`` (e.g. Qwen3.5 base) still tokenizes via the
    tokenizer path and produces token-identical ids, so it must keep full
    proxy-token fidelity. Content is OpenAI-format here (``_anthropic_to_openai_messages``
    already ran): a list of parts, with multimodal parts typed image_url/image/etc.
    """
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") in (
                    "image_url",
                    "image",
                    "video_url",
                    "video",
                    "input_audio",
                ):
                    return True
    return False


def _messages_before_first_assistant(messages: list[dict]) -> list[dict]:
    """Return messages strictly before the first ``assistant`` turn (system/user/tool/…).

    Used so ``initial_prompt_token_len`` reflects the true initial prompt, not the
    full multi-turn history tokenized as one block.
    """
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            return messages[:i]
    return list(messages)


def _normalize_openai_messages_for_template(messages: list[dict]) -> list[dict]:
    """Return a shallow-copied message list safe for `apply_chat_template`.

    OpenHands / LiteLLM may send `content` as OpenAI multimodal blocks
    `[{"type":"text","text":"..."}]`. Most HF chat templates expect string
    `content` for text-only models; passing raw lists can drop or distort
    user/system text.
    """

    def _flatten_content(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and item.get("text") is not None:
                    parts.append(str(item["text"]))
            return "".join(parts)
        return str(content)

    normalized: list[dict] = []
    for raw in messages:
        m = dict(raw)
        m["content"] = _flatten_content(m.get("content"))
        # String function.arguments → dict for templating consistency
        if m.get("tool_calls"):
            new_tcs = []
            for tc in m["tool_calls"]:
                new_tc = dict(tc)
                func = dict(new_tc.get("function") or {})
                args = func.get("arguments")
                if isinstance(args, str):
                    try:
                        func["arguments"] = json.loads(args)
                    except json.JSONDecodeError:
                        func["arguments"] = {}
                new_tc["function"] = func
                new_tcs.append(new_tc)
            m["tool_calls"] = new_tcs
        normalized.append(m)
    return normalized


# ============================================================================
# Anthropic <-> OpenAI codec (Design A: let the proxy speak Anthropic directly)
# ----------------------------------------------------------------------------
# These run INSIDE the proxy, BEFORE tokenization, so the sampled token_ids /
# log_probs are archived with full fidelity — the same as the OpenAI path. The
# entry-direction block-unpacking MUST stay consistent with
# ``builtin_cc_agent_loop._normalize_anthropic_messages`` (the retokenize
# fallback), otherwise the two paths would build different message shapes.
# ============================================================================


def _anthropic_to_openai_messages(body: dict) -> tuple[list[dict], list[dict] | None]:
    """Anthropic Messages API body -> (openai_messages, openai_tools).

    - system: Anthropic top-level field (str or list of text blocks) -> a
      leading {role:system} message.
    - assistant.content tool_use blocks -> OpenAI tool_calls.
    - user.content tool_result blocks -> separate {role:tool} messages placed
      before any trailing user text (OpenAI ordering convention).
    - tools: {name, description, input_schema} -> {type:function, function:{...}}.
    """
    openai_msgs: list[dict] = []

    system = body.get("system")
    if system:
        if isinstance(system, list):
            sys_text = "".join(
                b.get("text", "")
                for b in system
                if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            sys_text = str(system)
        if sys_text:
            openai_msgs.append({"role": "system", "content": sys_text})

    for msg in body.get("messages", []) or []:
        role = msg.get("role")
        content = msg.get("content")

        if isinstance(content, str):
            openai_msgs.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            openai_msgs.append({"role": role, "content": ""})
            continue

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        tool_results: list[dict] = []
        for blk in content:
            if not isinstance(blk, dict):
                continue
            bt = blk.get("type")
            if bt == "text":
                text_parts.append(blk.get("text", "") or "")
            elif bt == "tool_use" and role == "assistant":
                tool_calls.append(
                    {
                        "id": blk.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": blk.get("name", ""),
                            "arguments": blk.get("input", {}),
                        },
                    }
                )
            elif bt == "tool_result" and role == "user":
                tr = blk.get("content", "")
                if isinstance(tr, list):
                    tr = "".join(
                        i.get("text", "")
                        for i in tr
                        if isinstance(i, dict) and i.get("type") == "text"
                    )
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": blk.get("tool_use_id", ""),
                        "content": str(tr) if tr is not None else "",
                    }
                )

        joined = "".join(text_parts)
        if role == "assistant":
            m: dict = {"role": "assistant", "content": joined}
            if tool_calls:
                m["tool_calls"] = tool_calls
            openai_msgs.append(m)
        elif role == "user":
            # tool results must precede any trailing user text (OpenAI convention)
            openai_msgs.extend(tool_results)
            if joined.strip() or not tool_results:
                openai_msgs.append({"role": "user", "content": joined})
        else:
            openai_msgs.append({"role": role, "content": joined})

    tools = None
    a_tools = body.get("tools")
    if a_tools:
        tools = []
        for t in a_tools:
            if not isinstance(t, dict):
                continue
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.get("name", ""),
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {}) or {},
                    },
                }
            )
    return openai_msgs, tools


def _openai_message_to_anthropic(
    message: dict,
    finish_reason: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    session_id: str,
) -> dict:
    """OpenAI assistant message -> Anthropic Messages API response body."""
    blocks: list[dict] = []
    if message.get("content"):
        blocks.append({"type": "text", "text": message["content"]})
    for tc in message.get("tool_calls") or []:
        func = tc.get("function") or {}
        args = func.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        blocks.append(
            {
                "type": "tool_use",
                "id": tc.get("id") or f"toolu_{uuid4().hex[:24]}",
                "name": func.get("name", ""),
                "input": args or {},
            }
        )
    if not blocks:
        blocks.append({"type": "text", "text": ""})

    stop_reason = (
        "tool_use"
        if finish_reason == "tool_calls"
        else ("max_tokens" if finish_reason == "length" else "end_turn")
    )
    return {
        "id": f"msg_{session_id}_{uuid4().hex[:8]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": prompt_tokens, "output_tokens": completion_tokens},
    }


def _tools_equal(a: list | None, b: list | None) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return json.dumps(a, sort_keys=True, default=str) == json.dumps(
            b, sort_keys=True, default=str
        )
    except Exception:
        return a == b


def _default_advertise_ipv4() -> str:
    """Pick this machine's primary non-loopback IPv4 for `api_base`."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        finally:
            s.close()
        if ip and not ip.startswith("127."):
            return ip
    except OSError:
        pass
    try:
        hostname = socket.gethostname()
        for res in socket.getaddrinfo(hostname, None, socket.AF_INET, socket.SOCK_STREAM):
            ip = res[4][0]
            if ip and not ip.startswith("127."):
                return ip
    except OSError:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if ip and not ip.startswith("127."):
                return ip
    except OSError:
        pass
    raise RuntimeError(
        "Cannot determine a non-loopback IPv4 for the in-process vLLM proxy URL; "
        "assign this host a routable address or fix /etc/hosts and DNS."
    )


def _advertise_ipv4_for_proxy() -> str:
    """Return the IPv4 Harbor should place in `api_base`."""
    try:
        import ray

        if ray.is_initialized():
            ip = ray.util.get_node_ip_address()
            if ip and not str(ip).startswith("127."):
                logger.debug("[Proxy] advertise IP from Ray: %s", ip)
                return str(ip)
    except Exception as exc:
        logger.debug("[Proxy] Ray node IP unavailable (%s); using heuristic", exc)
    return _default_advertise_ipv4()


def _turn_routing_per_token(routed_experts: Any, n_tokens: int) -> list:
    """Slice one generate turn's vLLM ``routed_experts`` down to its generated tokens.

    vLLM returns ``routed_experts`` spanning ``[this-turn prompt + generated]`` with
    shape ``[m, num_layers, topk]``; the generated tokens are the LAST ``n_tokens``
    rows (mirrors verl fully_async agent_loop's ``[-len(token_ids):]``). Returns a
    per-token list aligned 1:1 with the turn's output token ids -- each element a
    ``[num_layers, topk]`` list, or ``None`` when routing is unavailable (R3 off /
    non-MoE). Kept as plain Python lists to mirror ``traj_response_logprobs``;
    assembled into a tensor only in ``BuiltinCCAgentLoop._make_output``.
    """
    if routed_experts is None or n_tokens <= 0:
        return [None] * max(0, n_tokens)
    arr = routed_experts
    # --- probe: are the GENERATED (last n) rows actually non-zero as vLLM returns them? ---
    global _R3_TURN_PROBE
    if _R3_TURN_PROBE < 0:  # [R3-turn] probe disabled (set >0 to re-enable)
        _R3_TURN_PROBE += 1
        try:
            import numpy as _np
            _a = _np.asarray(routed_experts)
            _tail = _a[-n_tokens:] if _a.shape[0] >= n_tokens else _a
            _nz = int((_tail.reshape(_tail.shape[0], -1) != 0).any(axis=1).sum())
            print(
                f"[R3-turn] gen_rows={n_tokens} nonzero_in_gen_tail={_nz} full_shape={tuple(_a.shape)}",
                flush=True,
            )
        except Exception as _e:  # pragma: no cover
            print(f"[R3-turn] probe error: {_e}", flush=True)
    try:
        arr = arr.tolist() if hasattr(arr, "tolist") else list(arr)
    except Exception:
        return [None] * n_tokens
    if len(arr) >= n_tokens:
        return list(arr[len(arr) - n_tokens:])
    return [None] * (n_tokens - len(arr)) + list(arr)


def _proxy_error(status: int, message: str) -> aiohttp.web.Response:
    return aiohttp.web.json_response(
        {"error": {"message": message, "type": "invalid_request"}}, status=status
    )


def _is_context_overflow_exc(exc: Exception) -> bool:
    """True when ``exc`` is a context-window overflow (prompt exceeds the model's max
    context). vLLM raises a bare ValueError ("Prompt length N leaves no room to generate
    within the model's maximum context length M"); match on the message text so the signal
    is robust across the Ray/proxy/SDK boundaries that strip the exception type."""
    text = f"{getattr(exc, 'message', '') or ''} {exc}".lower()
    return (
        "leaves no room to generate" in text
        or "maximum context length" in text
        or "exceeds the model" in text
        or "context_length_exceeded" in text
    )


class _AdmissionPreempted(Exception):
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(
            f"Request preempted by a newer retry for session {session_id}"
        )


class _SessionAttempt:
    def __init__(self, session_id: str, seq: int) -> None:
        self.session_id = session_id
        self.seq = seq
        self.abort_event = asyncio.Event()
        self.phase = "created"


def _proxy_error_from_exception(exc: Exception) -> aiohttp.web.Response:
    """Best-effort passthrough of upstream (vLLM/server) exception fields."""
    if isinstance(exc, _AdmissionPreempted):
        return _proxy_error(409, str(exc))

    status = (
        getattr(exc, "status_code", None)
        or getattr(exc, "http_status", None)
        or getattr(exc, "status", None)
        or 500
    )
    try:
        status = int(status)
    except (TypeError, ValueError):
        status = 500

    # Context-window overflow is PERMANENT, not transient. vLLM raises a bare
    # ValueError ("Prompt length N exceeds the model's maximum context length")
    # with no status attr, so the fallback above tags it 500. The Claude Code SDK
    # treats 500 as retryable and hammers the SAME (un-shrinkable) prompt ~10x with
    # exponential backoff before giving up (~3 min wasted per overflow). Force a
    # non-retryable 400 so the SDK fails fast — the trajectory still ends as
    # context-overflow/reward-0 with its real pre-failure logprobs intact (the
    # is_context_error classification keys off the message text, not the status).
    _body = getattr(exc, "body", None)
    _body_err = ""
    if isinstance(_body, dict) and isinstance(_body.get("error"), dict):
        _body_err = str(_body["error"].get("message", ""))
    _ovf_text = f"{getattr(exc, 'message', '') or ''} {exc} {_body_err}".lower()
    if "maximum context length" in _ovf_text or "exceeds the model" in _ovf_text:
        status = 400

    if hasattr(exc, "body") and isinstance(getattr(exc, "body"), dict):
        body = getattr(exc, "body")
        if "error" in body:
            err = body.get("error")
            if isinstance(err, dict):
                err_msg = str(err.get("message", ""))
                if "maximum context length" in err_msg.lower():
                    if "exceed context limit!" not in err_msg.lower():
                        err["message"] = f"{err_msg} exceed context limit!"
            return aiohttp.web.json_response(body, status=status)

    message = (
        getattr(exc, "message", None)
        or getattr(exc, "detail", None)
        or str(exc)
    )
    message = str(message)
    if "maximum context length" in message.lower():
        if "exceed context limit!" not in message.lower():
            message = f"{message} exceed context limit!"
    return _proxy_error(status, message)


_PROXY_INIT_LOCK = threading.Lock()


def _env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in (
        "", "0", "false", "no", "off"
    )


def _env_float(name: str, default: str) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


class _SessionAdmissionLease:
    def __init__(
        self,
        scheduler: "_SessionLevelScheduler",
        session_id: str,
        wait_sec: float,
        reason: str,
        usage: float | None,
        prefix_hit_ratio: float | None,
        retain_session: bool,
    ) -> None:
        self._scheduler = scheduler
        self.session_id = session_id
        self.wait_sec = wait_sec
        self.reason = reason
        self.usage = usage
        self.prefix_hit_ratio = prefix_hit_ratio
        # Whether the session should keep its high-pressure ownership window
        # after this single in-flight vLLM request returns.
        self.retain_session = retain_session
        self._released = False

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._scheduler.release(self.session_id, self.retain_session)


class _NoopSessionAdmissionLease:
    wait_sec = 0.0
    reason = "disabled"
    usage = None
    prefix_hit_ratio = None
    retain_session = False

    async def release(self) -> None:
        return None


class _SessionLevelScheduler:
    """Proxy-local session admission gate for high KV-cache pressure.

    In an agent loop, a session typically has at most one in-flight LLM request.
    The useful scheduling unit is therefore not the in-flight request, but the
    session's short ownership window after a turn completes: the tool result is
    likely to arrive soon and should keep using the same resident prefix. A
    session releases that ownership when it explicitly finishes or stays idle
    past ``idle_grace_sec``.
    """

    def __init__(self, server_manager: Any) -> None:
        self.enabled = _env_bool("HARBOR_SESSION_LEVEL_SCHEDULER", "0")
        self._server_manager = server_manager
        self.high_watermark = _env_float("HARBOR_SESSION_SCHEDULER_HIGH_WATERMARK", "0.85")
        self.hit_threshold = _env_float("HARBOR_SESSION_SCHEDULER_HIT_THRESHOLD", "0.80")
        self.wait_poll_sec = max(5, _env_float("HARBOR_SESSION_SCHEDULER_WAIT_POLL_SEC", "5"))
        self.idle_grace_sec = max(0.0, _env_float("HARBOR_SESSION_SCHEDULER_IDLE_GRACE_SEC", "60"))

        self._cond = asyncio.Condition()
        # High-pressure session ownership: session_id -> expiry perf_counter.
        # A reservation is refreshed on release and purged lazily by waiters.
        self._reserved_sessions: dict[str, float] = {}

    async def acquire(
        self,
        session_id: str,
        prompt_ids: list[int],
        *,
        abort_event: asyncio.Event | None = None,
    ) -> _SessionAdmissionLease | _NoopSessionAdmissionLease:
        if not self.enabled:
            return _NoopSessionAdmissionLease()

        start = time.perf_counter()
        last_usage: float | None = None
        last_hit_ratio: float | None = None

        while True:
            if abort_event is not None and abort_event.is_set():
                raise _AdmissionPreempted(session_id)

            usage_probe = await self._kv_cache_usage_probe(session_id)
            if abort_event is not None and abort_event.is_set():
                raise _AdmissionPreempted(session_id)
            last_usage = self._extract_usage(usage_probe)
            high_pressure = (
                last_usage is not None and last_usage >= self.high_watermark
            )

            if not high_pressure:
                async with self._cond:
                    self._purge_expired_locked(time.perf_counter())
                    return self._admit_locked(
                        session_id, start, "open", last_usage, last_hit_ratio,
                        retain_session=False,
                    )

            async with self._cond:
                self._purge_expired_locked(time.perf_counter())
                if abort_event is not None and abort_event.is_set():
                    raise _AdmissionPreempted(session_id)
                if self._has_reservation_locked(session_id):
                    return self._admit_locked(
                        session_id, start, "reserved_session", last_usage, last_hit_ratio,
                        retain_session=True,
                    )

            prefix_probe = await self._prefix_cache_probe(session_id, prompt_ids)
            if abort_event is not None and abort_event.is_set():
                raise _AdmissionPreempted(session_id)
            last_hit_ratio = self._extract_hit_ratio(prefix_probe)
            async with self._cond:
                now = time.perf_counter()
                self._purge_expired_locked(now)
                if abort_event is not None and abort_event.is_set():
                    raise _AdmissionPreempted(session_id)
                if last_hit_ratio is None:
                    return self._admit_locked(
                        session_id, start, "prefix_probe_unavailable", last_usage, last_hit_ratio,
                        retain_session=False,
                    )
                if last_hit_ratio >= self.hit_threshold:
                    return self._admit_locked(
                        session_id, start, "resident_prefix", last_usage, last_hit_ratio,
                        retain_session=True,
                    )
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=self.wait_poll_sec)
                except asyncio.TimeoutError:
                    pass
                if abort_event is not None and abort_event.is_set():
                    raise _AdmissionPreempted(session_id)

    def _admit_locked(
        self,
        session_id: str,
        start: float,
        reason: str,
        usage: float | None,
        hit_ratio: float | None,
        *,
        retain_session: bool,
    ) -> _SessionAdmissionLease:
        if retain_session:
            # Keep ownership throughout the in-flight request. release() will
            # convert it to an idle grace window.
            self._reserved_sessions[session_id] = float("inf")
        wait_sec = time.perf_counter() - start
        if wait_sec > 0.01 or reason not in ("open", "reserved_session"):
            logger.debug(
                "[ProxyScheduler] admit session=%s reason=%s wait=%.3fs usage=%s hit_ratio=%s reserved=%d",
                session_id, reason, wait_sec, usage, hit_ratio,
                len(self._reserved_sessions),
            )
        return _SessionAdmissionLease(
            self, session_id, wait_sec, reason, usage, hit_ratio, retain_session
        )

    async def release(self, session_id: str, retain_session: bool) -> None:
        async with self._cond:
            if retain_session and session_id in self._reserved_sessions:
                self._reserved_sessions[session_id] = (
                    time.perf_counter() + self.idle_grace_sec
                )
            self._purge_expired_locked(time.perf_counter())
            self._cond.notify_all()

    async def finish_session(self, session_id: str, reason: str = "finished") -> None:
        if not self.enabled:
            return
        async with self._cond:
            removed = self._reserved_sessions.pop(session_id, None) is not None
            if removed:
                logger.debug(
                    "[ProxyScheduler] release session reservation session=%s reason=%s",
                    session_id, reason,
                )
            self._cond.notify_all()

    def _has_reservation_locked(self, session_id: str) -> bool:
        return session_id in self._reserved_sessions

    def _purge_expired_locked(self, now: float) -> None:
        expired = [
            sid for sid, expires_at in self._reserved_sessions.items()
            if expires_at <= now
        ]
        for sid in expired:
            self._reserved_sessions.pop(sid, None)
        if expired:
            self._cond.notify_all()

    async def _kv_cache_usage_probe(self, session_id: str) -> dict:
        fn = getattr(self._server_manager, "kv_cache_usage_probe", None)
        if fn is None:
            return {}
        try:
            probe = await fn(session_id)
            return probe if isinstance(probe, dict) else {}
        except Exception as exc:
            logger.debug("[ProxyScheduler] kv usage probe failed session=%s: %s", session_id, exc)
            return {}

    async def _prefix_cache_probe(self, session_id: str, prompt_ids: list[int]) -> dict:
        fn = getattr(self._server_manager, "prefix_cache_probe", None)
        if fn is None:
            return {}
        try:
            probe = await fn(session_id, prompt_ids=list(prompt_ids))
            return probe if isinstance(probe, dict) else {}
        except Exception as exc:
            logger.debug("[ProxyScheduler] prefix probe failed session=%s: %s", session_id, exc)
            return {}

    @staticmethod
    def _extract_usage(probe: dict) -> float | None:
        try:
            if not probe or probe.get("kv_cache_usage") is None:
                return None
            return float(probe["kv_cache_usage"])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_hit_ratio(probe: dict) -> float | None:
        try:
            if not probe or probe.get("hit_ratio") is None:
                return None
            return float(probe["hit_ratio"])
        except (TypeError, ValueError):
            return None


def _message_calls_finish_tool(message: dict) -> bool:
    for tc in message.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function") or {}
        name = str(func.get("name") or "").strip().lower()
        if name == "finish" or name.endswith(".finish"):
            return True
    return False


class _VLLMChatCompletionsProxy:
    """aiohttp server that forwards Chat Completions to `server_manager.generate`."""

    def __init__(self, agent_loop: Any, tool_parser_name: str, max_consecutive_no_tool: int = 0):
        # Keep a handle to the agent loop for `apply_chat_template` and parsers.
        self._agent_loop = agent_loop
        self._tokenizer = agent_loop.tokenizer
        self._processor = agent_loop.processor
        self._server_manager = agent_loop.server_manager
        self._tool_parser_name = tool_parser_name
        # When >0, terminate the session after this many consecutive assistant
        # turns that do not contain tool_calls. 0 = disabled.
        self._max_consecutive_no_tool = max_consecutive_no_tool
        self._vllm_tool_parser = _build_vllm_tool_parser(tool_parser_name, self._tokenizer)
        self._loop = agent_loop.loop

        # Token length of the trailing ``add_generation_prompt`` block (e.g.
        # ``<|im_start|>assistant\n`` for Qwen). Used by the fallback rebuild
        # in ``_compute_prompt_ids`` to strip the redundant gen_prompt that
        # ``apply_chat_template`` always appends to each per-message tokenization.
        self._gen_prompt_len = _compute_gen_prompt_len(
            self._tokenizer, getattr(agent_loop, "apply_chat_template_kwargs", None)
        )

        # vLLM stops at the assistant EOS token without emitting the trailing
        # tokens that the chat template would otherwise put after it (e.g. the
        # ``\n`` after ``<|im_end|>`` for Qwen). ``_finalize_generation`` uses
        # these to backfill ``rolling_acc_ids`` / ``traj_acc_ids`` whenever a
        # completion ended at EOS, so the incremental path becomes
        # byte-equivalent to a one-shot ``apply_chat_template``.
        self._assistant_eos_token_id, self._assistant_eos_tail = _compute_assistant_eos_tail(
            self._tokenizer, getattr(agent_loop, "apply_chat_template_kwargs", None)
        )

        self._sessions: dict[str, dict] = {}
        # Registry lock only: protects creation/removal of session entries and
        # per-session locks. Session state itself is protected by
        # _session_state_locks so unrelated sessions never queue behind a large
        # trajectory append from another session.
        self._sessions_lock = asyncio.Lock()
        # Serialize the full prompt->generate->finalize critical section per
        # effective session. LiteLLM may retry a timed-out HTTP request while the
        # original vLLM generation is still running; without this, both requests
        # can generate and append to the same trajectory concurrently.
        self._session_generation_locks: dict[str, asyncio.Lock] = {}
        # Short critical-section locks for each effective session's state dict.
        self._session_state_locks: dict[str, asyncio.Lock] = {}
        self._session_scheduler = _SessionLevelScheduler(self._server_manager)
        self._session_attempts: dict[str, _SessionAttempt] = {}
        self._session_attempt_seq = 0
        # Sub-agent routing (see _route_subagent_session):
        # base session_id -> the main agent's system prompt (first seen).
        self._main_system: dict[str, str] = {}
        # base session_id -> derived sub-agent state keys, for cleanup on pop.
        self._sub_session_keys: dict[str, set] = {}

        self._start_lock = asyncio.Lock()
        self._site: aiohttp.web.TCPSite | None = None
        self._runner: aiohttp.web.AppRunner | None = None
        self._port: int | None = None
        self._advertise_host: str | None = None

        self._app = aiohttp.web.Application()
        self._app.router.add_post("/sess/{session_id}/v1/chat/completions", self._handle_chat)
        self._app.router.add_get("/sess/{session_id}/v1/models", self._handle_models)
        self._app.router.add_post("/v1/chat/completions", self._handle_chat_anon)
        self._app.router.add_get("/v1/models", self._handle_models)
        # Anthropic Messages API endpoints (Design A: Claude Code connects here
        # directly, no LiteLLM). Both per-session and anonymous variants.
        self._app.router.add_post("/sess/{session_id}/v1/messages", self._handle_messages)
        self._app.router.add_post("/v1/messages", self._handle_messages_anon)

    async def ensure_started(self) -> "_VLLMChatCompletionsProxy":
        if self._site is not None:
            return self
        async with self._start_lock:
            if self._site is not None:
                return self
            self._runner = aiohttp.web.AppRunner(self._app, access_log=None)
            await self._runner.setup()
            self._advertise_host = _advertise_ipv4_for_proxy()
            self._site = aiohttp.web.TCPSite(self._runner, host="0.0.0.0", port=0)
            await self._site.start()
            sockets = list(self._site._server.sockets) if self._site._server else []
            assert sockets, "aiohttp TCPSite did not bind a socket"
            self._port = sockets[0].getsockname()[1]
            logger.info(
                "[Proxy] vLLM proxy listening on 0.0.0.0:%d — api_base uses http://%s:%d/...",
                self._port,
                self._advertise_host,
                self._port,
            )
        return self

    async def stop(self) -> None:
        if self._site is not None:
            await self._site.stop()
            self._site = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        self._port = None
        self._advertise_host = None

    @property
    def base_url(self) -> str:
        if self._port is None or self._advertise_host is None:
            raise RuntimeError("Proxy is not started; call ensure_started() first")
        return f"http://{self._advertise_host}:{self._port}"

    def session_url(self, session_id: str) -> str:
        return f"{self.base_url}/sess/{quote(session_id, safe='')}/v1"

    def session_anthropic_base(self, session_id: str) -> str:
        """ANTHROPIC_BASE_URL for a session. The Claude Code SDK appends
        ``/v1/messages``, so we hand it the ``/sess/{sid}`` prefix only."""
        return f"{self.base_url}/sess/{quote(session_id, safe='')}"

    async def _get_session_generation_lock(self, session_id: str) -> asyncio.Lock:
        async with self._sessions_lock:
            lock = self._session_generation_locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._session_generation_locks[session_id] = lock
            return lock

    async def _get_session_state_lock(self, session_id: str) -> asyncio.Lock:
        async with self._sessions_lock:
            lock = self._session_state_locks.get(session_id)
            if lock is None:
                lock = asyncio.Lock()
                self._session_state_locks[session_id] = lock
            return lock

    async def open_session(
        self, session_id: str, trajectory_dir: str | Path | None = None
    ) -> None:
        async with self._sessions_lock:
            self._session_state_locks.setdefault(session_id, asyncio.Lock())
            st = self._fresh_session_state()
            if trajectory_dir is not None:
                st["trajectory_dir"] = str(Path(trajectory_dir).expanduser().resolve())
            self._sessions[session_id] = st

    async def pop_session(self, session_id: str) -> dict:
        async with self._sessions_lock:
            meta = self._sessions.pop(session_id, None) or self._fresh_session_state()
            # Drop the main-system fingerprint and any isolated sub-agent threads
            # (CC Task sub-agents) that shared this session; never trained, so
            # just discard their state to avoid leaking across trials.
            self._main_system.pop(session_id, None)
            self._session_generation_locks.pop(session_id, None)
            self._session_state_locks.pop(session_id, None)
            self._session_attempts.pop(session_id, None)
            sub_keys = tuple(self._sub_session_keys.pop(session_id, ()))
            for sub_key in sub_keys:
                self._sessions.pop(sub_key, None)
                self._session_generation_locks.pop(sub_key, None)
                self._session_state_locks.pop(sub_key, None)
                self._session_attempts.pop(sub_key, None)
        await self._session_scheduler.finish_session(session_id, reason="pop_session")
        for sub_key in sub_keys:
            await self._session_scheduler.finish_session(sub_key, reason="pop_session")
        traj_dir = meta.get("trajectory_dir")
        # One-shot assemble: final snapshot already holds full chat prefix (no per-step duplication).
        if traj_dir and (
            meta.get("messages_snapshot") is not None or int(meta.get("num_calls") or 0) > 0
        ):
            payload = {
                "session_id": session_id,
                "messages_snapshot": _copy_messages_for_snapshot(
                    meta.get("messages_snapshot") or []
                ),
                "message_snap_debug": deepcopy(meta.get("message_snap_debug") or []),
                "tools": deepcopy(meta["last_tools"])
                if meta.get("last_tools") is not None
                else None,
            }
            await asyncio.to_thread(_write_proxy_trajectory_file, Path(traj_dir), payload)
        return meta

    @staticmethod
    def _fresh_session_state() -> dict:
        return {
            "min_global_steps": None,
            "max_global_steps": None,
            "num_calls": 0,
            "num_aborts": 0,
            "num_preempted": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "messages_snapshot": None,
            "message_snap_debug": [],
            "last_tools": None,
            "rolling_acc_ids": [],
            "initial_prompt_token_len": None,
            "traj_acc_ids": [],
            "traj_response_mask": [],
            "traj_response_logprobs": [],
            # R3: per-response-token routed experts, parallel to traj_response_logprobs.
            # Each element is a [num_layers, topk] list (sampled token) or None placeholder.
            "traj_response_routing": [],
            "incremental_tokenize_ok": True,
            "disable_proxy_trajectory": False,
            # Set True when a generate call is refused because the prompt overflowed the
            # model context window. Read by the agent loop to classify the trajectory as
            # termination_reason="overlong" from an explicit signal (no exc string-matching
            # across process boundaries). pop_session() returns it in the session meta.
            "context_overflow": False,
            # Consecutive assistant turns without tool_calls (for early termination).
            "consecutive_no_tool": 0,
            # Cumulative wall-time spent in server_manager.generate (vLLM inference).
            "proxy_model_inference_sec": 0.0,
            # Wall-time between returning one completion and receiving the next HTTP
            # request for this session (Harbor-side tools + client/network).
            "proxy_tool_call_sec": 0.0,
            # Per-turn version of the above: one entry per inter-completion gap (≈ one
            # round of the agent's in-pod tool execution + reasoning). The agent loop turns
            # this into per-trajectory mean/max/min/p90, which verl aggregates per step —
            # a hung/slow tool round shows up as a large max here.
            "tool_gap_secs": [],
            "last_completion_perf": None,
            "proxy_scheduler_wait_sec": 0.0,
            "proxy_scheduler_num_waits": 0,
            "proxy_scheduler_last_reason": None,
            "proxy_scheduler_last_usage": None,
            "proxy_scheduler_last_hit_ratio": None,
            "trajectory_dir": None,
        }

    async def _route_subagent_session(
        self, session_id: str, messages: list[dict]
    ) -> tuple[str, bool]:
        """Map a request to its conversation thread's state key.

        The main agent and every CC Task sub-agent (Explore, web-search, ...)
        share one proxy ``session_id`` (inherited ANTHROPIC_BASE_URL) yet are
        distinct conversations with distinct system prompts. Folding them onto
        one state thrashes the main thread's prefix chain (full rebuilds on every
        switch). So: the first system prompt seen on a session is the main thread
        (keeps the bare ``session_id``); any request whose system prompt is
        *substantially different* is a sub-agent, routed to an isolated
        ``{session_id}::sub::{fp}`` key with archiving disabled — served so CC
        works, but never folded into the trained trajectory.

        A small same-agent variation (dynamic <system-reminder>/<env> edit) keeps
        a long common prefix and stays on the main thread (handled there by
        rebuild + recompute), so reminders are NOT mistaken for sub-agents.
        Returns ``(effective_session_key, is_subagent)``.
        """
        sys_text = _system_text(messages)
        async with self._sessions_lock:
            if _is_opencode_hidden_system_prompt(sys_text):
                # OpenCode may issue title/summary-style internal requests before
                # the coding agent first turn. Serve them, but keep the bare
                # session_id available for the real main thread.
                fp = f"{len(sys_text)}_{abs(hash(sys_text)) & 0xffffffff:08x}"
                eff = f"{session_id}::hidden::{fp}"
                st = self._sessions.get(eff)
                if st is None:
                    st = self._fresh_session_state()
                    st["disable_proxy_trajectory"] = True
                    self._sessions[eff] = st
                    self._session_state_locks.setdefault(eff, asyncio.Lock())
                    logger.info(
                        "[Proxy] routed OpenCode hidden conversation off main session "
                        "(session=%s -> %s, system_len=%d)",
                        session_id, eff, len(sys_text),
                    )
                self._sub_session_keys.setdefault(session_id, set()).add(eff)
                return eff, True

            main_sys = self._main_system.get(session_id)
            if main_sys is None:
                # First request on this session establishes the main thread.
                self._main_system[session_id] = sys_text
                return session_id, False
            if (
                sys_text == main_sys
                or _common_prefix_len(sys_text, main_sys) >= _MAIN_AGENT_PREFIX_MATCH
            ):
                return session_id, False
            # Sub-agent: isolate on a derived key, never archive.
            fp = f"{len(sys_text)}_{abs(hash(sys_text)) & 0xffffffff:08x}"
            eff = f"{session_id}::sub::{fp}"
            st = self._sessions.get(eff)
            if st is None:
                st = self._fresh_session_state()
                st["disable_proxy_trajectory"] = True
                self._sessions[eff] = st
                self._session_state_locks.setdefault(eff, asyncio.Lock())
                logger.info(
                    "[Proxy] routed sub-agent conversation off main session "
                    "(session=%s -> %s, sub_system_len=%d, main_system_len=%d)",
                    session_id, eff, len(sys_text), len(main_sys),
                )
            self._sub_session_keys.setdefault(session_id, set()).add(eff)
            return eff, True

    async def _register_session_attempt(self, session_id: str) -> _SessionAttempt:
        async with self._sessions_lock:
            previous = self._session_attempts.get(session_id)
            if previous is not None and previous.phase in ("created", "admission_wait"):
                previous.abort_event.set()
                logger.info(
                    "[Proxy] preempt stale queued request for session=%s old_seq=%d phase=%s",
                    session_id, previous.seq, previous.phase,
                )
            self._session_attempt_seq += 1
            attempt = _SessionAttempt(session_id, self._session_attempt_seq)
            self._session_attempts[session_id] = attempt
            return attempt

    async def _clear_session_attempt(
        self, session_id: str, attempt: _SessionAttempt
    ) -> None:
        async with self._sessions_lock:
            if self._session_attempts.get(session_id) is attempt:
                self._session_attempts.pop(session_id, None)

    async def _record_admission_preempted(
        self, session_id: str, state_lock: asyncio.Lock
    ) -> None:
        async with state_lock:
            sess = self._sessions.setdefault(session_id, self._fresh_session_state())
            sess["num_preempted"] = int(sess.get("num_preempted") or 0) + 1

    async def _initial_prompt_token_len_for_messages(
        self, messages: list[dict], tools: list[dict] | None
    ) -> int:
        """Token length of the opening prompt: messages before the first assistant turn."""
        prefix = _messages_before_first_assistant(messages)
        if not prefix:
            return 0
        ids = await self._agent_loop.apply_chat_template(prefix, tools=tools)
        return len(list(ids))

    async def _handle_models(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        return aiohttp.web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": "vllm-rollout",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "verl",
                    }
                ],
            }
        )

    async def _handle_chat_anon(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        request.match_info["session_id"] = "anon-" + uuid4().hex
        return await self._handle_chat(request)

    async def _logged_json_response(
        self,
        request: aiohttp.web.Request,
        payload: dict,
        *,
        session_id: str,
        api: str,
    ) -> aiohttp.web.StreamResponse:
        try:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        except Exception:
            logger.exception(
                "[Proxy] %s response json dumps failed (session=%s)",
                api,
                session_id,
            )
            raise

        resp = aiohttp.web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Content-Length": str(len(body)),
            },
        )
        try:
            t0 = time.perf_counter()
            await resp.prepare(request)
            await resp.write(body)
            await resp.write_eof()
            return resp
        except asyncio.CancelledError:
            logger.exception(
                "[Proxy] %s response send cancelled (session=%s, total=%.3fs)",
                api,
                session_id,
                time.perf_counter() - t0,
            )
            raise
        except (ConnectionResetError, BrokenPipeError) as exc:
            logger.exception(
                "[Proxy] %s response send failed: client disconnected "
                "(session=%s, bytes=%d, total=%.3fs): %r",
                api,
                session_id,
                len(body),
                time.perf_counter() - t0,
                exc,
            )
            raise
        except Exception:
            logger.exception(
                "[Proxy] %s response send failed unexpectedly "
                "(session=%s, bytes=%d, total=%.3fs): %r",
                api,
                session_id,
                len(body),
                time.perf_counter() - t0,
                exc,
            )
            raise

    async def _generate_assistant_message(
        self,
        session_id: str,
        messages: list[dict],
        tools: list[dict] | None,
        body: dict,
    ) -> tuple[dict, str, int, int]:
        """Token-fidelity generation core shared by the OpenAI and Anthropic
        endpoints.

        Tokenizes ``messages`` via the verl tokenizer, runs
        ``server_manager.generate(prompt_ids=...)`` (token-in/token-out), and
        archives the sampled ``token_ids`` + ``log_probs`` into the session via
        ``_finalize_generation``. Returns the assembled assistant message plus
        ``(finish_reason, prompt_tokens, completion_tokens)``.

        Raises on tokenization / generate errors; callers translate via
        ``_proxy_error_from_exception``. ``last_completion_perf`` is updated on
        both success and generate-failure so inter-call timing stays accurate.
        """
        # Route CC Task sub-agents off the main thread so the main conversation
        # accumulates cleanly (no rebuild thrash). For the main agent this is a
        # no-op returning the bare session_id; sub-agents get an isolated,
        # non-archived state key. All session-state and the vLLM request_id below
        # use the routed key.
        session_id, _is_subagent = await self._route_subagent_session(session_id, messages)
        session_lock = await self._get_session_generation_lock(session_id)
        state_lock = await self._get_session_state_lock(session_id)
        attempt = await self._register_session_attempt(session_id)
        try:
            if attempt.abort_event.is_set():
                await self._record_admission_preempted(session_id, state_lock)
                raise _AdmissionPreempted(session_id)
            async with session_lock:
                if attempt.abort_event.is_set():
                    await self._record_admission_preempted(session_id, state_lock)
                    raise _AdmissionPreempted(session_id)
                async with state_lock:
                    sess = self._sessions.setdefault(session_id, self._fresh_session_state())
                    t_enter = time.perf_counter()
                    last_cmp = sess.get("last_completion_perf")
                    if last_cmp is not None:
                        _gap = t_enter - float(last_cmp)
                        sess["proxy_tool_call_sec"] = float(sess.get("proxy_tool_call_sec") or 0.0) + _gap
                        sess.setdefault("tool_gap_secs", []).append(_gap)
                    prompt_ids = await self._compute_prompt_ids(sess, messages, tools, session_id)

                sampling_params = self._translate_sampling_params(body)

                attempt.phase = "admission_wait"
                try:
                    lease = await self._session_scheduler.acquire(
                        session_id, list(prompt_ids), abort_event=attempt.abort_event
                    )
                except _AdmissionPreempted:
                    await self._record_admission_preempted(session_id, state_lock)
                    raise
                attempt.phase = "generating"
                if getattr(lease, "wait_sec", 0.0) > 0.0:
                    async with state_lock:
                        sched_sess = self._sessions.setdefault(session_id, self._fresh_session_state())
                        sched_sess["proxy_scheduler_wait_sec"] = (
                            float(sched_sess.get("proxy_scheduler_wait_sec") or 0.0)
                            + float(getattr(lease, "wait_sec", 0.0))
                        )
                        sched_sess["proxy_scheduler_num_waits"] = (
                            int(sched_sess.get("proxy_scheduler_num_waits") or 0) + 1
                        )
                        sched_sess["proxy_scheduler_last_reason"] = getattr(lease, "reason", None)
                        sched_sess["proxy_scheduler_last_usage"] = getattr(lease, "usage", None)
                        sched_sess["proxy_scheduler_last_hit_ratio"] = getattr(lease, "prefix_hit_ratio", None)

                t_gen_start = time.perf_counter()
                try:
                    try:
                        output: TokenOutput = await self._server_manager.generate(
                            request_id=session_id,
                            prompt_ids=list(prompt_ids),
                            sampling_params=sampling_params,
                        )
                    finally:
                        await lease.release()
                    global _R3_PROXY_DIAG
                    if _R3_PROXY_DIAG < 0:  # [R3-proxy] probe disabled (set >0 to log first N)
                        _R3_PROXY_DIAG += 1
                        _re = getattr(output, "routed_experts", "NOATTR")
                        _nz = -1
                        try:
                            import numpy as _np

                            if hasattr(_re, "shape"):
                                _f = _np.asarray(_re)
                                _f = _f.reshape(_f.shape[0], -1)
                                _nz = int((_f != 0).any(axis=1).sum())
                        except Exception:
                            pass
                        print(
                            f"[R3-proxy] received TokenOutput.routed_experts is_none={_re is None} "
                            f"type={type(_re).__name__} shape={getattr(_re, 'shape', None)} "
                            f"nonzero_rows={_nz} "
                            f"ntok={len(getattr(output, 'token_ids', []) or [])}",
                            flush=True,
                        )
                except Exception as exc:
                    logger.exception(
                        "[Proxy] server_manager.generate failed (session=%s, prompt_len=%d)",
                        session_id,
                        len(prompt_ids),
                    )
                    async with state_lock:
                        fin = self._sessions.setdefault(session_id, self._fresh_session_state())
                        fin["last_completion_perf"] = time.perf_counter()
                        # Record context-window overflow as an explicit session signal so the agent
                        # loop can classify termination_reason="overlong" without re-matching the
                        # exception text after it has crossed the Ray/SDK boundary.
                        if _is_context_overflow_exc(exc):
                            fin["context_overflow"] = True
                    raise
                inference_sec = time.perf_counter() - t_gen_start

                await self._finalize_generation(session_id, sess, output, len(prompt_ids))

                async with state_lock:
                    s = self._sessions.setdefault(session_id, self._fresh_session_state())
                    s["proxy_model_inference_sec"] = float(s.get("proxy_model_inference_sec") or 0.0) + inference_sec

                message: dict[str, Any] = {"role": "assistant"}
                finish_reason = "stop" if output.stop_reason != "length" else "length"

                try:
                    decoded_out = self._tokenizer.decode(
                        output.token_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                except TypeError:
                    decoded_out = self._tokenizer.decode(output.token_ids, skip_special_tokens=False)

                if self._vllm_tool_parser is not None:
                    try:
                        from vllm.entrypoints.openai.chat_completion.protocol import (
                            ChatCompletionRequest,
                        )

                        _dummy_req = ChatCompletionRequest.model_validate(
                            {
                                "messages": [{"role": "user", "content": "-"}],
                                "model": body.get("model") or "vllm-rollout",
                            }
                        )
                        extracted = self._vllm_tool_parser.extract_tool_calls(decoded_out, _dummy_req)
                    except Exception as exc:
                        logger.exception(
                            "[Proxy] vLLM tool parser failed (session=%s): %s",
                            session_id,
                            exc,
                        )
                        extracted = None

                    if extracted is not None and extracted.tools_called and extracted.tool_calls:
                        message["content"] = extracted.content
                        message["tool_calls"] = [
                            tc.model_dump(exclude_none=True) for tc in extracted.tool_calls
                        ]
                        finish_reason = "tool_calls"
                    elif extracted is not None and extracted.content is not None:
                        message["content"] = extracted.content
                    else:
                        # Fallback: vllm parser returned no tool_calls (often due to
                        # strict JSON parsing of control characters). Re-try with
                        # lenient json.loads(strict=False).
                        fallback = _lenient_hermes_extract(decoded_out)
                        if fallback is not None:
                            fb_content, fb_tool_calls = fallback
                            message["content"] = fb_content
                            message["tool_calls"] = fb_tool_calls
                            finish_reason = "tool_calls"
                            logger.info(
                                "[Proxy] lenient fallback recovered %d tool call(s) "
                                "(session=%s)",
                                len(fb_tool_calls),
                                session_id,
                            )
                        else:
                            message["content"] = decoded_out

                else:
                    message["content"] = decoded_out

                # ``messages_snapshot`` after ``_compute_prompt_ids`` only reflects the *incoming*
                # request (prompt side). The assistant reply for this round exists only in the HTTP
                # response; if the trial ends here, no further request arrives to extend history.
                # Append this turn so ``pop_session`` / ``proxy_trajectory.json`` include the last reply.
                async with state_lock:
                    sess = self._sessions.setdefault(session_id, self._fresh_session_state())
                    prev_snap = sess.get("messages_snapshot") or []
                    normalized_assistant = _normalize_openai_messages_for_template([message])[0]
                    sess["messages_snapshot"] = list(prev_snap) + [normalized_assistant]
                    prev_debug = list(sess.get("message_snap_debug") or [])
                    if len(prev_debug) != len(prev_snap):
                        logger.warning(
                            "[Proxy] message_snap_debug length mismatch before assistant append "
                            "(session=%s, debug_len=%d, snapshot_len=%d); rebuilding debug metadata",
                            session_id,
                            len(prev_debug),
                            len(prev_snap),
                        )
                        prev_debug = _message_snap_debug_for(prev_snap, "existing_snapshot")
                    sess["message_snap_debug"] = prev_debug + [
                        _message_snap_debug_entry(len(prev_snap), normalized_assistant, "assistant_append")
                    ]

                    # Track consecutive no-tool turns and optionally terminate early.
                    if finish_reason == "tool_calls":
                        sess["consecutive_no_tool"] = 0
                    else:
                        sess["consecutive_no_tool"] = int(sess.get("consecutive_no_tool") or 0) + 1

                    if (
                        self._max_consecutive_no_tool > 0
                        and sess["consecutive_no_tool"] >= self._max_consecutive_no_tool
                    ):
                        logger.warning(
                            "[Proxy] session=%s terminated: %d consecutive turns without tool_calls "
                            "(limit=%d)",
                            session_id,
                            sess["consecutive_no_tool"],
                            self._max_consecutive_no_tool,
                        )
                        message["content"] = (
                            "Session terminated: model produced "
                            f"{sess['consecutive_no_tool']} consecutive responses "
                            "without calling any tools."
                        )
                        finish_reason = "stop"

                    sess["last_completion_perf"] = time.perf_counter()

                if _message_calls_finish_tool(message):
                    await self._session_scheduler.finish_session(session_id, reason="finish_tool")

                return message, finish_reason, len(prompt_ids), len(output.token_ids)
        finally:
            attempt.phase = "done"
            await self._clear_session_attempt(session_id, attempt)

    async def _handle_chat(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        try:
            body = await request.json()
        except Exception as exc:
            return _proxy_error(400, f"Invalid JSON body: {exc}")

        if body.get("stream"):
            return _proxy_error(400, "stream=True is not supported by the verl proxy")

        n_choices = int(body.get("n") or 1)
        if n_choices != 1:
            return _proxy_error(400, f"n={n_choices} not supported; only n=1 works")

        session_id = request.match_info.get("session_id") or "anon-" + uuid4().hex
        messages = _normalize_openai_messages_for_template(body.get("messages") or [])
        tools = body.get("tools")

        try:
            message, finish_reason, p_tok, c_tok = await self._generate_assistant_message(
                session_id, messages, tools, body
            )
        except _AdmissionPreempted as exc:
            logger.info("[Proxy] chat request preempted (session=%s)", session_id)
            return _proxy_error_from_exception(exc)
        except Exception as exc:
            logger.exception("[Proxy] chat generation failed (session=%s)", session_id)
            return _proxy_error_from_exception(exc)

        resp = {
            "id": f"chatcmpl-{session_id}-{uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model") or "vllm-rollout",
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "total_tokens": p_tok + c_tok,
            },
        }
        return await self._logged_json_response(
            request, resp, session_id=session_id, api="chat"
        )

    async def _handle_messages_anon(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        request.match_info["session_id"] = "anon-" + uuid4().hex
        return await self._handle_messages(request)

    async def _handle_messages(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        """Anthropic Messages API endpoint — reuses the token-fidelity core.

        Lets Claude Code talk to the proxy directly (no LiteLLM): the Anthropic
        request body is converted to OpenAI-shaped (messages, tools) BEFORE
        tokenization, so the sampled token_ids/log_probs are archived exactly as
        with the OpenAI path. Supports both non-streaming and (fake) SSE, since
        the Claude Code SDK defaults to stream=True.
        """
        try:
            body = await request.json()
        except Exception as exc:
            return _proxy_error(400, f"Invalid JSON body: {exc}")

        stream = bool(body.get("stream"))
        session_id = request.match_info.get("session_id") or "anon-" + uuid4().hex
        messages, tools = _anthropic_to_openai_messages(body)
        messages = _normalize_openai_messages_for_template(messages)

        try:
            msg, finish_reason, p_tok, c_tok = await self._generate_assistant_message(
                session_id, messages, tools, body
            )
        except _AdmissionPreempted as exc:
            logger.info("[Proxy] messages request preempted (session=%s)", session_id)
            return _proxy_error_from_exception(exc)
        except Exception as exc:
            logger.exception("[Proxy] messages generation failed (session=%s)", session_id)
            return _proxy_error_from_exception(exc)

        model = body.get("model") or "vllm-rollout"
        anthropic_resp = _openai_message_to_anthropic(
            msg, finish_reason, model, p_tok, c_tok, session_id
        )

        if not stream:
            return await self._logged_json_response(
                request, anthropic_resp, session_id=session_id, api="messages"
            )
        return await self._anthropic_sse_response(request, anthropic_resp)

    async def _anthropic_sse_response(
        self, request: aiohttp.web.Request, msg: dict
    ) -> aiohttp.web.StreamResponse:
        """Emit a fully-computed Anthropic message as an SSE event sequence.

        The generation already finished (token fidelity is unaffected); this
        only replays the result in the streaming wire-format the Claude Code SDK
        expects when it sends ``stream=True``.
        """
        resp = aiohttp.web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        try:
            await resp.prepare(request)
        except Exception as exc:
            # Client (Claude Code) already gone — generation/token-fidelity is
            # done, the SSE replay is cosmetic. Don't let a dead socket crash
            # the agent-loop worker.
            logger.warning("[Proxy] SSE prepare failed (client gone?): %s", exc)
            return resp

        async def send(event: str, data: dict) -> None:
            payload = json.dumps(data, ensure_ascii=False)
            try:
                await resp.write(f"event: {event}\ndata: {payload}\n\n".encode("utf-8"))
            except Exception as exc:
                logger.warning("[Proxy] SSE write failed (client gone?): %s", exc)

        in_tok = msg["usage"]["input_tokens"]
        out_tok = msg["usage"]["output_tokens"]

        await send(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    **{k: v for k, v in msg.items() if k != "content"},
                    "content": [],
                    "usage": {"input_tokens": in_tok, "output_tokens": 0},
                },
            },
        )
        for idx, blk in enumerate(msg["content"]):
            if blk.get("type") == "tool_use":
                await send(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": blk["id"],
                            "name": blk["name"],
                            "input": {},
                        },
                    },
                )
                await send(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(blk.get("input", {}), ensure_ascii=False),
                        },
                    },
                )
            else:
                await send(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                await send(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "text_delta", "text": blk.get("text", "")},
                    },
                )
            await send(
                "content_block_stop",
                {"type": "content_block_stop", "index": idx},
            )
        await send(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": msg["stop_reason"], "stop_sequence": None},
                "usage": {"output_tokens": out_tok},
            },
        )
        await send("message_stop", {"type": "message_stop"})
        try:
            await resp.write_eof()
        except Exception as exc:
            logger.warning("[Proxy] SSE write_eof failed (client gone?): %s", exc)
        return resp

    async def _recompute_prefix_logprobs(
        self,
        session_id: str,
        acc_ids: list[int],
        mask: list[int],
        lp_out: list[float],
    ) -> None:
        """Recompute real per-token logprobs for a just-rebuilt prefix, in place.

        Called only on the rebuild path — a genuine sequence divergence that
        id-anchoring could not absorb (e.g. Claude Code inserting a dynamic
        ``<system-reminder>`` mid-history, which shifts every following token).
        The incrementally-archived logprobs no longer align, so instead of
        training on zeros we run a throwaway 1-token generation over ``acc_ids``
        with vLLM ``prompt_logprobs`` enabled and fill the assistant
        (``mask==1``) positions of ``lp_out`` under the current (sampling-time)
        weights — exactly what the training-time actor recompute will later see,
        keeping ``rollout_probs_diff`` ~0 by construction.

        Best-effort and ISOLATED from the real generation: any failure (engine
        error, prompt_logprobs/prefix-cache incompatibility, shape mismatch) is
        logged and leaves ``lp_out`` as zeros (== prior behavior). Never raises.

        Alignment: ``extract_prompt_logprobs`` (num_prompt_logprobs=0) drops the
        first prompt token (no preceding context → no logprob) and re-bases the
        list, so element ``pl[m]`` holds the logprob of the token at sequence
        position ``m+1``. Hence the token at position ``p`` (p>=1) reads
        ``pl[p-1]``; position 0 has no logprob. A trailing dummy keeps
        ``len(pl) == len(acc_ids)``.
        """
        try:
            out = await self._server_manager.generate(
                request_id=f"{session_id}-lprecompute",
                prompt_ids=list(acc_ids),
                sampling_params={
                    "prompt_logprobs": 0,
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "logprobs": False,
                },
            )
        except Exception as exc:
            logger.warning(
                "[Proxy] prefix logprob recompute failed (session=%s, len=%d): %s",
                session_id, len(acc_ids), exc,
            )
            return

        pl = (getattr(out, "extra_fields", None) or {}).get("prompt_logprobs")
        n_assist = sum(1 for m in mask if m == 1)
        if not pl:
            logger.warning(
                "[Proxy] prefix logprob recompute: no prompt_logprobs returned "
                "(session=%s, acc_len=%d)", session_id, len(acc_ids),
            )
            return

        filled = missing = 0
        for p in range(1, len(acc_ids)):
            if mask[p] != 1:
                continue
            j = p - 1  # extract is shifted by one: token at p reads pl[p-1].
            if j >= len(pl):
                missing += 1
                continue
            v = pl[j]
            val = v[0] if isinstance(v, (list, tuple)) and v else v
            if val is None:
                missing += 1
                continue
            lp_out[p] = float(val)
            filled += 1
        logger.info(
            "[Proxy] prefix logprob recompute (session=%s): filled=%d/%d assistant "
            "tokens (missing=%d, prompt_logprobs_len=%d, acc_len=%d)",
            session_id, filled, n_assist, missing, len(pl), len(acc_ids),
        )

    async def _compute_prompt_ids(self, sess: dict, messages: list[dict], tools: list[dict] | None, session_id: str) -> list[int]:
        # Only disable token-level trajectory archiving for *genuinely* multimodal
        # requests. A processor object being loaded (Qwen3.5 base ships
        # processor_config.json) does NOT imply multimodal: text-only requests
        # tokenize identically via the tokenizer path, so keep proxy-token fidelity.
        if self._processor is not None and _messages_have_multimodal(messages):
            sess["disable_proxy_trajectory"] = True
            ids = await self._agent_loop.apply_chat_template(messages, tools=tools)
            sess["rolling_acc_ids"] = list(ids)
            sess["messages_snapshot"] = _copy_messages_for_snapshot(messages)
            sess["message_snap_debug"] = _message_snap_debug_for(messages, "incoming_multimodal")
            sess["last_tools"] = tools
            sess["incremental_tokenize_ok"] = False
            return list(ids)

        if sess.get("messages_snapshot") is None:
            ids = await self._agent_loop.apply_chat_template(messages, tools=tools)
            ids_list = list(ids)
            sess["rolling_acc_ids"] = ids_list
            sess["messages_snapshot"] = _copy_messages_for_snapshot(messages)
            sess["message_snap_debug"] = _message_snap_debug_for(messages, "incoming_initial")
            sess["last_tools"] = tools
            sess["initial_prompt_token_len"] = len(ids_list)
            sess["traj_acc_ids"] = list(ids_list)
            sess["traj_response_mask"] = []
            sess["traj_response_logprobs"] = []
            sess["traj_response_routing"] = []
            sess["incremental_tokenize_ok"] = True
            return ids_list

        snap = sess["messages_snapshot"]
        if not _messages_is_prefix(snap, messages) or not _tools_equal(sess.get("last_tools"), tools):
            trajectory_dir = sess.get("trajectory_dir")
            logger.error(
                "messages are not a prefix extension of the snapshot or tools are not the same as tools "
                "in the snapshot, falling back to per-message rebuild "
                "(session_id=%s, trajectory_dir=%s)",
                session_id,
                trajectory_dir,
            )
            if trajectory_dir:
                snap_for_dump = deepcopy(snap)
                snap_debug_for_dump = deepcopy(sess.get("message_snap_debug") or [])
                messages_for_dump = deepcopy(messages)

                def _dump_prefix_mismatch_files() -> None:
                    base = Path(trajectory_dir)
                    base.mkdir(parents=True, exist_ok=True)
                    ts = _utc_debug_timestamp()
                    (base / f"prefix_mismatch_snap_{ts}.json").write_text(
                        json.dumps(snap_for_dump, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8",
                    )
                    (base / f"prefix_mismatch_snap_debug_{ts}.json").write_text(
                        json.dumps(snap_debug_for_dump, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8",
                    )
                    (base / f"prefix_mismatch_message_{ts}.json").write_text(
                        json.dumps(messages_for_dump, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8",
                    )

                try:
                    await asyncio.to_thread(_dump_prefix_mismatch_files)
                except Exception as dump_exc:
                    logger.warning(
                        "[Proxy] failed to persist prefix-mismatch debug files "
                        "(session_id=%s, dir=%s): %s",
                        session_id,
                        trajectory_dir,
                        dump_exc,
                    )

            # Rebuild rolling/traj state piece-by-piece so the result matches
            # what the incremental (prefix-matched) path would have produced
            # token-for-token, modulo the trailing ``\n`` after each historical
            # ``<|im_end|>`` that vLLM omits when stopping at the EOS marker.
            #
            # Layout (same as the incremental path):
            #
            #   prompt segment      = apply_chat_template(prefix_msgs, tools)
            #                         (ends with a single gen_prompt block)
            #   for each assistant: format(msg) stripped of its leading
            #                       <|im_start|>assistant\n (already present as
            #                       the previous segment's trailing gen_prompt)
            #                       AND its trailing gen_prompt; marked mask=1.
            #   for each batch of   apply_chat_template(batch, tools=None,
            #     consecutive          remove_system_prompt=True) → format(batch)
            #     non-assistant       + gen_prompt; marked mask=0 (the trailing
            #     messages:           gen_prompt is the start-of-assistant marker
            #                       for the next assistant turn / the next
            #                       vLLM call).
            #
            # Batching consecutive non-assistant messages (instead of tokenizing
            # them one by one) keeps the result bit-equivalent to the
            # incremental path's ``delta = messages[len(snap):]`` tokenization.
            #
            # Snapshot the PRIOR archive before we overwrite it: the dominant
            # rebuild trigger is abort/retry, where the clean incoming history is
            # a token *prefix* of this (stale, phantom-carrying) archive. We use
            # it below to salvage real logprobs for the unchanged prefix instead
            # of training the whole trajectory on zeros.
            _old_acc = list(sess.get("traj_acc_ids") or [])
            _old_lp = list(sess.get("traj_response_logprobs") or [])
            _old_routing = list(sess.get("traj_response_routing") or [])
            _old_plen = int(sess.get("initial_prompt_token_len") or 0)

            prefix_msgs = _messages_before_first_assistant(messages)
            prompt_ids_list = list(
                await self._agent_loop.apply_chat_template(prefix_msgs, tools=tools)
            )

            gp_len = self._gen_prompt_len
            rebuilt_acc: list[int] = list(prompt_ids_list)
            rebuilt_mask: list[int] = []
            rebuilt_lp: list[float] = []
            rebuilt_routing: list = []

            idx = len(prefix_msgs)
            while idx < len(messages):
                if messages[idx].get("role") == "assistant":
                    asst_ids = list(
                        await self._agent_loop.apply_chat_template(
                            [messages[idx]], tools=None, remove_system_prompt=True
                        )
                    )
                    # asst_ids == <|im_start|>assistant\n + content + <|im_end|>\n + gen_prompt.
                    # The leading <|im_start|>assistant\n is already in
                    # ``rebuilt_acc`` as the previous segment's trailing
                    # gen_prompt; the trailing gen_prompt is redundant and
                    # would duplicate the start-of-assistant marker for the
                    # *next* turn. Strip both.
                    if gp_len > 0 and len(asst_ids) >= 2 * gp_len:
                        asst_ids = asst_ids[gp_len:-gp_len]
                    rebuilt_acc.extend(asst_ids)
                    rebuilt_mask.extend([1] * len(asst_ids))
                    rebuilt_lp.extend([0.0] * len(asst_ids))
                    rebuilt_routing.extend([None] * len(asst_ids))
                    idx += 1
                    continue

                run_end = idx
                while run_end < len(messages) and messages[run_end].get("role") != "assistant":
                    run_end += 1
                batch = messages[idx:run_end]
                batch_ids = list(
                    await self._agent_loop.apply_chat_template(
                        batch, tools=None, remove_system_prompt=True
                    )
                )
                rebuilt_acc.extend(batch_ids)
                rebuilt_mask.extend([0] * len(batch_ids))
                rebuilt_lp.extend([0.0] * len(batch_ids))
                rebuilt_routing.extend([None] * len(batch_ids))
                idx = run_end

            sess["rolling_acc_ids"] = list(rebuilt_acc)
            sess["messages_snapshot"] = _copy_messages_for_snapshot(messages)
            sess["message_snap_debug"] = _message_snap_debug_for(messages, "incoming_rebuild")
            sess["last_tools"] = tools
            sess["initial_prompt_token_len"] = len(prompt_ids_list)
            sess["traj_acc_ids"] = list(rebuilt_acc)
            sess["traj_response_mask"] = rebuilt_mask

            # --- Salvage real logprobs for the unchanged token prefix. ---
            # ``rebuilt_lp`` is placeholder zeros. The dominant rebuild cause is
            # abort/retry: a turn vLLM generated but Claude Code dropped/replaced
            # leaves the prior archive one (or more) assistant turns AHEAD of the
            # clean incoming history, so the rebuilt token sequence is a *prefix*
            # of the stale archive. Every leading token they share has a
            # byte-identical left context, hence the previously-archived logprob
            # is exactly correct — copy it back rather than train on a zero. We
            # salvage ONLY the common prefix, never a common suffix: a token's
            # logprob is conditioned on its full left context, so a turn
            # re-generated after a phantom carries logprobs computed under that
            # phantom and must not be reused once the phantom is gone (zeros are
            # the correct value there). This recovers ~all logprobs for the
            # abort/retry case and everything up to the divergence otherwise,
            # with no recompute (zero OOM risk).
            n_salvaged = 0
            cp = None
            if _old_acc and _old_plen == len(prompt_ids_list):
                cp = 0
                cp_max = min(len(rebuilt_acc), len(_old_acc))
                while cp < cp_max and rebuilt_acc[cp] == _old_acc[cp]:
                    cp += 1
                n_resp_prefix = cp - len(prompt_ids_list)
                for k in range(max(0, min(n_resp_prefix, len(rebuilt_lp), len(_old_lp)))):
                    if rebuilt_mask[k]:
                        rebuilt_lp[k] = _old_lp[k]
                        if k < len(_old_routing):
                            rebuilt_routing[k] = _old_routing[k]
                        n_salvaged += 1
            n_resp_total = sum(rebuilt_mask)

            def _decode_token_for_log(token_ids: list[int]) -> str:
                try:
                    return self._tokenizer.decode(
                        token_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                except TypeError:
                    return self._tokenizer.decode(token_ids, skip_special_tokens=False)
                except Exception as exc:
                    return f"<decode-error {type(exc).__name__}: {exc}>"

            def _token_window_for_log(acc: list[int], center: int | None) -> list[tuple[int, int, str]]:
                if center is None or not acc:
                    return []
                start = max(0, center - 5)
                end = min(len(acc), center + 5)
                return (center, acc[start:end], _decode_token_for_log(acc[start:end]))

            mismatch_pos = cp
            if mismatch_pos is not None and mismatch_pos >= min(len(rebuilt_acc), len(_old_acc)):
                mismatch_pos = min(len(rebuilt_acc), len(_old_acc))
            logger.info(
                "[Proxy] rebuild salvaged %d/%d response logprobs from the prior "
                "archive's common prefix (session=%s, len_old_acc=%d, old_plen=%d, "
                "len_prompt_ids=%d, len_rebuilt_acc=%d, cp=%s, mismatch_pos=%s, "
                "rebuilt_window=%r, old_window=%r)",
                n_salvaged,
                n_resp_total,
                session_id,
                len(_old_acc),
                _old_plen,
                len(prompt_ids_list),
                len(rebuilt_acc),
                cp,
                mismatch_pos,
                _token_window_for_log(rebuilt_acc, mismatch_pos),
                _token_window_for_log(_old_acc, mismatch_pos),
            )

            # Optional recompute fallback for the still-zero (divergent-tail)
            # positions. Off by default (HARBOR_RECOMPUTE_MISMATCH_LOGPROBS=0)
            # because vLLM prompt_logprobs over the full sequence OOMs; the
            # prefix salvage above already recovers the bulk. Best-effort +
            # isolated: on any failure the affected positions stay zero.
            if (
                _RECOMPUTE_MISMATCH_LP
                and not sess.get("disable_proxy_trajectory")
                and any(rebuilt_mask)
            ):
                await self._recompute_prefix_logprobs(
                    session_id, rebuilt_acc, rebuilt_mask, rebuilt_lp
                )
            sess["traj_response_logprobs"] = rebuilt_lp
            sess["traj_response_routing"] = rebuilt_routing
            return list(rebuilt_acc)

        # ``snap`` must end with the prior turn's assistant message (appended in
        # ``_handle_chat`` after each successful generation) so ``delta`` is only
        # new user/tool content—never a duplicate assistant span already emitted as completion ids.
        delta = messages[len(snap) :]
        if not delta:
            return list(sess["rolling_acc_ids"])

        delta_ids = await self._agent_loop.apply_chat_template(
            delta, tools=None, remove_system_prompt=True
        )
        delta_ids = list(delta_ids)
        new_prompt = list(sess["rolling_acc_ids"]) + delta_ids

        sess["rolling_acc_ids"] = new_prompt
        sess["messages_snapshot"] = _copy_messages_for_snapshot(messages)
        prev_debug = list(sess.get("message_snap_debug") or [])
        if len(prev_debug) != len(snap):
            logger.warning(
                "[Proxy] message_snap_debug length mismatch before delta append "
                "(session=%s, debug_len=%d, snapshot_len=%d); rebuilding debug metadata",
                session_id,
                len(prev_debug),
                len(snap),
            )
            prev_debug = _message_snap_debug_for(snap, "existing_snapshot")
        sess["message_snap_debug"] = prev_debug + _message_snap_debug_for(
            delta, "incoming_delta", start_index=len(snap)
        )
        sess["last_tools"] = tools

        sess["traj_acc_ids"] = list(sess["traj_acc_ids"]) + delta_ids
        sess["traj_response_mask"] = list(sess["traj_response_mask"]) + [0] * len(delta_ids)
        sess["traj_response_logprobs"] = list(sess["traj_response_logprobs"]) + [0.0] * len(delta_ids)
        sess["traj_response_routing"] = list(sess.get("traj_response_routing") or []) + [None] * len(delta_ids)
        return new_prompt

    async def _finalize_generation(self, session_id: str, sess: dict, output: TokenOutput, prompt_len: int) -> None:
        state_lock = await self._get_session_state_lock(session_id)
        async with state_lock:
            s = self._sessions.setdefault(session_id, self._fresh_session_state())
            s.update(sess)
            s["num_calls"] += 1
            s["total_prompt_tokens"] += prompt_len
            s["total_completion_tokens"] += len(output.token_ids)
            if output.num_preempted is not None and output.num_preempted > 0:
                s["num_preempted"] += int(output.num_preempted)
            if output.stop_reason in ("aborted", "abort"):
                s["num_aborts"] += 1

            extra = output.extra_fields or {}
            mn = extra.get("min_global_steps")
            mx = extra.get("max_global_steps")
            if mn is not None:
                cur = s["min_global_steps"]
                s["min_global_steps"] = mn if cur is None else min(cur, mn)
            if mx is not None:
                cur = s["max_global_steps"]
                s["max_global_steps"] = mx if cur is None else max(cur, mx)

            output_ids = list(output.token_ids)
            s["rolling_acc_ids"] = list(s.get("rolling_acc_ids") or []) + output_ids

            if not s.get("disable_proxy_trajectory"):
                lp = output.log_probs
                if lp is None:
                    lp = [0.0] * len(output_ids)
                elif len(lp) != len(output_ids):
                    lp = list(lp) + [0.0] * max(0, len(output_ids) - len(lp))

                s["traj_acc_ids"] = list(s.get("traj_acc_ids") or []) + output_ids
                s["traj_response_mask"] = list(s.get("traj_response_mask") or []) + [1] * len(output_ids)
                s["traj_response_logprobs"] = list(s.get("traj_response_logprobs") or []) + list(lp)
                # R3: archive this turn's per-token routing (vLLM returns the last
                # len(output_ids) rows for the generated tokens). Parallel to logprobs.
                s["traj_response_routing"] = list(s.get("traj_response_routing") or []) + _turn_routing_per_token(
                    getattr(output, "routed_experts", None), len(output_ids)
                )

            # Backfill the trailing tokens that the chat template puts after
            # the assistant EOS marker (typically ``\n``). vLLM stops at EOS
            # without emitting them, so without this step the incremental
            # ``rolling_acc_ids`` would diverge from a one-shot
            # ``apply_chat_template`` result by exactly these tokens per
            # assistant turn. Only triggered when the completion actually
            # ended at EOS — length/stop-string terminations are left alone.
            eos_id = self._assistant_eos_token_id
            tail = self._assistant_eos_tail
            if eos_id is not None and tail and output_ids and output_ids[-1] == eos_id:
                s["rolling_acc_ids"] = list(s["rolling_acc_ids"]) + list(tail)
                if not s.get("disable_proxy_trajectory"):
                    s["traj_acc_ids"] = list(s["traj_acc_ids"]) + list(tail)
                    s["traj_response_mask"] = list(s["traj_response_mask"]) + [0] * len(tail)
                    s["traj_response_logprobs"] = list(s["traj_response_logprobs"]) + [0.0] * len(tail)
                    s["traj_response_routing"] = list(s.get("traj_response_routing") or []) + [None] * len(tail)

    @staticmethod
    def _translate_sampling_params(body: dict) -> dict:
        sp: dict[str, Any] = {}
        for src_key, dst_key, conv in (
            ("temperature", "temperature", float),
            ("top_p", "top_p", float),
            ("top_k", "top_k", int),
            ("max_tokens", "max_tokens", int),
            ("max_completion_tokens", "max_tokens", int),
            ("seed", "seed", int),
        ):
            v = body.get(src_key)
            if v is None:
                continue
            try:
                sp[dst_key] = conv(v)
            except (TypeError, ValueError):
                continue
        v = body.get("stop")
        if v is not None:
            sp["stop"] = v if isinstance(v, list) else [v]
        # if body.get("logprobs"):
        sp["logprobs"] = True
        return sp


async def _ensure_proxy_started(
    agent_loop: Any,
) -> _VLLMChatCompletionsProxy:
    """Get-or-init the per-server_manager singleton proxy and start it."""
    tool_parser_name = getattr(agent_loop, "_tool_parser_name", None)
    max_consecutive_no_tool = int(getattr(agent_loop, "_max_consecutive_no_tool", 0))
    sm = agent_loop.server_manager
    proxy = getattr(sm, "_harbor_swe_proxy", None)
    if proxy is None:
        with _PROXY_INIT_LOCK:
            proxy = getattr(sm, "_harbor_swe_proxy", None)
            if proxy is None:
                proxy = _VLLMChatCompletionsProxy(
                    agent_loop, tool_parser_name,
                    max_consecutive_no_tool=max_consecutive_no_tool,
                )
                setattr(sm, "_harbor_swe_proxy", proxy)
    elif getattr(proxy, "_tool_parser_name", None) != tool_parser_name:
        logger.warning(
            "[Proxy] Existing singleton parser name=%s differs from requested=%s; "
            "keeping existing parser for this worker.",
            getattr(proxy, "_tool_parser_name", None),
            tool_parser_name,
        )
    await proxy.ensure_started()
    return proxy