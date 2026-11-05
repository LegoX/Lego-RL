"""
In-process vLLM Chat Completions proxy for Harbor.

This module hosts OpenAI ``/v1/chat/completions`` / ``/v1/models`` and
Anthropic ``/v1/messages`` surfaces. Harnesses send HTTP requests directly to
their per-session route; the proxy normalizes both protocols, tokenizes through
the same ``apply_chat_template`` path, and forwards generation to verl's
server manager while retaining exact token-level trajectory metadata.

Kept as a standalone module to avoid bloating `builtin_swe_agent_loop.py`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
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

from .trajectory import TrajectoryRecorder
from .responses_transform import (
    openai_message_to_responses,
    responses_sse_events,
    responses_to_openai_messages,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# R3 diagnostic counter: a few VISIBLE (AgentLoopWorker process) lines confirming
# whether the TokenOutput the proxy receives actually carries routed_experts.
_R3_PROXY_DIAG = 0
# R3 probe: per-turn, how many of the generated (last n) rows are actually non-zero —
# splits "vLLM read returns zeros for gen tokens" from "we lose it in archiving".
_R3_TURN_PROBE = 0



_MAIN_AGENT_PREFIX_MATCH = int(
    os.environ.get("HARBOR_MAIN_AGENT_PREFIX_MATCH", "256")
)
# Gate 3, compaction continuation: a request that reproduces at least this many
# characters of the reply this session generated last is the same conversation
# resuming from its own summary (CC / Codex quote it in a user turn, OpenCode
# opens the new history with the summary as an assistant turn) -- never a
# sub-agent. 0 disables the quote check.
_COMPACTION_QUOTE_MIN_CHARS = int(
    os.environ.get("HARBOR_COMPACTION_QUOTE_MIN_CHARS", "256")
)


def _load_compaction_markers() -> tuple[str, ...]:
    """Continuation markers harnesses put in front of a compacted history.

    Fallback for the quote check when the summary is shorter than the threshold.
    Extra markers: JSON list in ``HARBOR_COMPACTION_MARKERS``.
    """
    builtin = (
        # Claude Code, /compact and autocompact continuation prompt.
        "This session is being continued from a previous conversation that ran out of context",
    )
    raw = os.environ.get("HARBOR_COMPACTION_MARKERS", "").strip()
    if not raw:
        return builtin
    try:
        extra = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[Proxy] HARBOR_COMPACTION_MARKERS is not a JSON list; ignored")
        return builtin
    return builtin + tuple(str(x) for x in extra if str(x))


_COMPACTION_CONTINUATION_MARKERS = _load_compaction_markers()


def _env_int(name: str, default: str) -> int:
    try:
        return int(os.environ.get(name, default) or default)
    except ValueError:
        logger.warning("[Proxy] %s is not an int; using %s", name, default)
        return int(default)


# Cross-compaction budgets. Compaction removes the context window as a natural
# brake (a compacted trial otherwise runs to the harness turn cap or the wall
# clock), so the proxy enforces a trial-level budget instead: the request that
# would start the (N+1)-th compaction, or any request once the trial generated
# this many tokens, is refused with the context-overflow error the harnesses
# already handle (Claude Code fails fast on the 400) and the agent loop
# classifies the trajectory as ``overlong`` through the ``context_overflow``
# signal, keeping its real verifier reward. 0 = unlimited.
_COMPACTION_MAX = _env_int("HARBOR_COMPACTION_MAX", "0")
_GEN_TOKEN_BUDGET = _env_int("HARBOR_TRIAL_GEN_TOKEN_BUDGET", "0")
# CompactionRL trains the summary tokens under the task reward (their ablation
# "w/o summary training" is the main thing this switch reproduces): 0 archives
# every summary with response_mask=0 so it stays context, not action.
_TRAIN_SUMMARY = os.environ.get("HARBOR_COMPACTION_TRAIN_SUMMARY", "1").strip().lower() not in (
    "0", "false", "no", "off", ""
)

_OPENCODE_HIDDEN_SYSTEM_PROMPT_PREFIXES = (
    "You are a title generator. You output ONLY a thread title.",
)




def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _message_text(message: dict) -> str:
    """Plain text of a message's content (a string, or OpenAI text parts)."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return "" if content is None else str(content)


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


_SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)

def _first_user_identity(messages: list[dict]) -> str:
    """The opening user turn with harness-injected ``<system-reminder>`` blocks
    removed and whitespace collapsed.

    Claude Code edits those blocks in place between turns (context-window notes,
    todo state, ...); the task text underneath is what identifies the
    conversation, and OpenCode Task sub-agents differ from the main thread
    exactly there while sharing its system prompt.
    """
    for m in messages:
        if m.get("role") == "user":
            return _normalize_ws(_SYSTEM_REMINDER_RE.sub("", _message_text(m)))
    return ""


def _messages_quote_text(messages: list[dict], needle: str, min_chars: int) -> bool:
    """True when ``messages`` reproduce a sizeable chunk of ``needle`` verbatim.

    Recognises a compacted history: every harness re-injects the summary it
    just generated (CC / Codex inside a user turn, OpenCode as the assistant
    turn that opens the new history). Three probe windows (head / middle / tail
    of the needle) tolerate a harness trimming or prefixing the quoted text.
    """
    needle = _normalize_ws(needle or "")
    if min_chars <= 0 or len(needle) < min_chars:
        return False
    haystack = _normalize_ws(" ".join(_message_text(m) for m in messages))
    mid = len(needle) // 2
    probes = (
        needle[:min_chars],
        needle[max(0, mid - min_chars // 2) : mid + (min_chars - min_chars // 2)],
        needle[-min_chars:],
    )
    return any(p and p in haystack for p in probes)


def _messages_have_continuation_marker(messages: list[dict], markers=None) -> bool:
    markers = tuple(markers) if markers is not None else _COMPACTION_CONTINUATION_MARKERS
    if not markers:
        return False
    for m in messages:
        if m.get("role") != "user":
            continue
        text = _message_text(m)
        if any(marker in text for marker in markers):
            return True
    return False


def _assistant_turn_key(message: dict) -> tuple:
    """Identity of an assistant turn for subsequence matching: tool_call ids when
    present (id-anchored like ``_tool_calls_equiv``), else the exact text."""
    tcs = message.get("tool_calls") or []
    ids = tuple(str((tc or {}).get("id") or "") for tc in tcs)
    if any(ids):
        return ("ids", ids)
    return ("text", _message_text(message))


def _is_subsequence(short: list, long: list) -> bool:
    it = iter(long)
    return all(any(x == y for y in it) for x in short)


def _classify_rebuild(
    snap: list[dict], messages: list[dict], *, pending_summary_texts: tuple[str, ...] = (), initial_prefix: list[dict] | None = None
) -> str:
    """Why an incoming history is not a prefix extension of the archived snapshot.

    * ``retry``: the incoming history is a message-prefix of the snapshot -- the
      harness dropped trailing phantom turns after an abort / retry. The token
      prefix salvage in ``_compute_prompt_ids`` recovers the logprobs.
    * ``compaction_inplace``: the opening prefix survived, but assistant turns
      were removed from the middle while the most recent one is still there
      (OpenHands SDK condenser: keep ``keep_first`` events, summarise the middle,
      keep the recent tail). Token-based condensation can drop that tail too;
      in that case require an exact newly inserted summary from this session
      and preserve its opening prefix and any surviving assistant identities.
      The archive is frozen as a segment and a new one
      starts from the condensed history.
    * ``history_edit``: anything else (a dynamic reminder edit, re-serialised
      content the id-anchoring cannot absorb, ...): the lossy rebuild + salvage.

    ``compaction_replace`` (the whole history replaced by ``[system, summary,
    ...]``) is recognised earlier, by ``_route_subagent_session``, because the
    opening prefix itself changes there.
    """
    if _messages_is_prefix(messages, snap):
        return "retry"
    snap_keys = [_assistant_turn_key(m) for m in snap if m.get("role") == "assistant"]
    inc_keys = [_assistant_turn_key(m) for m in messages if m.get("role") == "assistant"]
    if (
        snap_keys
        and inc_keys
        and len(inc_keys) < len(snap_keys)
        and inc_keys[-1] == snap_keys[-1]
        and _is_subsequence(inc_keys, snap_keys)
    ):
        return "compaction_inplace"
    if (
        pending_summary_texts
        and snap_keys
        and len(inc_keys) < len(snap_keys)
        and _is_subsequence(inc_keys, snap_keys)
        and _messages_is_prefix(initial_prefix if initial_prefix is not None else _messages_before_first_assistant(snap), messages)
        and any(
            m.get("role") == "user"
            and m not in snap
            and _message_text(m) in pending_summary_texts
            for m in messages
        )
    ):
        return "compaction_inplace"
    return "history_edit"


class _CompactionBudgetExceeded(Exception):
    """A trial exhausted its cross-compaction budget (see ``_COMPACTION_MAX`` /
    ``_GEN_TOKEN_BUDGET``). Worded as a context-window overflow on purpose:
    ``_is_context_overflow_exc`` / ``_proxy_error_from_exception`` turn it into
    the permanent 400 the harnesses already give up on, and the agent loop maps
    the session's ``context_overflow`` flag to termination_reason=overlong."""

    def __init__(self, session_id: str, what: str) -> None:
        self.session_id = session_id
        super().__init__(
            f"Prompt exceeds the model's maximum context length: {what} "
            f"(session {session_id})"
        )


_COMPACTION_REBUILD_CAUSES = ("compaction_inplace", "compaction_replace")

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

_HERMES_TOOL_CALL_RE = re.compile(
    r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL
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
    in order to keep ``rolling_acc_ids`` and the trajectory recorder byte-equivalent
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
    logprobs for this turn already live in the trajectory recorder (appended verbatim
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


def _messages_fingerprint(messages: list[dict]) -> str:
    """Return a stable, compact fingerprint for a JSON-shaped message list."""
    encoded = json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return f"{len(encoded)}_{hashlib.sha256(encoded).hexdigest()[:16]}"


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
    non-MoE). Kept as plain Python lists alongside segment response logprobs;
    assembled into a tensor only in ``BuiltinSWEAgentLoop.run``.
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
        # these to backfill ``rolling_acc_ids`` / the trajectory recorder whenever a
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
        # Sub-agent routing (see _route_subagent_session): base session_id ->
        # the main conversation's complete prefix before its first assistant
        # turn. OpenCode task sub-agents reuse the main system prompt, so the
        # system prompt alone is not a conversation identity.
        self._main_identity: dict[str, dict[str, Any]] = {}
        self._summ_seq: dict[str, int] = {}
        self._gen_seq = 0
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
        self._app.router.add_post("/sess/{session_id}/summ/v1/chat/completions", self._handle_chat_summ)
        self._app.router.add_get("/sess/{session_id}/summ/v1/models", self._handle_models)
        self._app.router.add_post("/v1/chat/completions", self._handle_chat_anon)
        self._app.router.add_get("/v1/models", self._handle_models)
        # Anthropic Messages API endpoints (Design A: Claude Code connects here
        # directly, no LiteLLM). Both per-session and anonymous variants.
        self._app.router.add_post("/sess/{session_id}/v1/messages", self._handle_messages)
        self._app.router.add_post("/v1/messages", self._handle_messages_anon)
        self._app.router.add_post("/sess/{session_id}/v1/responses", self._handle_responses)
        self._app.router.add_post("/v1/responses", self._handle_responses_anon)

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
        self,
        session_id: str,
        trajectory_dir: str | Path | None = None,
        compaction: dict | None = None,
        sampling_params: dict | None = None,
        harness_name: str | None = None,
        max_turns: int | None = None,
    ) -> None:
        """Register a trial's main session. ``compaction`` is the harness's
        compaction card (see harness.COMPACTION_PROFILES): its continuation
        markers feed the routing, its ``mode`` / ``summary`` let the rebuild
        validate the compaction shape and the summary source it sees."""
        async with self._sessions_lock:
            self._session_state_locks.setdefault(session_id, asyncio.Lock())
            st = self._fresh_session_state()
            st["session_id"] = session_id
            st["compaction_profile"] = dict(compaction) if compaction else None
            # Codex omits sampling fields on the wire. Keep the caller's
            # train/validation settings scoped to this trial.
            st["harness_name"] = harness_name
            # Codex CLI does not consume Harbor's max_turns kwarg. Enforce it
            # here, at the model-request boundary, for the whole trial.
            if harness_name == "codex" and max_turns is not None:
                if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 0:
                    raise ValueError("Codex max_turns must be a non-negative integer or None")
                st["max_turns"] = max_turns or None
            st["sampling_defaults"] = (
                deepcopy(sampling_params or {}) if harness_name == "codex" else {}
            )
            if trajectory_dir is not None:
                st["trajectory_dir"] = str(Path(trajectory_dir).expanduser().resolve())
            self._sessions[session_id] = st


    async def pop_session(self, session_id: str) -> dict:
        async with self._sessions_lock:
            meta = self._sessions.pop(session_id, None) or self._fresh_session_state()
            meta["recorder_closed"] = True
            self._summ_seq.pop(session_id, None)
            # Export enabled child archives before closing all trial states.
            self._main_identity.pop(session_id, None)
            self._session_generation_locks.pop(session_id, None)
            self._session_state_locks.pop(session_id, None)
            self._session_attempts.pop(session_id, None)
            sub_keys = list(self._sub_session_keys.pop(session_id, ()))
            seen = set(sub_keys)
            for key in sub_keys:
                for descendant in self._sub_session_keys.pop(key, ()):
                    if descendant not in seen:
                        seen.add(descendant)
                        sub_keys.append(descendant)
            child_segments = []
            train_children = os.environ.get("HARBOR_TRAIN_SUBAGENTS", "0") == "1"
            incomplete = False
            for sub_key in sub_keys:
                child = self._sessions.pop(sub_key, None)
                self._main_identity.pop(sub_key, None)
                self._summ_seq.pop(sub_key, None)
                if train_children and child is not None and child.get("trainable_subagent"):
                    segments = child["trajectory_recorder"].export() + list(child.get("segments") or [])
                    if child.get("num_calls") and child.get("disable_proxy_trajectory"):
                        incomplete = True
                    for segment in segments:
                        count = len(segment["response_ids"])
                        if segment.get("response_logprobs") is None or any(
                            len(segment[key]) != count for key in
                            ("response_mask", "response_logprobs", "response_routing")
                        ):
                            incomplete = True
                        child_segments.append({**segment, "kind": "subagent", "subagent_session_id": sub_key,
                                               "subagent_segment_kind": segment.get("kind", "exec")})
                    meta.setdefault("subagent_discarded_generations", []).extend(
                        child["trajectory_recorder"].discarded_generations)
                    meta["subagent_num_calls"] = int(meta.get("subagent_num_calls") or 0) + int(child.get("num_calls") or 0)

                if child is not None:
                    child["recorder_closed"] = True
                self._session_generation_locks.pop(sub_key, None)
                self._session_state_locks.pop(sub_key, None)
                self._session_attempts.pop(sub_key, None)
        await self._session_scheduler.finish_session(session_id, reason="pop_session")
        for sub_key in sub_keys:
            await self._session_scheduler.finish_session(sub_key, reason="pop_session")
        meta["subagent_archive_incomplete"] = incomplete
        if incomplete:
            meta["disable_proxy_trajectory"] = True
        recorder = meta.pop("trajectory_recorder")
        meta["episode_id"] = session_id
        meta["trajectory_segments"] = sorted(
            recorder.export() + list(meta.get("segments") or []) + child_segments,
            key=lambda segment: int(segment.get("seq_end") or 0),
        )
        meta["discarded_generations"] = recorder.discarded_generations
        traj_dir = meta.get("trajectory_dir")
        # One-shot assemble: final snapshot already holds full chat prefix (no per-step duplication).
        if traj_dir and (
            meta.get("messages_snapshot") is not None or int(meta.get("num_calls") or 0) > 0
        ):
            payload = {
                "session_id": session_id,
                "sampling_defaults": deepcopy(meta.get("sampling_defaults") or {}),
                "last_sampling_params": deepcopy(meta.get("last_sampling_params") or {}),
                "trajectory_segments": meta["trajectory_segments"],
                "discarded_generations": meta["discarded_generations"],
                "subagent_discarded_generations": meta.get("subagent_discarded_generations", []),
                "subagent_archive_incomplete": meta["subagent_archive_incomplete"],
                "subagent_num_calls": int(meta.get("subagent_num_calls") or 0),
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
            'record_pending': None,
            'session_id': None,
            'last_assistant_text': "",
            'last_assistant_tokens': 0,
            'pending_rebuild_cause': None,
            'rebuild_counts': {},
            'compaction_count': 0,
            'segments': [],
            'carried_over_keys': [],
            'summarizer_of': None,
            'last_gen_seq': 0,
            'gen_tokens_total': 0,
            'budget_refusals': 0,
            'compaction_profile': None,
            'summaries_since_compaction': 0,
            'compaction_shape_mismatch': 0,
            'compaction_summary_missing': 0,

            "turn_limit_reached": False,
            "completed_assistant_turns": 0,
            "max_turns": None,
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
            "trajectory_recorder": TrajectoryRecorder(),
            "last_generation_messages": None,
            "last_generation_tools": None,
            "last_generation_snapshot": None,
            "pending_retry": False,
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

        The main agent and Task sub-agents share one proxy ``session_id``
        (inherited base URL) yet are distinct conversations. The first non-hidden
        request establishes the main thread's identity; a later request stays on
        the bare, trainable ``session_id`` only if it passes three gates, and is
        otherwise a sub-agent served from an isolated, non-archived
        ``{session_id}::sub::{fp}`` state:

        1. **System prompt** (``_MAIN_AGENT_PREFIX_MATCH``): CC sub-agents
           (Explore, general-purpose, web-search, ...) run under their own system
           prompt and diverge within the first sentence; a dynamic
           ``<system-reminder>`` / ``<env>`` edit keeps a long common prefix.
        2. **Opening user turn**, reminder blocks stripped: OpenCode Task
           sub-agents reuse the main system prompt but open with their own task.
        3. **Compaction continuation**: the opening turn changed under the same
           system prompt *and* the request quotes the reply this session wrote
           last (or carries a harness continuation marker) -- the main thread
           resuming from its own summary. The identity is re-anchored to the new
           opening prefix and the next ``_compute_prompt_ids`` rebuild is told to
           freeze the pre-compaction archive as a segment.

        Returns ``(effective_session_key, is_subagent)``.
        """
        sys_text = _system_text(messages)
        incoming_initial = _messages_before_first_assistant(messages)
        async with self._sessions_lock:
            if _is_opencode_hidden_system_prompt(sys_text):
                # OpenCode may issue title/summary-style internal requests before
                # the coding agent first turn. Serve them, but keep the bare
                # session_id available for the real main thread.
                fp = _messages_fingerprint(incoming_initial)
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

            main = self._main_identity.get(session_id)
            if main is None:
                # First request on this session establishes the main thread.
                self._main_identity[session_id] = {
                    "system": sys_text,
                    "first_user": _first_user_identity(messages),
                    "initial": deepcopy(incoming_initial),
                }
                return session_id, False
            same_system = sys_text == main["system"] or (
                _common_prefix_len(sys_text, main["system"]) >= _MAIN_AGENT_PREFIX_MATCH
            )
            if same_system:
                first_user = _first_user_identity(messages)
                if incoming_initial == main["initial"] or first_user == main["first_user"]:
                    # Codex keeps the original task while replacing the rest of
                    # the conversation with a summary. Identity alone therefore
                    # cannot tell whether the live token archive is continuous.
                    st = self._sessions.get(session_id) or {}
                    snap = st.get("messages_snapshot") or []
                    profile = st.get("compaction_profile") or {}
                    new_opening = [
                        m for m in incoming_initial
                        if m.get("role") == "user" and m not in main["initial"]
                    ]
                    incoming_turns = {
                        _assistant_turn_key(m) for m in messages
                        if m.get("role") == "assistant"
                    }
                    removed_turns = any(
                        _assistant_turn_key(m) not in incoming_turns
                        for m in snap if m.get("role") == "assistant"
                    )
                    # OpenCode can retain assistant/tool turns before its new
                    # summary, leaving a system-only opening after every later
                    # compaction. Require a freshly archived same-session
                    # summary, newly inserted into the incoming history.
                    pending_summaries = int(st.get("summaries_since_compaction") or 0)
                    fresh_summary_inserted = False
                    if (
                        profile.get("mode") == "replace"
                        and profile.get("summary") == "summ_route"
                        and pending_summaries > 0
                    ):
                        summaries = [
                            s for s in st.get("segments", []) if s.get("kind") == "summary"
                        ]
                        texts = {
                            str(s["summary_text"]) for s in summaries[-pending_summaries:]
                            if s.get("summary_text")
                        }
                        old_texts = {_message_text(m) for m in snap if m.get("role") == "assistant"}
                        fresh_summary_inserted = any(
                            m.get("role") == "assistant"
                            and _message_text(m) in texts - old_texts
                            for m in messages
                        )
                    if (
                        profile.get("mode") == "replace"
                        and (
                            fresh_summary_inserted
                            or (new_opening and self._is_compaction_continuation(session_id, new_opening))
                        )
                        and removed_turns
                        # A normal extension or exact-prefix retry is not a
                        # compaction, even if its context contains a summary.
                        and not _messages_is_prefix(snap, messages)
                        and not _messages_is_prefix(messages, snap)
                    ):
                        st["pending_rebuild_cause"] = "compaction_replace"
                        main.update(
                            system=sys_text,
                            first_user=first_user,
                            initial=deepcopy(incoming_initial),
                        )
                        logger.info(
                            "[Proxy] compaction retained opening identity on main session "
                            "(session=%s, messages=%d)", session_id, len(messages),
                        )
                    return session_id, False
                sub_key = f"{session_id}::sub::{_messages_fingerprint(incoming_initial)}"
                if (sub_key in self._sub_session_keys.get(session_id, ())
                        and sub_key in self._sessions):
                    return sub_key, True
                # Same system prompt, different opening turn: an OpenCode Task
                # sub-agent, or the main thread resuming from a compacted
                # history. Only the latter quotes a reply this session wrote.
                if self._is_compaction_continuation(session_id, messages):
                    main.update(
                        system=sys_text,
                        first_user=first_user,
                        initial=deepcopy(incoming_initial),
                    )
                    st = self._sessions.get(session_id)
                    if st is not None:
                        st["pending_rebuild_cause"] = "compaction_replace"
                    logger.info(
                        "[Proxy] compaction continuation kept on main session "
                        "(session=%s, messages=%d, initial_messages=%d)",
                        session_id, len(messages), len(incoming_initial),
                    )
                    return session_id, False
            # Sub-agent: isolate its conversation and optionally train its sampled
            # tokens as a separate segment under the parent trial's reward.
            fp = _messages_fingerprint(incoming_initial)
            eff = f"{session_id}::sub::{fp}"
            st = self._sessions.get(eff)
            if st is None:
                st = self._fresh_session_state()
                st["trainable_subagent"] = os.environ.get("HARBOR_TRAIN_SUBAGENTS", "0") == "1"
                st["disable_proxy_trajectory"] = not st["trainable_subagent"]
                if st["trainable_subagent"]:
                    st["session_id"] = eff
                    st["budget_parent"] = session_id
                    st["compaction_profile"] = deepcopy((self._sessions.get(session_id) or {}).get("compaction_profile"))
                self._sessions[eff] = st
                self._session_state_locks.setdefault(eff, asyncio.Lock())
                logger.info(
                    "[Proxy] routed sub-agent conversation off main session "
                    "(session=%s -> %s, same_system=%s, sub_initial_messages=%d, "
                    "main_initial_messages=%d)",
                    session_id, eff, same_system, len(incoming_initial),
                    len(main["initial"]),
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
        except Exception as exc:
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
        summarizer: bool = False,
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
        # use the routed key. Summarizer requests (``/summ/`` route) get their own
        # archived key instead and are filed as summary segments afterwards.
        # Capture main-trial defaults before routing: sub-agents, hidden
        # conversations and summarizers inherit this sampling policy.
        async with self._sessions_lock:
            main_session = self._sessions.get(session_id, {})
            is_codex = main_session.get("harness_name") == "codex"
            sampling_defaults = dict(main_session.get("sampling_defaults") or {})
            request_user_round = main_session.get("user_round_id")
        controller = main_session.get("native_round_controller")
        if controller is not None:
            if summarizer:
                raise ValueError("Native no-compaction session cannot summarize")
            messages, tools, body = await controller.prepare(messages, tools, body)
        trial_session_id = session_id
        turn_budget = main_session if is_codex and main_session.get("max_turns") else None
        session_id, _is_subagent = await self._route_generation_session(
            session_id, messages, tools, summarizer=summarizer
        )
        # A Codex trial shares one cap across compaction, summaries and routed
        # calls. Use the trial lock so concurrent requests cannot overshoot it.
        session_lock = await self._get_session_generation_lock(
            trial_session_id if turn_budget is not None else session_id
        )
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
                if turn_budget is not None and (
                    turn_budget["completed_assistant_turns"] >= turn_budget["max_turns"]
                ):
                    turn_budget["turn_limit_reached"] = True
                    # A terminal control response lets Codex exit normally and
                    # Harbor run the verifier. It is not a sampled completion:
                    # never add it to tokens, masks, logprobs, or the trajectory.
                    return {
                        "role": "assistant",
                        "content": f"[Lego-RL] Maximum agent turns reached ({turn_budget['max_turns']}).",
                    }, "stop", 0, 0
                async with state_lock:
                    self._check_recorder_session_open(main_session)
                    sess = self._sessions.setdefault(session_id, self._fresh_session_state())
                    if request_user_round and sess.get("user_round_id") != request_user_round:
                        if sess.get("user_round_id"):
                            self._mark_user_boundary(sess, "user_round")
                        sess["user_round_id"] = request_user_round
                    t_enter = time.perf_counter()
                    last_cmp = sess.get("last_completion_perf")
                    if last_cmp is not None:
                        _gap = t_enter - float(last_cmp)
                        sess["proxy_tool_call_sec"] = float(sess.get("proxy_tool_call_sec") or 0.0) + _gap
                        sess.setdefault("tool_gap_secs", []).append(_gap)
                    prompt_ids = await self._compute_prompt_ids(sess, messages, tools, session_id)

                sampling_params = self._translate_sampling_params(
                    body, defaults=sampling_defaults if is_codex else None
                )
                async with state_lock:
                    sess["last_sampling_params"] = dict(sampling_params)
                    if not sess.get("num_calls"):
                        logger.info("[Proxy] effective sampling (session=%s): %s", session_id, sampling_params)

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
                        self._check_recorder_session_open(sess)
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
                    self._check_recorder_session_open(sess)
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

                        _parser_req = ChatCompletionRequest.model_validate(
                            {
                                "messages": [{"role": "user", "content": "-"}],
                                "model": body.get("model") or "vllm-rollout",
                                # Scope typed XML recovery to Codex; preserve
                                # other harnesses until separately validated.
                                "tools": tools if is_codex else None,
                            }
                        )
                        extracted = self._vllm_tool_parser.extract_tool_calls(decoded_out, _parser_req)
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
                    self._check_recorder_session_open(sess)
                    sess = self._sessions.setdefault(session_id, self._fresh_session_state())
                    prev_snap = sess.get("messages_snapshot") or []
                    normalized_assistant = _normalize_openai_messages_for_template([message])[0]
                    sess["messages_snapshot"] = list(prev_snap) + [normalized_assistant]
                    if sess.get("trajectory_recorder") is not None:
                        sess["last_generation_snapshot"] = _copy_messages_for_snapshot(sess["messages_snapshot"])
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
                    # Compaction bookkeeping: remember this reply so a follow-up
                    # request that quotes it (a compacted history) is recognised
                    # as the same conversation, not a sub-agent.
                    sess["last_assistant_text"] = str(message.get("content") or "")
                    sess["last_assistant_tokens"] = len(output.token_ids)
                    is_summarizer = bool(sess.get("summarizer_of"))
                    if turn_budget is not None:
                        turn_budget["completed_assistant_turns"] += 1

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

                if is_summarizer:
                    await self._archive_summary_segment(sess)

                if _message_calls_finish_tool(message):
                    await self._session_scheduler.finish_session(session_id, reason="finish_tool")

                return message, finish_reason, len(prompt_ids), len(output.token_ids)
        finally:
            attempt.phase = "done"
            await self._clear_session_attempt(session_id, attempt)


    async def _handle_chat(
        self, request: aiohttp.web.Request, summarizer: bool = False
    ) -> aiohttp.web.Response:
        try:
            body = await request.json()
        except Exception as exc:
            return _proxy_error(400, f"Invalid JSON body: {exc}")

        stream = bool(body.get("stream"))

        n_choices = int(body.get("n") or 1)
        if n_choices != 1:
            return _proxy_error(400, f"n={n_choices} not supported; only n=1 works")

        session_id = request.match_info.get("session_id") or "anon-" + uuid4().hex
        messages = _normalize_openai_messages_for_template(body.get("messages") or [])
        tools = body.get("tools")

        try:
            message, finish_reason, p_tok, c_tok = await self._generate_assistant_message(
                session_id, messages, tools, body, summarizer=summarizer
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
        if stream:
            return await self._chat_sse_response(request, resp)
        return await self._logged_json_response(
            request, resp, session_id=session_id, api="chat"
        )


    async def _chat_sse_response(self, request, completion):
        """Replay an archived completion as OpenAI SSE, preserving sampling tokens.

        Like the Anthropic/Responses surfaces, generation and trajectory capture
        finish before HTTP replay. Tool argument strings and reasoning content
        travel unchanged; no second generation or tokenization is performed.
        """
        response = aiohttp.web.StreamResponse(headers={
            "Content-Type": "text/event-stream", "Cache-Control": "no-cache",
        })
        choice = completion["choices"][0]
        delta = deepcopy(choice["message"])
        if delta.get("tool_calls"):
            delta["tool_calls"] = [
                {**call, "index": i} for i, call in enumerate(delta["tool_calls"])
            ]
        base = {k: completion[k] for k in ("id", "created", "model")}
        base["object"] = "chat.completion.chunk"
        chunks = [
            {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
            {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}]},
            {**base, "choices": [], "usage": completion["usage"]},
        ]
        try:
            await response.prepare(request)
            for chunk in chunks:
                await response.write(("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode())
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
        except (ConnectionError, RuntimeError) as exc:
            logger.warning("[Proxy] chat SSE client disconnected after generation: %s", exc)
        return response


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

    async def _handle_responses_anon(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        request.match_info["session_id"] = "anon-" + uuid4().hex
        return await self._handle_responses(request)


    async def _handle_responses(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        """OpenAI Responses API endpoint — reuses the token-fidelity core.

        Lets Codex CLI talk to the proxy directly: the Responses request body
        is converted to OpenAI-shaped (messages, tools) BEFORE tokenization, so
        the sampled token_ids/log_probs are archived exactly as on the OpenAI
        and Anthropic paths. Codex always streams (``accept:
        text/event-stream``), so the reply is replayed as SSE unless the caller
        explicitly opted out.
        """
        try:
            body = await request.json()
        except Exception as exc:
            return _proxy_error(400, f"Invalid JSON body: {exc}")

        session_id = request.match_info.get("session_id") or "anon-" + uuid4().hex
        try:
            messages, tools = responses_to_openai_messages(body)
            messages = _normalize_openai_messages_for_template(messages)
        except (TypeError, ValueError) as exc:
            return _proxy_error(400, str(exc))

        # _translate_sampling_params only knows the chat spelling.
        gen_body = dict(body)
        if gen_body.get("max_output_tokens") is not None:
            gen_body.setdefault("max_tokens", gen_body["max_output_tokens"])

        try:
            msg, finish_reason, p_tok, c_tok = await self._generate_assistant_message(
                session_id, messages, tools, gen_body
            )
        except _AdmissionPreempted as exc:
            logger.info("[Proxy] responses request preempted (session=%s)", session_id)
            return _proxy_error_from_exception(exc)
        except Exception as exc:
            logger.exception("[Proxy] responses generation failed (session=%s)", session_id)
            return _proxy_error_from_exception(exc)

        model = body.get("model") or "vllm-rollout"
        responses_body = openai_message_to_responses(
            msg, finish_reason, model, p_tok, c_tok, session_id
        )

        if not bool(body.get("stream", True)):
            return await self._logged_json_response(
                request, responses_body, session_id=session_id, api="responses"
            )
        return await self._responses_sse_response(request, responses_body)


    async def _responses_sse_response(
        self, request: aiohttp.web.Request, response_body: dict
    ) -> aiohttp.web.StreamResponse:
        """Emit a fully-computed Responses body as its SSE event sequence.

        The generation already finished (token fidelity is unaffected); this
        only replays the result in the streaming wire-format Codex expects.
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
            # Client already gone — generation/token-fidelity is done, the SSE
            # replay is cosmetic. Don't let a dead socket crash the worker.
            logger.warning("[Proxy] responses SSE prepare failed (client gone?): %s", exc)
            return resp

        for event_name, payload in responses_sse_events(response_body):
            try:
                data = json.dumps(payload, ensure_ascii=False)
                await resp.write(f"event: {event_name}\ndata: {data}\n\n".encode("utf-8"))
            except Exception as exc:
                logger.warning("[Proxy] responses SSE write failed (client gone?): %s", exc)
                break
        try:
            await resp.write_eof()
        except Exception as exc:
            logger.warning("[Proxy] responses SSE write_eof failed (client gone?): %s", exc)
        return resp


    def _is_compaction_continuation(self, session_id: str, messages: list[dict]) -> bool:
        """Does this request resume the main thread from a summary this session wrote?

        Checks the last reply of the main thread (Claude Code / Codex compact on
        the main thread) and of every isolated sub-thread of the session
        (OpenCode's compaction agent runs under its own system prompt). Falls back
        to the harness continuation markers when the summary is shorter than the
        quote threshold. Caller holds ``_sessions_lock``.
        """
        for key in (session_id, *self._sub_session_keys.get(session_id, ())):
            st = self._sessions.get(key)
            if st is None:
                continue
            # A trainable sub-agent may compact its own history. Its summary
            # must not turn that continuation into the parent conversation.
            if st.get("trainable_subagent"):
                continue
            if _messages_quote_text(
                messages, st.get("last_assistant_text") or "", _COMPACTION_QUOTE_MIN_CHARS
            ):
                return True
        # Markers: the global list plus the harness card's own continuation text.
        profile = (self._sessions.get(session_id) or {}).get("compaction_profile") or {}
        markers = list(_COMPACTION_CONTINUATION_MARKERS) + [
            str(m) for m in (profile.get("markers") or []) if m
        ]
        return _messages_have_continuation_marker(messages, markers)


    def session_summ_url(self, session_id: str) -> str:
        """OpenAI-style base URL of a session's summarizer route: the harness's
        compaction/condenser LLM client points here so its summaries are archived
        as ``kind="summary"`` segments of the trial (see _open_summarizer_session)."""
        return f"{self.base_url}/sess/{quote(session_id, safe='')}/summ/v1"


    def _is_native_summary_request(self, session_id: str, messages: list[dict], tools) -> bool:
        """Recognize a harness-declared summary request without rewriting main history.

        Codex drops tool definitions when asking for its checkpoint summary. It
        therefore needs a separate prompt even though the messages are a prefix
        extension. Other edits, retries and sub-agent histories keep their routes.
        """
        main = self._sessions.get(session_id) or {}
        prefix = (main.get("compaction_profile") or {}).get("summary_request_prefix")
        snapshot = main.get("messages_snapshot")
        if not prefix or tools or not snapshot or main.get("disable_proxy_trajectory"):
            return False
        if len(messages) <= len(snapshot) or not _messages_is_prefix(snapshot, messages):
            return False
        last = messages[-1]
        return (
            last.get("role") == "user"
            and isinstance(last.get("content"), str)
            and last["content"].startswith(prefix)
        )


    async def _route_generation_session(self, session_id, messages, tools, summarizer=False):
        native_summary = not summarizer and self._is_native_summary_request(session_id, messages, tools)
        if (summarizer or native_summary) and session_id in self._sessions:
            # Check the parent budget before allocating a new state. A summary
            # request must not bypass the trial's compaction or generation limit.
            self._check_compaction_budget(self._sessions[session_id], session_id, about_to_compact=True)
        if summarizer or native_summary:
            effective, is_subagent = await self._open_summarizer_session(session_id)
            self._sessions[effective]["native_summary"] = native_summary
            return effective, is_subagent
        return await self._route_subagent_session(session_id, messages)


    async def _open_summarizer_session(self, session_id: str) -> tuple[str, bool]:
        """State key for one request on the ``/summ/`` route of ``session_id``.

        Every summarizer request gets a fresh, archived state key so its prompt
        (the history to summarize) and the sampled summary form one complete
        segment; ``_archive_summary_segment`` files it under the main session
        once generated. The key is registered like a sub-thread so the routing's
        compaction check sees the summary and ``pop_session`` cleans it up.
        """
        async with self._sessions_lock:
            n = int(self._summ_seq.get(session_id) or 0) + 1
            self._summ_seq[session_id] = n
            eff = f"{session_id}::summ::{n}"
            main = self._sessions.get(session_id)
            st = self._fresh_session_state()
            st["session_id"] = eff
            st["summarizer_of"] = session_id
            st["budget_parent"] = main.get("budget_parent") if main else None
            # Archive only when the main thread is archived (and exists): an
            # anonymous / multimodal session has no trainable trajectory to join.
            st["disable_proxy_trajectory"] = (
                main is None or bool(main.get("disable_proxy_trajectory"))
            )
            self._sessions[eff] = st
            self._session_state_locks.setdefault(eff, asyncio.Lock())
            self._sub_session_keys.setdefault(session_id, set()).add(eff)
            return eff, True


    async def _archive_summary_segment(self, sess: dict) -> None:
        """File a finished summarizer request as a ``kind="summary"`` segment of
        the main session it belongs to (CompactionRL: summary tokens are policy
        actions optimized under the task reward; ``HARBOR_COMPACTION_TRAIN_SUMMARY=0``
        keeps them as context instead)."""
        base = sess.get("summarizer_of")
        if sess.get("trajectory_recorder") is not None:
            if not base or sess.get("recorder_closed") or sess.get("disable_proxy_trajectory") or sess.get("summary_archived"):
                return
            segments = sess["trajectory_recorder"].export()
            if not segments:
                return
            for seg in segments:
                seg.update(kind="summary", boundary="summarizer",
                    summary_text=sess.get("last_assistant_text") or "",
                    summary_tokens=sum(seg["response_mask"]))
                if not _TRAIN_SUMMARY:
                    seg["response_mask"] = [0] * len(seg["response_mask"])
            async with self._sessions_lock:
                main = self._sessions.get(base)
                if main is None or main.get("disable_proxy_trajectory") or sess.get("summary_archived"):
                    return
                main.setdefault("segments", []).extend(segments)
                sess["summary_archived"] = True
                if sess.get("native_summary"):
                    main["native_summary_pending"] = True
                main["gen_tokens_total"] += int(sess.get("gen_tokens_total") or 0)
                main["summaries_since_compaction"] += 1
            return

    def _check_compaction_budget(self, sess: dict, session_id: str, about_to_compact: bool) -> None:
        """Refuse the request when the trial's cross-compaction budget is spent.

        Raises ``_CompactionBudgetExceeded`` (a context-overflow-shaped error) and
        flags ``context_overflow`` on the session so the agent loop classifies the
        trajectory as overlong. Called before any state is modified, so the live
        archive is left exactly as it was.
        """
        if sess.get("summarizer_of") or sess.get("disable_proxy_trajectory"):
            return
        n_comp = int(sess.get("compaction_count") or 0)
        if about_to_compact and _COMPACTION_MAX > 0 and n_comp >= _COMPACTION_MAX:
            what = f"compaction budget exhausted (compactions={n_comp}, max={_COMPACTION_MAX})"
        elif _GEN_TOKEN_BUDGET > 0 and int(sess.get("gen_tokens_total") or 0) >= _GEN_TOKEN_BUDGET:
            what = (
                f"generation budget exhausted (generated={int(sess.get('gen_tokens_total') or 0)}, "
                f"budget={_GEN_TOKEN_BUDGET})"
            )
        else:
            return
        sess["context_overflow"] = True
        sess["budget_refusals"] = int(sess.get("budget_refusals") or 0) + 1
        logger.warning("[Proxy] refusing request: %s (session=%s)", what, session_id)
        raise _CompactionBudgetExceeded(session_id, what)


    def _record_successful_generation(self, sess, output, routes, mn, mx):
        recorder = sess["trajectory_recorder"]
        output_ids = list(output.token_ids)
        prompt_ids = list(sess.get("rolling_acc_ids") or [])
        tail = list(self._assistant_eos_tail) if (output_ids and
            self._assistant_eos_token_id is not None and output_ids[-1] == self._assistant_eos_token_id) else []
        pending = sess.get("record_pending") or {}
        cause = pending.get("cause")
        compaction = cause in _COMPACTION_REBUILD_CAUSES
        if output_ids and not sess.get("disable_proxy_trajectory"):
            # The caller has checked lengths before mutating budget or archive.
            if pending.get("retry"):
                recorder.rollback_last_generation()
            if cause:
                counts = sess.setdefault("rebuild_counts", {})
                counts[cause] = int(counts.get(cause) or 0) + 1
            if recorder.current is not None and cause:
                recorder.current.metadata.update(boundary=cause, messages_snapshot=pending["snapshot"])
            if compaction:
                profile = sess.get("compaction_profile") or {}
                expected = profile.get("mode")
                if expected and cause != "compaction_" + expected:
                    sess["compaction_shape_mismatch"] += 1
                if profile.get("summary") == "summ_route" and not sess.get("summaries_since_compaction"):
                    sess["compaction_summary_missing"] += 1
                on_thread = (cause == "compaction_replace" and
                    profile.get("summary", "on_thread") == "on_thread" and
                    not pending.get("native_summary_pending"))
                current = recorder.current
                if current is not None:
                    n_summary = pending["summary_tokens"] if on_thread else 0
                    current.metadata.update(summary_tokens=n_summary,
                        summary_text=pending.get("summary_text") if on_thread else None)
                    if on_thread and not _TRAIN_SUMMARY and current.generations:
                        start, end, _, _ = current.generations[-1]
                        current.response_mask[start:end] = [0] * (end-start)
                sess["compaction_count"] += 1
                sess["summaries_since_compaction"] = 0
                sess["native_summary_pending"] = False
            recorder.record(prompt_ids, output_ids, None if output.log_probs is None else list(output.log_probs), routes,
                context_tail=tail, min_global_steps=mn, max_global_steps=mx,
                force_boundary=compaction or bool(sess.get("user_round_boundary_pending", False)),
                metadata={"kind": "exec", "boundary": "live", "seq_end": sess["last_gen_seq"],
                          **({"user_round_id": sess["user_round_id"]} if sess.get("user_round_id") else {})})
            sess.pop("user_round_boundary_pending", None)
            sess["last_generation_messages"] = _copy_messages_for_snapshot(sess.get("messages_snapshot") or [])
            sess["last_generation_tools"] = deepcopy(sess.get("last_tools"))
            sess["last_generation_snapshot"] = None
            sess["record_pending"] = None
        sess["rolling_acc_ids"] = prompt_ids + output_ids + tail


    async def begin_user_round(self, session_id: str, round_id: int) -> None:
        """Mark a round boundary without resetting episode budgets or rebuilding actions."""
        lock = await self._get_session_state_lock(session_id)
        async with lock:
            sess = self._sessions[session_id]
            self._check_recorder_session_open(sess)
            rounds = sess.setdefault("user_rounds", [])
            if round_id != len(rounds) + 1:
                raise ValueError("User rounds must start at 1 and advance consecutively")
            if rounds:
                self._mark_user_boundary(sess, "user_round")
            sess["user_round_id"] = round_id
            sess["consecutive_no_tool"] = 0
            rounds.append({"round_id": round_id, "gen_tokens_before": int(sess.get("gen_tokens_total") or 0)})

    @staticmethod
    def _mark_user_boundary(sess, cause):
        current = sess["trajectory_recorder"].current
        if current is not None:
            current.metadata["boundary"] = cause
        sess["user_round_boundary_pending"] = True
        sess["record_pending"] = None
        sess["messages_snapshot"] = None
        sess["carried_over_keys"] = []

    async def begin_tool_handoff(self, session_id: str) -> None:
        lock = await self._get_session_state_lock(session_id)
        async with lock:
            sess = self._sessions[session_id]
            self._check_recorder_session_open(sess)
            if not sess.get("user_round_id"):
                raise ValueError("Tool handoff requires an active user round")
            self._mark_user_boundary(sess, "tool_handoff")

    async def _handle_chat_summ(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        """``/sess/{id}/summ/v1/chat/completions``: the harness's summarizer client."""
        return await self._handle_chat(request, summarizer=True)


    @staticmethod
    def _check_recorder_session_open(sess: dict) -> None:
        # A timeout may export/pop this same state while inference is in flight.
        # Late completions must neither mutate the exported batch nor recreate
        # a session that the harness has already closed.
        if sess.get("recorder_closed"):
            raise asyncio.CancelledError("Recorder session closed during generation")


    async def _compute_prompt_ids(self, sess, messages, tools, session_id):
        """Prepare engine input without overwriting previously sampled actions."""
        pending_cause = sess.pop("pending_rebuild_cause", None)
        parent_id = sess.get("budget_parent")
        if parent_id:
            parent = self._sessions.get(parent_id)
            if parent is None or parent.get("recorder_closed"):
                raise asyncio.CancelledError("Parent recorder session closed")
            self._check_compaction_budget(parent, parent_id, about_to_compact=False)
        self._check_compaction_budget(sess, session_id, about_to_compact=False)
        if self._processor is not None and _messages_have_multimodal(messages):
            sess["disable_proxy_trajectory"] = True
        snap = sess.get("messages_snapshot")
        previous_request = sess.get("last_generation_messages")
        successful_snap = sess.get("last_generation_snapshot")
        retry = (previous_request is not None
                 and _tools_equal(sess.get("last_generation_tools"), tools)
                 and _messages_is_prefix(previous_request, messages))
        suffix = messages[len(previous_request):] if retry else []
        retry = retry and (not suffix or (len(suffix) == 1
                 and suffix[0].get("role") == "assistant"
                 and (successful_snap is None or not _messages_is_prefix(successful_snap, messages))))
        sess["pending_retry"] = bool(retry and not pending_cause)
        changed = snap is not None and (retry or pending_cause or
                  not _messages_is_prefix(snap, messages) or not _tools_equal(sess.get("last_tools"), tools))
        if changed:
            profile = sess.get("compaction_profile") or {}
            n_summary = int(sess.get("summaries_since_compaction") or 0)
            summary_texts = ()
            if profile.get("mode") == "inplace" and profile.get("summary") == "summ_route" and n_summary:
                summary_texts = tuple(s["summary_text"] for s in
                    [s for s in sess.get("segments", []) if s.get("kind") == "summary"][-n_summary:]
                    if s.get("summary_text"))
            cause = pending_cause or ("retry" if retry else _classify_rebuild(
                snap, messages, pending_summary_texts=summary_texts,
                initial_prefix=(self._main_identity.get(session_id) or {}).get("initial")))
            is_compaction = cause in _COMPACTION_REBUILD_CAUSES
            if is_compaction:
                self._check_compaction_budget(sess, session_id, about_to_compact=True)
            # Commit boundary/counts only after a nonempty generation succeeds.
            # A failed request, including a failed retry, preserves the archive.
            sess["record_pending"] = {
                "cause": cause, "retry": bool(retry and not is_compaction),
                "snapshot": _copy_messages_for_snapshot(snap),
                "summary_text": sess.get("last_assistant_text"),
                "summary_tokens": int(sess.get("last_assistant_tokens") or 0),
                "native_summary_pending": bool(sess.get("native_summary_pending")),
            }
        if snap is None or changed or sess.get("disable_proxy_trajectory"):
            recorder = sess["trajectory_recorder"]
            if retry and not suffix and recorder.current is not None and recorder.current.generations:
                start = recorder.current.generations[-1][0]
                ids = recorder.current.prompt_ids + recorder.current.response_ids[:start]
            else:
                ids = list(await self._agent_loop.apply_chat_template(messages, tools=tools))
        else:
            delta = messages[len(snap):]
            delta_ids = list(await self._agent_loop.apply_chat_template(
                delta, tools=None, remove_system_prompt=True)) if delta else []
            ids = list(sess.get("rolling_acc_ids") or []) + delta_ids
        sess["rolling_acc_ids"] = list(ids)
        sess["messages_snapshot"] = _copy_messages_for_snapshot(messages)
        sess["message_snap_debug"] = _message_snap_debug_for(messages, "incoming_recorder")
        sess["last_tools"] = deepcopy(tools)
        return ids


    async def _finalize_generation(self, session_id: str, sess: dict, output: TokenOutput, prompt_len: int) -> None:
        self._check_recorder_session_open(sess)
        state_lock = await self._get_session_state_lock(session_id)
        async with state_lock:
            self._check_recorder_session_open(sess)
            s = self._sessions.setdefault(session_id, self._fresh_session_state())
            s.update(sess)
            s["num_calls"] += 1
            s["total_prompt_tokens"] += prompt_len
            s["total_completion_tokens"] += len(output.token_ids)
            self._gen_seq += 1
            s["last_gen_seq"] = self._gen_seq
            s["gen_tokens_total"] += len(output.token_ids)
            parent_id = s.get("budget_parent")
            if parent_id:
                parent_lock = await self._get_session_state_lock(parent_id)
                async with parent_lock:
                    parent = self._sessions.get(parent_id)
                    if parent is None or parent.get("recorder_closed"):
                        raise asyncio.CancelledError("Parent recorder session closed")
                    parent["gen_tokens_total"] += len(output.token_ids)
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
            prompt_ids = list(s.get("rolling_acc_ids") or [])
            eos_id = self._assistant_eos_token_id
            tail = (
                list(self._assistant_eos_tail)
                if (eos_id is not None and output_ids and output_ids[-1] == eos_id)
                else []
            )
            if not s.get("disable_proxy_trajectory"):
                if output.log_probs is not None and len(output.log_probs) != len(output_ids):
                    raise ValueError("Generated token ids and logprobs must have identical lengths")
                routes = _turn_routing_per_token(getattr(output, "routed_experts", None), len(output_ids))
                self._record_successful_generation(s, output, routes, mn, mx)
            s["rolling_acc_ids"] = prompt_ids + output_ids + tail
            if output_ids:
                s["last_generation_messages"] = _copy_messages_for_snapshot(
                    s.get("messages_snapshot") or []
                )
                s["last_generation_tools"] = deepcopy(s.get("last_tools"))
                s["last_generation_snapshot"] = None
            s["pending_retry"] = False


    @staticmethod
    def _translate_sampling_params(body: dict, defaults: dict | None = None) -> dict:
        sp: dict[str, Any] = {}
        # Explicit non-null request values win. Translating separately also
        # handles aliases such as max_completion_tokens overriding max_tokens.
        for source in (defaults or {}, body):
            for src_key, dst_key, conv in (
                ("temperature", "temperature", float),
                ("top_p", "top_p", float),
                ("top_k", "top_k", int),
                ("max_tokens", "max_tokens", int),
                ("max_completion_tokens", "max_tokens", int),
                ("seed", "seed", int),
            ):
                v = source.get(src_key)
                if v is None:
                    continue
                try:
                    sp[dst_key] = conv(v)
                except (TypeError, ValueError):
                    continue
            # This additional field belongs to the opt-in Codex path only.
            if defaults is not None and source.get("repetition_penalty") is not None:
                try:
                    sp["repetition_penalty"] = float(source["repetition_penalty"])
                except (TypeError, ValueError):
                    pass
            v = source.get("stop")
            if v is not None:
                sp["stop"] = v if isinstance(v, list) else [v]
        # Token-fidelity training requires generated-token log probabilities.
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
