"""Harness configuration and weighted harness selection.

This module intentionally has no Harbor or verl imports. Keeping harness
configuration normalization independent of trial creation makes it cheap to
validate mixed configurations before a trial is created. Task-level choices
are deterministic; trajectory-level choices are independent random draws
because fully-async workers do not receive a stable sibling-rollout identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

import numpy as np

HarnessProtocol = Literal["openai", "anthropic"]
HarnessSelectionSource = Literal[
    "metadata", "weighted_random", "default", "validation"
]


_KNOWN_HARBOR_AGENT_NAMES = {
    "aider",
    "claude-code",
    "cline-cli",
    "codex",
    "copilot-cli",
    "cursor-cli",
    "gemini-cli",
    "goose",
    "hermes",
    "kimi-cli",
    "mini-swe-agent",
    "nop",
    "opencode",
    "openhands",
    "openhands-sdk",
    "oracle",
    "pi",
    "qwen-coder",
    "rovodev-cli",
    "swe-agent",
    "terminus",
    "terminus-1",
    "terminus-2",
    "trae-agent",
}


@dataclass(frozen=True)
class HarnessDefinition:
    """One named Harbor configuration and its HTTP ingress protocol."""

    name: str
    protocol: HarnessProtocol
    harbor_cfg: dict[str, Any]
    render_contract: dict[str, Any] | None = None
    # Context-compaction profile ("card"), see COMPACTION_PROFILES. None = the
    # harness has no known compaction behaviour; the proxy still handles any
    # compaction it sees generically, but no harness knob is derived from the budget.
    compaction: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Context-compaction profiles: one card per harness (CompactionRL support).
#
# Every harness compacts differently. The card answers the four questions the
# training stack needs, so the rest of the pipeline stays harness-agnostic:
#
#   mode     what the history looks like after a compaction --
#            "replace": the whole history becomes [system, summary, ...];
#            "inplace": the opening prefix and the recent tail survive, the middle
#                       is replaced by a summary event.
#   summary  who samples the summary --
#            "on_thread":  the harness appends a "summarize" user turn to the main
#                          conversation; the summary is a normal archived turn;
#            "summ_route": a separate LLM client of the harness; it is pointed at
#                          the proxy's /sess/{id}/summ/v1 route so the summary is
#                          archived as a kind="summary" segment.
#   markers  fixed text the harness puts in the post-compaction opening turn;
#            an extra hint for the routing that this is the same conversation.
#   budget   how the single context budget C (HARBOR_CONTEXT_BUDGET_TOKENS)
#            reaches this harness's own compaction knob. Targets:
#              env:NAME              -> agent env var
#              kwargs:NAME           -> agent kwarg (typed)
#              opencode:limit.context / opencode:limit.output
#                                    -> the provider model limit in OPENCODE_CONFIG_CONTENT
#              opencode:compaction_provider=summ
#                                    -> a second provider on the summarizer route for
#                                       OpenCode's compaction agent
#            Values are templates: {budget}, {budget:0.9} (a fraction of C),
#            {temperature} (the policy sampling temperature), or a literal.
#
# A scaffold can override its card with a "compaction" object inside its render
# contract (HARBOR_HARNESS_RENDER_CONTRACT) or a mixed harness definition.
# ---------------------------------------------------------------------------
COMPACTION_PROFILES: dict[str, dict[str, Any]] = {
    "claude_code": {
        "mode": "replace",
        "summary": "on_thread",
        "markers": [
            "This session is being continued from a previous conversation that ran out of context",
        ],
        # Verify the env name against the pinned CC runtime (grep AUTO_COMPACT in
        # its cli.js) whenever HARBOR_AGENT_VERSION moves.
        "budget": {"env:CLAUDE_CODE_AUTO_COMPACT_WINDOW": "{budget}"},
    },
    "codex": {
        "mode": "replace",
        "summary": "on_thread",
        # Native checkpoint requests omit tools; archive their sampling under
        # that exact context independently of the unchanged execution history.
        "summary_request_prefix": "You are performing a CONTEXT CHECKPOINT COMPACTION.",
        # Observed in the pinned Codex 0.153.4 compacted user message.
        "markers": [
            "Another language model started to solve this problem and produced a summary of its thinking process.",
        ],
        # config.toml keys written by image_mounted_codex/codex.py from these env vars.
        "budget": {
            "env:CODEX_MODEL_CONTEXT_WINDOW": "{budget}",
            "env:CODEX_AUTO_COMPACT_TOKEN_LIMIT": "{budget:0.9}",
        },
    },
    "opencode": {
        "mode": "replace",
        "summary": "summ_route",
        "markers": [],
        # OpenCode auto-compacts only when the model declares limit.context.
        "budget": {
            "opencode:limit.context": "{budget}",
            "opencode:limit.output": "16384",
            "opencode:compaction_provider": "summ",
        },
    },
    "openhands_sdk": {
        "mode": "inplace",
        "summary": "summ_route",
        "markers": [],
        # LLMSummarizingCondenser: token trigger just under C, event-count trigger
        # as a safety net; the condenser LLM samples at the policy temperature.
        "budget": {
            "kwargs:condenser": "summarizer",
            "kwargs:condenser_max_tokens": "{budget}",
            "kwargs:condenser_max_size": "2147483647",
            "kwargs:condenser_keep_first": "4",
            "kwargs:condenser_llm_temperature": "{temperature}",
        },
    },
}

_COMPACTION_MODES = ("replace", "inplace")
_COMPACTION_SUMMARY_SOURCES = ("on_thread", "summ_route")


def compaction_profile_for(name: str, override: Any = None) -> dict[str, Any] | None:
    """The compaction card of harness ``name``: an explicit override (a mapping
    or JSON string, validated) or the built-in profile, None when unknown."""
    if override in (None, ""):
        profile = COMPACTION_PROFILES.get(name)
        return deepcopy(profile) if profile else None
    if isinstance(override, str):
        try:
            override = json.loads(override)
        except json.JSONDecodeError as exc:
            raise ValueError(f"compaction profile for {name!r} must be a JSON object") from exc
    if not isinstance(override, Mapping):
        raise ValueError(f"compaction profile for {name!r} must be a mapping")
    profile = deepcopy(dict(override))
    if profile.get("mode", "replace") not in _COMPACTION_MODES:
        raise ValueError(f"compaction.mode for {name!r} must be one of {_COMPACTION_MODES}")
    if profile.get("summary", "on_thread") not in _COMPACTION_SUMMARY_SOURCES:
        raise ValueError(
            f"compaction.summary for {name!r} must be one of {_COMPACTION_SUMMARY_SOURCES}"
        )
    profile.setdefault("mode", "replace")
    profile.setdefault("summary", "on_thread")
    profile.setdefault("markers", [])
    profile.setdefault("budget", {})
    if not isinstance(profile["markers"], list) or not isinstance(profile["budget"], Mapping):
        raise ValueError(f"compaction.markers must be a list and compaction.budget a mapping ({name!r})")
    return profile


_BUDGET_TEMPLATE_RE = re.compile(r"^\{budget(?::([0-9.]+))?\}$")


def _render_budget_value(template: Any, *, budget: int, temperature: Any) -> Any:
    """{budget} -> C, {budget:0.9} -> int(0.9*C), {temperature} -> the policy
    temperature; anything else is returned as-is (int-looking strings as int)."""
    if not isinstance(template, str):
        return template
    match = _BUDGET_TEMPLATE_RE.fullmatch(template.strip())
    if match:
        fraction = float(match.group(1)) if match.group(1) else 1.0
        return int(budget * fraction)
    if template.strip() == "{temperature}":
        return float(temperature)
    text = template.strip()
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    try:
        return float(text) if re.fullmatch(r"-?\d+\.\d+", text) else template
    except ValueError:  # pragma: no cover - defensive
        return template


def apply_compaction_profile(
    agent_cfg: dict[str, Any],
    profile: Mapping[str, Any] | None,
    *,
    budget: int,
    temperature: Any,
    served: str,
) -> dict[str, Any]:
    """Derive a harness's own compaction knobs from the context budget ``C``.

    Applies the card's ``budget`` mapping to a Harbor agent config in place (env
    vars, agent kwargs, or the OpenCode provider document) and returns what was
    set, keyed by target. A budget of 0 (compaction off) or no card is a no-op.
    """
    applied: dict[str, Any] = {}
    if not profile or budget <= 0:
        return applied
    env = agent_cfg.setdefault("env", {})
    kwargs = agent_cfg.setdefault("kwargs", {})
    opencode_config: dict[str, Any] | None = None
    for target, template in dict(profile.get("budget") or {}).items():
        value = _render_budget_value(template, budget=budget, temperature=temperature)
        kind, _, key = str(target).partition(":")
        if kind == "env":
            env[key] = str(value)
        elif kind == "kwargs":
            kwargs[key] = value
        elif kind == "opencode":
            if opencode_config is None:
                raw = env.get("OPENCODE_CONFIG_CONTENT")
                try:
                    opencode_config = json.loads(raw) if raw else {}
                except json.JSONDecodeError as exc:
                    raise ValueError("OPENCODE_CONFIG_CONTENT is not valid JSON") from exc
                if not isinstance(opencode_config, dict):
                    raise ValueError("OPENCODE_CONFIG_CONTENT must be a JSON object")
            _apply_opencode_budget(opencode_config, key, value, served=served)
        else:
            raise ValueError(f"Unknown compaction budget target {target!r}")
        applied[str(target)] = value
    if opencode_config is not None:
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(opencode_config, separators=(",", ":"))
    return applied


def _apply_opencode_budget(config: dict[str, Any], key: str, value: Any, *, served: str) -> None:
    """One ``opencode:*`` budget target on the OpenCode provider document."""
    providers = config.setdefault("provider", {})
    # The provider that serves the policy: the first one that lists the served
    # model, else the first provider, else a fresh "hosted_vllm" entry.
    main_name = next(
        (n for n, p in providers.items() if isinstance(p, dict) and served in (p.get("models") or {})),
        next(iter(providers), "hosted_vllm"),
    )
    main = providers.setdefault(main_name, {})
    model_entry = main.setdefault("models", {}).setdefault(served, {"name": served})
    if key in ("limit.context", "limit.output"):
        model_entry.setdefault("limit", {})[key.split(".", 1)[1]] = int(value)
        return
    if key == "compaction_provider":
        if str(value) != "summ":
            raise ValueError(f"opencode:compaction_provider supports only 'summ'; got {value!r}")
        summ_name = f"{main_name}_summ"
        options = dict(main.get("options") or {})
        # The summarizer route is injected per session as HOSTED_VLLM_SUMM_BASE_URL
        # (the main endpoint again in eval), resolved by OpenCode at start-up.
        options["baseURL"] = "{env:HOSTED_VLLM_SUMM_BASE_URL}"
        providers[summ_name] = {
            "npm": main.get("npm", "@ai-sdk/openai-compatible"),
            "name": f"{main.get('name', main_name)} (summarizer)",
            "options": options,
            "models": {served: deepcopy(model_entry)},
        }
        enabled = config.setdefault("enabled_providers", list(providers))
        if summ_name not in enabled:
            enabled.append(summ_name)
        config.setdefault("agent", {}).setdefault("compaction", {})["model"] = f"{summ_name}/{served}"
        return
    raise ValueError(f"Unknown opencode compaction target {key!r}")


def render_harness_contract(
    contract: Mapping[str, Any],
    *,
    openai_base: str,
    anthropic_base: str,
    openai_api_key: str,
    anthropic_api_key: str,
    served: str,
    temperature: Any,
    environment: Mapping[str, str] | None = None,
    summarizer_base: str | None = None,
) -> dict[str, Any]:
    """Render one declarative harness contract against a runtime endpoint.

    ``{llm_summ_base}`` is the OpenAI-style base URL a harness's compaction /
    condenser LLM client should use: the proxy's per-session ``/summ/v1`` route
    during training (its summaries are then archived as trainable segments), or
    simply ``openai_base`` again when no such route exists (eval against plain
    vLLM).
    """

    def without_v1(url: str) -> str:
        value = url.rstrip("/")
        return value[:-3] if value.endswith("/v1") else value

    values: dict[str, Any] = {
        "llm_base": openai_base,
        "llm_base_no_v1": without_v1(anthropic_base),
        "llm_summ_base": summarizer_base or openai_base,
        "endpoint": openai_base,
        "endpoint_no_v1": without_v1(anthropic_base),
        "llm_key": openai_api_key,
        "openai_key": openai_api_key,
        "anthropic_key": anthropic_api_key,
        "served": served,
        "temperature": temperature,
    }
    env_values = dict(environment or os.environ)
    token_pattern = re.compile(
        r"\{(?:env:([A-Za-z_][A-Za-z0-9_]*)|([A-Za-z_][A-Za-z0-9_]*))\}"
    )

    def render(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        match = token_pattern.fullmatch(value)
        if match:
            env_name, value_name = match.groups()
            if env_name:
                return env_values.get(env_name, "")
            if value_name not in values:
                raise ValueError(f"Unknown harness contract token: {value_name}")
            return values[value_name]

        def replace(match: re.Match[str]) -> str:
            env_name, value_name = match.groups()
            if env_name:
                return str(env_values.get(env_name, ""))
            if value_name not in values:
                raise ValueError(f"Unknown harness contract token: {value_name}")
            return str(values[value_name])

        return token_pattern.sub(replace, value)

    endpoint = render(contract.get("endpoint_template", "{llm_base}"))
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("Harness endpoint_template must render to a URL")
    endpoint_host = urlsplit(endpoint).hostname
    if not endpoint_host:
        raise ValueError(f"Harness endpoint has no hostname: {endpoint!r}")
    values["endpoint"] = endpoint
    values["endpoint_no_v1"] = without_v1(endpoint)

    def mapping(name: str) -> dict[str, Any]:
        value = contract.get(name, {}) or {}
        if not isinstance(value, Mapping):
            raise ValueError(f"Harness contract {name} must be a mapping")
        return {str(key): render(item) for key, item in value.items()}

    model_name = render(contract.get("model_template", "{served}"))
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("Harness model_template must render to a model name")
    return {
        "model_name": model_name,
        "kwargs": mapping("kwargs"),
        "env": {key: str(value) for key, value in mapping("env").items()},
        "endpoint_host": endpoint_host,
        "allow_endpoint_host": bool(contract.get("allow_endpoint_host", True)),
    }


@dataclass(frozen=True)
class HarnessSelection:
    definition: HarnessDefinition
    source: HarnessSelectionSource


def _plain_value(value: Any, *, field: str) -> Any:
    """Unwrap NumPy scalars and one-element object arrays from DataProto."""

    while isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(
                f"Harness metadata field {field!r} must be scalar; got array shape {value.shape}"
            )
        value = value.reshape(()).item()
    while isinstance(value, np.generic):
        value = value.item()
    return value


def _mapping_value(value: Any, *, field: str) -> Mapping[str, Any]:
    value = _plain_value(value, field=field)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(  # noqa: TRY004 - callers handle invalid sample/config values
            f"{field} must be a mapping; got {type(value).__name__}"
        )
    return value


def normalize_sample_mapping(value: Any, *, field: str) -> dict[str, Any]:
    """Return a plain dict for dataset mappings wrapped in NumPy containers."""

    return dict(_mapping_value(value, field=field))


def infer_harness_protocol(name: str, harbor_cfg: Mapping[str, Any]) -> HarnessProtocol:
    agent_cfg = _mapping_value(
        harbor_cfg.get("agent", {}), field=f"harnesses.{name}.harbor_cfg.agent"
    )
    descriptor = " ".join(
        str(value or "")
        for value in (name, agent_cfg.get("name"), agent_cfg.get("import_path"))
    ).lower()
    return "anthropic" if "claude" in descriptor else "openai"


def infer_implicit_harness_name(harbor_cfg: Mapping[str, Any]) -> str:
    agent_cfg = _mapping_value(harbor_cfg.get("agent", {}), field="harbor_cfg.agent")
    descriptor = " ".join(
        str(value or "")
        for value in (agent_cfg.get("name"), agent_cfg.get("import_path"))
    ).lower()
    if "claude" in descriptor:
        return "claude_code"
    if "codex" in descriptor:
        return "codex"
    if "opencode" in descriptor:
        return "opencode"
    if "openhands_sdk" in descriptor or "openhands-sdk" in descriptor:
        return "openhands_sdk"
    if "openhands" in descriptor:
        return "openhands"
    return "default"


def _validate_agent_factory_dispatch(name: str, harbor_cfg: Mapping[str, Any]) -> None:
    agent_cfg = _mapping_value(
        harbor_cfg.get("agent", {}), field=f"harnesses.{name}.harbor_cfg.agent"
    )
    agent_name = agent_cfg.get("name")
    import_path = agent_cfg.get("import_path")
    if (
        isinstance(agent_name, str)
        and agent_name in _KNOWN_HARBOR_AGENT_NAMES
        and import_path
    ):
        raise ValueError(
            f"Harness {name!r} configures recognized Harbor agent.name={agent_name!r} "
            f"together with import_path={import_path!r}; Harbor would ignore import_path. "
            "Remove agent.name to use the patched/custom agent."
        )


def normalize_harness_definitions(
    *,
    harbor_cfg: Mapping[str, Any] | None,
    harnesses: Mapping[str, Any] | None,
    render_contract: Mapping[str, Any] | None = None,
) -> dict[str, HarnessDefinition]:
    """Normalize mixed or legacy single-harness configuration."""

    def normalize_contract(value: Any, *, field: str) -> dict[str, Any] | None:
        value = _plain_value(value, field=field)
        if value in (None, ""):
            return None
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{field} must be a mapping or JSON object") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"{field} must be a mapping or JSON object")
        return deepcopy(dict(value))

    if harnesses is None:
        cfg = deepcopy(dict(harbor_cfg or {}))
        name = infer_implicit_harness_name(cfg)
        _validate_agent_factory_dispatch(name, cfg)
        contract = normalize_contract(render_contract, field="render_contract")
        return {
            name: HarnessDefinition(
                name=name,
                protocol=infer_harness_protocol(name, cfg),
                harbor_cfg=cfg,
                render_contract=contract,
                compaction=compaction_profile_for(
                    name, (contract or {}).get("compaction")
                ),
            )
        }

    harnesses = _mapping_value(harnesses, field="harnesses")
    if not harnesses:
        raise ValueError("harnesses must contain at least one harness definition")
    if harbor_cfg:
        raise ValueError(
            "Configure either legacy harbor_cfg or mixed harnesses, not both"
        )

    definitions: dict[str, HarnessDefinition] = {}
    for raw_name, raw_definition in harnesses.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(
                f"Harness name must be a non-empty string; got {raw_name!r}"
            )
        name = raw_name.strip()
        definition = _mapping_value(raw_definition, field=f"harnesses.{name}")
        unknown = set(definition) - {"protocol", "harbor_cfg", "render_contract", "compaction"}
        if unknown:
            raise ValueError(
                f"Harness {name!r} has unsupported control fields: {sorted(unknown)}"
            )
        cfg = deepcopy(
            dict(
                _mapping_value(
                    definition.get("harbor_cfg"), field=f"harnesses.{name}.harbor_cfg"
                )
            )
        )
        protocol = definition.get("protocol") or infer_harness_protocol(name, cfg)
        raw_contract = normalize_contract(
            definition.get("render_contract"),
            field=f"harnesses.{name}.render_contract",
        )
        if protocol not in ("openai", "anthropic"):
            raise ValueError(
                f"Harness {name!r} protocol must be 'openai' or 'anthropic'; got {protocol!r}"
            )
        _validate_agent_factory_dispatch(name, cfg)
        # The card: an explicit "compaction" control on the definition, else the
        # contract's, else the built-in profile for this harness name.
        compaction_override = definition.get("compaction")
        if compaction_override in (None, "") and raw_contract:
            compaction_override = raw_contract.get("compaction")
        definitions[name] = HarnessDefinition(
            name=name,
            protocol=protocol,
            harbor_cfg=cfg,
            render_contract=raw_contract,
            compaction=compaction_profile_for(name, _plain_value(compaction_override, field=f"harnesses.{name}.compaction")),
        )
    return definitions


class HarnessResolver:
    """Resolve exactly one harness from metadata or a weighted fallback."""

    def __init__(
        self,
        definitions: Mapping[str, HarnessDefinition],
        policy: Mapping[str, Any] | None = None,
    ) -> None:
        self.definitions = dict(definitions)
        policy = dict(policy or {})
        self.metadata_keys = tuple(
            policy.get("metadata_keys") or ("agent_harness", "harness")
        )
        if not self.metadata_keys or any(
            not isinstance(key, str) or not key for key in self.metadata_keys
        ):
            raise ValueError(
                "harness_selection.metadata_keys must contain non-empty strings"
            )

        self.default = policy.get("default")
        if self.default is None and len(self.definitions) == 1:
            self.default = next(iter(self.definitions))
        if self.default is not None and self.default not in self.definitions:
            raise ValueError(
                f"Unknown harness_selection.default={self.default!r}; "
                f"configured harnesses: {sorted(self.definitions)}"
            )

        # Validation in a mixed run should use one stable harness for every
        # validation sample.  ``val_harness`` is intentionally separate from
        # the training fallback policy: training may select per-task or
        # per-trajectory, while validation is a comparable fixed benchmark.
        # Keep the OpenHands SDK as the mixed-harness default. Configurations
        # without that harness must set ``val_harness`` explicitly if they also
        # require fixed validation selection.
        self.validation_harness = policy.get("val_harness")
        if self.validation_harness is None:
            self.validation_harness = policy.get("validation_harness")
        if self.validation_harness is None and "openhands_sdk" in self.definitions:
            self.validation_harness = "openhands_sdk"
        if self.validation_harness is not None:
            if (
                not isinstance(self.validation_harness, str)
                or not self.validation_harness.strip()
            ):
                raise ValueError(
                    "harness_selection.val_harness must be a non-empty string"
                )
            self.validation_harness = self.validation_harness.strip()
            if self.validation_harness not in self.definitions:
                raise ValueError(
                    "Unknown harness_selection.val_harness="
                    f"{self.validation_harness!r}; "
                    f"configured harnesses: {sorted(self.definitions)}"
                )

        self.fallback = policy.get("fallback", "default")
        if self.fallback not in ("default", "weighted_random"):
            raise ValueError(
                "harness_selection.fallback must be 'default' or 'weighted_random'"
            )
        self.random_seed = int(policy.get("random_seed", 0))
        self.granularity = policy.get("granularity", "task")
        if self.granularity not in ("task", "trajectory"):
            raise ValueError(
                "harness_selection.granularity must be 'task' or 'trajectory'"
            )

        raw_weights = dict(policy.get("weights") or {})
        unknown_weights = set(raw_weights) - set(self.definitions)
        if unknown_weights:
            raise ValueError(
                f"harness_selection.weights contains unknown harnesses: {sorted(unknown_weights)}"
            )
        self.weights: tuple[tuple[str, float], ...] = tuple(
            (name, float(raw_weights.get(name, 0.0)))
            for name in sorted(self.definitions)
        )
        if any(not math.isfinite(weight) or weight < 0 for _, weight in self.weights):
            raise ValueError(
                "harness_selection.weights must be finite and non-negative"
            )
        if (
            self.fallback == "weighted_random"
            and sum(weight for _, weight in self.weights) <= 0
        ):
            raise ValueError(
                "weighted_random fallback requires at least one positive weight"
            )
        if self.fallback == "default" and self.default is None:
            raise ValueError(
                "Multiple harnesses require harness_selection.default or weighted_random fallback"
            )

    def resolve(
        self, kwargs: Mapping[str, Any], *, is_val: bool | None = None
    ) -> HarnessSelection:
        if is_val is None:
            is_val = bool(kwargs.get("validate", False))

        extra_info = _mapping_value(kwargs.get("extra_info", {}), field="extra_info")

        # Metadata-key order is authoritative. For each key, a directly
        # forwarded dataset column takes precedence over the nested value.
        # With the default keys this yields:
        # direct agent_harness > extra_info.agent_harness > direct harness >
        # extra_info.harness.
        metadata_selection: HarnessDefinition | None = None
        for key in self.metadata_keys:
            if key in kwargs:
                metadata_selection = self._explicit(key, kwargs[key])
                break
            if key in extra_info:
                metadata_selection = self._explicit(
                    f"extra_info.{key}", extra_info[key]
                )
                break

        if is_val and self.validation_harness is not None:
            return HarnessSelection(
                self.definitions[self.validation_harness], "validation"
            )
        if metadata_selection is not None:
            return HarnessSelection(metadata_selection, "metadata")

        if self.fallback == "weighted_random":
            return HarnessSelection(
                self._weighted_choice(kwargs, extra_info), "weighted_random"
            )
        assert self.default is not None
        return HarnessSelection(self.definitions[self.default], "default")

    def _explicit(self, field: str, value: Any) -> HarnessDefinition:
        value = _plain_value(value, field=field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Harness metadata field {field!r} must be a non-empty string; got {value!r}"
            )
        name = value.strip()
        if name not in self.definitions:
            raise ValueError(
                f"Unknown harness {name!r} in {field}; configured harnesses: "
                f"{sorted(self.definitions)}"
            )
        return self.definitions[name]

    def _weighted_choice(
        self,
        kwargs: Mapping[str, Any],
        extra_info: Mapping[str, Any],
    ) -> HarnessDefinition:
        if self.granularity == "trajectory":
            # Fully-async dispatch repeats a task and then sends each trajectory
            # to a worker as a singleton batch. The worker consequently observes
            # rollout_n=0 for every sibling, so it is not a usable trajectory ID.
            point = secrets.randbits(256) / float(1 << 256)
        else:
            task_identity = kwargs.get("index")
            if task_identity is None:
                trajectory_info = _mapping_value(
                    kwargs.get("trajectory_info", {}), field="trajectory_info"
                )
                task_identity = trajectory_info.get("sample_index")
            if task_identity is None:
                task_identity = extra_info.get("index")
            if task_identity is None:
                task_identity = extra_info.get("instance_id")
            if task_identity is None:
                task_identity = extra_info.get("harbor_task_path") or extra_info.get(
                    "task_path"
                )
            if task_identity is None:
                task_identity = kwargs.get("raw_prompt")
            task_identity = _plain_value(task_identity, field="task_identity")

            stable_input = {
                "seed": self.random_seed,
                "global_step": int(kwargs.get("global_steps", 0) or 0),
                "task": task_identity,
            }
            encoded = json.dumps(
                stable_input, sort_keys=True, separators=(",", ":"), default=str
            ).encode("utf-8")
            point = int.from_bytes(hashlib.sha256(encoded).digest(), "big") / float(
                1 << 256
            )
        total = sum(weight for _, weight in self.weights)
        threshold = point * total
        cumulative = 0.0
        last_positive: str | None = None
        for name, weight in self.weights:
            if weight <= 0:
                continue
            last_positive = name
            cumulative += weight
            if threshold < cumulative:
                return self.definitions[name]
        assert last_positive is not None
        return self.definitions[last_positive]


def _inject_compaction(agent_cfg: dict[str, Any], definition: HarnessDefinition, summ_base: str, temperature: float | None = None) -> None:
    """Context compaction for one trial: derive the harness's own compaction knobs
    from the budget (its card, see COMPACTION_PROFILES) and point its separate
    summarizer LLM client, if any, at the session's summarizer route.

    * OpenHands SDK: ``agent.kwargs.condenser`` (e.g. "summarizer") builds a
      dedicated condenser LLM; its ``condenser_llm_base_url`` selects the route.
    * OpenCode: the provider document references ``{env:HOSTED_VLLM_SUMM_BASE_URL}``
      for the compaction agent's provider; the env var carries the route.
    """
    budget = int(os.environ.get("HARBOR_CONTEXT_BUDGET_TOKENS", "0") or 0)
    apply_compaction_profile(
        agent_cfg,
        getattr(definition, "compaction", None),
        budget=budget,
        temperature=float(os.environ.get("HARBOR_AGENT_TEMPERATURE", "1.0")) if temperature is None else temperature,
        served=os.environ.get("SERVED_MODEL_NAME", "vllm_model"),
    )
    kwargs = agent_cfg.setdefault("kwargs", {})
    if kwargs.get("condenser") and not kwargs.get("condenser_llm_base_url"):
        kwargs["condenser_llm_base_url"] = summ_base
    env = agent_cfg.setdefault("env", {})
    env.setdefault("HOSTED_VLLM_SUMM_BASE_URL", summ_base)


def inject_harness_endpoint(
    harbor_cfg: dict[str, Any],
    definition: HarnessDefinition,
    *,
    openai_base: str,
    anthropic_base: str,
    openai_api_key: str,
    anthropic_api_key: str,
    summarizer_base: str | None = None,
    temperature: float | None = None,
) -> None:
    """Inject the selected session endpoint into a copied Harbor config.

    ``summarizer_base`` is the session's ``/summ/v1`` proxy route (context
    compaction). When the harness runs a separate summarizer LLM client -- the
    OpenHands SDK condenser (``agent.kwargs.condenser``) or OpenCode's compaction
    agent (``HOSTED_VLLM_SUMM_BASE_URL`` provider) -- it is pointed there so the
    summaries it samples are archived as ``kind="summary"`` segments of the
    trial. None falls back to the main endpoint (nothing archived).
    """

    agent_cfg = harbor_cfg.setdefault("agent", {})
    if definition.name == "openhands_sdk" and os.environ.get("HARBOR_OPENHANDS_ENABLE_SUBAGENT", "0") == "1":
        agent_cfg.setdefault("kwargs", {})["enable_subagent"] = True
    endpoint_base = (
        anthropic_base if definition.protocol == "anthropic" else openai_base
    )
    summ_base = summarizer_base or openai_base
    endpoint_host = urlsplit(endpoint_base).hostname
    if not endpoint_host:
        raise ValueError(f"Harness endpoint has no hostname: {endpoint_base!r}")
    # Agent pods switch to an egress allowlist before inference, so preserve
    # access to the exact per-worker proxy host without opening its whole subnet.
    extra_allowed_hosts = agent_cfg.setdefault("extra_allowed_hosts", [])
    if not isinstance(extra_allowed_hosts, list):
        raise ValueError("agent.extra_allowed_hosts must be a list")
    contract = definition.render_contract
    if contract:
        rendered = render_harness_contract(
            contract,
            openai_base=openai_base,
            anthropic_base=anthropic_base,
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            served=os.environ.get("SERVED_MODEL_NAME", "vllm_model"),
            temperature=float(os.environ.get("HARBOR_AGENT_TEMPERATURE", "1.0")) if temperature is None else temperature,
            environment=os.environ,
            summarizer_base=summ_base,
        )
        agent_cfg["model_name"] = rendered["model_name"]
        agent_cfg.setdefault("kwargs", {}).update(rendered["kwargs"])
        agent_cfg.setdefault("env", {}).update(rendered["env"])
        _inject_compaction(agent_cfg, definition, summ_base, temperature)
        if rendered["allow_endpoint_host"] and endpoint_host not in extra_allowed_hosts:
            extra_allowed_hosts.append(endpoint_host)
        return

    if endpoint_host not in extra_allowed_hosts:
        extra_allowed_hosts.append(endpoint_host)

    agent_cfg.setdefault("kwargs", {})["api_base"] = openai_base
    agent_env = agent_cfg.setdefault("env", {})
    _inject_compaction(agent_cfg, definition, summ_base, temperature)

    if definition.protocol == "anthropic":
        # Claude Code appends /v1/messages itself.  anthropic_base therefore
        # intentionally ends at /sess/{session_id}, unlike openai_base.
        agent_env["ANTHROPIC_BASE_URL"] = anthropic_base.rstrip("/")
        agent_env.setdefault("ANTHROPIC_API_KEY", anthropic_api_key)
        model_name = str(agent_cfg.get("model_name") or "")
        if "/" in model_name:
            agent_cfg["model_name"] = model_name.rsplit("/", 1)[-1]
        return

    agent_env["LLM_BASE_URL"] = openai_base
    agent_env.setdefault("LLM_API_KEY", openai_api_key)
    descriptor = f"{definition.name} {agent_cfg.get('name', '')} {agent_cfg.get('import_path', '')}".lower()
    if "opencode" in descriptor:
        agent_env["HOSTED_VLLM_BASE_URL"] = openai_base
        agent_env.setdefault("HOSTED_VLLM_API_KEY", openai_api_key)
    if "codex" in descriptor:
        # Codex CLI speaks ONLY the Responses API and appends "/responses" to
        # OPENAI_BASE_URL, so the per-session ".../sess/{id}/v1" is exactly the
        # proxy route registered for it. Codex ignores the LLM_BASE_URL set
        # above, so without this branch it keeps cx.env's inert placeholder
        # (http://in-process-proxy/v1), which does not resolve inside the
        # sandbox pod: every trial then dies in agent setup. This replaces the
        # hook that lived in builtin_cc_agent_loop.py:636 before the refactor.
        agent_env["OPENAI_BASE_URL"] = openai_base
        agent_env.setdefault("OPENAI_API_KEY", openai_api_key)
