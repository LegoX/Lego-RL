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
import secrets
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

HarnessProtocol = Literal["openai", "anthropic"]
HarnessSelectionSource = Literal["metadata", "weighted_random", "default"]


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
) -> dict[str, HarnessDefinition]:
    """Normalize mixed or legacy single-harness configuration."""

    if harnesses is None:
        cfg = deepcopy(dict(harbor_cfg or {}))
        name = infer_implicit_harness_name(cfg)
        _validate_agent_factory_dispatch(name, cfg)
        return {
            name: HarnessDefinition(
                name=name,
                protocol=infer_harness_protocol(name, cfg),
                harbor_cfg=cfg,
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
        unknown = set(definition) - {"protocol", "harbor_cfg"}
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
        if protocol not in ("openai", "anthropic"):
            raise ValueError(
                f"Harness {name!r} protocol must be 'openai' or 'anthropic'; got {protocol!r}"
            )
        _validate_agent_factory_dispatch(name, cfg)
        definitions[name] = HarnessDefinition(
            name=name, protocol=protocol, harbor_cfg=cfg
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

    def resolve(self, kwargs: Mapping[str, Any]) -> HarnessSelection:
        extra_info = _mapping_value(kwargs.get("extra_info", {}), field="extra_info")

        # Metadata-key order is authoritative. For each key, a directly
        # forwarded dataset column takes precedence over the nested value.
        # With the default keys this yields:
        # direct agent_harness > extra_info.agent_harness > direct harness >
        # extra_info.harness.
        for key in self.metadata_keys:
            if key in kwargs:
                return HarnessSelection(self._explicit(key, kwargs[key]), "metadata")
            if key in extra_info:
                return HarnessSelection(
                    self._explicit(f"extra_info.{key}", extra_info[key]), "metadata"
                )

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


def inject_harness_endpoint(
    harbor_cfg: dict[str, Any],
    definition: HarnessDefinition,
    *,
    openai_base: str,
    anthropic_base: str,
    openai_api_key: str,
    anthropic_api_key: str,
) -> None:
    """Inject the selected session endpoint into a copied Harbor config."""

    agent_cfg = harbor_cfg.setdefault("agent", {})
    agent_cfg.setdefault("kwargs", {})["api_base"] = openai_base
    agent_env = agent_cfg.setdefault("env", {})

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
