import json
import os
import subprocess
import sys
from copy import deepcopy

import numpy as np
import pytest

from verl_patch.agent_loop.harness import (
    HarnessDefinition,
    HarnessResolver,
    inject_harness_endpoint,
    normalize_harness_definitions,
)


@pytest.fixture
def definitions():
    return {
        "openhands_sdk": HarnessDefinition(
            "openhands_sdk",
            "openai",
            {"agent": {"import_path": "example:OpenHandsSDK"}},
        ),
        "claude_code": HarnessDefinition(
            "claude_code",
            "anthropic",
            {"agent": {"import_path": "example:ClaudeCode"}},
        ),
    }


def test_metadata_precedence_and_numpy_values(definitions):
    resolver = HarnessResolver(
        definitions,
        {"default": "openhands_sdk", "fallback": "default"},
    )
    selected = resolver.resolve(
        {
            "agent_harness": np.array("claude_code", dtype=object),
            "extra_info": {
                "agent_harness": "openhands_sdk",
                "harness": "openhands_sdk",
            },
        }
    )
    assert selected.definition.name == "claude_code"
    assert selected.source == "metadata"

    nested = resolver.resolve(
        {
            "extra_info": {
                "agent_harness": np.str_("claude_code"),
                "harness": "openhands_sdk",
            }
        }
    )
    assert nested.definition.name == "claude_code"

    nested_primary_beats_direct_alias = resolver.resolve(
        {
            "harness": "openhands_sdk",
            "extra_info": {"agent_harness": "claude_code"},
        }
    )
    assert nested_primary_beats_direct_alias.definition.name == "claude_code"

    wrapped_extra_info = resolver.resolve(
        {"extra_info": np.array({"agent_harness": "claude_code"}, dtype=object)}
    )
    assert wrapped_extra_info.definition.name == "claude_code"


@pytest.mark.parametrize(
    "value", ["unknown", "", np.array(["claude_code", "openhands_sdk"])]
)
def test_malformed_explicit_metadata_is_rejected(definitions, value):
    resolver = HarnessResolver(definitions, {"default": "openhands_sdk"})
    with pytest.raises(ValueError, match="Harness|harness"):
        resolver.resolve({"extra_info": {"agent_harness": value}})


def test_task_granularity_is_stable_across_rollouts(definitions):
    resolver = HarnessResolver(
        definitions,
        {
            "fallback": "weighted_random",
            "random_seed": 42,
            "granularity": "task",
            "weights": {"openhands_sdk": 1, "claude_code": 1},
        },
    )
    names = {
        resolver.resolve(
            {"index": 17, "global_steps": 9, "rollout_n": rollout_n, "extra_info": {}}
        ).definition.name
        for rollout_n in range(8)
    }
    assert len(names) == 1


def test_weighted_choice_is_stable_across_processes():
    code = r"""
import json
from verl_patch.agent_loop.harness import HarnessDefinition, HarnessResolver
d = {
    "a": HarnessDefinition("a", "openai", {"agent": {"import_path": "x:A"}}),
    "b": HarnessDefinition("b", "anthropic", {"agent": {"import_path": "x:B"}}),
}
r = HarnessResolver(d, {"fallback": "weighted_random", "random_seed": 91,
                        "granularity": "task", "weights": {"a": 1, "b": 3}})
print(json.dumps([r.resolve({"index": i, "global_steps": 2}).definition.name
                  for i in range(12)]))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(os.path.abspath("src")), env.get("PYTHONPATH", "")]
    )
    first = subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
    second = subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
    assert json.loads(first) == json.loads(second)


def test_weighted_distribution_uses_all_positive_harnesses(definitions):
    resolver = HarnessResolver(
        definitions,
        {
            "fallback": "weighted_random",
            "random_seed": 7,
            "weights": {"openhands_sdk": 1, "claude_code": 3},
        },
    )
    choices = [
        resolver.resolve({"index": index, "global_steps": 1}).definition.name
        for index in range(1000)
    ]
    cc_ratio = choices.count("claude_code") / len(choices)
    assert 0.68 < cc_ratio < 0.82


def test_trajectory_granularity_draws_without_rollout_identity(definitions, monkeypatch):
    resolver = HarnessResolver(
        definitions,
        {
            "fallback": "weighted_random",
            "granularity": "trajectory",
            "weights": {"openhands_sdk": 1, "claude_code": 1},
        },
    )
    points = iter((0, (1 << 256) - 1))
    monkeypatch.setattr(
        "verl_patch.agent_loop.harness.secrets.randbits", lambda _: next(points)
    )

    first = resolver.resolve({"index": 1})
    second = resolver.resolve({"index": 1})

    assert first.definition.name == "claude_code"
    assert second.definition.name == "openhands_sdk"


def test_endpoint_injection_preserves_protocol_paths(definitions):
    openai_cfg = deepcopy(definitions["openhands_sdk"].harbor_cfg)
    inject_harness_endpoint(
        openai_cfg,
        definitions["openhands_sdk"],
        openai_base="http://proxy/sess/a/v1",
        anthropic_base="http://proxy/sess/a",
        openai_api_key="openai-key",
        anthropic_api_key="anthropic-key",
    )
    assert openai_cfg["agent"]["kwargs"]["api_base"].endswith("/v1")
    assert openai_cfg["agent"]["env"]["LLM_BASE_URL"].endswith("/v1")
    assert openai_cfg["agent"]["extra_allowed_hosts"] == ["proxy"]
    assert "ANTHROPIC_BASE_URL" not in openai_cfg["agent"]["env"]

    anthropic_cfg = deepcopy(definitions["claude_code"].harbor_cfg)
    inject_harness_endpoint(
        anthropic_cfg,
        definitions["claude_code"],
        openai_base="http://proxy/sess/b/v1",
        anthropic_base="http://proxy/sess/b",
        openai_api_key="openai-key",
        anthropic_api_key="anthropic-key",
    )
    assert anthropic_cfg["agent"]["env"]["ANTHROPIC_BASE_URL"] == "http://proxy/sess/b"
    assert anthropic_cfg["agent"]["extra_allowed_hosts"] == ["proxy"]
    assert not anthropic_cfg["agent"]["env"]["ANTHROPIC_BASE_URL"].endswith("/v1")


def test_endpoint_host_is_added_to_existing_allowlist_once(definitions):
    cfg = deepcopy(definitions["openhands_sdk"].harbor_cfg)
    cfg["agent"]["extra_allowed_hosts"] = ["registry.internal"]

    for _ in range(2):
        inject_harness_endpoint(
            cfg,
            definitions["openhands_sdk"],
            openai_base="http://22.54.171.134:41797/sess/a/v1",
            anthropic_base="http://22.54.171.134:41797/sess/a",
            openai_api_key="openai-key",
            anthropic_api_key="anthropic-key",
        )

    assert cfg["agent"]["extra_allowed_hosts"] == [
        "registry.internal",
        "22.54.171.134",
    ]


def test_opencode_gets_compatible_provider_endpoint():
    definition = HarnessDefinition(
        "opencode", "openai", {"agent": {"import_path": "example:OpenCode"}}
    )
    cfg = deepcopy(definition.harbor_cfg)
    inject_harness_endpoint(
        cfg,
        definition,
        openai_base="http://proxy/sess/oc/v1",
        anthropic_base="http://proxy/sess/oc",
        openai_api_key="key",
        anthropic_api_key="key",
    )
    assert cfg["agent"]["env"]["HOSTED_VLLM_BASE_URL"].endswith("/v1")


def test_legacy_config_normalizes_to_one_implicit_harness():
    definitions = normalize_harness_definitions(
        harbor_cfg={"agent": {"import_path": "pkg:ClaudeCode"}},
        harnesses=None,
    )
    assert list(definitions) == ["claude_code"]
    assert definitions["claude_code"].protocol == "anthropic"


def test_recognized_name_cannot_silently_override_custom_import():
    with pytest.raises(ValueError, match="ignore import_path"):
        normalize_harness_definitions(
            harbor_cfg=None,
            harnesses={
                "opencode": {
                    "harbor_cfg": {
                        "agent": {"name": "opencode", "import_path": "patched:OpenCode"}
                    }
                }
            },
        )
