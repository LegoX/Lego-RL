import json
from copy import deepcopy
import pytest
from verl_patch.agent_loop.harness import normalize_harness_definitions, inject_harness_endpoint

def _agent_cfg_with_opencode_doc(served="vllm_model"):
    doc = {
        "$schema": "https://opencode.ai/config.json",
        "model": f"hosted_vllm/{served}",
        "enabled_providers": ["hosted_vllm"],
        "provider": {
            "hosted_vllm": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Hosted vLLM",
                "options": {"baseURL": "{env:HOSTED_VLLM_BASE_URL}", "apiKey": "{env:HOSTED_VLLM_API_KEY}"},
                "models": {served: {"name": served}},
            }
        },
        "agent": {"build": {"steps": 200}, "compaction": {"model": f"hosted_vllm/{served}"}},
    }
    return {"kwargs": {}, "env": {"OPENCODE_CONFIG_CONTENT": json.dumps(doc)}}


def test_compaction_cards_derive_each_harness_knob_from_the_budget():
    from verl_patch.agent_loop.harness import (
        COMPACTION_PROFILES,
        apply_compaction_profile,
        compaction_profile_for,
    )

    assert set(COMPACTION_PROFILES) == {"claude_code", "codex", "opencode", "openhands_sdk"}

    # Claude Code: one env var carrying C.
    cfg = {"kwargs": {}, "env": {}}
    applied = apply_compaction_profile(
        cfg, compaction_profile_for("claude_code"), budget=100000, temperature=1.0, served="vllm_model"
    )
    assert cfg["env"] == {"CLAUDE_CODE_AUTO_COMPACT_WINDOW": "100000"}
    assert applied == {"env:CLAUDE_CODE_AUTO_COMPACT_WINDOW": 100000}

    # Codex: the window plus a 0.9 C auto-compact limit, both read by codex.py.
    cfg = {"kwargs": {}, "env": {}}
    apply_compaction_profile(cfg, compaction_profile_for("codex"), budget=100000, temperature=1.0, served="vllm_model")
    assert cfg["env"] == {"CODEX_MODEL_CONTEXT_WINDOW": "100000", "CODEX_AUTO_COMPACT_TOKEN_LIMIT": "90000"}

    # OpenHands SDK: typed condenser kwargs, summarizer temperature = policy temperature.
    cfg = {"kwargs": {}, "env": {}}
    apply_compaction_profile(cfg, compaction_profile_for("openhands_sdk"), budget=100000, temperature=0.7, served="vllm_model")
    assert cfg["kwargs"] == {
        "condenser": "summarizer",
        "condenser_max_tokens": 100000,
        "condenser_max_size": 2147483647,
        "condenser_keep_first": 4,
        "condenser_llm_temperature": 0.7,
    }

    # OpenCode: the provider document is patched in place.
    cfg = _agent_cfg_with_opencode_doc()
    apply_compaction_profile(cfg, compaction_profile_for("opencode"), budget=100000, temperature=1.0, served="vllm_model")
    doc = json.loads(cfg["env"]["OPENCODE_CONFIG_CONTENT"])
    assert doc["provider"]["hosted_vllm"]["models"]["vllm_model"]["limit"] == {"context": 100000, "output": 16384}
    assert doc["provider"]["hosted_vllm"]["options"]["baseURL"] == "{env:HOSTED_VLLM_BASE_URL}"
    assert doc["enabled_providers"] == ["hosted_vllm", "hosted_vllm_summ"]
    summ = doc["provider"]["hosted_vllm_summ"]
    assert summ["options"]["baseURL"] == "{env:HOSTED_VLLM_SUMM_BASE_URL}"
    assert summ["models"]["vllm_model"]["limit"]["context"] == 100000
    assert doc["agent"]["compaction"]["model"] == "hosted_vllm_summ/vllm_model"

    # Compaction off (budget 0) and harnesses without a card are no-ops.
    cfg = {"kwargs": {}, "env": {}}
    assert apply_compaction_profile(cfg, compaction_profile_for("claude_code"), budget=0, temperature=1.0, served="m") == {}
    assert cfg == {"kwargs": {}, "env": {}}
    assert compaction_profile_for("openhands") is None
    assert compaction_profile_for("terminus_2") is None


def test_compaction_card_override_is_validated():
    from verl_patch.agent_loop.harness import compaction_profile_for

    card = compaction_profile_for(
        "claude_code",
        '{"mode": "replace", "summary": "on_thread", "markers": ["RESUMED"], "budget": {"env:X": "{budget:0.5}"}}',
    )
    assert card["markers"] == ["RESUMED"]
    assert card["budget"] == {"env:X": "{budget:0.5}"}
    with pytest.raises(ValueError, match="compaction.mode"):
        compaction_profile_for("x", {"mode": "sideways"})
    with pytest.raises(ValueError, match="compaction.summary"):
        compaction_profile_for("x", {"summary": "nowhere"})
    with pytest.raises(ValueError, match="JSON object"):
        compaction_profile_for("x", "{not json")


def test_definitions_carry_their_compaction_card():
    legacy = normalize_harness_definitions(
        harbor_cfg={"agent": {"import_path": "harbor_patch.agents.image_mounted_claude_code:ClaudeCode"}},
        harnesses=None,
    )
    assert legacy["claude_code"].compaction["mode"] == "replace"
    assert legacy["claude_code"].compaction["summary"] == "on_thread"

    mixed = normalize_harness_definitions(
        harbor_cfg=None,
        harnesses={
            "openhands_sdk": {
                "harbor_cfg": {"agent": {"import_path": "harbor.agents.custom.openhands_sdk:CustomOpenHandsSDK"}}
            },
            "opencode": {
                "harbor_cfg": {"agent": {"import_path": "harbor_patch.agents.image_mounted_opencode:OpenCode"}},
                "compaction": {"mode": "replace", "summary": "summ_route", "markers": ["<compacted>"]},
            },
            "claude_code": {
                "harbor_cfg": {"agent": {"import_path": "harbor_patch.agents.image_mounted_claude_code:ClaudeCode"}},
                "render_contract": {
                    "compaction": {"budget": {"env:CLAUDE_CODE_AUTO_COMPACT_WINDOW": "{budget:0.8}"}}
                },
            },
        },
    )
    assert mixed["openhands_sdk"].compaction["mode"] == "inplace"
    assert mixed["opencode"].compaction["markers"] == ["<compacted>"]
    assert mixed["opencode"].compaction["budget"] == {}  # an override replaces the whole card
    assert mixed["claude_code"].compaction["budget"] == {"env:CLAUDE_CODE_AUTO_COMPACT_WINDOW": "{budget:0.8}"}


def test_endpoint_injection_applies_the_card_when_a_budget_is_set(monkeypatch):
    monkeypatch.setenv("HARBOR_CONTEXT_BUDGET_TOKENS", "50000")
    monkeypatch.setenv("HARBOR_AGENT_TEMPERATURE", "1.0")
    definition = normalize_harness_definitions(
        harbor_cfg={"agent": {"import_path": "harbor.agents.custom.openhands_sdk:CustomOpenHandsSDK"}},
        harnesses=None,
    )["openhands_sdk"]

    cfg = deepcopy(definition.harbor_cfg)
    inject_harness_endpoint(
        cfg, definition, openai_base="http://proxy/sess/a/v1", anthropic_base="http://proxy/sess/a",
        openai_api_key="k", anthropic_api_key="k", summarizer_base="http://proxy/sess/a/summ/v1",
    )
    kw = cfg["agent"]["kwargs"]
    assert kw["condenser"] == "summarizer"
    assert kw["condenser_max_tokens"] == 50000
    assert kw["condenser_llm_temperature"] == 1.0
    assert kw["condenser_llm_base_url"] == "http://proxy/sess/a/summ/v1"

    # Compaction off: the card is not applied and no condenser is configured.
    monkeypatch.setenv("HARBOR_CONTEXT_BUDGET_TOKENS", "0")
    cfg = deepcopy(definition.harbor_cfg)
    inject_harness_endpoint(
        cfg, definition, openai_base="http://proxy/sess/a/v1", anthropic_base="http://proxy/sess/a",
        openai_api_key="k", anthropic_api_key="k",
    )
    assert "condenser" not in cfg["agent"]["kwargs"]


def test_condenser_uses_per_trial_validation_temperature(monkeypatch):
    monkeypatch.setenv('HARBOR_CONTEXT_BUDGET_TOKENS','96000')
    monkeypatch.setenv('HARBOR_AGENT_TEMPERATURE','1.0')
    definition=normalize_harness_definitions(
        harbor_cfg={'agent':{'import_path':'harbor.agents.custom.openhands_sdk:CustomOpenHandsSDK'}},harnesses=None)['openhands_sdk']
    cfg=deepcopy(definition.harbor_cfg)
    inject_harness_endpoint(cfg,definition,openai_base='http://proxy/sess/s/v1',
        anthropic_base='http://proxy/sess/s',openai_api_key='test',anthropic_api_key='test',
        summarizer_base='http://proxy/sess/s/summ/v1',temperature=.7)
    assert cfg['agent']['kwargs']['condenser_llm_temperature']==.7


def test_opencode_summary_provider_preserves_default_execution_provider():
    from verl_patch.agent_loop.harness import apply_compaction_profile, compaction_profile_for
    cfg={'env':{},'kwargs':{}}
    apply_compaction_profile(cfg,compaction_profile_for('opencode'),budget=96000,temperature=.7,served='policy')
    doc=json.loads(cfg['env']['OPENCODE_CONFIG_CONTENT'])
    assert set(doc['enabled_providers'])=={'hosted_vllm','hosted_vllm_summ'}
