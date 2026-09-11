from types import SimpleNamespace
from copy import deepcopy

import pytest
from omegaconf import OmegaConf

from verl_patch.agent_loop.builtin_swe_agent_loop import BuiltinSWEAgentLoop


CURRENT_LOOP_CONFIGS = (
    "src/verl_patch/config/agent_loop_config_oh.yaml",
    "src/verl_patch/config/agent_loop_config_cc.yaml",
    "src/verl_patch/config/agent_loop_config_oh_docker.yaml",
    "src/verl_patch/config/agent_loop_config_cc_docker.yaml",
    "src/verl_patch/config/agent_loop_config_mixed.yaml",
)
SHARED_LOOP_CONTROLS = {
    "tool_parser",
    "max_consecutive_no_tool",
    "trials_dir",
    "max_retries",
}


class FakeProxy:
    def __init__(self):
        self.opened = []
        self.popped = []

    def session_url(self, session_id):
        return f"http://proxy/sess/{session_id}/v1"

    def session_anthropic_base(self, session_id):
        return f"http://proxy/sess/{session_id}"

    async def open_session(self, session_id, trajectory_dir=None):
        self.opened.append((session_id, trajectory_dir))

    async def pop_session(self, session_id):
        self.popped.append(session_id)
        return {
            "traj_acc_ids": [11, 12, 21],
            "initial_prompt_token_len": 2,
            "traj_response_mask": [1],
            "traj_response_logprobs": [-0.2],
            "traj_response_routing": [],
            "messages_snapshot": [{"role": "assistant", "content": "ok"}],
            "num_calls": 1,
        }


class FakeTrial:
    configs = []
    results = []

    @classmethod
    async def create(cls, config):
        cls.configs.append(config)
        return cls()

    async def run(self):
        return self.results.pop(0)


class FakeTrialConfig:
    @staticmethod
    def model_validate(config):
        assert "harnesses" not in config
        assert "harness_selection" not in config
        # TrialConfig.model_validate returns an independent pydantic model;
        # copy here so the next retry's endpoint mutation cannot rewrite the
        # previous attempt in our assertion fixture.
        return deepcopy(config)


def result(*, verifier=True):
    return SimpleNamespace(
        exception_info=None,
        verifier_result=(
            SimpleNamespace(rewards={"reward": 1.0}) if verifier else None
        ),
        agent_result=None,
        environment_setup=None,
        agent_setup=None,
        agent_execution=None,
        verifier=None,
    )


def make_loop(tmp_path):
    loop = BuiltinSWEAgentLoop.__new__(BuiltinSWEAgentLoop)
    loop.harnesses = {
        "openhands_sdk": SimpleNamespace(
            name="openhands_sdk",
            protocol="openai",
            harbor_cfg={
                "agent": {
                    "import_path": "example:OpenHandsSDK",
                    "kwargs": {},
                    "env": {},
                },
                "environment": {"kwargs": {}},
            },
        ),
        "claude_code": SimpleNamespace(
            name="claude_code",
            protocol="anthropic",
            harbor_cfg={
                "agent": {"import_path": "example:ClaudeCode", "kwargs": {}, "env": {}},
                "environment": {"kwargs": {}},
            },
        ),
    }
    loop.base_trials_dir = str(tmp_path)
    loop.max_retries = 2
    loop.val_max_retries = None
    loop.val_pod_startup_timeout = None
    loop.val_pod_active_deadline = None
    loop.val_agent_max_timeout = None
    return loop


@pytest.mark.parametrize("config_path", CURRENT_LOOP_CONFIGS)
def test_loop_configs_keep_shared_controls_outside_harbor_configs(config_path):
    config = OmegaConf.load(config_path)[0]

    assert SHARED_LOOP_CONTROLS <= set(config)
    harbor_configs = (
        [config.harbor_cfg]
        if "harbor_cfg" in config
        else [definition.harbor_cfg for definition in config.harnesses.values()]
    )
    for harbor_cfg in harbor_configs:
        assert SHARED_LOOP_CONTROLS.isdisjoint(harbor_cfg)
        assert SHARED_LOOP_CONTROLS.isdisjoint(harbor_cfg.get("agent", {}))


def test_mixed_yaml_uses_one_set_of_shared_worker_controls(tmp_path, monkeypatch):
    from verl_patch.agent_loop import builtin_swe_agent_loop as module

    monkeypatch.setattr(module, "HARBOR_AVAILABLE", True)
    config = OmegaConf.load("src/verl_patch/config/agent_loop_config_mixed.yaml")[0]
    loop = BuiltinSWEAgentLoop.__new__(BuiltinSWEAgentLoop)
    loop.rollout_config = SimpleNamespace(prompt_length=100, response_length=200)

    class Base:
        def __init__(self, *args, **kwargs):
            pass

    # Exercise the concrete initializer while avoiding AgentLoopBase's tokenizer
    # setup; its fields are already supplied above for this config-boundary test.
    original = BuiltinSWEAgentLoop.__mro__[1].__init__
    monkeypatch.setattr(BuiltinSWEAgentLoop.__mro__[1], "__init__", Base.__init__)
    try:
        BuiltinSWEAgentLoop.__init__(
            loop,
            harnesses=config.harnesses,
            harness_selection=config.harness_selection,
            tool_parser=config.tool_parser,
            max_consecutive_no_tool=config.max_consecutive_no_tool,
            trials_dir=config.trials_dir,
            max_retries=config.max_retries,
        )
    finally:
        monkeypatch.setattr(BuiltinSWEAgentLoop.__mro__[1], "__init__", original)

    assert sorted(loop.harnesses) == [
        "claude_code",
        "opencode",
        "openhands_sdk",
    ]
    assert loop.harnesses["claude_code"].protocol == "anthropic"
    assert loop._tool_parser_name == config.tool_parser
    assert loop._max_consecutive_no_tool == config.max_consecutive_no_tool
    assert loop.base_trials_dir == config.trials_dir
    assert loop.max_retries == config.max_retries
    for definition in loop.harnesses.values():
        assert "tool_parser" not in definition.harbor_cfg["agent"]
        assert "max_consecutive_no_tool" not in definition.harbor_cfg["agent"]
        assert "trials_dir" not in definition.harbor_cfg["agent"]
        assert "max_retries" not in definition.harbor_cfg["agent"]


def test_initializer_requires_shared_controls_at_loop_level(monkeypatch):
    from verl_patch.agent_loop import builtin_swe_agent_loop as module

    monkeypatch.setattr(module, "HARBOR_AVAILABLE", True)
    loop = BuiltinSWEAgentLoop.__new__(BuiltinSWEAgentLoop)
    loop.rollout_config = SimpleNamespace(prompt_length=100, response_length=200)

    class Base:
        def __init__(self, *args, **kwargs):
            pass

    original = BuiltinSWEAgentLoop.__mro__[1].__init__
    monkeypatch.setattr(BuiltinSWEAgentLoop.__mro__[1], "__init__", Base.__init__)
    legacy_cfg = {
        "agent": {
            "import_path": "example:OpenHands",
            "tool_parser": "hermes",
            "max_consecutive_no_tool": 3,
            "trials_dir": "/tmp/legacy-trials",
            "max_retries": 2,
        }
    }
    try:
        with pytest.raises(ValueError, match="top-level loop controls"):
            BuiltinSWEAgentLoop.__init__(loop, harbor_cfg=legacy_cfg)
    finally:
        monkeypatch.setattr(BuiltinSWEAgentLoop.__mro__[1], "__init__", original)

    assert legacy_cfg["agent"]["tool_parser"] == "hermes"
    assert legacy_cfg["agent"]["max_consecutive_no_tool"] == 3
    assert legacy_cfg["agent"]["trials_dir"] == "/tmp/legacy-trials"
    assert legacy_cfg["agent"]["max_retries"] == 2


@pytest.mark.asyncio
async def test_retry_keeps_selected_harness_and_anthropic_base(tmp_path, monkeypatch):
    from verl_patch.agent_loop import builtin_swe_agent_loop as module

    FakeTrial.configs = []
    FakeTrial.results = [result(verifier=False), result(verifier=True)]
    monkeypatch.setattr(module, "Trial", FakeTrial)
    monkeypatch.setattr(module, "TrialConfig", FakeTrialConfig)
    loop = make_loop(tmp_path)
    proxy = FakeProxy()
    selection = SimpleNamespace(
        definition=loop.harnesses["claude_code"], source="metadata"
    )

    reward, reason, meta = await loop._run_harbor_trial(
        task_path="/tasks/example",
        global_steps=3,
        sampling_params={"temperature": 0.7},
        proxy=proxy,
        metrics={},
        harness_selection=selection,
    )

    assert reward == 1.0
    assert reason == "agent_completed"
    assert meta["num_calls"] == 1
    assert len(FakeTrial.configs) == 2
    assert all("ClaudeCode" in cfg["agent"]["import_path"] for cfg in FakeTrial.configs)
    assert all(
        cfg["agent"]["env"]["ANTHROPIC_BASE_URL"].endswith(cfg["trial_name"])
        for cfg in FakeTrial.configs
    )
    assert all(
        not cfg["agent"]["env"]["ANTHROPIC_BASE_URL"].endswith("/v1")
        for cfg in FakeTrial.configs
    )
    assert all(
        cfg["agent"]["extra_allowed_hosts"] == ["proxy"]
        for cfg in FakeTrial.configs
    )
    assert len({cfg["trial_name"] for cfg in FakeTrial.configs}) == 2
    assert all(cfg["trial_name"].startswith("cc-") for cfg in FakeTrial.configs)
    assert proxy.popped == [session_id for session_id, _ in proxy.opened]


@pytest.mark.asyncio
async def test_openai_trial_uses_chat_completions_base(tmp_path, monkeypatch):
    from verl_patch.agent_loop import builtin_swe_agent_loop as module

    FakeTrial.configs = []
    FakeTrial.results = [result(verifier=True)]
    monkeypatch.setattr(module, "Trial", FakeTrial)
    monkeypatch.setattr(module, "TrialConfig", FakeTrialConfig)
    loop = make_loop(tmp_path)
    proxy = FakeProxy()
    selection = SimpleNamespace(
        definition=loop.harnesses["openhands_sdk"], source="default"
    )

    await loop._run_harbor_trial(
        task_path="/tasks/example",
        global_steps=3,
        sampling_params={},
        proxy=proxy,
        metrics={},
        harness_selection=selection,
    )

    cfg = FakeTrial.configs[0]
    assert cfg["agent"]["env"]["LLM_BASE_URL"].endswith("/v1")
    assert cfg["agent"]["extra_allowed_hosts"] == ["proxy"]
    assert "ANTHROPIC_BASE_URL" not in cfg["agent"]["env"]
    assert cfg["trial_name"].startswith("ohsdk-")
