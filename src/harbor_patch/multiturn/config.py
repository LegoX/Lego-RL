"""Only the outer trainer config owns multi-turn controls."""

from typing import Literal
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MultiTurnConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    task_profile: Literal["ccbench", "openswe", "swe_interact"] = "ccbench"
    workspace_dir: str | None = None

    @property
    def resolved_workspace(self) -> str:
        return self.workspace_dir or ("/testbed" if self.task_profile == "openswe" else "/app")

    # Includes the initial task: 5 rounds means at most 4 external follow-ups.
    max_rounds: int = Field(default=5, ge=1)
    stop_on_success: bool = True
    episode_timeout_sec: float = Field(default=7200, gt=0)
    simulator_model: str | None = None
    simulator_api_base: str | None = None
    # Comma-separated OpenAI-compatible bases for the conversation simulator.
    simulator_api_bases: str | None = None
    simulator_api_key_env: str = "MULTITURN_SIMULATOR_API_KEY"
    simulator_temperature: float = Field(default=0.2, ge=0, le=2)
    simulator_timeout_sec: float = Field(default=120, gt=0)
    simulator_max_retries: int = Field(default=2, ge=1)
    simulator_max_tokens: int | None = Field(default=None, gt=0)
    simulator_max_context_chars: int | None = Field(default=None, ge=1000)
    # Terminal mode preserves the earlier adapter; the training example opts in.
    reward_mode: Literal["terminal", "per_round"] = "terminal"
    judge_enabled: bool = False
    judge_model: str | None = None
    judge_api_base: str | None = None
    judge_api_key_env: str = "MULTITURN_SIMULATOR_API_KEY"
    judge_timeout_sec: float = Field(default=120, gt=0)
    judge_max_retries: int = Field(default=3, ge=1)
    judge_max_context_chars: int = Field(default=48000, ge=4000)
    test_weight: float = Field(default=0.5, ge=0, le=1)
    judge_weight: float = Field(default=0.5, ge=0, le=1)
    seed: int = 42
    # Scripted mode is for deterministic plumbing tests, not benchmark training.
    simulator_mode: Literal["llm", "scripted"] = "llm"
    scripted_followups: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_enabled(self):
        path = PurePosixPath(self.resolved_workspace)
        if (not path.is_absolute() or str(path) == "/" or ".." in path.parts
                or str(path) != self.resolved_workspace):
            raise ValueError("workspace_dir must be a normalized absolute task directory, not /")
        if self.enabled and self.max_rounds > 1:
            if self.simulator_mode == "llm" and not self.simulator_model:
                raise ValueError(
                    "multiturn.simulator_model is required for LLM user feedback"
                )
            if self.simulator_mode == "scripted" and not self.scripted_followups:
                raise ValueError("scripted mode requires scripted_followups")
        if self.simulator_api_bases:
            bases = [base.strip() for base in self.simulator_api_bases.split(",")]
            if (not all(base.startswith(("http://", "https://")) for base in bases)
                    or len(set(bases)) != len(bases)):
                raise ValueError("simulator_api_bases requires unique HTTP(S) URLs")
        if self.enabled and self.task_profile == "swe_interact":
            if self.reward_mode != "terminal":
                raise ValueError("SWE-Interact requires terminal verifier reward")
            if self.judge_enabled and (self.judge_weight > 0.10 or not self.judge_api_base):
                raise ValueError("SWE-Interact judge requires an API base and judge_weight <= 0.10")
        if self.enabled:
            if self.judge_enabled and not self.judge_model:
                raise ValueError("judge_model is required when judge_enabled=true")
            if self.reward_mode == "per_round" and not self.judge_enabled:
                raise ValueError("per_round rewards require judge_enabled=true")
            if abs(self.test_weight + self.judge_weight - 1.0) > 1e-8:
                raise ValueError("test_weight + judge_weight must equal 1")
        return self
