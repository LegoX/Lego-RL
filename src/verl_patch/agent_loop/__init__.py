"""Lego-RL agent-loop exports, loaded lazily for lightweight helper tests."""

from typing import Any

__all__ = ["BuiltinCCAgentLoop", "BuiltinSWEAgentLoop"]


def __getattr__(name: str) -> Any:
    if name == "BuiltinSWEAgentLoop":
        from .builtin_swe_agent_loop import BuiltinSWEAgentLoop

        return BuiltinSWEAgentLoop
    if name == "BuiltinCCAgentLoop":
        from .builtin_cc_agent_loop import BuiltinCCAgentLoop

        return BuiltinCCAgentLoop
    raise AttributeError(name)
