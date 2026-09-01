"""Backward-compatible Claude Code agent-loop symbol.

Claude Code now runs through :class:`BuiltinSWEAgentLoop`, including its
Anthropic Messages ingress and exact proxy-captured trajectory. Keep this
subclass temporarily so downstream YAML/imports do not break during migration.
"""

from .builtin_swe_agent_loop import BuiltinSWEAgentLoop


class BuiltinCCAgentLoop(BuiltinSWEAgentLoop):
    """Deprecated compatibility subclass with no behavioral overrides."""
