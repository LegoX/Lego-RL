---
name: rl-check
description: One-to-one Codex counterpart for Claude `/rl:check`. Preflight a Lego-RL train, eval, or infer config and answer whether it is safe to launch right now. Use when the user asks to check, preflight, validate, review launch readiness, list resolved run parameters, or asks `/rl:check config`.
---

# RL Check

Use this skill as the Codex entrypoint corresponding to Claude `/rl:check`.

Before taking task actions, read the canonical workflow in
`.claude/plugins/rl-plugin/skills/check/SKILL.md` completely and follow it as
the source of truth. Treat this file as a thin adapter for Codex skill
discovery, not a replacement for the detailed rules there.

## Invocation Mapping

- Claude: `/rl:check config`
- Codex: `$rl-check` or natural language such as "check this config",
  "preflight this run", "can I launch this run", or "validate the config".

## Codex Notes

Find the Lego-RL repo root first. It must contain
`scripts/lib/preflight.sh`, `scripts/lib/live_probe.sh`, and the `.claude`
plugin path above.

Follow the Claude skill's layering rule:

- scripts own deterministic behavior
- this skill owns orchestration, live-host judgement, and the final report

Remain read-only. Do not edit configs, launch workloads, kill processes, clear
shared memory, delete logs, SSH to other nodes, or mutate cluster state.
