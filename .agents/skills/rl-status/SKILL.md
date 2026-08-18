---
name: rl-status
description: One-to-one Codex counterpart for Claude `/rl:status`. Diagnose a Lego-RL run that is live or recently finished; identify the run, locate the real log, report progress, inspect metrics, and match known failure signatures. Use when the user asks how a run is doing, what step it is on, whether reward is improving, whether a run is broken, or asks `/rl:status`.
---

# RL Status

Use this skill as the Codex entrypoint corresponding to Claude `/rl:status`.

Before taking task actions, read the canonical workflow in
`.claude/plugins/rl-plugin/skills/status/SKILL.md` completely and follow it as
the source of truth.

## Invocation Mapping

- Claude: `/rl:status`
- Codex: `$rl-status` or natural language such as "how's the run doing",
  "what step is it on", "is reward going up", "is this run broken", or
  "diagnose the training run".

## Codex Notes

Find the Lego-RL repo root first. It must contain
`scripts/lib/live_probe.sh` and the `.claude` plugin path above.

Remain read-only. Do not kill, restart, clean, rotate, edit configs, delete
logs, SSH to other nodes, or mutate Kubernetes. When evidence points off-box,
report the symptom and the node or layer to confirm rather than asserting a
cause you cannot observe locally.
