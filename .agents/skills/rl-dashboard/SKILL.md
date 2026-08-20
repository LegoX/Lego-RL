---
name: rl-dashboard
description: One-to-one Codex counterpart for Claude `/rl:dashboard`. Start or diagnose the Lego-RL webui dashboard on the current machine; choose log directories, port, frontend build state, existing instances, and optional Cloudflare quick tunnel after confirmation. Use when the user asks to start, deploy, expose, fix, or diagnose the dashboard/webui, missing curves, empty board, tunnel 502, or asks `/rl:dashboard [log-dir]`.
---

# RL Dashboard

Use this skill as the Codex entrypoint corresponding to Claude
`/rl:dashboard`.

Before taking task actions, read the canonical workflow in
`.claude/plugins/rl-plugin/skills/dashboard/SKILL.md` completely and follow it
as the source of truth.

## Invocation Mapping

- Claude: `/rl:dashboard [log-dir]`
- Codex: `$rl-dashboard` or natural language such as "start the dashboard",
  "deploy the webui", "bring the dashboard up", "the dashboard will not open",
  "tunnel 502", "why does this run have no curves", or "training curves not
  showing".

## Codex Notes

Find the Lego-RL repo root first. It must contain `webui/server.py`,
`scripts/lib/dashboard_probe.sh`, and the `.claude` plugin path above.

Never open a public tunnel, start a new background server, rebuild frontend
assets, or create symlinks without the confirmations required by the canonical
workflow. Do not kill existing servers or tunnels that this conversation did
not start.
