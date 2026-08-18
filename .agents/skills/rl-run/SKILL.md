---
name: rl-run
description: One-to-one Codex counterpart for Claude `/rl:run`. Preflight and launch a Lego-RL train, eval, or infer run only after blocking checks pass, resolved parameters are shown, and the user explicitly confirms. Use when the user asks to launch, run, start, kick off training/eval/infer, get a config running, or asks `/rl:run config`.
---

# RL Run

Use this skill as the Codex entrypoint corresponding to Claude `/rl:run`.

Before taking task actions, read the canonical workflow in
`.claude/plugins/rl-plugin/skills/run/SKILL.md` completely and follow it as the
source of truth. That workflow depends on the `/rl:check` behavior; if needed,
also read `.claude/plugins/rl-plugin/skills/check/SKILL.md`.

## Invocation Mapping

- Claude: `/rl:run config`
- Codex: `$rl-run` or natural language such as "launch this config",
  "run harbor training", "start the eval", "kick off training", or "get this
  config running".

## Codex Notes

Find the Lego-RL repo root first. It must contain
`scripts/train/train.sh`, `scripts/eval/eval.sh`, `scripts/infer/infer.sh`, and
the `.claude` plugin path above.

Never launch without an explicit yes in the current conversation. Default to
background launch when the user approves. For multi-node runs, print the
per-node command; do not SSH to other nodes.

Do not force past a blocking preflight, kill a live run, edit configs to make a
launch pass, delete logs/checkpoints, or mutate the cluster.
