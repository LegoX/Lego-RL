---
name: rl-k8s-sandbox-install
description: One-to-one Codex counterpart for Claude `/rl:k8s-sandbox-install`. Guided install or scale-out of a sandbox Kubernetes cluster for the Lego-RL k8s backend, using probes before decisions and confirmations for risky mutations. Use when the user asks to install Kubernetes, set up a cluster, add or join worker nodes, create a sandbox cluster, prepare kubeadm/containerd/flannel/ImageVolume, or asks `/rl:k8s-sandbox-install`.
---

# RL K8s Sandbox Install

Use this skill as the Codex entrypoint corresponding to Claude
`/rl:k8s-sandbox-install`.

Before taking task actions, read the canonical workflow in
`.claude/plugins/rl-plugin/skills/k8s-sandbox-install/SKILL.md` completely and
follow it as the source of truth.

## Invocation Mapping

- Claude: `/rl:k8s-sandbox-install`
- Codex: `$rl-k8s-sandbox-install` or natural language such as "install
  kubernetes", "set up a cluster", "add a worker node", "join worker",
  "new cluster", or "sandbox cluster deployment".

## Codex Notes

Find the Lego-RL repo root first. It must contain
`scripts/lib/site.example.env`, the training runners, and the `.claude` plugin
path above.

Probe before deciding. Ask only for information that cannot be discovered
safely, usually how to reach the target nodes over SSH. Any `kubeadm reset`,
edit to an existing `/etc/kubernetes`, package install, service mutation, or
operation touching a cluster someone else may be using requires explicit user
confirmation and, where needed, sandbox escalation approval.

Generate a new `scripts/lib/site.<name>.env` rather than overwriting the default
`site.env`.
