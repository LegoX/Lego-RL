<div align="center">

<img src="img/legorl_emblem.png" alt="Lego-RL" width="120"/>

# Lego-RL

**Online RL for real coding agents — in their native harnesses, on real repositories.**

[![Docs](https://img.shields.io/badge/docs-lego--rl.pages.dev-2563eb?logo=readthedocs&logoColor=white)](https://lego-rl.pages.dev)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![verl](https://img.shields.io/badge/built%20on-verl-orange)](https://github.com/verl-project/verl)
[![Harbor](https://img.shields.io/badge/built%20on-Harbor-8b5cf6)](https://github.com/SWE-Lego/harbor)

**[Documentation](https://lego-rl.pages.dev)** · **[Getting Started](https://lego-rl.pages.dev/docs/getting-started)** · **[Why Lego-RL](https://lego-rl.pages.dev/docs/why-lego-rl)**

</div>

Lego-RL is an open-source framework for reinforcement learning of **coding agents in their own
harnesses**. It connects agents such as **OpenHands**, **Claude Code** and **OpenCode**
to scalable policy-gradient training (PPO / GRPO / GSPO) while preserving their original control flow
and their exact rollout trajectories.

It integrates [**verl**](https://github.com/verl-project/verl) (trainer + rollout) with
[**Harbor**](https://github.com/SWE-Lego/harbor) (sandboxed task execution + verifier reward): the agent
solves a real issue in a real repository inside a fresh sandbox, the task's own test suite produces the
reward, and the trainer turns the scored trajectory into a gradient step.

Lego-RL is **faithful** with:

- **Unmodified agent harnesses**: each scaffold is a thin adapter, not a fork. The control flow that produced a rollout is the control flow you deploy.
- **Generation-time trajectory capture**: token ids, masks and log-probabilities are recorded inside the serving path by an in-process proxy — nothing is re-tokenized from a rendered transcript.
- **Executable rewards**: each task's own test suite runs in a fresh sandbox and its exit status is the reward — no reference-patch similarity, no model judge.
- **Rollout/training consistency on sparse models**: rollout-time expert routing is recorded and replayed during the update, holding rollout-vs-training log-prob correlation at `pearson ≈ 0.999`.

Lego-RL is **production-ready** with:

- **Scale-out sandboxes**: Kubernetes (inline build, nydus image cache, agent-runtime image mounting) or plain local/remote Docker, thousands of task containers per run.
- **Fully-async training**: partial rollout and staleness control keep GPUs busy while long agent trajectories finish.
- **Observability first**: validate a run before any GPU is committed, then diagnose it from metrics, termination causes and full agent trajectories in a live dashboard.

<div align="center">
 <img src="docs/public/framework.png" width="820" alt="Lego-RL architecture">
</div>

## News

- [2026/08] **First public release.** Lego-RL trains real coding agents — OpenHands, Claude Code, OpenCode — in their own unmodified harnesses on real repositories: token ids, masks and log-probabilities are captured inside the serving path by an in-process proxy, and the reward is each task's own test suite run in a fresh sandbox. Kubernetes (thousands of task containers per run) or plain Docker, synchronous or fully-async.

## Key Features

- **Agents**: OpenHands (`openhands-sdk` / `openhands-ai`), Claude Code, OpenCode — all driving the *same* model being trained. Any harness speaking the OpenAI or Anthropic API can be added as a [custom adapter](https://lego-rl.pages.dev/docs/architecture/agent-loop-workers#agent-loop-classes).
- **Algorithms**: PPO, GRPO, GSPO, with token-level / sequence-level importance sampling, adaptive KL, and trajectory filtering by `termination_reason` (broken or over-long trajectories are dropped from the *loss*, not the batch).
- **Training backends**: FSDP, Megatron-LM and **VeOmni**; synchronous single- and multi-node, or **fully-async** with partial rollout and staleness control.
- **Rollout**: vLLM behind an [in-process proxy](https://lego-rl.pages.dev/docs/architecture/in-process-proxy) serving both the OpenAI and Anthropic `/v1/messages` surfaces, with exact token/logprob capture, prefix-mismatch recovery for harnesses that rewrite their own history, and a [global load balancer](https://lego-rl.pages.dev/docs/architecture/global-load-balancer) with sticky session routing.
- **MoE**: R3 routing replay for Qwen3-MoE / Qwen3.5, expert parallelism, and a checkpoint merger with a mandatory key-set verifier.
- **Sandbox backends**: [Kubernetes or Docker](https://lego-rl.pages.dev/docs/training-run/sandbox-backends), with prebuilt or inline-built per-task images, nydus lazy pull, and per-phase network policy.
- **Reward**: each task's verifier suite, run in the sandbox, with [reward-hacking audits](https://lego-rl.pages.dev/docs/reference/reward-hacking) for task sets that leak their own answer.
- **Data**: build a parquet [task index](https://lego-rl.pages.dev/docs/data/task-index) from Harbor task folders (SWE-bench-style or OpenSWE-style), with [sandbox image](https://lego-rl.pages.dev/docs/data/sandbox-images) build pipelines and difficulty filtering.
- **Observability**: [run validation](https://lego-rl.pages.dev/docs/run-validation) before launch, a [live dashboard](https://lego-rl.pages.dev/docs/dashboard) over reward / entropy / KL / MFU / lengths / validation plus per-trial trajectories and their verifier reward, wandb-backed, publishable to Cloudflare Pages.
- **No-patch integration**: all glue lives in `src/verl_patch` and `src/harbor_patch` — nothing is patched into upstream verl or Harbor.
- **Optional agent plugin**: `/rl:check`, `/rl:run`, `/rl:status`, `/rl:dashboard` wrap the runners for [Claude Code](https://claude.com/claude-code) and Codex ([details](https://lego-rl.pages.dev/docs/reference/agent-plugin)) — convenience, never a requirement.

## Getting Started

> [!TIP]
> Everything — installation, configuration, the full loop, the failure playbook — is at
> **[lego-rl.pages.dev/docs](https://lego-rl.pages.dev/docs)**.

Install and launch the [demo run](https://lego-rl.pages.dev/docs/demo), a real training run at
1/16 scale (`8 prompts × 4 responses = 32 trials/step`):

```bash
git clone https://github.com/LegoX/Lego-RL.git && cd Lego-RL
bash scripts/setup_env.sh                                          # pinned upstreams + self-contained venv

cp scripts/train/examples/demo.env scripts/train/configs/demo.env
$EDITOR scripts/train/configs/demo.env                             # fill the CHANGEME values:
                                                                   #   checkpoint, train/val index, kubeconfig
PREFLIGHT_ONLY=1 bash scripts/train/train.sh scripts/train/configs/demo.env   # validate
bash scripts/train/train.sh scripts/train/configs/demo.env                   # launch
```

Needs 8× A100/H100-class GPUs, [uv](https://docs.astral.sh/uv/), a policy checkpoint, two task
indexes, and a reachable Kubernetes cluster (`BACKEND=docker` drives one machine's daemon instead).
The validate step launches nothing. In Claude Code the same run is `/rl:run scripts/train/configs/demo.env`.

> [!IMPORTANT]
> `setup_env.sh` clones Harbor [`SWE-Lego/harbor`](https://github.com/SWE-Lego/harbor) @ `ydu_dev`
> (harbor `0.3.1`), which **is not public yet**, so that clone fails without access. `HARBOR_REF=main`
> gets vanilla upstream (`0.1.45`) and works for everyone, minus the per-phase network-policy
> framework, the OpenHands-SDK 1.33 runtime, the OpenSWE adapter and the git-history restore hook.

## License

[Apache License 2.0](LICENSE).

## Acknowledgement

Built on [verl](https://github.com/verl-project/verl) for RL training and
[Harbor](https://github.com/SWE-Lego/harbor) for sandboxed task execution and verifier rewards.
Coding agents: [Claude Code](https://github.com/anthropics/claude-code),
[OpenHands](https://github.com/All-Hands-AI/OpenHands),
and [OpenCode](https://github.com/sst/opencode).
</content>
