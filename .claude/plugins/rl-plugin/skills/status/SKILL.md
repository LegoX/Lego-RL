---
name: status
description: >
  Diagnose a SWE-Lego-RL run that is already in flight (or just finished):
  which run is alive, how far it has got, and whether its numbers are healthy.
  Reads the process table, the run log's metric lines and the trials directory,
  then checks the metrics against this cluster's known failure signatures —
  R3 pearson collapse, lr=0, grad starvation, env_setup_failed avalanches,
  val fake-zeros, no-tool-call collapse — and says which one matches. Read-only,
  local host only: never kills, never restarts, never edits. Triggers on
  "how's the run doing", "what step is it on", "is reward going up",
  "is this run broken", "diagnose the training run".
---

# /rl:status — is the live run healthy?

Read-only diagnosis of a run in flight. Answers three things in order:
**which run · how far · is it sick**. Never kills, restarts, cleans or edits
anything — a wrong intervention here costs more than a slow answer.

## Step 1 — Which run?

```bash
bash scripts/lib/live_probe.sh train 2>&1 | grep -E '^(WARN|OK) +(job|gpu):'
ls -t logs/*.log | head -5
```

Identify the live run from the `job:trainer` / `job:runner` lines, then map it
to its log. The runner's own log is `logs/<exp_name>.log` (train/eval) — take
`<exp_name>` from the `run configuration` block near the top of the matching
`logs/launch_*.log`, not from a guess.

If nothing is alive, say so and offer the **last finished** run instead
(newest `logs/*.log`); make it explicit in the report which of the two you are
describing. If several runs are alive, list them and ask which one — do not
merge metrics from two runs.

## Step 2 — How far has it got?

```bash
LOG=logs/<exp_name>.log
grep -oE 'step:[0-9]+ ' "$LOG" | tail -1          # latest step
grep -cE ' step:[0-9]+ - training/global_step' "$LOG"
ls -t harbor_trials/<project>/<exp_name> 2>/dev/null | head -3
tail -40 "$LOG"
```

Report: latest step, wall-clock since launch (`ps -p <pid> -o etime=`), average
minutes/step, and whether the tail is still moving (compare `stat -c %Y "$LOG"`
against now). **A log that has not been written to in >30 min while the process
is alive is itself the finding** — that is the deadlock shape, not a slow step.

## Step 3 — Are the numbers healthy?

Metrics live on the step lines as `key:value` pairs. Pull the latest step line
and read the keys below (these names are exact — they come from the real logs):

```bash
grep -E ' step:[0-9]+ - training/global_step' "$LOG" | tail -1 \
  | grep -oE '(training/rollout_actor_probs_pearson_corr|actor/(lr|grad_norm|kl_coef)|critic/rewards/mean|num_turns/mean|trajectory_filter/[a-z_/]+|response_length/(mean|clip_ratio)):[0-9.e+-]+'
```

| Metric key | Healthy | What a bad value means |
|---|---|---|
| `training/rollout_actor_probs_pearson_corr` | ≈ 0.999 (≥ 0.99) | **R3 routing replay is misaligned** — training on corrupted logprobs. The single most important gate; a run below this is already wasted. |
| `actor/lr` | = the configured lr | `0` → the fully-async + cosine + `total_training_steps=-1` bug; the model is frozen. Runner forces `constant`, so a 0 here means something overrode it. |
| `actor/grad_norm` | same order as prior runs (~0.2–0.5) | ~0.03 with very long responses = gradient starvation from token dilution, not a bug to fix mid-run. |
| `critic/rewards/mean` | non-zero, trending up | Flat 0 from step 1 = infrastructure, not the model — go to the filter reasons below before touching hyperparameters. |
| `num_turns/mean` | tens of turns | Collapsing toward ~1 with reward dropping = the model stopped emitting tool calls and just ends the episode; a real training pathology, not infra. |
| `trajectory_filter/reason/env_setup_failed` | ~0 | Non-trivial count = pods cannot start: image unpullable, registry down, or a node missing its insecure-registry trust. |
| `trajectory_filter/reason/timeout` | small fraction | A large share means the agent budget is too tight for these tasks, or env exec is stalling. |
| `trajectory_filter/invalid_ratio` | < ~0.1 | High = most of the batch is being dropped; the effective batch is far smaller than configured. |
| `response_length/clip_ratio` | low | High = responses hitting the window; the tail is being truncated. |
| `val-core/…`, `val-aux/num_turns/…` | non-zero at test_freq steps | All-zero val while train reward is fine = the val split's images are unpullable, **not** a model regression. |

Also worth a line each when present: `fully_async/processing_time/tp99` (long
tail), `fully_async/count/dropped_stale_samples` (staleness pressure),
`rollout_corr/kl`.

## Step 4 — Match against known failure signatures

Only claim a signature when its **specific** evidence is present. Say
"no known signature matched" rather than forcing a match — a wrong diagnosis here sends
the user chasing the wrong layer for hours.

| Signature | Evidence that must be present |
|---|---|
| **R3 misalignment** | pearson well below 0.99 on recent steps |
| **frozen model** | `actor/lr:0` |
| **grad starvation** | `actor/grad_norm` an order below the run's own earlier steps, alongside very long `response_length/mean` |
| **env avalanche** | `trajectory_filter/reason/env_setup_failed` climbing across steps; reward down in step |
| **val fake-zero** | val metrics 0 while `critic/rewards/mean` is healthy |
| **no-tool-call collapse** | `num_turns/mean` falling toward 1 over consecutive steps + reward falling; filter reasons *normal* |
| **deadlock / stall** | process alive, log mtime old, no new step line; check whether the tail sits in val or in a rollout wait |
| **step slowdown** | minutes/step up sharply — compare the node/replica counts in the run's own config block before blaming the tasks |

For anything that points off-box (registry, kyverno, node disk, image pulls),
**report the symptom and stop**. This skill does not SSH, does not touch the
cluster, and must not assert a cluster-side cause it cannot see from here —
phrase it as "the symptom points at X; confirm on <node>", and let the user decide.

## Step 5 — Report

Keep metric keys verbatim so they can be grepped.

```
## harbor status — <exp_name>

**<🟢 healthy | 🟡 at risk | 🔴 recommend stopping>** — <one-line conclusion>

  stage    step <N> (<epoch>) · running <etime> · ~<M> min/step · log last written <X> min ago
  procs    runner=<pid>  trainer=<pid>  GPUs in use: <n>
  reward   critic/rewards/mean=<v> (last <k> steps: <trend>)
  grads    actor/grad_norm=<v>   actor/lr=<v>   pearson=<v>
  traj     num_turns/mean=<v>  invalid_ratio=<v>  env_setup_failed=<v> timeout=<v>
  val      <value from the most recent val, or "test_freq not reached yet">

**Diagnosis**
<the matched signature + its supporting evidence; otherwise "no known signature matched">

**Recommendations**
1. <at most 3, cheapest first; irreversible actions such as stopping a run are always
   phrased as recommendations for the user to carry out>
```

Never end with an action you already took — this skill takes none.

## Guardrails

- read-only: no `kill`, no `ray stop`, no restart, no config edit, no log
  deletion or rotation (a dangling symlink under a run dir breaks the webui)
- local host only: no SSH, no `kubectl` mutation
- never merge two runs' metrics into one report
- never state a cause you did not read out of the log or the process table
- when the evidence is thin, say the evidence is thin
