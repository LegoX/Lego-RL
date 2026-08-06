---
name: run
description: >
  Preflight and launch a Lego-RL run (train, eval or infer). Runs
  /rl:check internally and refuses on any blocking failure, shows the fully
  resolved run parameters and waits for explicit confirmation, then launches the
  runner in the background by default — these runs take hours, so a foreground
  default would hold the session hostage. For multi-node train/infer it prints
  the exact per-node command instead of SSH-ing anywhere. Before launching it
  states where the run's log will actually land and whether the running dashboard
  can see it, asking when it cannot. After launch it surfaces the real exp_name,
  the PIDs, the resolved log paths and the first-step gates worth watching.
  Triggers on "launch this config", "run harbor training", "start the eval",
  "kick off training", "get this config running".
---

# /rl:run — launch one Lego-RL run

Preflight, review the parameters, confirm, then launch in the background.
Refuses if any check blocks, or if a run is already live on this box.

## Step 0 — Orient

Same as `/rl:check` Step 0: find the repo root, resolve the config
(path / bare name / ask when absent or ambiguous), infer the kind
(`train` | `eval` | `infer`). Never guess which config the user meant — a
wrong guess here burns a cluster for hours.

## Step 1 — Refuse if a run is already in flight

```bash
bash scripts/lib/live_probe.sh <kind> 2>&1 | grep -E '^WARN +job:'
```

If any `job:trainer` or `job:runner` line is alive, abort:

```
A run is already in flight; refusing to start another:
  pids:  <pid + etime, from ps -p <pid> -o pid=,comm=,etime=>
  log:   <most recent *.log under logs/>
Let it finish, or stop it yourself (kill -INT <pid>), then re-run /rl:run.
```

A bare `job:vllm` with no runner above it is a **foreign serving job**, not
ours — that is Step 2's business (it blocks on GPU ownership), not an abort
message about "our" run.

## Step 2 — Preflight via /rl:check

Run the `check` skill's logic on this config (invoke that skill, or inline its
Steps 1–3). If the verdict is **SAFE TO RUN: ❌ NO**, abort with the same
consolidated report, prefixed:

```
Preflight failed — fix the items below before /rl:run.
```

Do **not** invent values, skip a check, or pass a force flag; there is no force
flag. If the user says "just launch it anyway", explain exactly which check blocks
and what it costs to ignore it (each `✗ FATAL` maps to a real past incident, written
up in the Troubleshooting section of the docs site), then ask them to fix the config.
Configs are user-owned; this skill is a launcher, not an editor.

## Step 3 — Show the parameters and confirm  (mandatory, never skip)

`PREFLIGHT_ONLY=1` already printed the `run configuration (<kind>)` block in
Step 2. Show that block — verbatim — and then draw attention to the handful of
fields that decide whether the run is worth its hours. Keep this short; the
block above already has the detail.

```
Worth double-checking before this run:
  • data     <train/val index, or dataset, or shard>   ← is this the one you meant?
  • scale    <N nodes × 8 gpu>, <trials/step>, epochs=<N>  ← roughly <estimate> hours/step
  • resume   <RESUME_FROM, or "auto: picks up the newest ckpt in the exp dir">
  • naming   project=<...>  exp=<...>
  • logs     <the real TRAIN_LOG path>
             dashboard <visible / not visible: it is serving <log-dir>>
  • first step  <R3 on → pearson ≈ 0.999; otherwise routing misalignment silently
                destroys the gradients>
```

**Where the log lands, and whether the board will see it** — this line is not
cosmetic. `scripts/templates/verl/common.env` derives
`TRAIN_LOG=${HARBOR_LOG_DIR}/${TRAINER_EXPERIMENT_NAME}.log`, and a real config
overrides `HARBOR_LOG_DIR` to a per-experiment directory under the shared trials
root. Only the template *default* is `<repo>/logs`. The dashboard globs its
`--log-dir` **one level, no recursion** (`webui/server.py`) — so a config that
overrides `HARBOR_LOG_DIR` produces a run that trains fine and is **invisible on
the board for its whole life** unless something links it in. Resolve both sides
before asking for the yes:

```bash
bash scripts/train/train.sh --dry-run <config> 2>&1 | grep -E 'train log|vLLM log'
pgrep -af 'server\.py.*--log-dir'      # which dirs a board actually serves
```

If `dirname(TRAIN_LOG)` is not one of the served dirs — and "no board running at
all" counts as not served — **say so and ask**. Do not launch silently, and do not
pick for them:

```
This config writes its training log to
  <TRAIN_LOG>
but the dashboard (pid=<P>, port=<P>) is serving
  <served log-dir>
One-level glob, no match → this run will never appear on the board
(training itself is unaffected).

  [1] symlink it into <served-dir>/<name>.log right after launch
      (no dashboard restart — recommended)
  [2] leave it, watch wandb only
  [3] stop and change HARBOR_LOG_DIR in the config first
Which one?
```

Option 1 is one `ln -s` in Step 6, after the real `exp_name` is known. It is the
only choice needing neither a restart nor a config edit, and the symlink name
becomes the run's id in the UI — the analysis panels get their exp-dir mapping
from the log *contents*, so any readable name works. Never *move* the log (the
runner holds it open through `tee`), and never leave a dangling symlink under a
log dir — that breaks `/api/runs` for every run.

Then ask, in one line:

```
Launch with this configuration? [Y]/[N]  (N = edit the config first)
```

Never launch without an explicit yes. If the user says no, stop cleanly and
tell them to edit the config, then re-run `/rl:run`.

**exp_name gotcha** — unless the config pins `EXP_NAME`, `train.sh` derives it
as `harbor-<scaffold>-<EXP_TAG>-<timestamp>` **at launch time**, so the name in
the preflight output is *not* the name the run will have. Never report the
preflight-time name as final; read the real one back in Step 6. If the user
needs a stable name (resuming, comparing runs), suggest pinning `EXP_NAME=` in
the config before launching.

## Step 4 — Decide launch mode

Default = **background**. These runs last hours; foreground holds the session.

```
Launch mode? [B]ackground (default: nohup setsid, logs to logs/launch_<ts>.log)
             [F]oreground (holds this session until the run ends)
```

Accept `B` / `F` / `<enter>` (= background).

## Step 5 — Multi-node: print the per-node command, do not SSH

Read `topology` (train) or `vllm topo` (infer) out of the parameter block.

- **train, `NNODES > 1`** — `ray_bringup.sh` branches on local IP: the node whose
  IP equals `MASTER_ADDR` (default: the launching node's first IP) becomes the ray
  head and drives training; every other node joins and exits. So the *same*
  command must be run on each of the `NNODES` nodes.
- **infer, `VLLM_NNODES > 1`** — same shape, keyed on `VLLM_HEAD_HOST`; rank-0
  serves the API, the others run `--headless` DP shards.
- **eval** — always single-node.

Print the command block and tell the user to run it on the other nodes
themselves. **Never SSH to another node**, and never assume the workers are up
just because the head started.

```
Multi-node: run the command below once on every node (this machine is already covered
by the launch just performed)
  nodes <ip2>, <ip3>, ...:
    cd <repo_root> && MASTER_ADDR=<head_ip> nohup setsid bash scripts/train/train.sh <config> > logs/launch_<ts>_$(hostname -s).log 2>&1 &
The head node waits for all <NNODES> nodes to join before training starts; a worker that
never came up shows as the head spinning in its ray status poll.
```

## Step 6 — Launch

Timestamp: `TS=$(date -u +%Y%m%d-%H%M%S)`. Log: `logs/launch_${TS}.log`
(`mkdir -p logs` first).

### Background (default)

```bash
cd <repo_root> && nohup setsid bash scripts/<kind>/<kind>.sh <config> > "logs/launch_${TS}.log" 2>&1 < /dev/null &
PARENT_PID=$!
disown
```

`setsid` matters: it detaches the run from this session so a disconnect does not
take the training down with it.

Then, without polling in a tight loop:

1. Wait up to 60s (say, 5s sleeps) for the launch log to grow past 0 bytes.
2. Read its first ~120 lines and pull out:
   - the real `exp=` value from the `run configuration` block,
   - the preflight verdict line (it runs again inside the real launch),
   - any early `[FATAL]`.
3. Once they appear, capture the PIDs:
   `pgrep -f 'scripts/<kind>/<kind>.sh'` and, for train,
   `pgrep -f 'fully_async_main|main_ppo'` (the trainer takes a few minutes to
   show up — report it as pending rather than waiting for it).
4. If Step 3 chose `[1]`, link the log into the served dir now that `exp_name` is
   real. Verify with the API rather than assuming — the run only shows up once it
   has training steps, so an empty `steps` right after launch is expected:

   ```bash
   ln -sfn "<TRAIN_LOG>" "<served-dir>/<readable-name>.log"
   curl -s -m 300 "http://127.0.0.1:<P>/api/runs" | grep -c '<readable-name>'
   ```

   For a multi-node run, link only the **ray-head** node's log — that is the one
   carrying the trainer output; the worker exp dirs hold just
   `*_train_gpu_wandb.log`, and the analysis panels recover the sibling exp dirs
   on their own.

If the log stays empty for 60s, or the runner exits non-zero within 60s, print
the tail of the launch log and stop. **Do not retry automatically** and do not
"fix" the config to get past it.

### Foreground (only if chosen)

```bash
cd <repo_root> && bash scripts/<kind>/<kind>.sh <config> 2>&1 | tee "logs/launch_${TS}.log"
```

Stream it; on exit report the exit code and the log tail.

## Step 7 — Report

Tight summary, ≤ 15 lines:

```
Launched (background)
  kind / config:  <kind> · <config path relative to the repo>
  exp_name:       <the real name read back from the launch log; if not there yet,
                   "starting — check the log in 30s">
  launch log:     logs/launch_<TS>.log
  train log:      <absolute TRAIN_LOG path, read from the launch log's "train log:" line>
  vllm log:       <absolute VLLM_LOG path, same source>
  trials:         <HARBOR_TRIALS_DIR>
  dashboard:      <URL, run name = <readable-name> / not visible (user chose [2])>
  pids:           runner=<pid>  trainer=<pid or "starting">
  stage:          vLLM loading weights + cuda-graph capture; ~10–20 min before step 1

Watch:
  tail -F logs/launch_<TS>.log
  nvidia-smi
  /rl:status            # once it is running, check that the first step is healthy
Stop:
  kill -INT <runner_pid>    # let it wind down; do not kill -9
```

For train, close with the first-step gates — they are cheap to check and each
one has cost a full run before:

```
Check these at the first step (~20–40 minutes in):
  • R3 on → pearson ≈ 0.999 in the log. Clearly below that means the routing replay is
    misaligned and the gradients are already garbage — stop and investigate.
  • lr is not 0 (fully-async + cosine collapses to 0; this runner forces constant, but
    it is worth confirming).
  • First-step reward / num_turns are non-zero — all zeros usually means the environment
    images cannot be pulled, not a model problem.
```

## What this skill must NOT do

- never force past a blocking preflight, and never add a flag to bypass one
- never `kill` a live run to make room — ask the user to stop it themselves
- never edit the config, `lib/site.env`, or `src/` to make a check pass
- never SSH to another node, or launch anything on a node other than this one
- never run in the foreground without asking — it locks the session for hours
- never report the preflight-time `exp_name` as the run's real name (Step 3)
- never print `logs/<exp_name>.log` as the training log without checking — that is
  the *old* scripts' path; the config decides, and it is usually under
  `harbor_trials/`. Read the runner's own `train log:` line instead of composing one
- never launch without telling the user whether the dashboard will see this run
- never delete or move logs, `harbor_trials/`, or checkpoints; a dangling
  symlink under a run dir is enough to break the webui run list
