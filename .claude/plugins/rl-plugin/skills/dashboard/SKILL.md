---
name: dashboard
description: >
  Bring up the Lego-RL training dashboard (webui/) on whatever machine you are on,
  adapting to that box's layout instead of assuming this repo's paths. Probes for
  the repo root, the directories that actually hold run logs, a free port, whether
  dist/ needs rebuilding and whether an instance is already serving, then proposes
  one concrete plan, waits for confirmation, and starts it in the background —
  optionally behind a Cloudflare quick tunnel. Also diagnoses a board that is up
  but showing nothing, and a run that trains fine yet never appears because the
  config wrote its log outside the served directory. Never kills someone else's
  instance. Triggers on "start the dashboard", "deploy the webui", "put the board
  online", "bring the dashboard up", "deploy the dashboard", "start the webui",
  "the dashboard will not open", "tunnel 502", "why does this run have no
  curves", "wandb has it but the board does not", "training curves not showing".
---

# /rl:dashboard — serve the training dashboard here

Gets `webui/` running **on this machine**, whatever its paths look like. The
dashboard is a stdlib-only Python server plus a prebuilt static bundle, so the
work is never "install things" — it is deciding **which logs to show, on which
port, and whether the bundle is current**.

```
/rl:dashboard [log-dir]
   │
   ├─ 1. lib/dashboard_probe.sh ..... layout · running instances · tunnels · ports
   │                                  · toolchain · dist freshness · log dirs
   ├─ 2. judgement ................. reuse or start? which log dir? rebuild?
   ├─ 3. plan + confirm ............. one block, explicit yes (a tunnel is public)
   └─ 4. start in background + report the URLs
```

Deterministic facts come from the probe; this skill only decides. It never kills
a process it did not start.

## Step 0 — Orient

Repo root is the checkout containing `webui/server.py` and
`scripts/lib/dashboard_probe.sh`. If the user invoked this from elsewhere, locate
it. An argument, if given, is the **log directory** to serve.

## Step 1 — Probe

```bash
bash scripts/lib/dashboard_probe.sh 2>&1
```

Read it as-is. Everything below is a judgement over these lines; never re-probe a
fact the script already reported.

## Step 2 — The four decisions

### 2a — Is one already serving?

| Probe lines | Meaning | What to do |
|---|---|---|
| `OK srv:none` | nothing up | start one |
| `WARN srv:running` with a `log-dir` that matches what the user wants | **already done** | do not start a second one — report its port + tunnel and stop |
| `WARN srv:running` on a *different* log-dir | someone else's board (or another checkout's) | leave it alone; start yours on a free port |

Two servers on two ports is fine and common. Two servers on the *same* log dir is
just confusion. **Never kill an existing instance to take its port** — the probe
cannot tell you whose it is, and on a shared box it is usually not yours.

<!-- A `server.py` sometimes shows as two pids (parent + thread); same port and
same log-dir means it is one instance, not two. -->

### 2b — Which log directory?

This is the decision that actually moves between machines, and the one to get
right before anything else: a board serving the wrong directory looks perfectly
healthy and shows nothing.

Rank the `logdir:*` lines by evidence, not by name:

1. an explicit argument or `LOG_DIR` from the user — always wins;
2. `logdir:*` with the **most run logs and the freshest** newest-write;
3. `logdir:default` (`<repo>/logs`) when nothing else has logs.

`INFO logdir:… exists but holds no .log/.out` means serving it yields an empty
board — say so rather than starting it. If two candidates both look live (a
sibling checkout is the usual case), **ask** which one; do not merge them silently.
When the user wants several at once, that is what `--extra-log-dir` is for — the
primary stays the one with the runs they care about.

#### `<repo>/logs` is NOT where the runner writes any more

Since the config/template refactor, `scripts/templates/verl/common.env:126` derives

```
TRAIN_LOG=${HARBOR_LOG_DIR}/${TRAINER_EXPERIMENT_NAME}.log
```

and **a real config overrides `HARBOR_LOG_DIR`** to a per-experiment directory
under the shared trials root (`<trials-root>/<project>/<exp>/logs`). Only the
template *default* falls back to `<repo>/logs`, which is why serving `<repo>/logs`
alone can look correct and still show nothing.

`server.py` globs **one level, no recursion** (`webui/server.py:133`), so a board
on `<repo>/logs` shows nothing for those runs while training is perfectly healthy.

The probe reports this directly — act on these lines, never assume:

| Probe line | Meaning | What to do |
|---|---|---|
| `INFO logdir:trialsroot` | the shared root the configs point `TRAIN_LOG` at | context for the two lines below |
| `OK logdir:trials … already visible via a link` | a real run log that a served dir already reaches through a symlink | nothing — it will show up |
| `WARN logdir:offrepo … NOT visible from any candidate dir` | **a real run whose curve the board cannot show** | surface it in the plan and pick one of the two fixes below |

Two fixes, both fine; pick by how many runs and whether the board is already up:

- `--extra-log-dir <exp>/logs` per run dir — explicit, but needs a restart, and
  you add a flag for every new exp;
- `ln -s <exp>/logs/<exp>.log <served-dir>/<readable-name>.log` — no restart (run
  discovery re-globs per request), and the name becomes the run's id in the UI.
  The trials/exp-dir mapping for the analysis panels is parsed out of the **log
  contents**, not the filename, so any readable name works.

Never *move* a run log to make it visible — the runner is holding it open through
`tee`, and a stale symlink left under a log dir is its own outage
(`webui/server.py:141`).

#### Say which directories you scanned

Whatever you decide, the plan and the final report must **name the directories
being served** and account for any `logdir:offrepo` line — including the ones you
chose to leave out. "The board is up" while a live run is invisible is the exact
failure this skill exists to prevent, and the user cannot see the probe output.

### 2c — Does the frontend need rebuilding?

| Probe line | Judgement |
|---|---|
| `OK dist:fresh` | serve as-is; do not rebuild "just in case" — it costs minutes and changes nothing |
| `WARN dist:stale` | src/ is newer than the bundle. `start_dashboard.sh` will **not** rebuild it (it only checks that `index.html` exists), so recent UI edits stay invisible until you rebuild |
| `WARN dist:absent` | must build before serving |

Rebuilding needs `OK node` (>= 20). On `WARN node`, activate nvm in the same
shell rather than declaring it impossible:

```bash
export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh"; nvm use 22
```

If no Node >= 20 exists at all and `dist:absent`, stop and say the frontend cannot
be built here — serving is impossible without a bundle. If `dist:stale` and Node is
unavailable, serving the stale bundle is a legitimate fallback; just say it is stale.

### 2d — Port, and whether to expose it

Take `OK port:free`. Honour an explicit `PORT=` from the user even if the probe
warns it is busy — but then say who holds it.

**The deliverable is a URL the user can actually open.** They are usually not on
this box, so `http://127.0.0.1:<P>` alone is not an answer — default to proposing
a tunnel, and only skip it if they say local-only.

It is still an outward-facing action: a quick tunnel makes the board reachable by
**anyone with the URL**, so it goes in the plan and waits for the yes like
everything else. Never open one silently.

Before opening a new one, check the `WARN tunnel` lines: if one already targets
the port you are about to serve, **reuse it** — that URL is the answer, and a
second tunnel to the same port just creates a second URL to keep track of.

**A quick tunnel needs no Cloudflare account, no login, and no token.**
`cloudflared tunnel --url http://127.0.0.1:<P>` gets a random
`*.trycloudflare.com` hostname anonymously. Never ask the user for credentials
for this, and never tell them to run `cloudflared tunnel login` — it is not
required and does not make a quick tunnel any more stable.

What can actually stop you, in the order it bites on a new machine:

| Probe line | Meaning | What to do |
|---|---|---|
| `INFO cloudflared … not installed` | just the binary is missing | local + `ssh -L <P>:127.0.0.1:<P> <host>`; do not install it yourself |
| `WARN egress` | binary is there, the box cannot reach Cloudflare's edge | the tunnel will hang or die at startup — go straight to `ssh -L`, do not retry |
| `OK egress` | good | propose the tunnel |

`INFO cf:account` is informational only: it reports whether a login happens to
exist, never a requirement.

If the user asks for a **stable** URL, be straight about the three options and
their real costs — none of them is "the same thing but permanent":

- **quick tunnel** (what this skill opens): no account, new URL on every restart.
- **named tunnel**: needs a Cloudflare account *and* a domain on it, plus
  `cloudflared tunnel login` or an API token. Gives a hostname you control.
  This skill does not set one up; say what it takes and let the user decide.
- **Cloudflare Pages** (`webui/functions/`): a fixed `*.pages.dev` URL, but it
  reads **wandb**, not local log files — so it shows different data than the
  board you just started. Do not offer it as a drop-in replacement.

## Step 3 — Plan, then confirm  (never skip)

Print exactly what will happen, then ask. Prose in Chinese; keep paths and flags
verbatim.

````
## Dashboard deployment plan

  log dir   <path>            (<N> run logs, newest <T> minutes old)
  extra dir <each --extra-log-dir path, or —>
  port      <P>               (<free / user-specified, currently held by pid X>)
  frontend  <reuse the existing dist / rebuild needed (<reason>)>
  tunnel    <on (publicly reachable — anyone with the link can open it) /
             reuse existing pid=<P> / off: this machine only>
  existing  <none / pid=<P> on <port>, serving <log-dir>, leaving it alone>

  runs the board cannot see (logdir:offrepo):
    <exp>/logs/<exp>.log        <T> min ago   → <symlink it / add --extra-log-dir / leave it>
    <write "none" when there are none>

<one line on why this log dir was chosen — from evidence, not guesswork>

Start with this plan? [Y]/[N]
````

If a tunnel is part of the plan, the confirmation line must say so explicitly —
the user is publishing an internal training board to a public URL.

If any `logdir:offrepo` run is fresh (written in the last hour — i.e. probably the
run the user actually wants), **do not silently write it off**: propose the
symlink or the `--extra-log-dir`, and let them decline.

## Step 4 — Start

Background, always: this process must outlive the session.

```bash
cd <repo_root>/webui
# only when 2c said so:
export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh"; nvm use 22 && npm run build

TS=$(date -u +%Y%m%d-%H%M%S)
nohup setsid python3 server.py --port <P> --host 127.0.0.1 \
  --log-dir <log-dir> [--extra-log-dir <dir>]... --static-dir dist \
  > "/tmp/dashboard_${TS}.log" 2>&1 < /dev/null &
```

Do **not** use `start_dashboard.sh` for this: it runs in the foreground, traps
`EXIT` to kill the server, and only rebuilds when `dist/index.html` is missing.
It is fine for an interactive one-off; it is wrong for a board meant to stay up.

Verify before claiming success — a server that binds and then finds nothing is
the failure this skill exists to catch:

```bash
curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:<P>/"          # fast
curl -s -o /dev/null -w '%{http_code}' -m 300 "http://127.0.0.1:<P>/api/runs"
```

`/api/runs` returning `[]` means the log dir was wrong — go back to 2b rather
than reporting a working board.

> **Cold-cache first call.** The *first* `/api/runs` on a fresh instance parses
> every log in the directory with an empty cache; over a few hundred runs that
> takes minutes, and every later call returns instantly. Give it a long timeout.
> A short one returns `000` and leaves a `BrokenPipeError` in the server log —
> the server finished the scan and was writing the response when the client left.
> That is not a crash: do not restart the server, and do not go looking for a
> dangling symlink because of it.

Then, if the tunnel was approved:

```bash
nohup setsid cloudflared tunnel --url "http://127.0.0.1:<P>" --no-autoupdate \
  > "/tmp/cf_dashboard_${TS}.log" 2>&1 < /dev/null &
```

Poll that log (5s sleeps, up to 60s) for `https://<...>.trycloudflare.com`, then
`curl` the URL and report the status code you actually got. A tunnel that is up
before the server is ready returns 502 for a few seconds — retry once before
calling it broken.

## Step 5 — Report

**Lead with the link the user can open.** Put it on the first line, and only
report a URL whose status code you have actually seen.

```
Dashboard is up → <public URL>        ← the one to click
  local    http://127.0.0.1:<P>
  log dir  <path> (<N> runs)
           <one line per --extra-log-dir>
  pid      server=<pid>  tunnel=<pid or —>
  log      <launch log path>
  not served  <run logs still outside every served dir, or "none">
Stop: kill <server_pid> <tunnel_pid>
```

The `log dir` lines are the answer to "why is my run missing" three days from now,
so spell out every directory this instance serves — not just the primary.

If no tunnel was opened, the first line instead says the board is local-only and
gives the SSH port-forward command — never leave the user without a way in.

Quick-tunnel URLs are ephemeral and change on every restart. If the user wants it
written down somewhere (docs, README), say that it will go stale — a stable URL
means a Cloudflare Pages deployment, which reads wandb rather than local logs and
therefore shows different data.

A tunnel URL that resolves but 502s for a few seconds right after launch is
normal (the tunnel registered before the server was ready); re-curl once before
reporting it as broken.

## Diagnosing a board that is already up

Two shapes come up constantly; both are answered from the probe alone.

**"The URL 502s."** A `WARN tunnel` line whose target port has no matching
`srv:running` is a tunnel outliving its server — the URL is valid, nothing is
behind it. Fix by starting a server on that port, or by opening a new tunnel to
the port that does have one. Do not assume the URL expired.

**"The board is empty / my run is missing."** Check in this order — the first one
is now the common answer and costs one probe line to rule out:

1. **`WARN logdir:offrepo` names the run** → the runner wrote it under the trials
   root, the served dir never had it. Fix per 2b (symlink or `--extra-log-dir`).
   Confirm by resolving the config's own `HARBOR_LOG_DIR` rather than guessing:
   `bash scripts/train/train.sh --dry-run <config> | grep -E 'train log|vLLM log'`.
   Do **not** go looking at the training itself first — a run at step 12 with
   healthy metrics in its own log is not a training problem.
2. Wrong served dir entirely (compare `logdir:*` counts).
3. The run's file is `*_vllm.log` / `*_train_gpu_wandb.log` / `launch_*` — all
   deliberately skipped (`webui/server.py:86-92`).
4. The run reached ≤ 5 training steps and is not currently being written to —
   `_handle_runs` hides those on purpose. A run visible in wandb but not here,
   with a cold log, is usually this.
5. A dangling symlink under the log dir breaking `/api/runs` — curl it and look
   for a non-200.

Only then suspect the UI.

A multi-node async run has one exp dir **per node** (`EXP_NAME=…$(date …)` is
evaluated on each node, so the timestamps differ by seconds). Only the ray-head
dir has the full trainer log; the others hold just `*_train_gpu_wandb.log`. Link
the head's log — the analysis panels recover the sibling dirs themselves
(`_expand_sibling_exp_dirs`, 600s window).

**"My UI change isn't showing."** `WARN dist:stale` — rebuild, per 2c.

## Guardrails — never do these

- `kill` a server or tunnel this skill did not start — report the pid, let the
  user decide, even when it looks abandoned
- take a port from a running instance
- open a public tunnel without explicit confirmation in this conversation
- run the server in the foreground, or under `start_dashboard.sh`'s EXIT trap
- delete or move `dist.bak-*`, `snapshots/`, `val_judgments/`, or anything under
  a log dir — a dangling symlink there breaks `/api/runs` for every run
- report a URL as working without having curled it
- edit `server.py`, `start_dashboard.sh`, or a config to make the launch work
