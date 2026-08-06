---
name: check
description: >
  Preflight a SWE-Lego-RL config: answer "is it safe to launch this run
  right now?". Runs every deterministic check through the runner's own
  PREFLIGHT_ONLY path (config summary + scripts/lib/preflight.sh), adds the
  local live checks a config-only script cannot judge (is a run already in
  flight? is that GPU/port mine or someone else's?), and always ends with one
  structured report: a SAFE-TO-RUN verdict, a status table, the resolved run
  parameters, and numbered next steps. Read-only — never edits a config, never
  launches. Works for train, eval and infer configs. Triggers on "check this
  config", "preflight this run", "can I launch this run", "pre-launch check",
  "validate the config", "list the key parameters for this run".
---

# /rl:check — preflight one SWE-Lego-RL run

Answers one question: **is it safe to launch this config right now?**

Deterministic checks come from the runner itself (never re-implemented here);
this skill adds the local live layer and prints **one report** ending in a
clear YES / NO. Read-only — it never edits a config and never launches.

```
/rl:check <config>
   │
   ├─ 1. PREFLIGHT_ONLY=1 <kind>.sh <config> ... resolved parameters + OK/WARN/FATAL
   │
   ├─ 2. lib/live_probe.sh <kind> .......... local facts; this skill judges ownership
   │
   └─ 3. report ............ verdict + status table + run parameters + next steps
```

Each check lives in **exactly one** layer:

| Layer | Run by | Covers |
|---|---|---|
| **Deterministic** | `scripts/lib/preflight.sh` via the runner | tool_parser × model × scaffold · topology · SP/device-mesh · VRAM · veomni constraints · R3 × engine × verl · agent_name × scaffold · image source · kubeconfig · lr_scheduler · val timeout · rollout_is · path existence |
| **Live (judgment)** | this skill, from `scripts/lib/live_probe.sh` facts | is a run already in flight? · is that GPU/port mine, stale, or a foreign job's? · venv imports · disk / shm headroom |

Everything is **local host only** — no SSH to 221/222/240/243, no cluster
mutation. Cluster-side faults (registry down, kyverno crash-looping, a node
missing its insecure-registry config) are out of scope here; they surface as
`env_setup_failed` once a run is live, which is `/rl:status`'s job.

## Step 0 — Orient

Repo root is the `SWE-Lego-RL` checkout (contains `scripts/lib/preflight.sh`).
If the user ran this from elsewhere, locate it; if there is none, abort — but
still print the Step 3 report with a `NO` verdict whose reason is the abort.

**Resolve the config.** The argument may be a full path, a bare filename, or
absent:

- full/relative path that exists → use it;
- bare name (`qwen35a3b_4n_mix1582`, `smoke_sync_cc_qwen35a3b_243.env`) → search
  `scripts/{train,eval,infer}/configs/` then `.../templates/`;
- **absent, or more than one match** → list the candidates
  (`ls scripts/*/configs/*.env`) and ask which one. Do not guess.

**Infer the kind** from the resolved path: `scripts/train/…` → `train`,
`scripts/eval/…` → `eval`, `scripts/infer/…` → `infer`. If the path is
ambiguous, read the config: `TRAIN_MODE`/`N_NODES_ROLLOUT` → train,
`DATASET_PATH`/`DATASET_NAME` → eval, `RESULTS_DIR`/`OUTPUT_INDEX` → infer.

Read `scripts/README.md` for context if you need the axis vocabulary. Do not
echo config or README content into the report.

## Step 1 — Deterministic checks

```bash
PREFLIGHT_ONLY=1 bash scripts/<kind>/<kind>.sh <config> 2>&1; echo "EXIT=$?"
```

This one command resolves the model preset, the harbor agent env and the site
layer, prints the **run configuration block**, then runs `preflight.sh`. Read
its output as-is — never re-run a probe separately and never overrule an `OK`.

| Marker | Status | Blocks launch? |
|---|:---:|:---:|
| `✓ OK` | pass | — |
| `⚠ WARN` | warning | no |
| `✗ FATAL` | fail | **yes** |

Two things need follow-up:

- the `run configuration (<kind>)` block → copy **verbatim** into the report;
- a non-zero `EXIT` with no `✗ FATAL` line → the runner died before preflight
  (bad config syntax, missing `PROJECT_NAME`/`EXP_TAG`, unreadable venv). Treat
  as a blocking failure and quote the error.

`preflight.sh` failing is never something to work around. Every `✗ FATAL` has
an entry in the Troubleshooting section of the docs site; point the user at it.

## Step 2 — Live checks

```bash
bash scripts/lib/live_probe.sh <kind> 2>&1
```

The probe prints facts only (`OK` / `WARN` / `INFO`); deciding whether a
process is **ours** is this skill's job. Four outcomes block a launch —
`job:running`, `gpu:foreign`, `port:conflict`, `import:missing`. Everything
else is advisory.

### 2a — Is a run already in flight?

A second run on the same GPUs OOMs or corrupts both.

| Probe lines | Name | Status |
|---|---|:---:|
| any `WARN job:trainer` or `WARN job:runner` | `job:running` | ✗ blocks |
| only `WARN job:vllm` / `job:harbor` (no trainer/runner) | `job:orphan` | ⚠ — a leftover server or someone else's serving job; classify in 2b/2c |
| `OK job:none` | `job:none` | ✓ |

On `job:running`, record the PID set — call it the **run tree**; it anchors the
ownership tests below. Report pid + etime, and tell the user to let it finish or
stop it **themselves** (`kill -INT <pid>`). Never kill anything.

### 2b — Are the GPUs free, mine, or a foreign job's?

Input: each `WARN gpu:busy pid=<pid> comm=<comm> mem=<mem>` line.

| Owner | Name | Status |
|---|---|:---:|
| in the run tree from 2a | `gpu:mine` | ⚠ |
| any other pid | `gpu:foreign` | ✗ blocks |
| `OK gpu:idle` | `gpu:idle` | ✓ |

Ownership test: walk the parent chain (`ps -o ppid= -p <pid>`, repeated) and see
whether it reaches a PID in the run tree. A `VLLM::Worker_TP*` / `EngineCore`
holding ~130GB with no runner above it is someone else's serving job — block,
report pid + memory + etime, and let the user decide whether to wait or ask its
owner. **This box is shared; a foreign vLLM squatting on all 8 GPUs is the
common case, not an anomaly.**

If `nvidia-smi` is absent (`INFO gpu:absent`), emit `gpu:unknown` (⚠) and say
the GPU layer could not be judged — do not silently pass it.

### 2c — Who owns a busy port?

Input: each `WARN port:<P>` line. Only ports **this config will bind** can
block; the rest are informational.

| kind | ports that matter |
|---|---|
| train | `6379` (ray), `8265` (ray dashboard) |
| eval | the `port=` in the summary's `serving` line (default 8000) |
| infer | `VLLM_PORT` / `VLLM_MASTER_PORT` / `VLLM_DP_RPC_PORT` from the summary |

| Owner | Name | Status |
|---|---|:---:|
| in the run tree | `port:mine` | ⚠ |
| anything else, on a port this run needs | `port:conflict` | ✗ blocks |
| busy but irrelevant to this kind (e.g. 8090 webui) | `port:other` | ✓ note only |

For `port:conflict`, name the pid and suggest either stopping it or moving this
run's port in the config — never suggest killing a process you cannot attribute.

### 2d — Environment sanity

| Probe line | Name | Status |
|---|---|:---:|
| `WARN import:veomni` on a `train` run | `import:missing` | ✗ blocks — the runner exits on this |
| `WARN import:<mod>` otherwise | `import:degraded` | ⚠ |
| `WARN venv` | `venv:fallback` | ⚠ — runner will use a non-.venv python |
| `WARN shm` | `shm:dirty` | ⚠ — `train.sh` clears it at bring-up; only worrying if a live run owns it |
| `WARN disk:<mnt>` | `disk:low` | ⚠ — quote the mount; a full root disk evicts pods and truncates logs |

## Step 3 — The report  (always the last thing you print)

The report **is** the deliverable. Print it every time, including on an abort
(then: heading + `NO` verdict with the abort reason, nothing else). Prose in
Chinese — the field labels and probe names stay as written here.

````
## harbor check — <kind> · <config basename>

**SAFE TO RUN: <✅ YES | ❌ NO>** — <R> blocking · <W> warnings

| Layer | Check | Status | Detail |
|-------|-------|:------:|--------|
| det  | preflight (tool_parser · topology · device-mesh · VRAM · veomni · R3 · agent · image · paths) | <✓/✗> | ok=<N> warn=<N> fatal=<N> |
| det  | <one row per ✗ FATAL or ⚠ WARN> | <✗/⚠> | <verbatim text> |
| live | job   | <✓/⚠/✗> | <job:none / job:running pid=<P> etime=<T> / job:orphan> |
| live | gpu   | <✓/⚠/✗> | <gpu:idle / gpu:mine / gpu:foreign pid=<P> mem=<M>> |
| live | ports | <✓/⚠/✗> | <port:free / port:mine / port:conflict:<P>> |
| live | env   | <✓/⚠/✗> | <imports ok / import:missing:<mod> / venv:fallback> |
| live | disk  | <✓/⚠>   | <shm=<pct> root=<pct> / disk:low:<mnt>> |

**Key parameters for this run**
```
<paste the whole "run configuration (<kind>)" block from PREFLIGHT_ONLY, verbatim>
```

**Log destination / dashboard visibility**
```
train log   <TRAIN_LOG, taken from the runner's "train log:" line — never assembled by hand>
dashboard   <pid=<P> port=<P> serving <log-dir> → visible / not visible / no instance running>
```

Not a pass/fail check, but it belongs in the report because it is invisible
otherwise: `scripts/templates/verl/common.env` puts `TRAIN_LOG` under
`${HARBOR_LOG_DIR}`, and a real config overrides that to a per-experiment
directory under the shared trials root — **not** `<repo>/logs`. The webui globs
its `--log-dir` one level with no recursion (`webui/server.py`), so such a run
trains normally and never appears on the board. Read the served dirs from
`pgrep -af 'server\.py.*--log-dir'` and say which way it falls. `/rl:run` turns
this line into a question before launching; `/rl:check` only has to report it.

**Next steps**
1. <one line per blocker, blockers before warnings; quote kubectl/curl/git errors verbatim>
2. ...
Re-run `/rl:check <config>` once they are fixed.
````

**The four invariants:**

1. **Verdict** — `✅ YES` iff `R == 0`, where `R` = `✗ FATAL` count **+** any
   `job:running` **+** any `gpu:foreign` **+** any `port:conflict` **+** any
   `import:missing`. Warnings *never* change the verdict.
2. **Glyphs** — `✓` pass · `✗` blocks · `⚠` advisory · `·` skipped.
3. **Collapse** — fold all passing preflight checks into the first row; add a
   row only for each check that is `✗` or `⚠`.
4. **Verbatim** — the parameter block and every quoted error are copied
   character-for-character. Never paraphrase a config value into the report;
   if it is not in the runner's output, do not state it.

Add one italic line under the table when it applies:

- train with `NNODES > 1` → *(multi-node: this check covers the head node only —
  run `/rl:check` once on every worker node too)*
- infer with `VLLM_NNODES > 1` → same, keyed on `node_rank`.

## Guardrails — never do these

- edit the config, `lib/site.env`, or anything under `src/` to make a check pass
- launch anything, including a "quick" `DRY_RUN` that starts vLLM or ray
- re-implement or overrule a `preflight.sh` check — add only the live layer
- `kill` a process, `ray stop`, clear `/dev/shm`, or delete a log — surface it,
  let the user decide
- SSH to another node, or mutate the k8s cluster in any way
- report a value you did not read out of the runner's own output
