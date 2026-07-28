# rl plugin — preflight / launch / diagnose / serve / install

Skills wrapping the runner system under `scripts/`, with one layering rule:

> **the script owns every deterministic check; the skill owns only what needs
> judgement, and always ends in one structured report.**

| Skill | Question it answers | Mutates anything? |
|---|---|---|
| `/rl:check <config>` | Can this be launched right now? | no |
| `/rl:run <config>` | Start a run once the parameters are confirmed | launches (after explicit confirmation) |
| `/rl:status` | Is the running job healthy? | no |
| `/rl:dashboard [log-dir]` | Bring the dashboard up on this machine | starts a server (after explicit confirmation) |
| `/rl:k8s-sandbox-install` | Install or scale out a sandbox Kubernetes cluster | installs (with confirmation at every risky step) |

`/rl:check`, `/rl:run` and `/rl:status` work on **train, eval and infer** configs — the kind is inferred
from the config path (`scripts/{train,eval,infer}/configs/*.env`). `/rl:dashboard`
takes no config: it adapts to whatever layout the current machine has, which is
the whole reason it exists.

## Layering

```
config.env
   │
   ├─ scripts/<kind>/<kind>.sh          resolves preset × axes × site × harbor env
   │     ├─ lib/run_summary.sh          → "run configuration" block  (the parameters)
   │     └─ lib/preflight.sh            → ✓OK / ⚠WARN / ✗FATAL       (the assertions)
   │
   ├─ scripts/lib/live_probe.sh         → local facts: jobs, GPUs, ports, shm, disk, imports
   │
   └─ skill                             → ownership judgement + one report + verdict

webui/
   │
   ├─ scripts/lib/dashboard_probe.sh    → local facts: instances, tunnels, ports,
   │                                      toolchain, dist freshness, candidate log dirs
   │
   └─ /rl:dashboard                     → which logs / which port / rebuild? + one report
```

The skills never re-implement a check. If a rule belongs in the config layer it
goes into `lib/preflight.sh`; if it is a fact about this host it goes into
`lib/live_probe.sh`; only *mine vs foreign vs stale* judgement lives in a skill.

## Using the scripts without Claude

Everything the skills call is a normal command:

```bash
PREFLIGHT_ONLY=1 bash scripts/train/train.sh scripts/train/configs/<cfg>.env   # parameters + assertions
DRY_RUN=1        bash scripts/train/train.sh scripts/train/configs/<cfg>.env   # + final launch command
bash scripts/lib/live_probe.sh train                                          # local live facts
bash scripts/lib/dashboard_probe.sh                                           # webui: where/what/which port
```

`PREFLIGHT_ONLY=1` and `DRY_RUN=1` both print the full parameter block first, so
a rejected config is still shown in full.

## Scope

Local host only — no SSH, no cluster mutation. Cluster-side faults (registry,
kyverno, per-node containerd trust, node disk) are deliberately out of scope:
they show up as `env_setup_failed` / val zeros once a run is live, and
`/rl:status` reports the symptom and points at the node instead of guessing.
