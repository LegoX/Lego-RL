# scripts/ — unified runners for training · inference · evaluation

One-liner: **pick a template → fill a few required fields → preflight → run.**
The other ~80 knobs all have defaults, hidden in `lib/` and the runners, so fixing a
bug in one place fixes it for every experiment.

```
pick template   cp <kind>/templates/<scenario>.env  <kind>/configs/my_run.env
fill required   edit the CHANGEME lines (data, topology, naming)
preflight       PREFLIGHT_ONLY=1 bash scripts/train/train.sh train/configs/my_run.env   (train only)
see the cmd     DRY_RUN=1        bash scripts/<kind>/<kind>.sh <kind>/configs/my_run.env
run                              bash scripts/<kind>/<kind>.sh <kind>/configs/my_run.env
```

> The old monolithic `fully_async_*/sync_*/eval_*/infer_*` scripts are still in the
> repo, **untouched** — keep using them if you like. The new system lives in
> `lib/ train/ infer/ eval/`; retire the old scripts once this is proven.

---

## Layout

```
scripts/
├── lib/                         # shared building blocks (every runner sources these)
│   ├── common_env.sh            #   legacy process env + venv + python setup
│   ├── site.env                 #   * THIS cluster's infra: registry/mounts/kubeconfig/accel/MODEL_ROOT
│   ├── site.example.env         #   * copy to site.env when moving to a different cluster
│   ├── harbor_env.sh            #   harbor agent (§8) + backend (§7), branches on SCAFFOLD × BACKEND
│   ├── ray_bringup.sh           #   ray cluster bring-up + /dev/shm cleanup (train)
│   ├── vllm_serve.sh            #   vLLM kill/ready helpers (eval/infer)
│   └── preflight.sh             #   pre-launch check: 9 classes of fatal assertions, fail-fast
├── train/{train.sh, configs/, templates/}
├── infer/{infer.sh, configs/, templates/}
└── eval/ {eval.sh,  configs/, templates/}
```

- **templates/** = structural skeletons, named by axes + model (`async_k8s_ohsdk_qwen35a3b.env`). Copy and edit.
- **configs/**   = concrete, runnable experiments, named by content (`qwen35a3b_4n_mix1582.env`).

---

## The three kinds

| kind | runner | template you copy | one live config |
|---|---|---|---|
| **train** | `train/train.sh` | `train/templates/<mode>_<backend>_<scaffold>_<model>.env` | `qwen35a3b_4n_mix1582.env` |
| **eval**  | `eval/eval.sh`  | `eval/templates/<model>_<profile>.env` | `qwen36-27b_official_verified.env` |
| **infer** | `infer/infer.sh` | `infer/templates/<model>_<topo>.env` | `qwen36-27b_14415_node0.env` |

---

## Axes (a template is a combination of axes)

Only these axes require picking a template / changing a value; everything else is a
parameter or a default.

| axis | values | what it changes |
|---|---|---|
| `TRAIN_MODE` | `async` \| `sync` | entrypoint `fully_async_main`+`fully_async_fsdp.yaml` / `main_ppo`+`sync.yaml`; async splits train/rollout, sync is one colocated pool |
| `ENGINE` | `veomni` \| `fsdp` | veomni: `model_engine=veomni`+`veomni.*` (required for hybrid GDN); fsdp: `strategy=fsdp2`+`fsdp_config.*` (R3 forces SP=1). **Chosen by the model preset** — usually leave it alone |
| `SCAFFOLD` | `ohsdk` \| `oh` \| `cc` \| `oc` | agent class + runtime image + loop config. ohsdk = primary; cc = Claude-Code; oc = OpenCode (R3 off by default) |
| `BACKEND` | `k8s` \| `docker` | task-env backend. k8s = pull prebuilt only (`force_build` ignored, never builds); docker = pull prebuilt (`force_build=False`) or build on a remote daemon. docker currently only has an oh loop config. See **Environment images** |
| model | `MODEL_PATH` + `TOOL_PARSER` + `ENGINE` + `SP_SIZE` + `MAX_PROMPT`/`MAX_RESP` | set explicitly per model; the per-model values worth copying are tabulated in the docs under Configuration |
| node count | `NNODES`/`N_NODES_TRAIN`/`N_NODES_ROLLOUT` | pure parameters, not a template axis |

**tool_parser is not a pure function of the model**: qwen3.5/3.6 → `qwen3_coder` (XML);
30B → `qwen3_coder` under cc, `hermes` under ohsdk/oh. The preset handles this by
SCAFFOLD automatically; preflight double-checks it.

---

## Parameter tiers (which knobs to expose, which to leave alone)

| tier | who changes it | examples | where it lives |
|---|---|---|---|
| **T0 axes** | picked when choosing a template | `TRAIN_MODE` `SCAFFOLD` `BACKEND` `MODEL_ENGINE` | template filename + first lines |
| **T1 required** | every run | `PROJECT_NAME` `EXP_TAG` `TRAIN_INDEX`/`VAL_INDEX` (or `DATASET_PATH`, or `RESULTS_DIR`+`OUTPUT_INDEX`) `NNODES`+topology | the CHANGEME lines in the config |
| **T2 often tuned** | frequently | `SAVE_FREQ` `TOTAL_EPOCHS` `TRAIN_BSZ` `N_RESP` `MAX_RESP` `VAL_BEFORE_TRAIN` `N_CONCURRENT` `EVAL_TEMPERATURE` | config, add as needed |
| **T3 advanced** | rarely, know why | `SP_SIZE` `ENABLE_R3` `ROLLOUT_IS` `TRAJ_FILTER_*` `GPU_MEM_UTIL` `VAL_TIMEOUT` `KL_LOSS_COEF` `CLIP_*` | config, override the default |
| **T4 site** | once per cluster | registry/nydus/mounts/kubeconfig/`MODEL_ROOT`/`DOCKER_HOST` | **`lib/site.env`** |
| **T5 hidden defaults** | basically never | ~80 harbor_env / hyperparameter defaults (loop name, offload, entropy chunking, tail-kill, pod timeouts…) | `lib/*` + runner |

Templates carry only T0–T1 (plus a few T2 examples). To tune a T2/T3 knob, add one
`KEY=VALUE` line to the config; the runner supplies a default for anything the config
omits. **Multi-word values must be quoted** (`EVAL_VLLM_EXTRA_ARGS="--a --b"`).

---

## site.env (read this before moving clusters / handing scripts to others)

The runners and libs **hardcode no cluster address**. Everything cluster-specific
(prebuilt-image registry, nydus mirror, hostPath mounts, kubeconfig, docker daemon,
model/code roots) lives in `lib/site.env`:

- with `site.env` → your accelerations are on (e.g. the 221 prebuilt images, fast pulls);
- without `site.env` → it falls back to a **portable vanilla** path (in-pod inline-build,
  no special mounts, no image rewrite) — slower, but runs anywhere.

To move to a different cluster: `cp lib/site.example.env lib/site.env` and edit the
values per the comments. **Do not hand your own `site.env` to anyone else.**

---

## Environment images (read this before running on a fresh cluster)

Every task instance needs an **environment image** (the container the agent acts in).
Where it comes from is the single biggest portability gotcha. Two modes:

| mode | how to enable | speed | needs |
|---|---|---|---|
| **prebuilt + registry** (recommended) | set `HARBOR_OPENSWE_IMAGE_REGISTRY` in `site.env`; prebuild per-instance images (`build_*.sh`) and push them | instant pull, no per-rollout setup | a registry every consumer can reach |
| **vanilla in-pod build** (portable fallback) | leave `HARBOR_OPENSWE_IMAGE_REGISTRY` empty | slow — re-installs deps every rollout | egress to pull the Dockerfile's `FROM` base image + install deps (pip/apt/conda) |

**Backends do NOT behave the same when the image isn't prebuilt:**

- **k8s** — `force_build` is **ignored** (`KubernetesEnvironment` only *pulls*, never builds).
  It needs either a prebuilt image in the registry, or a Dockerfile whose `FROM` line is a
  pullable image (harbor infers `docker_image` from that `FROM`). If even the base can't be
  pulled → `ImagePullBackOff`, the run can't start.
- **docker** — can `force_build=True` to build the Dockerfile on the daemon (no registry
  needed, but the daemon's builder must work and have egress), or `force_build=False` to
  pull a prebuilt image (same source as k8s).

**So "no local registry" means:**
- Small scale / debugging → vanilla works, just slow, and only if the cluster has egress.
- Real training (hundreds–thousands of instances) → **you must prebuild + push to a
  registry**; per-rollout dep installs don't scale, and restricted-network clusters will
  `env_setup_failed`/reward=0 without prebuilt images.

**If your registry is plain HTTP**, every consumer must trust it as *insecure*:
- k8s: each node's containerd (`certs.d/hosts.toml`, or `HARBOR_NYDUS_MIRROR`);
- docker: the daemon's `/etc/docker/daemon.json` → `"insecure-registries": ["<host:port>"]`
  then restart `docker.service` (restarts dockerd only, not the shared containerd — verify
  the containerd PID is unchanged and `kubectl get nodes` still Ready). Otherwise you get
  `http: server gave HTTP response to HTTPS client`.

Naming must match what harbor expects: `<registry>/openswe-<instance_id>:latest`, or the
Dockerfile's `FROM` line. See `the Troubleshooting section of the docs site` §D for the docker-backend
failure modes (`No pre-built rootfs`, `KeyError: 'config'`, empty-`position_ids` crash).

---

## Common tasks

| goal | how |
|---|---|
| preflight (no cluster) | `PREFLIGHT_ONLY=1 bash scripts/train/train.sh <config>` |
| print the final launch command | `DRY_RUN=1 bash scripts/train/train.sh <config>` |
| swap model | change `MODEL_PATH` + `TOOL_PARSER` + `ENGINE`/`SP_SIZE` + the context window |
| swap scaffold | change `SCAFFOLD=` (ohsdk/cc/oc) |
| sync ↔ async | change `TRAIN_MODE=` and use the matching template |
| change cluster | edit `lib/site.env` (or a one-off `K8S_KUBECONFIG=... bash ...`) |
| smoke run | in the config: `TRAIN_BSZ=8 N_RESP=4` |
| eval a checkpoint | in the eval config: `MODEL_PATH=/path/global_step_N_hf` |

---

## When something breaks

See `the Troubleshooting section of the docs site` — organized as symptom → root cause → fix,
one entry per class of preflight assertion. Every `✗ FATAL` preflight prints has its
explanation there.
