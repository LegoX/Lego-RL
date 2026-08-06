---
name: k8s-sandbox-install
description: Guided install / scale-out of a sandbox Kubernetes cluster for the Lego-RL k8s backend (kubeadm 1.32 + containerd + flannel + ImageVolume, optionally nydus / a shared registry / an isolated dockerd). Probes the target machines for differences (OS, network, disk, pre-existing cluster) and never asks for what it can detect itself; finishes by generating a site.<name>.env wired for the training side. Triggers: install kubernetes, set up a cluster, add a worker node, join worker, new cluster, sandbox cluster deployment.
---

# Sandbox cluster install wizard

You are an install wizard. The goal: stand up a cluster on the user's machines that can
run the Lego-RL Kubernetes backend, with versions, features and guardrails aligned to
the baseline below.

**Core principle: never ask for anything you can probe.** The user should usually have to
supply exactly one thing — how to reach the nodes over SSH. Everything else (OS, disks,
subnets, egress, an existing cluster, shared storage) you determine yourself. Only when a
decision is ambiguous or risky do you go back to the user, and then with **probe results
plus a recommendation**, not an open question. Never copy commands blindly: probe first,
then decide, at every stage.

Background reading before you start:

- `scripts/lib/site.example.env` — every cluster-specific value the training side consumes,
  and how the `site.env` layer works.
- `docs/content/docs/run-training/backends.mdx` — what the k8s backend expects.
- `docs/content/docs/data-preparation.mdx` — the environment-image story (prebuilt registry
  vs. in-pod inline build), which decides whether this cluster needs a registry at all.

## Version baseline (do not change unless the user explicitly asks)

kubeadm/kubelet/kubectl **1.32.13** (ImageVolume needs ≥1.32; 1.31 has a readonly bug and is
unusable), containerd 2.2.x, flannel, pause:3.10, CNI plugins ≥v1.5. `apt-mark hold` all of them.

## Stage 0 — the one question, then probe everything else

**Ask the user only**: how to log into the nodes (IP list + SSH user/password or key, whether
root is available). If the current shell's `known_hosts`, `~/.ssh/config` or command history
already hint at it, try that first — if it works, you do not even need to ask.

With SSH in hand, **probe everything** (collect per node, present one table before continuing):

| Item | How to probe | Automatic decision rule |
|---|---|---|
| OS / kernel / arch | `/etc/os-release`, `uname -rm` | Ubuntu 22.04/24.04 + amd64 → proceed; CentOS/arm64 → list the differences and confirm with the user |
| Which node is control-plane | compare cores / memory / disk | default to the balanced node, not the one with the largest disk (keep that for builds/storage). **State the choice and the reason**; change it only if the user objects |
| Large-disk path | `lsblk` / `df -h` for the biggest writable mount | one obvious large disk → use `<mount>/storage`; if the root disk *is* the largest, warn that images will consume hundreds of GB to TB and ask whether that is acceptable or another disk should be attached |
| Subnet conflicts | `ip route` vs `10.244.0.0/16`, `10.96.0.0/12` | on conflict, switch podSubnet automatically (e.g. `10.245.0.0/16`) and update flannel's `Network` to match — inform, do not ask |
| Egress | `curl -sI --max-time 5` against pkgs.k8s.io / download.docker.com / registry-1.docker.io / ghcr.io | all reachable → proceed; otherwise probe `env | grep -i proxy`, common internal apt mirrors, and any offline package directory — only ask for a proxy/mirror if all of that fails |
| Existing cluster / leftovers | `kubectl get nodes`, `systemctl status kubelet`, `ss -ltn \| grep 6443`, `ls /etc/kubernetes /var/lib/etcd` | live cluster → switch to the add-node flow; leftovers → list them, and `kubeadm reset` **always** requires user confirmation |
| Shared storage | `mount \| grep -E 'nfs\|alinas\|cpfs\|gpfs\|lustre'` | present → nydus / hostPath mounts can be enabled; absent → skip nydus, leave hostPath empty |
| swap / time / hostname | `swapon --show`, `timedatectl`, `hostname` | fix all of these directly (disable swap + fstab, install chrony, lowercase uppercase hostnames) and report afterwards |
| Usable registry | probe known addresses (already configured under `certs.d`, or seen in the user's shell history) with `curl /v2/` | list what answers as a recommendation; if none, default to the pure inline-build path (no registry needed) without asking |

**Optional-component defaults** (do not ask; explain in the final report, the user can request
more): isolated dockerd = only if the machine already has docker or the user mentioned building
images; nydus = only with a shared FS and obtainable binaries; metrics-server = install (harmless);
Docker Hub pull secret = only ask for a PAT if private images turn out to be needed.

## Stage 1 — per-node preflight (read-only; run everything before reporting)

Work through the Stage 0 table item by item and produce a node × check status table.
Resource limits: CPU < 16 warns (work around with kubeadm `--ignore-preflight-errors=NumCPU`);
ports 6443 / 10250 / 2379-2380 / 8472(udp) must be free; nodes must reach each other with
ping + nc. Fix what you find — do not enter Stage 2 with known problems.

## Stage 2 — per-node base configuration

Make every step idempotent so a re-run is safe:

1. Write all nodes into `/etc/hosts` (use a marked block, e.g. `# >>> k8s-sandbox-cluster >>>`,
   and delete the old block on re-run). **Write the full list on every node** so resolution
   works in both directions. Add any registry alias you detected a need for at the same time.
2. Kernel modules overlay / br_netfilter + sysctl (`bridge-nf-call-iptables`, `ip_forward`)
   + **inotify limits** (`max_user_watches=1048576`, `max_user_instances=8192`, required for
   high pod density).
3. Create the large-disk directories: `<disk>/containerd` (plus optional `<disk>/nydus-cache`,
   `<disk>/docker`).
4. apt sources + exact-version install + hold: containerd (either the Docker repo's
   `containerd.io=2.2.x` or Ubuntu's `containerd=2.2.1` — pick one, but **keep it identical
   cluster-wide**), kubelet/kubeadm/kubectl=1.32.13, kubernetes-cni.
5. containerd config (start from `containerd config default`, then edit — note **2.x emits
   single-quoted TOML, so `sed` must match single quotes**):
   - `root = '<disk>/containerd'`
   - `SystemdCgroup = true`
   - align pause: `s|pause:3.10.1|pause:3.10|g`
   - write `/etc/containerd/certs.d/<host:port>/hosts.toml` for each detected registry
     (add `skip_verify = true` for a plain-HTTP registry).
6. Install CNI plugins into `/opt/cni/bin` (flannel does not bundle them; without them nodes
   sit at `NetworkPluginNotReady`).
7. `systemctl restart containerd && systemctl enable containerd kubelet`.
8. Optional nydus: install nydusd / containerd-nydus-grpc + the snapshotter service, and
   register the proxy plugin. **Enable `sync_remove` GC**, and register the unpack platform
   with the containerd transfer service.

## Stage 3 — control-plane init (master only)

Generate `/root/kubeadm-config.yaml`, filling IPs / hostnames / podSubnet from the Stage 0
probe results. Key points:

- `apiServer.extraArgs: feature-gates=ImageVolume=true` **and**
  `KubeletConfiguration.featureGates.ImageVolume: true` — both sides are required.
- `maxPods` derived from the detected core count: 16c → 32; 64c+ → 128–200. Err on the low side.
- Set `controlPlaneEndpoint` even for a single master, so a later HA expansion does not need
  certificates re-signed.
- Include `imageGCHighThresholdPercent: 90` / `Low: 80`, `containerLogMaxSize: 100Mi`, and
  `evictionPressureTransitionPeriod: 5m` (`0s` amplifies DiskPressure evictions).

```bash
kubeadm config images pull --config=/root/kubeadm-config.yaml   # pull first: surfaces network problems early
kubeadm init --config=/root/kubeadm-config.yaml --upload-certs --ignore-preflight-errors=NumCPU
# save the join command; set up ~/.kube/config; apply flannel
# (if podSubnet changed, edit the flannel yml's Network first)
```

## Stage 4 — worker join + verification

`kubeadm join` (if the token expired, regenerate with `kubeadm token create --print-join-command`).
After each join, verify `ImageVolume: true` propagated into `/var/lib/kubelet/config.yaml`;
if not, add it by hand and restart kubelet. Once every node is Ready, a small cluster can drop
the master's `NoSchedule` taint.

## Stage 5 — acceptance (all green, or the install is not done)

1. Basics: `kubectl get nodes -o wide` all Ready, coredns Running, cross-node pod ping works
   (flannel VXLAN 8472/udp open).
2. **ImageVolume end to end**: a busybox test pod with `volumes[].image`. A
   `readonly must be true` error means the kubelet version is wrong.
3. **Registry trust**: `crictl pull <registry>/<known-image>` from any node. When probing
   whether an image exists, `curl` must send a full `Accept` header including the OCI index
   type, or you get a false 404.
4. maxPods took effect: `kubectl describe nodes | grep -A1 pods:`.
5. Verify each optional component: nydus (`ctr plugins ls | grep nydus` plus pulling one nydus
   image), isolated dockerd (Root Dir on the large disk; the k8s containerd namespace is only
   `k8s.io`), pull secret (start a pod from a private image).

## Stage 6 — wire it to the training side (generate, do not make the user fill it in)

1. Copy the master's `admin.conf` to the training machine (if `server:` is `127.0.0.1`, replace
   it with the real IP). **Check first whether a kubeconfig pointing at the same server already
   exists** and reuse it rather than creating a duplicate.
2. **Generate** `scripts/lib/site.<name>.env` from `scripts/lib/site.example.env`, filling every
   value from the probe results and noting where each came from:
   - `K8S_KUBECONFIG` ← the path from the previous step;
   - `HARBOR_OPENSWE_IMAGE_REGISTRY` ← a registry detected in Stage 0 **and** confirmed with a
     real `crictl pull`; none → leave empty (inline build is the fallback);
   - `HARBOR_NYDUS_MIRROR` ← only if nydus / a mirror was installed;
   - `HARBOR_HOSTPATH_MOUNTS` ← only paths you verified exist with `ls` on every node, else `null`;
   - `HARBOR_CLUSTER_DNS_IP` ← `kubectl -n kube-system get svc kube-dns -o jsonpath='{.spec.clusterIP}'`,
     needed explicitly only when it is not `10.96.0.10`;
   - `MODEL_ROOT` / `NEW_VERL_DIR` ← carry over from the existing `site.env` (these belong to the
     training machine and do not change with the cluster).

   Select it at run time with `SITE_ENV_FILE=scripts/lib/site.<name>.env`. **Do not overwrite the
   default `site.env`** — another run may still be using the old cluster.
3. If egress isolation is on, confirm on every node that `HARBOR_NETADMIN_IMAGE` is pullable
   (if it is not, every pod hangs). If it is not, mirror it into the user's registry first.
4. Health-check with `SITE_ENV_FILE=... PREFLIGHT_ONLY=1` against the runner (or `/rl:check`).
   Do a 1-node smoke run before a real one.

## Hard rules

- Any `kubeadm reset`, any edit to an existing `/etc/kubernetes`, or touching a cluster someone
  else is running → confirm with the user first.
- Never `apt upgrade`. Remind the user afterwards that the relevant packages are held.
- Never leave a `.bak` file in `/etc/kubernetes/manifests/` — the kubelet loads it as a static pod.
- Stop and diagnose at the first failing stage; do not skip ahead.
- Every IP and path in the generated site file must come from this run's probes. Never copy
  addresses from another user's cluster.
