import asyncio
import atexit
import hashlib
import io
import json
import os
import re
import shlex
import signal
import socket
import sys
import tarfile
import threading
import time
from pathlib import Path
from typing import Optional

from kubernetes import client as k8s_client
from kubernetes import config as k8s_config
from kubernetes.client.rest import ApiException
from kubernetes.stream import stream
from tenacity import retry, stop_after_attempt, wait_exponential

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths
from harbor.utils.logger import logger

# Nydus mirror rewrite: opt-in. Set HARBOR_NYDUS_MIRROR=<host:port> on
# clusters whose nodes are configured (via nydus-snapshotter + containerd
# hosts.toml) to serve the listed prefixes from a local registry. Default
# empty = no rewrite, pull from docker.io.
# NOTE: the implicit 600s per-command exec deadline (HARBOR_ENV_EXEC_TIMEOUT_SEC) was removed.
# In the image-mounted OpenHands scaffold the WHOLE agent loop is a single exec(), so a per-
# command default silently capped the entire trajectory at 600s and killed productive runs.
# timeout_sec is now opt-in: the agent loop passes its own trajectory budget
# (HARBOR_AGENT_MAX_TIMEOUT_SEC), and the trial layer's asyncio.wait_for is the outer backstop.

NYDUS_MIRROR_HOST = os.environ.get("HARBOR_NYDUS_MIRROR", "")
NYDUS_MIRROR_PREFIXES = (
    "slimshetty/swebench-verified:",
    "docker.io/slimshetty/swebench-verified:",
    "swebench/sweb.eval.x86_64.",
    "docker.io/swebench/sweb.eval.x86_64.",
)


def _maybe_rewrite_to_nydus_mirror(image: str) -> str:
    if not NYDUS_MIRROR_HOST:
        return image
    for pfx in NYDUS_MIRROR_PREFIXES:
        if image.startswith(pfx):
            return f"{NYDUS_MIRROR_HOST}/{image.removeprefix('docker.io/')}"
    return image


# OpenSWE base-image namespace rewrite. The OpenSWE dataset's per-instance
# Dockerfiles inherit from bare `openswe-python-<ver>` tags (e.g.
# `FROM openswe-python-3.11`). These base images cannot live in the docker.io
# `library/` namespace (only Docker, Inc. can push there), so a bare ref
# resolves to `docker.io/library/openswe-python-3.11` and fails with
# ImagePullBackOff. Set HARBOR_OPENSWE_BASE_REGISTRY to the namespace or private
# registry holding your prebuilt base images (e.g. "myorg" for Docker Hub, or
# "registry.example.com:5000/ns") and this rewrite redirects the bare ref there.
# Empty (the default) disables the rewrite and leaves the ref untouched.
OPENSWE_BASE_REGISTRY = os.environ.get(
    "HARBOR_OPENSWE_BASE_REGISTRY", ""
).strip().rstrip("/")


def _maybe_rewrite_openswe_base(image: str) -> str:
    if not OPENSWE_BASE_REGISTRY:
        return image
    # Only rewrite *bare* openswe base refs (no registry/namespace prefix).
    # An already-qualified ref like `myorg/openswe-python-3.11` contains a
    # "/" and is left untouched.
    if "/" not in image and image.startswith("openswe-"):
        return f"{OPENSWE_BASE_REGISTRY}/{image}"
    return image


# hostPath path-prefix rewrite: opt-in. Set
# HARBOR_HOSTPATH_NODE_REWRITE='<host_prefix>:<node_prefix>[,<host_prefix>:<node_prefix>]'
# when the data the training process accesses under one path is the same
# physical storage that the K8s nodes mount at a different path (e.g. CPFS
# mounted as /path/to/shared on the training host and /path/to/node-local on the
# cluster nodes). Used by inline-build mode when constructing the
# /build-context hostPath. With no rewrite configured the training-host
# path is passed through unchanged (correct when both sides share the
# same absolute path).
def _rewrite_host_path_for_node(host_path: str) -> str:
    raw = os.environ.get("HARBOR_HOSTPATH_NODE_REWRITE", "").strip()
    if not raw:
        return host_path
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        host_pfx, _, node_pfx = entry.partition(":")
        host_pfx = host_pfx.rstrip("/")
        node_pfx = node_pfx.rstrip("/")
        if not host_pfx or not node_pfx:
            continue
        if host_path == host_pfx or host_path.startswith(host_pfx + "/"):
            return node_pfx + host_path[len(host_pfx):]
    return host_path


class _GenericK8sClientManager:
    """
    Process-level singleton for a generic Kubernetes CoreV1Api client.

    Unlike KubernetesClientManager (GKE-specific), this manager does not
    require cluster_name / region / project_id — it loads kubeconfig from:
      1. An explicit kubeconfig file path (kubeconfig_path arg)
      2. The KUBECONFIG environment variable
      3. The default ~/.kube/config
      4. In-cluster service-account token (KUBERNETES_SERVICE_HOST)
    """

    _instance: "_GenericK8sClientManager | None" = None
    _lock = asyncio.Lock()

    # Signals we attempt to trap so surviving pods are torn down before we die.
    # SIGKILL / SIGSTOP cannot be handled — activeDeadlineSeconds on the pod
    # spec is the ultimate safety net for those cases.
    _CLEANUP_SIGNALS: tuple[int, ...] = tuple(
        s for s in (
            getattr(signal, "SIGTERM", None),
            getattr(signal, "SIGHUP", None),
            getattr(signal, "SIGQUIT", None),
        ) if s is not None
    )

    def __init__(self):
        self._reference_count = 0
        self._client_lock = asyncio.Lock()
        self._initialized = False
        self._cleanup_registered = False
        self._logger = logger.getChild(__name__)

        # Process-wide registry of pods this process created.
        # Maps pod_name -> (namespace, kubeconfig_path) so that an emergency
        # cleanup path running outside the asyncio loop (atexit / signal
        # handler) can still talk to the right cluster.
        #
        # Guarded by a threading.Lock (NOT asyncio.Lock) because it is
        # touched from atexit / signal-handler contexts where the event
        # loop may be closed or non-existent.
        self._pod_registry: dict[str, tuple[str, Optional[str]]] = {}
        self._registry_lock = threading.Lock()

        # Previous signal handlers, preserved so we can chain to them.
        self._previous_signal_handlers: dict[int, object] = {}

        # Guards against recursive entry into the emergency cleanup from
        # the signal handler re-raise path.
        self._emergency_cleanup_done = False

    @classmethod
    async def get_instance(cls) -> "_GenericK8sClientManager":
        """Get or create the singleton instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()

        assert cls._instance is not None

        return cls._instance

    def _load_kubeconfig(self, kubeconfig_path: str | None):
        """Load kubeconfig using the best available source."""
        if kubeconfig_path:
            k8s_config.load_kube_config(config_file=kubeconfig_path)
            return

        kubeconfig_env = os.environ.get("KUBECONFIG")
        if kubeconfig_env:
            k8s_config.load_kube_config(config_file=kubeconfig_env)
            return

        # Try default ~/.kube/config, fall back to in-cluster
        try:
            k8s_config.load_kube_config()
        except k8s_config.ConfigException:
            k8s_config.load_incluster_config()

    async def get_client(
        self, kubeconfig_path: str | None = None
    ) -> k8s_client.CoreV1Api:
        """Load kubeconfig once (shared), then return a fresh CoreV1Api per caller.

        Each caller gets its own CoreV1Api (and underlying ApiClient) so that
        concurrent kubernetes.stream.stream() calls — which temporarily monkey-patch
        ApiClient.request — cannot interfere with each other.
        """
        async with self._client_lock:
            if not self._initialized:
                self._logger.debug("Initializing generic Kubernetes client")
                await asyncio.to_thread(self._load_kubeconfig, kubeconfig_path)
                self._initialized = True

                if not self._cleanup_registered:
                    self._register_cleanup_hooks()
                    self._cleanup_registered = True

            self._reference_count += 1
            self._logger.debug(
                f"Kubernetes client reference count incremented to {self._reference_count}"
            )
            return k8s_client.CoreV1Api()

    async def release_client(self):
        """
        Decrement the reference count for the client.
        Note: Actual cleanup happens at program exit via atexit.
        """
        async with self._client_lock:
            if self._reference_count > 0:
                self._reference_count -= 1
                self._logger.debug(
                    f"Kubernetes client reference count decremented to {self._reference_count}"
                )

    def register_pod(
        self,
        pod_name: str,
        namespace: str,
        kubeconfig_path: Optional[str] = None,
    ) -> None:
        """Track a pod so it will be torn down if this process dies unexpectedly."""
        with self._registry_lock:
            self._pod_registry[pod_name] = (namespace, kubeconfig_path)

    def unregister_pod(self, pod_name: str) -> None:
        """Drop a pod from the registry (call after it has been deleted cleanly)."""
        with self._registry_lock:
            self._pod_registry.pop(pod_name, None)

    def _register_cleanup_hooks(self) -> None:
        """Install atexit + signal handlers for emergency pod cleanup.

        The asyncio-aware ``stop()`` path is still the preferred cleanup route.
        These hooks exist to catch the cases it cannot handle — uncaught
        exceptions, SIGTERM from a job scheduler / OOM killer, SIGHUP on
        terminal close, SIGQUIT, and Ctrl+C after the event loop is torn down.
        """
        atexit.register(self._emergency_cleanup_sync)

        # Signal handlers can only be installed from the main thread. Skip
        # silently otherwise — atexit + activeDeadlineSeconds still apply.
        if threading.current_thread() is not threading.main_thread():
            return

        for sig in self._CLEANUP_SIGNALS:
            try:
                previous = signal.signal(sig, self._handle_cleanup_signal)
                self._previous_signal_handlers[sig] = previous
            except (OSError, ValueError) as e:
                # e.g. running in a restricted context; fall back to atexit only
                self._logger.debug(
                    f"Could not install handler for signal {sig}: {e}"
                )

    def _handle_cleanup_signal(self, signum, frame) -> None:
        """Signal handler: tear down pods, chain to prior handler, re-raise."""
        try:
            self._emergency_cleanup_sync()
        finally:
            previous = self._previous_signal_handlers.get(signum)
            # Restore & invoke the previous handler so normal termination
            # semantics (including any handler the CLI / framework installed)
            # still fire.
            try:
                if callable(previous):
                    signal.signal(signum, previous)  # type: ignore[arg-type]
                    try:
                        previous(signum, frame)  # type: ignore[misc]
                    except SystemExit:
                        raise
                    except BaseException:
                        pass
                    # If the previous handler did not terminate the process,
                    # fall through and raise the signal with the default
                    # disposition so we actually exit.
                elif previous in (signal.SIG_IGN,):
                    # Caller explicitly wanted to ignore it; honor that.
                    return

                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)
            except Exception:
                # As a last resort make sure we die.
                os._exit(128 + int(signum))

    def _emergency_cleanup_sync(self) -> None:
        """Synchronously delete every still-registered pod.

        Must be safe to invoke from atexit and signal-handler contexts, so it
        deliberately avoids asyncio and uses a fresh, synchronous Kubernetes
        client built from whichever kubeconfig each pod was registered with.
        """
        with self._registry_lock:
            if self._emergency_cleanup_done or not self._pod_registry:
                self._emergency_cleanup_done = True
                return
            snapshot = dict(self._pod_registry)
            self._pod_registry.clear()
            self._emergency_cleanup_done = True

        # Group pods by kubeconfig so we only load each config once.
        grouped: dict[Optional[str], list[tuple[str, str]]] = {}
        for pod_name, (namespace, kubeconfig_path) in snapshot.items():
            grouped.setdefault(kubeconfig_path, []).append((pod_name, namespace))

        for kubeconfig_path, entries in grouped.items():
            try:
                self._load_kubeconfig(kubeconfig_path)
                api = k8s_client.CoreV1Api()
            except Exception as e:
                print(
                    f"[harbor.k8s] Emergency cleanup could not initialize k8s client "
                    f"(kubeconfig={kubeconfig_path!r}): {e}. "
                    f"{len(entries)} pod(s) may leak — rely on activeDeadlineSeconds.",
                    file=sys.stderr,
                )
                continue

            for pod_name, namespace in entries:
                try:
                    api.delete_namespaced_pod(
                        name=pod_name,
                        namespace=namespace,
                        body=k8s_client.V1DeleteOptions(
                            grace_period_seconds=0,
                            propagation_policy="Background",
                        ),
                    )
                    print(
                        f"[harbor.k8s] Emergency cleanup: deleted pod "
                        f"{namespace}/{pod_name}",
                        file=sys.stderr,
                    )
                except ApiException as e:
                    if e.status == 404:
                        continue
                    print(
                        f"[harbor.k8s] Emergency cleanup failed for "
                        f"{namespace}/{pod_name}: status={e.status} reason={e.reason}",
                        file=sys.stderr,
                    )
                except Exception as e:
                    print(
                        f"[harbor.k8s] Emergency cleanup error for "
                        f"{namespace}/{pod_name}: {e}",
                        file=sys.stderr,
                    )


def _sanitize_pod_name(session_id: str) -> str:
    """Convert session_id to a valid Kubernetes pod name (DNS label rules).

    K8s pod names must be <=63 chars and start/end with [a-z0-9] (RFC 1123).
    A naive ``name[:63]`` truncation can land on a '-' (invalid trailing char,
    rejected with HTTP 422 "must end with an alphanumeric character") AND drop
    the uuid suffix that makes the name unique — two trials of the same
    long-named task would then collide. With a long experiment-name prefix this
    fires deterministically for long task names, silently dropping ~20% of
    rollouts as env_setup failures. So on truncation we strip stray hyphens and
    keep a short hash of the full session_id as a stable, unique suffix.
    """
    name = session_id.lower()
    name = re.sub(r"[^a-z0-9-]", "-", name)
    name = re.sub(r"-+", "-", name).strip("-")
    if len(name) <= 63:
        return name
    suffix = hashlib.md5(session_id.encode()).hexdigest()[:8]
    return name[: 63 - len(suffix) - 1].strip("-") + "-" + suffix


def _coerce_host_path_mounts(raw) -> list[dict]:
    """
    Normalize the ``host_path_mounts`` constructor argument.

    Accepts either a list of dicts (parsed YAML/JSON) or a JSON/YAML string
    (the env-var passthrough path), and returns a list of validated dicts:
    ``[{host_path, mount_path, read_only, type}, ...]``.

    Used by callers that need an offline-mounted ``/opt/rebench-v2`` etc. on
    clusters where pods cannot reach github.com.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        if not s or s.lower() in ("null", "none", "[]"):
            return []
        try:
            raw = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"host_path_mounts must be valid JSON, got {s!r}: {e}"
            ) from e
    if not isinstance(raw, list):
        raise ValueError(
            f"host_path_mounts must be a list, got {type(raw).__name__}"
        )
    out: list[dict] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(
                f"host_path_mounts[{i}] must be a dict, got {type(item).__name__}"
            )
        host_path = item.get("host_path") or item.get("hostPath")
        mount_path = item.get("mount_path") or item.get("mountPath")
        if not host_path or not str(host_path).startswith("/"):
            raise ValueError(
                f"host_path_mounts[{i}].host_path must be an absolute path"
            )
        if not mount_path or not str(mount_path).startswith("/"):
            raise ValueError(
                f"host_path_mounts[{i}].mount_path must be an absolute path"
            )
        out.append(
            {
                "host_path": str(host_path),
                "mount_path": str(mount_path),
                "read_only": bool(item.get("read_only", True)),
                "type": str(item.get("type", "Directory")),
            }
        )
    return out


def _pod_name_prefix_from_env() -> str:
    """Read HARBOR_POD_NAME_PREFIX, sanitize to a DNS-label-safe prefix.

    Used both as a pod-name prefix and as the value of the ``harbor-run`` label
    so external cleanup can target pods by `kubectl -l harbor-run=<prefix>`.
    Empty / unset disables prefixing (preserves the bare session_id behavior).
    """
    raw = os.environ.get("HARBOR_POD_NAME_PREFIX", "").strip()
    if not raw:
        return ""
    # K8s DNS-label rules: leave room for "-<session_id>" suffix below.
    return _sanitize_pod_name(raw)[:32]


class KubernetesEnvironment(BaseEnvironment):
    """
    Generic Kubernetes backend for Harbor sandboxes.

    Uses pre-built Docker images specified in task_env_config.docker_image —
    no image building step required. Works with any Kubernetes cluster
    (on-prem, managed, or in-cluster) without GCP dependencies.

    Kubeconfig is loaded in priority order:
      1. ``kubeconfig_path`` constructor arg
      2. ``KUBECONFIG`` environment variable
      3. ``~/.kube/config``
      4. In-cluster service-account token

    Configuration kwargs (passed via harbor trial config ``environment.kwargs``):
      namespace              Kubernetes namespace (default: "default")
      kubeconfig_path        Explicit path to a kubeconfig file (optional)
      image_pull_secrets     List of secret names for private registries (optional)
      tolerations            List of toleration dicts (optional)
      pod_startup_timeout_sec  Seconds to wait for pod Ready (default: 300)
      pod_active_deadline_seconds  Hard upper bound on pod lifetime in seconds,
                                   enforced by the kubelet. Acts as a K8s-side
                                   safety net so pods do not leak when the
                                   driver process is killed (SIGKILL / OOM /
                                   crash) before it can issue a delete. Pass
                                   ``0`` or a negative value to disable. Default:
                                   6 hours.

    Orphan protection:
      The pod spec carries ``harbor-owner-pid`` / ``harbor-owner-host`` labels
      identifying the creator process, the CoreV1 client manager keeps a
      process-wide registry of pods it created, and ``atexit`` + ``SIGTERM`` /
      ``SIGHUP`` / ``SIGQUIT`` handlers synchronously delete any still-
      registered pods when the process dies. ``activeDeadlineSeconds`` on the
      pod itself covers the remaining SIGKILL / segfault cases.

    Optional self-contained **agent runtime image** (Kubernetes 1.31+ image
    volume, ``spec.volumes[].image``). When set, the pod mounts the given OCI
    image as a read-only volume and injects ``CUSTOM_AGENT_RUNTIME_ROOT`` /
    ``CUSTOM_AGENT_PYTHON`` so ``image_mounted_openhands_ai`` can skip the
    in-task ``uv pip install`` path. Requires cluster support for image volumes
    (see Kubernetes feature ``ImageVolume``).

    Image volumes do **not** support ``volumeMounts.subPath`` (the K8s API
    server rejects the pod with HTTP 422 ``not allowed in image volume
    sources``). To work around this, the whole image rootfs is mounted at a
    fixed staging path (``_AGENT_RUNTIME_STAGE_DIR``) and the container's
    entrypoint creates a symlink at ``agent_runtime_mount_path`` pointing
    into the staging dir before exec'ing ``sleep infinity``. The original
    task image's ``/opt`` subtree is untouched — only one new symlink entry
    is added inside it.

      agent_runtime_image            Container image reference for the runtime
                                     (e.g. ``docker.io/you/harbor-openhands-ai-runtime:0.50.0``).
      agent_runtime_mount_path       Symlink path inside the task pod (default:
                                     ``/opt/custom-agent-runtime/openhands-ai``,
                                     must match how the runtime image was built).
      agent_runtime_image_subpath    Path *inside the image root* to expose at
                                     ``agent_runtime_mount_path``. If omitted, defaults
                                     to ``agent_runtime_mount_path`` with the leading
                                     ``/`` stripped (e.g. ``opt/custom-agent-runtime/openhands-ai``).
      agent_runtime_image_pull_policy  ``Always`` | ``Never`` | ``IfNotPresent``
                                     (default: ``IfNotPresent``).
      agent_runtime_volume_name      Kubernetes volume name (default: ``harbor-agent-runtime``).
    """

    # Default hard cap on pod lifetime (K8s-side safety net that fires even if
    # the driver process dies from SIGKILL / OOM / segfault and cannot run its
    # own cleanup). Set just ABOVE the trial hard timeout (TRIAL_HARD_TIMEOUT_SEC,
    # 75min) so the graceful trial-level cancel + pod delete runs FIRST; this
    # kubelet-side kill is the backstop for when that path can't fire — a driver
    # crash, or a trial blocked in an uncancellable exec reader thread (a hung
    # `pip install`). Killing the pod closes the exec stream, which unblocks that
    # thread and frees the slot. 90min = 75min trial cap + 15min margin.
    # (Was 6h — far too loose: a hung env-build held a slot for hours.)
    # Override with HARBOR_POD_ACTIVE_DEADLINE_SEC.
    _DEFAULT_POD_ACTIVE_DEADLINE_SEC: int = 90 * 60

    # Where the full agent-runtime image rootfs is mounted inside the task
    # container. Kubernetes image volumes do not support volumeMounts.subPath,
    # so we mount the whole rootfs here and symlink the desired subdirectory
    # to ``agent_runtime_mount_path`` from the container's entrypoint.
    _AGENT_RUNTIME_STAGE_DIR: str = "/mnt/harbor-agent-runtime"

    # --- Egress isolation (anti-reward-hacking, k8s per-phase network policy) ---
    # This cluster's CNI is flannel, which does NOT enforce k8s NetworkPolicy
    # objects, so egress control is done with iptables inside the pod's (shared)
    # network namespace. A dedicated sidecar holds NET_ADMIN and applies the rules;
    # the main (agent) container does NOT get NET_ADMIN, so the RL policy cannot
    # `iptables -F` its way out. The sidecar stays alive (sleep) so the policy can
    # be re-applied per phase (agent=allowlist, verifier=public) via
    # apply_network_policy. The image must provide `iptables`; a two-line
    # alpine+iptables build is enough. Set HARBOR_NETADMIN_IMAGE to a ref every
    # node can pull, or set HARBOR_K8S_EGRESS_ISOLATION=0 to run without it.
    # The task container is always named "main" (see _build_pod_spec). Raw
    # connect_get_namespaced_pod_exec / cp calls MUST pass this explicitly: once
    # the egress sidecar makes the pod multi-container, the k8s API rejects an
    # unspecified container with 400 (the kubectl.kubernetes.io/default-container
    # annotation is honored by kubectl only, NOT the raw API).
    _MAIN_CONTAINER_NAME: str = "main"
    _NETADMIN_SIDECAR_NAME: str = "netadmin"
    _NETADMIN_IMAGE: str = os.environ.get(
        "HARBOR_NETADMIN_IMAGE", ""
    )
    # CoreDNS / kube-dns service IP (allowed so name resolution keeps working).
    _CLUSTER_DNS_IP: str = os.environ.get("HARBOR_CLUSTER_DNS_IP", "10.96.0.10")

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        namespace: str = "default",
        kubeconfig_path: str | None = None,
        image_pull_secrets: list[str] | None = None,
        tolerations: list[dict] | None = None,
        pod_startup_timeout_sec: int = 300,
        pod_active_deadline_seconds: int | None = None,
        agent_runtime_image: str | None = None,
        agent_runtime_mount_path: str = "/opt/custom-agent-runtime/openhands-ai",
        agent_runtime_image_subpath: str | None = None,
        agent_runtime_image_pull_policy: str = "IfNotPresent",
        agent_runtime_volume_name: str = "harbor-agent-runtime",
        host_path_mounts: list | str | None = None,
        **kwargs,
    ):
        # Must be set BEFORE super().__init__ because BaseEnvironment.__init__
        # calls self._validate_definition(), which may set _inline_build=True.
        # If we initialized it after super().__init__, we'd overwrite it.
        self._inline_build = False
        # When True (set in _validate_definition for prebuilt OpenSWE images),
        # the task's Dockerfile RUN commands are NOT re-executed inside the pod:
        # the prebuilt image already contains the full environment (conda env +
        # repo + pip installs), so re-running conda create / pip install would be
        # pure redundant work — and at scale (~100 pods/step) it hammered the
        # node-local ext4 with conda/git/unzip writes (rename-lock + jbd2 journal
        # contention -> agents block on I/O -> GPUs idle).
        self._skip_dockerfile_run = False

        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            **kwargs,
        )

        self.namespace = namespace
        self.kubeconfig_path = kubeconfig_path
        self.image_pull_secrets = image_pull_secrets or []
        self.tolerations = tolerations or []
        self.pod_startup_timeout_sec = pod_startup_timeout_sec

        # None means "no hard deadline"; 0 also disables it. Otherwise we pass
        # the value straight through to spec.activeDeadlineSeconds so that
        # kubelet terminates the pod even if the driver cannot.
        # No explicit value from config -> allow an env override before falling
        # back to the class default.
        if pod_active_deadline_seconds is None:
            _env_deadline = os.environ.get("HARBOR_POD_ACTIVE_DEADLINE_SEC")
            if _env_deadline is not None:
                pod_active_deadline_seconds = int(_env_deadline)
        if pod_active_deadline_seconds is None:
            self.pod_active_deadline_seconds: int | None = (
                self._DEFAULT_POD_ACTIVE_DEADLINE_SEC
            )
        elif pod_active_deadline_seconds <= 0:
            self.pod_active_deadline_seconds = None
        else:
            self.pod_active_deadline_seconds = int(pod_active_deadline_seconds)

        self.agent_runtime_image = (agent_runtime_image or "").strip() or None
        self.agent_runtime_mount_path = agent_runtime_mount_path
        self.agent_runtime_image_subpath = agent_runtime_image_subpath
        self.agent_runtime_image_pull_policy = agent_runtime_image_pull_policy
        self.agent_runtime_volume_name = agent_runtime_volume_name

        # Optional read-only / read-write hostPath volumes injected into the
        # task pod. Used e.g. for offline-mounted /opt/rebench-v2 (SWE-rebench-V2
        # log parsers) on clusters where pods cannot reach github.com.
        # Accepts either a parsed list or a JSON/YAML string (env-var passthrough).
        self.host_path_mounts: list[dict] = _coerce_host_path_mounts(host_path_mounts)

        # Resource requests from task config
        self.cpu_request = str(task_env_config.cpus)
        self.memory_request = f"{task_env_config.memory_mb}Mi"

        # Resource LIMITS (defense-in-depth against node OOM / a node going unreachable).
        # The pod spec historically set requests ONLY -> pods were Burstable with
        # NO memory cap, so a runaway agent (conda/pip/test-suite builds) could
        # balloon past memory_mb and exhaust the node. On a cluster with little
        # system-reserved headroom this triggers the kernel OOM-killer against
        # kubelet/containerd -> node NotReady -> rollouts hang -> training
        # deadlock (the 0627 221 incident). A memory limit makes a runaway pod
        # cgroup-OOM-kill ITSELF (one failed trajectory) instead of the node.
        #   HARBOR_ENVIRONMENT_OVERRIDE_MEMORY_LIMIT_MB : explicit limit (MB).
        #     Unset -> request * HARBOR_ENVIRONMENT_MEMORY_LIMIT_FACTOR (def 1.5).
        #     Factor <= 0 (or explicit "0") -> no memory limit (old behavior).
        #   HARBOR_ENVIRONMENT_OVERRIDE_CPU_LIMIT : optional CPU limit (cores).
        #     Unset -> no CPU limit (CPU is compressible; a cap only throttles
        #     builds, it does not protect the node, so off by default).
        # k8s rejects limit < request, so an explicit memory limit is clamped up
        # to the request.
        self.memory_limit: str | None = None
        self.cpu_limit: str | None = None
        _lim_mb_raw = os.environ.get(
            "HARBOR_ENVIRONMENT_OVERRIDE_MEMORY_LIMIT_MB", ""
        ).strip()
        if _lim_mb_raw:
            _lim_mb = int(_lim_mb_raw)
            if _lim_mb > 0:
                self.memory_limit = f"{max(_lim_mb, int(task_env_config.memory_mb))}Mi"
        else:
            try:
                _factor = float(
                    os.environ.get("HARBOR_ENVIRONMENT_MEMORY_LIMIT_FACTOR", "1.5")
                )
            except ValueError:
                _factor = 1.5
            if _factor > 0:
                _lim = max(
                    int(task_env_config.memory_mb * _factor),
                    int(task_env_config.memory_mb),
                )
                self.memory_limit = f"{_lim}Mi"
        _cpu_lim_raw = os.environ.get(
            "HARBOR_ENVIRONMENT_OVERRIDE_CPU_LIMIT", ""
        ).strip()
        if _cpu_lim_raw:
            self.cpu_limit = _cpu_lim_raw

        # Pod name must satisfy Kubernetes DNS label rules. When
        # HARBOR_POD_NAME_PREFIX is set, prepend "<prefix>-" so cleanup can
        # target pods by name pattern; also surfaced as the ``harbor-run``
        # label below for `kubectl -l harbor-run=<prefix>` selection.
        self.pod_name_prefix = _pod_name_prefix_from_env()
        bare_pod = _sanitize_pod_name(session_id)
        if self.pod_name_prefix:
            self.pod_name = _sanitize_pod_name(
                f"{self.pod_name_prefix}-{bare_pod}"
            )
        else:
            self.pod_name = bare_pod

        self._client_manager: _GenericK8sClientManager | None = None
        self._core_api: k8s_client.CoreV1Api | None = None

        # Identity of the process that owns this pod. Encoded as pod labels so
        # orphaned pods can be identified by external tooling (e.g. kubectl
        # label selectors) after a process crash. Hostname is sanitized to
        # satisfy DNS-label rules.
        self._owner_pid = str(os.getpid())
        self._owner_host = _sanitize_pod_name(socket.gethostname() or "unknown")


    @property
    def _api(self) -> k8s_client.CoreV1Api:
        """Return the Kubernetes API client, raising if not initialized."""
        if self._core_api is None:
            raise RuntimeError(
                "Kubernetes client not initialized. Call _ensure_client() first."
            )
        return self._core_api

    async def _ensure_client(self):
        """Ensure Kubernetes client is initialized via the singleton manager."""
        if self._client_manager is None:
            self._client_manager = await _GenericK8sClientManager.get_instance()
        if self._core_api is None:
            self._core_api = await self._client_manager.get_client(self.kubeconfig_path)

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.KUBERNETES

    @property
    def is_mounted(self) -> bool:
        """Cloud environments don't mount directories."""
        return False

    @property
    def supports_gpus(self) -> bool:
        return False

    @property
    def can_disable_internet(self) -> bool:
        # k8s egress isolation is enforced with an iptables sidecar (see
        # _apply_default_egress_policy). Report True so the framework accepts
        # non-public network_mode for this environment.
        return True

    @property
    def supports_dynamic_network_policy(self) -> bool:
        # apply_network_policy re-applies iptables in the netadmin sidecar, so the
        # per-phase policy switch (agent=allowlist, verifier=public) is supported.
        return self._egress_isolation_enabled

    @property
    def _egress_isolation_enabled(self) -> bool:
        # Anti-reward-hacking: block the pod's PUBLIC egress (github / raw.github /
        # pypi answer-fetch = hacking.md Mode A/C/F) by DEFAULT, keeping the private
        # LAN (in-cluster LLM proxy / registry / DNS) reachable. A task opts out via
        # [environment] network_mode = "public"; the global kill-switch is
        # HARBOR_K8S_EGRESS_ISOLATION=0 (emergency rollback, NOT a per-run knob).
        enabled = os.environ.get("HARBOR_K8S_EGRESS_ISOLATION", "1") in (
            "1",
            "true",
            "True",
        )
        if enabled and not self._NETADMIN_IMAGE:
            raise RuntimeError(
                "Egress isolation is enabled but HARBOR_NETADMIN_IMAGE is unset. "
                "Point it at an image that ships `iptables` and that every node can "
                "pull (an alpine base with `apk add iptables` is enough), or set "
                "HARBOR_K8S_EGRESS_ISOLATION=0 to run without the sidecar — note "
                "that without it, agents can reach the public internet and fetch "
                "the reference fix."
            )
        return enabled

    @property
    def _environment_definition_path(self) -> Path:
        return self.environment_dir / "Dockerfile"

    def _validate_definition(self):
        # OpenSWE prebuilt-image consumption: when HARBOR_OPENSWE_IMAGE_REGISTRY
        # is set, use the cached per-instance image
        #   <reg>/openswe-<lowercased instance_id>:latest
        # built ahead of time and pushed to that registry, instead of
        # inline-building the Dockerfile. instance_id is the task-dir name
        # (<harbor_task_path>/<instance_id>/environment). Gated: a no-op when the
        # env var is unset, so other datasets/runs are unaffected.
        if not self.task_env_config.docker_image:
            _reg = os.environ.get("HARBOR_OPENSWE_IMAGE_REGISTRY", "").strip().rstrip("/")
            # GUARD: only openswe tasks (task path under openswe_filtered) — never
            # rewrite val/swebench tasks, whose openswe-<id> image does not exist.
            _is_openswe = "openswe" in str(self.environment_dir).lower()
            if _reg and _is_openswe:
                try:
                    _iid = self.environment_dir.parent.name
                except Exception:
                    _iid = ""
                if _iid:
                    self.task_env_config.docker_image = f"{_reg}/openswe-{_iid.lower()}:latest"
                    logger.info(
                        "KubernetesEnvironment: using prebuilt OpenSWE image %s "
                        "(HARBOR_OPENSWE_IMAGE_REGISTRY set)",
                        self.task_env_config.docker_image,
                    )
                    # The prebuilt image already contains the full env (conda +
                    # repo + pip). Skipping the Dockerfile's RUN re-execution is
                    # opt-in via HARBOR_OPENSWE_SKIP_DOCKERFILE (default False, so
                    # behavior is unchanged unless the launch script enables it).
                    self._skip_dockerfile_run = os.environ.get(
                        "HARBOR_OPENSWE_SKIP_DOCKERFILE", "false"
                    ).strip().lower() in ("1", "true", "yes", "on")
        if not self.task_env_config.docker_image:
            # Try to extract the image from the Dockerfile's FROM line.
            # This supports tasks (e.g. swebench-verified, swebench_multilingual)
            # whose task.toml omits docker_image but whose Dockerfile is simply
            # ``FROM <pre-built-image>`` plus a few extra RUN tweaks.
            #
            # Inline-build mode (translate the entire Dockerfile to in-pod
            # setup commands — mkdir on WORKDIR, COPY from /build-context,
            # ENV propagation) is OPT-IN via HARBOR_K8S_INLINE_BUILD=true.
            # Default off so that datasets relying on the historical fallback
            # behavior (FROM-line image is a pre-built image; only extra
            # RUN lines are executed in /testbed) keep working byte-for-byte.
            # Selfmade-style datasets whose FROM line is a generic base image
            # (e.g. ubuntu:24.04) must opt in.
            if self._environment_definition_path.exists():
                for line in self._environment_definition_path.read_text().splitlines():
                    stripped = line.strip()
                    if stripped.upper().startswith("FROM ") and not stripped.startswith("#"):
                        # FROM <image> [AS <name>] — keep just the image ref.
                        self.task_env_config.docker_image = stripped.split()[1]
                        inline_opt_in = os.environ.get(
                            "HARBOR_K8S_INLINE_BUILD", ""
                        ).strip().lower() in ("1", "true", "yes", "on")
                        if inline_opt_in:
                            self._inline_build = True
                            logger.info(
                                "KubernetesEnvironment: inferred docker_image=%s "
                                "from Dockerfile; inline_build mode ENABLED "
                                "(HARBOR_K8S_INLINE_BUILD set): Dockerfile "
                                "RUN/COPY/ENV/WORKDIR will be executed inside "
                                "the pod after start.",
                                self.task_env_config.docker_image,
                            )
                        else:
                            logger.debug(
                                "KubernetesEnvironment: inferred docker_image=%s "
                                "from Dockerfile (legacy mode; set "
                                "HARBOR_K8S_INLINE_BUILD=true to translate the "
                                "full Dockerfile to in-pod setup).",
                                self.task_env_config.docker_image,
                            )
                        break
        if not self.task_env_config.docker_image:
            raise ValueError(
                "KubernetesEnvironment requires task_env_config.docker_image to be set. "
                "The Kubernetes backend uses pre-built images — no Dockerfile is needed."
            )

        original = self.task_env_config.docker_image
        # OpenSWE bare base ref (openswe-python-*) -> published namespace first,
        # then the Nydus-mirror rewrite for swebench-style prebuilt images.
        rewritten = _maybe_rewrite_openswe_base(original)
        if rewritten != original:
            self.task_env_config.docker_image = rewritten
            logger.info(
                "KubernetesEnvironment: rewrote image %s -> %s (OpenSWE base)",
                original,
                rewritten,
            )
            original = rewritten
        rewritten = _maybe_rewrite_to_nydus_mirror(original)
        if rewritten != original:
            self.task_env_config.docker_image = rewritten
            logger.info(
                "KubernetesEnvironment: rewrote image %s -> %s (Nydus mirror)",
                original,
                rewritten,
            )

    def _resolved_agent_runtime_image_subpath(self) -> str:
        """Subpath inside the runtime image root to expose at the mount path."""
        if self.agent_runtime_image_subpath is not None:
            return self.agent_runtime_image_subpath.strip().strip("/")
        return self.agent_runtime_mount_path.lstrip("/")

    def _build_pod_spec(self) -> k8s_client.V1Pod:
        """Build the V1Pod object for this sandbox."""
        # ephemeral-storage is intentionally omitted: it was only a scheduling
        # hint and has no equivalent in the Docker backend.  Storage is managed
        # via emptyDir volumes below instead.
        requests = {
            "cpu": self.cpu_request,
            "memory": self.memory_request,
        }
        # Optional limits (see __init__): memory cap protects the node from a
        # runaway pod; empty -> None -> reverts to the old requests-only spec.
        limits: dict[str, str] = {}
        if self.memory_limit:
            limits["memory"] = self.memory_limit
        if self.cpu_limit:
            limits["cpu"] = self.cpu_limit

        # Mount /logs and /tmp as emptyDir volumes so writes bypass the
        # container's overlayfs layer, eliminating lock contention when many
        # pods run concurrently.  This mirrors the Docker backend which
        # bind-mounts /logs/* directly to the host filesystem.
        # No size_limit on either volume — Docker imposes none on /logs either.
        volumes = [
            k8s_client.V1Volume(
                name="logs-dir",
                empty_dir=k8s_client.V1EmptyDirVolumeSource(),
            ),
            k8s_client.V1Volume(
                name="tmp-dir",
                empty_dir=k8s_client.V1EmptyDirVolumeSource(),
            ),
        ]
        volume_mounts: list[k8s_client.V1VolumeMount] = [
            k8s_client.V1VolumeMount(name="logs-dir", mount_path="/logs"),
            k8s_client.V1VolumeMount(name="tmp-dir", mount_path="/tmp"),
        ]

        # Optional hostPath volumes (e.g. offline /opt/rebench-v2 source on
        # 240 cluster where pods cannot reach github.com to clone it).
        # Volume names are auto-generated; type defaults to "Directory" so the
        # pod fails-loud at scheduling if the host path is missing on the node.
        for i, hp in enumerate(self.host_path_mounts):
            vol_name = f"extra-hostpath-{i}"
            volumes.append(
                k8s_client.V1Volume(
                    name=vol_name,
                    host_path=k8s_client.V1HostPathVolumeSource(
                        path=hp["host_path"],
                        type=hp["type"],
                    ),
                )
            )
            volume_mounts.append(
                k8s_client.V1VolumeMount(
                    name=vol_name,
                    mount_path=hp["mount_path"],
                    read_only=hp["read_only"],
                )
            )

        # Inline-build mode: mount the task's environment/ dir as /build-context
        # so _apply_dockerfile_run_commands can resolve COPY/ADD sources.
        #
        # The Python process reads files via `self.environment_dir` (a
        # training-host path like /path/to/shared/.../environment), but
        # hostPath in the pod spec needs the K8s-node path (e.g.
        # /path/to/node-local/.../environment) because the data lives on a
        # cluster-shared FS that's mounted at different prefixes on the two
        # sides. Set HARBOR_HOSTPATH_NODE_REWRITE='<host_prefix>:<node_prefix>'
        # to translate. With no rewrite configured the literal training-host
        # path is used (correct when both sides really do share the same
        # absolute path).
        if self._inline_build:
            build_ctx_path = _rewrite_host_path_for_node(str(self.environment_dir))
            volumes.append(
                k8s_client.V1Volume(
                    name="build-context",
                    host_path=k8s_client.V1HostPathVolumeSource(
                        path=build_ctx_path,
                        type="Directory",
                    ),
                )
            )
            volume_mounts.append(
                k8s_client.V1VolumeMount(
                    name="build-context",
                    mount_path="/build-context",
                    read_only=True,
                )
            )

        container_env: list[k8s_client.V1EnvVar] = [
            k8s_client.V1EnvVar(
                name="PATH",
                value=(
                    "/root/.local/bin"
                    ":/opt/miniconda3/envs/testbed/bin"
                    ":/opt/miniconda3/bin"
                    # continuumio/miniconda3 (used by OpenSWE tasks) installs
                    # conda at /opt/conda, not /opt/miniconda3 — include it so
                    # `conda create -n testbed` / `conda activate` resolve.
                    ":/opt/conda/envs/testbed/bin"
                    ":/opt/conda/bin"
                    ":/usr/local/go/bin"
                    ":/opt/node/bin"
                    ":/usr/local/node/bin"
                    ":/usr/local/sbin:/usr/local/bin"
                    ":/usr/sbin:/usr/bin:/sbin:/bin"
                ),
            ),
            k8s_client.V1EnvVar(name="TMPDIR", value="/tmp"),
        ]

        # Opt-in fix for cwd-shadowing of the runtime interpreter. The mounted
        # runtime detection probe (and the agent run) execute with cwd=/testbed,
        # so `import openhands.sdk` prepends /testbed to sys.path[0] and picks up
        # a task repo whose top-level package name collides with an SDK
        # dependency — e.g. the `requests` repo's /testbed/requests shadows the
        # runtime's own requests, the import raises, _detect_mounted_runtime()
        # returns None, and the agent falls back to the pre-3.12 in-pod venv,
        # dying with "Task image python3 is < 3.12 ...". PYTHONSAFEPATH tells
        # CPython (3.11+) not to prepend cwd/"" to sys.path, so the runtime's own
        # packages win. Gated (default off) so the main eval/train path is
        # unperturbed; enable per-run with HARBOR_K8S_PYTHONSAFEPATH=1.
        if os.environ.get("HARBOR_K8S_PYTHONSAFEPATH", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            container_env.append(
                k8s_client.V1EnvVar(name="PYTHONSAFEPATH", value="1")
            )

        # Default keep-alive command — same as before. Overridden below when
        # an agent-runtime image is mounted, so the symlink is in place
        # before any external exec runs against the pod.
        container_command: list[str] = ["sleep", "infinity"]

        if self.agent_runtime_image:
            if not self.agent_runtime_mount_path.startswith("/"):
                raise ValueError(
                    "agent_runtime_mount_path must be an absolute path "
                    f"(got {self.agent_runtime_mount_path!r})"
                )
            if self.agent_runtime_mount_path.rstrip("/") in ("", "/"):
                raise ValueError(
                    "agent_runtime_mount_path must not be '/' when using agent_runtime_image"
                )

            subpath = self._resolved_agent_runtime_image_subpath()
            if not subpath:
                raise ValueError(
                    "agent_runtime_image_subpath resolves to an empty string; "
                    "set agent_runtime_image_subpath explicitly if the runtime "
                    "layout in the image is non-standard."
                )

            # Kubernetes image volumes reject volumeMounts.subPath, so we
            # mount the whole image rootfs at a staging directory and let
            # the container's entrypoint create a symlink at the public
            # mount path. This preserves CUSTOM_AGENT_RUNTIME_ROOT semantics
            # without touching the rest of the task image's /opt subtree.
            stage_dir = self._AGENT_RUNTIME_STAGE_DIR
            volumes.append(
                k8s_client.V1Volume(
                    name=self.agent_runtime_volume_name,
                    image=k8s_client.V1ImageVolumeSource(
                        reference=self.agent_runtime_image,
                        pull_policy=self.agent_runtime_image_pull_policy,
                    ),
                )
            )
            volume_mounts.append(
                k8s_client.V1VolumeMount(
                    name=self.agent_runtime_volume_name,
                    mount_path=stage_dir,
                    read_only=True,
                )
            )

            mount = self.agent_runtime_mount_path.rstrip("/")
            link_parent = os.path.dirname(mount) or "/"
            link_target = f"{stage_dir.rstrip('/')}/{subpath}"
            # /bin/sh availability is the same assumption the previous
            # ``sleep infinity`` command already made about the task image.
            symlink_script = (
                "set -e; "
                f"mkdir -p {shlex.quote(link_parent)}; "
                f"rm -rf {shlex.quote(mount)} 2>/dev/null || true; "
                f"ln -sfn {shlex.quote(link_target)} {shlex.quote(mount)}; "
                "exec sleep infinity"
            )
            container_command = ["/bin/sh", "-c", symlink_script]

            container_env.extend(
                [
                    k8s_client.V1EnvVar(
                        name="CUSTOM_AGENT_RUNTIME_ROOT",
                        value=mount,
                    ),
                    k8s_client.V1EnvVar(
                        name="CUSTOM_AGENT_PYTHON",
                        value=f"{mount}/bin/python",
                    ),
                ]
            )
            logger.info(
                "KubernetesEnvironment: mounting agent runtime image %s at %s "
                "(staged at %s -> %s, volume=%s)",
                self.agent_runtime_image,
                self.agent_runtime_mount_path,
                stage_dir,
                link_target,
                self.agent_runtime_volume_name,
            )

        pull_secrets = (
            [k8s_client.V1LocalObjectReference(name=s) for s in self.image_pull_secrets]
            if self.image_pull_secrets
            else None
        )

        tolerations_objs = (
            [
                k8s_client.V1Toleration(
                    key=t.get("key"),
                    operator=t.get("operator", "Equal"),
                    value=t.get("value"),
                    effect=t.get("effect"),
                    toleration_seconds=t.get("toleration_seconds"),
                )
                for t in self.tolerations
            ]
            if self.tolerations
            else None
        )

        labels = {
            "app": "harbor-sandbox",
            "session": self.session_id[:63],
            "environment": self.environment_name[:63],
            # Owner metadata lets external tooling find orphans if both the
            # in-process registry and activeDeadlineSeconds somehow miss them.
            "harbor-owner-pid": self._owner_pid[:63],
            "harbor-owner-host": self._owner_host[:63],
        }
        if self.pod_name_prefix:
            # Selectable via `kubectl -l harbor-run=<prefix>` for targeted cleanup.
            labels["harbor-run"] = self.pod_name_prefix[:63]

        return k8s_client.V1Pod(
            api_version="v1",
            kind="Pod",
            metadata=k8s_client.V1ObjectMeta(
                name=self.pod_name,
                namespace=self.namespace,
                labels=labels,
                # With the egress sidecar the pod is multi-container; pin the
                # default exec target to "main" so exec(container=None) (agent
                # commands, git shim) deterministically hits the task container.
                annotations=(
                    {"kubectl.kubernetes.io/default-container": "main"}
                    if self._egress_isolation_enabled
                    else None
                ),
            ),
            spec=k8s_client.V1PodSpec(
                containers=[
                    k8s_client.V1Container(
                        name="main",
                        image=self.task_env_config.docker_image,
                        image_pull_policy="IfNotPresent",
                        command=container_command,
                        env=container_env,
                        resources=k8s_client.V1ResourceRequirements(
                            requests=requests,
                            limits=limits or None,
                        ),
                        volume_mounts=volume_mounts,
                    ),
                    # Egress-isolation sidecar (anti-reward-hacking). Holds
                    # NET_ADMIN and applies iptables to the shared pod netns; the
                    # main container above has no NET_ADMIN so the agent cannot
                    # undo the rules. Added only when isolation is enabled.
                    *(
                        [
                            k8s_client.V1Container(
                                name=self._NETADMIN_SIDECAR_NAME,
                                image=self._NETADMIN_IMAGE,
                                image_pull_policy="IfNotPresent",
                                command=["sh", "-c", "exec sleep infinity"],
                                security_context=k8s_client.V1SecurityContext(
                                    capabilities=k8s_client.V1Capabilities(
                                        add=["NET_ADMIN", "NET_RAW"]
                                    )
                                ),
                                # NEAR-ZERO requests so the sidecar does NOT eat
                                # into how many task pods bin-pack onto a node
                                # (the scheduler reserves by *requests*): it just
                                # applies iptables once then sleeps at ~0 CPU /
                                # ~2Mi RSS. No CPU limit (compressible; a cap
                                # would only throttle the brief apply). A small
                                # memory limit protects the node. All tunable via
                                # HARBOR_NETADMIN_* if a node is memory-starved.
                                resources=k8s_client.V1ResourceRequirements(
                                    requests={
                                        "cpu": os.environ.get(
                                            "HARBOR_NETADMIN_CPU_REQUEST", "1m"
                                        ),
                                        "memory": os.environ.get(
                                            "HARBOR_NETADMIN_MEM_REQUEST", "16Mi"
                                        ),
                                    },
                                    limits={
                                        "memory": os.environ.get(
                                            "HARBOR_NETADMIN_MEM_LIMIT", "64Mi"
                                        ),
                                    },
                                ),
                            )
                        ]
                        if self._egress_isolation_enabled
                        else []
                    ),
                ],
                restart_policy="Never",
                image_pull_secrets=pull_secrets,
                tolerations=tolerations_objs,
                volumes=volumes,
                # Hard upper bound on pod lifetime. The kubelet terminates the
                # pod after this many seconds regardless of whether the driver
                # process is still alive. This is the last line of defence for
                # SIGKILL / OOM / crash scenarios where neither stop() nor the
                # atexit / signal-handler fallbacks can run.
                active_deadline_seconds=self.pod_active_deadline_seconds,
            ),
        )

    async def _delete_pod_and_wait(self):
        """Delete the pod and block until it's gone (max 60 s)."""
        try:
            await asyncio.to_thread(
                self._api.delete_namespaced_pod,
                name=self.pod_name,
                namespace=self.namespace,
                body=k8s_client.V1DeleteOptions(
                    grace_period_seconds=5, propagation_policy="Foreground"
                ),
            )
            # Wait for deletion
            for _ in range(60):
                try:
                    await asyncio.to_thread(
                        self._api.read_namespaced_pod,
                        name=self.pod_name,
                        namespace=self.namespace,
                    )
                    await asyncio.sleep(1)
                except ApiException as del_e:
                    if del_e.status == 404:
                        break
            else:
                raise RuntimeError(
                    f"Pod {self.pod_name} was not deleted in time."
                )
        except ApiException as del_e:
            if del_e.status != 404:
                raise RuntimeError(f"Failed to delete existing pod: {del_e}")
        finally:
            if self._client_manager is not None:
                self._client_manager.unregister_pod(self.pod_name)

    async def _apply_dockerfile_run_commands(self):
        """
        Parse the task's Dockerfile and exec its instructions inside the
        running pod. Bridges the gap between Docker (which builds the
        Dockerfile) and Kubernetes (which uses the base image directly).

        Two modes:

        1. **Legacy mode** (default — used when task.toml supplies a pre-built
           docker_image): only RUN instructions are executed; COPY / ADD /
           ENV are skipped; WORKDIR merely changes cwd (no mkdir). Default
           workdir is ``/testbed`` to match the swebench-style convention
           baked into existing pre-built images. Failures are logged as
           warnings, never raised. Behavior is preserved verbatim so tasks
           like harbor_swe_tasks (prebuilt `sweb.eval.*` images) see no change.

        2. **Inline-build mode** (enabled when docker_image was inferred from
           the Dockerfile's FROM line — i.e. no pre-built image is available):
           the Dockerfile is treated as a setup script. WORKDIR creates the
           directory and changes cwd; ENV exports variables and propagates
           them to subsequent RUN; COPY/ADD copy from ``/build-context``
           (a hostPath mount of environment_dir injected by _build_pod_spec);
           RUN executes in the current cwd with accumulated ENV exports.
           Default workdir is ``/``. Failures are logged but don't raise.
        """
        dockerfile = self.environment_dir / "Dockerfile"
        if not dockerfile.exists():
            return
        if self._skip_dockerfile_run:
            logger.info(
                "KubernetesEnvironment: skipping Dockerfile RUN re-execution — "
                "prebuilt OpenSWE image already contains the full environment."
            )
            return

        if self._inline_build:
            await self._apply_dockerfile_inline_build(dockerfile)
        else:
            await self._apply_dockerfile_run_commands_legacy(dockerfile)

    async def _apply_dockerfile_run_commands_legacy(self, dockerfile: Path):
        """Pre-existing behavior: RUN-only, default workdir /testbed, no
        mkdir on WORKDIR, COPY/ADD/ENV ignored. Kept verbatim to preserve
        the behavior tasks with pre-built images depend on.
        """
        lines = dockerfile.read_text().splitlines()
        workdir = "/testbed"  # default
        # Each entry is (workdir_at_that_point, cmd)
        run_commands: list[tuple[str, str]] = []

        i = 0
        while i < len(lines):
            stripped = lines[i].strip()

            # Track WORKDIR so RUN commands execute in the right directory
            if stripped.upper().startswith("WORKDIR "):
                workdir = stripped.split(None, 1)[1].strip()
                i += 1
                continue

            # Collect RUN commands (handle multi-line with trailing \)
            if stripped.upper().startswith("RUN "):
                cmd = stripped[4:]  # strip "RUN "
                while cmd.endswith("\\") and i + 1 < len(lines):
                    nxt = lines[i + 1].strip()
                    i += 1
                    # A `#` comment line inside a `\`-continuation is stripped by
                    # Docker; skip it (rather than flattening it onto the command,
                    # which comments out the rest and leaves a dangling `&&` ->
                    # syntax error, and drops the following real lines). Same fix
                    # as _parse_dockerfile_for_inline_build. For Dockerfiles
                    # WITHOUT mid-continuation comments (e.g. swerebench V2) this
                    # is byte-for-byte identical to the old space-join.
                    if nxt.startswith("#"):
                        continue
                    cmd = cmd[:-1] + " " + nxt
                if cmd.endswith("\\"):
                    cmd = cmd[:-1].rstrip()
                run_commands.append((workdir, cmd))

            i += 1

        # SWE-rebench V2 dataset workaround: every task's Dockerfile sets
        # WORKDIR /<reponame> (e.g. /mbed-tools), but the actual swerebench
        # base images keep the source under /testbed (standard SWE-bench
        # convention). Without this symlink the RUN commands below — and the
        # task's verifier test.sh, which also `cd /<reponame>` — all fail
        # with "No such file or directory". `[ ! -e ... ]` ensures we don't
        # overwrite paths that already exist for other datasets.
        seen_workdirs: set[str] = set()
        for cwd, _ in run_commands:
            if cwd and cwd != "/testbed" and cwd not in seen_workdirs:
                seen_workdirs.add(cwd)
                await self.exec(
                    f"if [ -d /testbed ] && [ ! -e {cwd} ]; then "
                    f"ln -s /testbed {cwd}; fi"
                )

        if not run_commands:
            return

        self.logger.debug(
            "Applying %d Dockerfile RUN command(s) in pod %s",
            len(run_commands),
            self.pod_name,
        )

        for cwd, cmd in run_commands:
            shell_cmd = f"cd {cwd} && {cmd}"
            result = await self.exec(shell_cmd)
            if result.return_code != 0:
                self.logger.warning(
                    "Dockerfile RUN command failed (exit %d) in pod %s: %s\n"
                    "stdout: %s\nstderr: %s",
                    result.return_code,
                    self.pod_name,
                    cmd,
                    result.stdout[:500] if result.stdout else "",
                    result.stderr[:500] if result.stderr else "",
                )
                # Don't raise — best-effort, same as test.sh fallbacks

    async def _apply_dockerfile_inline_build(self, dockerfile: Path):
        """Treat the Dockerfile as a setup script and execute it inside the
        running pod (RUN + WORKDIR + COPY/ADD + ENV). Triggered when
        docker_image was inferred from FROM. See _apply_dockerfile_run_commands
        docstring for the full behavior contract.
        """
        instructions = self._parse_dockerfile_for_inline_build(dockerfile)
        # Drop the leading FROM — it set the base image already (via _validate_definition).
        instructions = [(k, p) for k, p in instructions if k != "FROM"]
        if not instructions:
            return

        self.logger.info(
            "Inline-build: applying %d Dockerfile instruction(s) in pod %s",
            len(instructions),
            self.pod_name,
        )

        # Per-command deadline for inline-build RUN/COPY exec. WITHOUT this a hung
        # `pip install` (network/dep-resolution stall) blocks the exec reader thread
        # forever: the trial-level asyncio.wait_for cancels the await but cannot kill
        # the worker thread (see exec()), so the slot stays occupied until the pod's
        # activeDeadlineSeconds backstop. Passing timeout_sec makes _read_exec_output
        # self-terminate at the deadline, freeing the slot. Independent of the agent's
        # execution budget (different caller/phase). 0/negative disables.
        run_timeout = int(os.environ.get("HARBOR_ENV_BUILD_RUN_TIMEOUT_SEC", "1200"))
        if run_timeout <= 0:
            run_timeout = None

        workdir = "/"
        # Accumulated 'export K=V' statements, prepended to every subsequent RUN
        # so ENV propagates across exec boundaries (each exec is a fresh shell).
        env_exports: list[str] = []

        for kind, payload in instructions:
            if kind == "WORKDIR":
                target = payload.strip()
                if not target:
                    continue
                res = await self.exec(f"mkdir -p {shlex.quote(target)}")
                if res.return_code == 0:
                    workdir = target
                else:
                    self.logger.warning(
                        "Inline-build WORKDIR mkdir failed in pod %s: %s "
                        "(stderr=%s)",
                        self.pod_name,
                        target,
                        (res.stderr or "")[:300],
                    )
            elif kind == "ENV":
                for k, v in self._parse_dockerfile_env(payload):
                    # Use double quotes (not shlex.quote) so shell expands
                    # variable references like ${PATH} and ${GOPATH}.
                    # shlex.quote wraps in single quotes which prevents
                    # expansion and breaks PATH accumulation.
                    env_exports.append(f'export {k}="{v}"')
            elif kind in ("COPY", "ADD"):
                srcs, dst = self._parse_dockerfile_copy(payload)
                if not srcs or dst is None:
                    self.logger.warning(
                        "Inline-build %s: cannot parse '%s'; skipping",
                        kind, payload,
                    )
                    continue
                # Resolve dst relative to current WORKDIR if not absolute.
                dst_full = dst if dst.startswith("/") else f"{workdir.rstrip('/')}/{dst}"
                # Dockerfile convention: trailing slash OR multiple sources
                # means dst is a directory.
                is_dir_dst = dst.endswith("/") or len(srcs) > 1
                parent = dst_full if is_dir_dst else (os.path.dirname(dst_full) or "/")
                cp_cmds = [
                    f"cp -a {shlex.quote('/build-context/' + s.lstrip('/'))} "
                    f"{shlex.quote(dst_full)}"
                    for s in srcs
                ]
                shell = (
                    f"mkdir -p {shlex.quote(parent)} && " + " && ".join(cp_cmds)
                )
                res = await self.exec(shell, timeout_sec=run_timeout)
                if res.return_code != 0:
                    self.logger.warning(
                        "Inline-build %s failed (exit %d) in pod %s: %s\n"
                        "stdout: %s\nstderr: %s",
                        kind,
                        res.return_code,
                        self.pod_name,
                        payload,
                        (res.stdout or "")[:500],
                        (res.stderr or "")[:500],
                    )
            elif kind == "RUN":
                env_prefix = "; ".join(env_exports)
                if env_prefix:
                    env_prefix += "; "
                shell = (
                    f"cd {shlex.quote(workdir)} && {env_prefix}{payload}"
                )
                res = await self.exec(shell, timeout_sec=run_timeout)
                if res.return_code != 0:
                    self.logger.warning(
                        "Inline-build RUN failed (exit %d) in pod %s: %s\n"
                        "stdout: %s\nstderr: %s",
                        res.return_code,
                        self.pod_name,
                        payload,
                        (res.stdout or "")[-500:] if res.stdout else "",
                        (res.stderr or "")[-500:] if res.stderr else "",
                    )
            # FROM/ARG/LABEL/EXPOSE/USER/ENTRYPOINT/CMD/VOLUME/SHELL/ONBUILD/
            # STOPSIGNAL/HEALTHCHECK — ignored (no-op for setup).

    @staticmethod
    def _parse_dockerfile_for_inline_build(dockerfile: Path) -> list[tuple[str, str]]:
        """Lightweight Dockerfile lexer used by inline-build mode. Returns an
        ordered list of ``(KIND, payload)`` pairs preserving Dockerfile order.
        Handles backslash continuations and ``#`` comments. Does NOT handle
        JSON-form CMD/ENTRYPOINT arrays, HEREDOC, or ARG substitution — those
        are not needed by the SWE-style self-made tasks this targets.
        """
        instructions: list[tuple[str, str]] = []
        lines = dockerfile.read_text().splitlines()
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if not stripped or stripped.startswith("#"):
                i += 1
                continue
            full = stripped
            while full.endswith("\\") and i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                i += 1
                # Docker strips comment lines during preprocessing, so a `#`
                # line INSIDE a `\`-continuation neither terminates the
                # instruction nor becomes part of the shell command. The old
                # code flattened it onto the command line, where the `#`
                # commented out the rest (leaving a dangling `&&` -> "syntax
                # error: unexpected end of file") AND dropped the following
                # real lines as bogus standalone instructions. Skip the comment
                # and keep the current trailing backslash so the continuation
                # proceeds to the next real line.
                if nxt.startswith("#"):
                    continue
                full = full[:-1] + " " + nxt
            # Defensive: if the instruction ran off the end of the file still on
            # a continuation, drop the dangling backslash.
            if full.endswith("\\"):
                full = full[:-1].rstrip()
            parts = full.split(None, 1)
            if not parts:
                i += 1
                continue
            kind = parts[0].upper()
            payload = parts[1] if len(parts) > 1 else ""
            if kind in ("RUN", "WORKDIR", "ENV", "COPY", "ADD", "FROM"):
                instructions.append((kind, payload))
            i += 1
        return instructions

    @staticmethod
    def _parse_dockerfile_env(payload: str) -> list[tuple[str, str]]:
        """Parse the payload of an ENV instruction. Two forms:
            ENV K=V K2=V2 ...   (modern, multi-pair)
            ENV K rest of line  (legacy, single-var with spaces in value)
        Quoted values like K="multi word" are handled via shlex.
        """
        payload = payload.strip()
        if not payload:
            return []
        first_tok = payload.split(None, 1)[0]
        if "=" not in first_tok:
            # Legacy single-var form: ENV K val with spaces in value
            k, _, v = payload.partition(" ")
            return [(k.strip(), v.strip().strip('"').strip("'"))]
        # Modern K=V K2=V2 form — shlex to respect quoting
        try:
            tokens = shlex.split(payload, posix=True)
        except ValueError:
            tokens = payload.split()
        out: list[tuple[str, str]] = []
        for tok in tokens:
            if "=" in tok:
                k, _, v = tok.partition("=")
                out.append((k, v))
        return out

    @staticmethod
    def _parse_dockerfile_copy(payload: str) -> tuple[list[str], Optional[str]]:
        """Parse ``COPY <src>... <dst>``. Ignores leading ``--flags`` (--chown=,
        --from=, --chmod=). Returns ``([src,...], dst)`` or ``([], None)`` if
        the line cannot be parsed.
        """
        tokens = payload.split()
        args = [t for t in tokens if not t.startswith("--")]
        if len(args) < 2:
            return [], None
        return args[:-1], args[-1]

    async def _ensure_musl_compatibility(self):
        """
        Ensure musl-based containers (Alpine Linux) have the toolchain
        needed for Node.js installation and native-addon compilation.

        The claude-code agent setup installs nvm which *overrides* any
        system Node.js and downloads specific versions.  On Alpine this
        fails because:

          1. Official Node.js does not publish musl prebuilt binaries (404).
          2. nvm falls back to source compilation, which needs gcc/g++/make.
          3. nvm.sh runs under ``set -u`` and crashes if TMPDIR is unbound.

        This method always installs the build toolchain on Alpine — even
        when a system ``node`` already exists — because nvm will bypass it.
        It also writes ``TMPDIR`` and ``NVM_NODEJS_ORG_MIRROR`` (pointing
        at the unofficial-builds mirror that *does* ship musl binaries)
        into shell profiles so every bash session picks them up.
        """
        detect = await self.exec(
            "cat /etc/os-release 2>/dev/null | grep -qi alpine && echo ALPINE || echo OTHER"
        )
        if "ALPINE" not in (detect.stdout or ""):
            return

        self.logger.debug(
            "Detected Alpine (musl) container in pod %s — "
            "installing build tools and configuring Node.js musl compatibility",
            self.pod_name,
        )

        result = await self.exec(
            "apk add --no-cache "
            "gcc g++ make python3 musl-dev linux-headers"
        )
        if result.return_code != 0:
            self.logger.warning(
                "apk install (build tools) failed in pod %s (exit %d): %s",
                self.pod_name,
                result.return_code,
                (result.stderr or "")[:500],
            )

        # Write env vars into shell profiles so they survive across exec
        # invocations, subshells, and nvm's own bash processes.
        # - TMPDIR: prevents nvm.sh crash under ``set -u``
        # - NVM_NODEJS_ORG_MIRROR: unofficial-builds provides musl binaries
        profile_script = (
            'export TMPDIR="${TMPDIR:-/tmp}"\n'
            'export NVM_NODEJS_ORG_MIRROR="${NVM_NODEJS_ORG_MIRROR:-https://unofficial-builds.nodejs.org/download/release}"\n'
        )
        await self.exec(
            f"mkdir -p /etc/profile.d && printf %s '{profile_script}' > /etc/profile.d/musl-node-compat.sh"
        )
        await self.exec(
            f"printf '\\n{profile_script}' >> /root/.bashrc 2>/dev/null; "
            f"printf '\\n{profile_script}' >> /root/.profile 2>/dev/null; "
            "true"
        )

    async def start(self, force_build: bool):
        """Create pod with pre-built image and wait until ready."""
        # Initialize Kubernetes client via singleton manager
        await self._ensure_client()

        # Hybrid build approach: build only if needed
        if force_build:
            self.logger.warning(
                "force_build=True is ignored by KubernetesEnvironment — "
                "pre-built images are used directly (set task_env_config.docker_image)."
            )

        pod = self._build_pod_spec()

        # Register the pod with the process-wide registry *before* creating
        # it so that if we crash between the create call returning and the
        # normal stop() path running, the atexit / signal handlers still
        # know to tear it down. (Registering an unknown pod is harmless —
        # the emergency cleanup tolerates 404s.)
        assert self._client_manager is not None
        self._client_manager.register_pod(
            self.pod_name, self.namespace, self.kubeconfig_path
        )

        async def create_pod_with_retry():
            retry_delays = (10, 30, 90)
            for attempt in range(len(retry_delays) + 1):
                try:
                    await asyncio.to_thread(
                        self._api.create_namespaced_pod,
                        namespace=self.namespace,
                        body=pod,
                    )
                    return
                except ApiException as e:
                    if e.status != 500 or attempt == len(retry_delays):
                        raise
                    delay = retry_delays[attempt]
                    self.logger.warning(
                        "Failed to create pod %s with Kubernetes API 500 "
                        "(attempt %d/%d); retrying in %ss: %s",
                        self.pod_name,
                        attempt + 1,
                        len(retry_delays) + 1,
                        delay,
                        e,
                    )
                    await asyncio.sleep(delay)

        try:
            await create_pod_with_retry()
        except ApiException as e:
            if e.status == 409:  # Already exists
                self.logger.debug(f"Pod {self.pod_name} already exists, recreating...")
                # Delete existing pod inline (don't call stop() as it releases the client)
                await self._delete_pod_and_wait()
                # _delete_pod_and_wait unregistered the pod; re-register for
                # the incoming recreate so the crash-safety net is restored.
                self._client_manager.register_pod(
                    self.pod_name, self.namespace, self.kubeconfig_path
                )
                try:
                    await create_pod_with_retry()
                except Exception:
                    self._client_manager.unregister_pod(self.pod_name)
                    raise
            else:
                # Never-created pod — drop the registration to avoid a
                # spurious delete attempt at exit.
                self._client_manager.unregister_pod(self.pod_name)
                raise RuntimeError(f"Failed to create pod: {e}")
        except Exception:
            self._client_manager.unregister_pod(self.pod_name)
            raise

        # Wait for pod to be ready
        await self._wait_for_pod_ready(timeout_sec=1800)
        await self._wait_for_container_exec_ready()

        # Run Dockerfile RUN commands that were skipped because K8s uses
        # pre-built images directly without building the Dockerfile.
        await self._apply_dockerfile_run_commands()

        # On Alpine (musl) containers, install Node.js and build tools so
        # that agent setup (nvm) does not fail.
        await self._ensure_musl_compatibility()

        # Create required directories and make them world-writable so
        # non-root agent/verifier users can write to them.
        # Right after the pod reports Ready the exec channel is occasionally
        # still flaky and a single mkdir can return non-zero (or the API drops
        # the exec). Previously the first failure raised "Failed to create log
        # directories", turning a transient hiccup into a hard env failure.
        # Retry a few times before giving up.
        mkdir_cmd = (
            f"mkdir -p {EnvironmentPaths.agent_dir} {EnvironmentPaths.verifier_dir} && "
            f"chmod 777 {EnvironmentPaths.agent_dir} {EnvironmentPaths.verifier_dir}"
        )
        mkdir_result = None
        for _attempt in range(3):
            mkdir_result = await self.exec(mkdir_cmd)
            if mkdir_result.return_code == 0:
                break
            await asyncio.sleep(2)
        if mkdir_result is None or mkdir_result.return_code != 0:
            raise RuntimeError(
                f"Failed to create log directories in pod {self.pod_name}: "
                f"stdout={mkdir_result.stdout if mkdir_result else ''}, "
                f"stderr={mkdir_result.stderr if mkdir_result else ''}"
            )

        # Anti-reward-hacking (git history): SINGLE-COMMIT REBUILD. After env
        # setup / before inference, strip ALL git history (save it to .git.orig,
        # re-init a fresh 1-commit repo from the working tree). This is the only
        # robust way to kill the gold-fix leak (hacking.md Mode D/E): a command
        # blacklist cannot — the fix is reachable via git log -p / blame / archive
        # / checkout <sha> / worktree / format-patch / ... (open-ended). Works for
        # BOTH image structures: SWE-bench (fix is an ANCESTOR of HEAD) and openswe
        # (fix on side branches refs/remotes/origin/*) — the fresh repo has neither.
        # The working tree is untouched so the agent still solves; the original
        # .git is restored before the verifier (restore_git_history(), called from
        # trial.py's verifier phase) so `git checkout <historical_sha> <testfile>`
        # works. Idempotent, gated by HARBOR_BLOCK_GIT_HISTORY_LEAK.
        await self._rebuild_git_single_commit()

        # Anti-reward-hacking (network): block the pod's PUBLIC egress (github /
        # raw.github / pypi answer-fetch) while keeping the private-LAN LLM proxy /
        # registry / DNS reachable. Runs AFTER env setup (musl/nvm installs, any
        # build-time github) so nothing legitimate breaks; the agent then runs
        # isolated. Enforced via the netadmin sidecar (see apply_network_policy).
        await self._apply_default_egress_policy()

    # Shell run at the end of setup(). Wraps every git on PATH to refuse (a) the
    # network ops clone/fetch/pull/remote/submodule AND (b) the history-exploration
    # ops that can reach the post-base-commit gold code from the LOCAL .git (the
    # eval/openswe images retain the fix commit, so `git log --all | grep <sha>` +
    # `git show <sha>` reads the answer with NO network — hacking.md Mode D/E):
    # log --all/--branches/--tags/--remotes/--reflog, show <sha>, cat-file,
    # rev-list, for-each-ref, ls-remote, reflog, diff <a>..<b>/<a> <b>,
    # branch --contains. The local git the agent AND verifier actually need still
    # works: plain `git log`, `git log -- <path>`, `git show HEAD:<path>`,
    # `git diff`/`git diff HEAD`, and crucially `git checkout <sha> <testfile>` +
    # `git apply` (the verifier resets the test file from a pinned commit then
    # applies the hidden test patch — validated against a real SWE-bench test.sh).
    # The subcommand is located AFTER git's global options, so `git --no-pager log
    # --all` / `git -C dir log --all` no longer bypass it. Idempotent
    # (harbor-anti-hack guard), best-effort, toggle with HARBOR_BLOCK_GIT_HISTORY_LEAK=0.
    # NOTE: the network vectors (curl/wget/python egress to github, Mode A/C/F) are
    # handled separately by the iptables egress sidecar (_apply_default_egress_policy),
    # so this shim is git-only. RESIDUAL (a string blacklist can't cleanly separate):
    # `git checkout <sha> -- .` / `git reset --hard <sha>` stay (the verifier needs
    # them). Validated on 240 with real git 2.43. Keep BYTE-IDENTICAL to the
    # docker.py twin.
    _DISABLE_GIT_OPS_SH = r"""
# 1) drop upstream remotes so fetch/pull have nothing to talk to
for g in $(find /testbed /workspace /repo /app -maxdepth 3 -type d -name .git 2>/dev/null); do
  r=$(dirname "$g")
  for rem in $(git -C "$r" remote 2>/dev/null); do
    git -C "$r" remote remove "$rem" 2>/dev/null
  done
done
# 2) wrap every git on PATH so network + history-leak ops are refused, while the
#    local git the agent/verifier need keeps working. The shim body is written via
#    a QUOTED heredoc (no expansion) so it matches the tested logic byte-for-byte;
#    only the real-binary path is injected on a separate __REAL= line.
for real in $(command -v git 2>/dev/null) $(which -a git 2>/dev/null) /usr/bin/git /bin/git; do
  [ -x "$real" ] || continue
  grep -q harbor-anti-hack "$real" 2>/dev/null && continue
  [ -e "$real.real" ] || cp -p "$real" "$real.real" 2>/dev/null || continue
  rm -f "$real" 2>/dev/null
  {
    printf '#!/usr/bin/env bash\n'
    echo "__REAL=\"$real.real\""
    cat <<'HARBOR_EOF'
# harbor-anti-hack
__blk(){ echo "git $1 is disabled in this environment (anti-reward-hacking)." >&2; exit 128; }
__i=1; __sub=""
while [ "$__i" -le "$#" ]; do
  eval "__a=\${$__i}"
  case "$__a" in
    -C|-c|--git-dir|--work-tree|--namespace|--exec-path|--super-prefix) __i=$((__i+2)); continue ;;
    --git-dir=*|--work-tree=*|--namespace=*|--exec-path=*|--super-prefix=*|-c=*) __i=$((__i+1)); continue ;;
    --) __i=$((__i+1)); eval "__sub=\${$__i}"; break ;;
    -*) __i=$((__i+1)); continue ;;
    *) __sub="$__a"; break ;;
  esac
done
__all="$*"
case "$__sub" in
  clone|fetch|pull|remote|submodule) __blk "$__sub" ;;
  branch)
    case " $__all " in *" --contains"*) __blk "branch --contains" ;; esac ;;
  cat-file|rev-list|for-each-ref|ls-remote|reflog) __blk "$__sub" ;;
  log)
    case " $__all " in
      *" --all"*|*" --branches"*|*" --tags"*|*" --remotes"*|*" --reflog"*|*" --source"*) __blk "log(cross-ref)" ;;
    esac ;;
  show)
    __j=$((__i+1))
    while [ "$__j" -le "$#" ]; do
      eval "__t=\${$__j}"; __j=$((__j+1))
      case "$__t" in -*) continue ;; esac
      __b=${__t%%:*}; __b=${__b%%^*}; __b=${__b%%\~*}
      if printf '%s' "$__b" | grep -qiE '^[0-9a-f]{7,40}$'; then __blk "show <commit>"; fi
    done ;;
  diff)
    __cnt=0; __j=$((__i+1))
    while [ "$__j" -le "$#" ]; do
      eval "__t=\${$__j}"; __j=$((__j+1))
      case "$__t" in
        *..*) printf '%s' "$__t" | grep -qiE '^[0-9a-f]{7,40}(\^|~[0-9]+)?\.\.\.?[0-9a-f]{7,40}' && __blk "diff <a>..<b>" ;;
        -*) : ;;
        *) __b=${__t%%^*}; __b=${__b%%\~*}
           if printf '%s' "$__b" | grep -qiE '^[0-9a-f]{7,40}$'; then __cnt=$((__cnt+1)); [ "$__cnt" -ge 2 ] && __blk "diff <a> <b>"; fi ;;
      esac
    done ;;
esac
exec "$__REAL" "$@"
HARBOR_EOF
  } > "$real"
  chmod +x "$real"
done
true
"""

    async def _disable_history_leak_git_ops(self):
        """Disable the git operations that could recover post-base-commit (gold)
        code: clone/fetch/pull/remote/submodule, plus removal of upstream remotes.
        Read-only/local git (log/show/diff/checkout/apply) still works, so the
        agent and verifier are unaffected. Other github access used during build
        (e.g. pip installing unrelated packages) is untouched because this runs
        only after setup completes.

        Toggle off with HARBOR_BLOCK_GIT_HISTORY_LEAK=0.
        """
        if os.environ.get("HARBOR_BLOCK_GIT_HISTORY_LEAK", "1") not in ("1", "true", "True"):
            return
        try:
            result = await self.exec(self._DISABLE_GIT_OPS_SH)
            if result.return_code != 0:
                self.logger.warning(
                    f"[anti-hack] disabling git ops in pod {self.pod_name} returned "
                    f"{result.return_code}: {result.stderr[:300]}"
                )
            else:
                self.logger.debug(f"[anti-hack] git ops disabled in {self.pod_name}")
        except Exception as e:
            # Never fail a rollout because hardening could not be applied.
            self.logger.warning(f"[anti-hack] could not disable git ops in {self.pod_name}: {e}")

    # -------------------------------------------------- git single-commit rebuild
    # Agent-phase: move .git aside and re-init a fresh 1-commit repo from the
    # working tree. Kills the gold-fix leak at the source (no history => nothing
    # to read via log -p / show / checkout / archive / worktree / ...). The
    # working tree is untouched so the agent still solves. Idempotent: skips a repo
    # that already has a .git.orig. `git add -A` respects .gitignore.
    _REBUILD_GIT_SH = r"""
for g in $(find /testbed /workspace /repo /app -maxdepth 3 -type d -name .git 2>/dev/null); do
  r=$(dirname "$g")
  [ -e "$r/.git.orig" ] && continue
  ( cd "$r" \
    && mv .git .git.orig \
    && git init -q \
    && git add -A \
    && git -c user.email=harbor@anti.hack -c user.name=harbor -c commit.gpgsign=false \
         commit -q -m "base (anti-reward-hacking: history stripped for agent phase)" \
  ) >/dev/null 2>&1 || { [ -d "$r/.git.orig" ] && [ ! -e "$r/.git" ] && mv "$r/.git.orig" "$r/.git" 2>/dev/null; }
done
true
"""

    # Verifier-phase: restore the original history so the verifier's
    # `git checkout <historical_sha> <testfile>` + `git apply <test_patch>` work.
    _RESTORE_GIT_SH = r"""
for o in $(find /testbed /workspace /repo /app -maxdepth 3 -type d -name .git.orig 2>/dev/null); do
  r=$(dirname "$o")
  ( cd "$r" && rm -rf .git && mv .git.orig .git ) >/dev/null 2>&1 || true
done
true
"""

    async def _rebuild_git_single_commit(self):
        """Strip git history (single-commit rebuild) so the agent cannot reach the
        gold fix. Saves the original .git to .git.orig; restore_git_history()
        puts it back before the verifier. Gated by HARBOR_BLOCK_GIT_HISTORY_LEAK,
        best-effort (never fails a rollout)."""
        if os.environ.get("HARBOR_BLOCK_GIT_HISTORY_LEAK", "1") not in ("1", "true", "True"):
            return
        try:
            result = await self.exec(self._REBUILD_GIT_SH)
            if result.return_code != 0:
                self.logger.warning(
                    f"[anti-hack] git single-commit rebuild in {self.pod_name} "
                    f"returned {result.return_code}: {(result.stderr or '')[:300]}"
                )
            else:
                self.logger.debug(f"[anti-hack] git history stripped in {self.pod_name}")
        except Exception as e:
            self.logger.warning(
                f"[anti-hack] could not rebuild git in {self.pod_name}: {e}"
            )

    async def restore_git_history(self) -> None:
        """Restore the original .git (saved by _rebuild_git_single_commit) so the
        verifier sees the full history. Called from trial.py's verifier phase.
        No-op if history was not stripped (idempotent, best-effort)."""
        if os.environ.get("HARBOR_BLOCK_GIT_HISTORY_LEAK", "1") not in ("1", "true", "True"):
            return
        try:
            await self.exec(self._RESTORE_GIT_SH)
        except Exception as e:
            self.logger.warning(
                f"[anti-hack] could not restore git in {self.pod_name}: {e}"
            )

    # ------------------------------------------------------------------ egress
    def _build_egress_iptables(self, policy: NetworkPolicy) -> str:
        """Return the sh script (run in the netadmin sidecar) that enforces
        ``policy`` on the shared pod netns. PUBLIC flushes to allow-all;
        ALLOWLIST allows lo + DNS + the private LAN (RFC1918: LLM proxy /
        registry / cluster services) + any explicit allowed_hosts and DROPs the
        rest (public internet = the github/raw/pypi answer-fetch vector);
        NO_NETWORK allows only lo + DNS."""
        dns = shlex.quote(self._CLUSTER_DNS_IP)
        if policy.network_mode == NetworkMode.PUBLIC:
            return (
                "iptables -P OUTPUT ACCEPT; iptables -F OUTPUT; "
                "ip6tables -P OUTPUT ACCEPT 2>/dev/null; ip6tables -F OUTPUT 2>/dev/null; true"
            )
        lines = [
            "iptables -F OUTPUT",
            "iptables -P OUTPUT ACCEPT",  # permissive while we build the ruleset
            "iptables -A OUTPUT -o lo -j ACCEPT",
            "iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT",
            f"iptables -A OUTPUT -p udp -d {dns} --dport 53 -j ACCEPT",
            f"iptables -A OUTPUT -p tcp -d {dns} --dport 53 -j ACCEPT",
        ]
        if policy.network_mode == NetworkMode.ALLOWLIST:
            # Private LAN is always reachable: the in-cluster LLM proxy (training
            # node), the shared registry, and cluster services all live here.
            for cidr in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"):
                lines.append(f"iptables -A OUTPUT -d {cidr} -j ACCEPT")
            # Explicit extra hosts (e.g. a public LLM endpoint), resolved to IPv4.
            for host in policy.allowed_hosts:
                h = shlex.quote(host)
                lines.append(
                    f"for ip in $(getent ahostsv4 {h} 2>/dev/null | awk '{{print $1}}' | sort -u); "
                    f'do iptables -A OUTPUT -d "$ip" -j ACCEPT; done'
                )
        # NO_NETWORK: only lo + DNS above. Then drop everything else.
        lines.append("iptables -P OUTPUT DROP")
        # No IPv6 allowlist support -> block v6 egress entirely (keep lo).
        lines.append(
            "ip6tables -F OUTPUT 2>/dev/null; ip6tables -A OUTPUT -o lo -j ACCEPT 2>/dev/null; "
            "ip6tables -P OUTPUT DROP 2>/dev/null; true"
        )
        return "; ".join(lines)

    async def apply_network_policy(self, policy: NetworkPolicy) -> None:
        """Enforce ``policy`` by (re-)running iptables in the netadmin sidecar.
        Called per phase by the trial layer (online eval) and once at the end of
        setup() for the RL path (see _apply_default_egress_policy). No-op if
        isolation is disabled."""
        if not self._egress_isolation_enabled:
            # Fall back to the base behavior: only PUBLIC is acceptable.
            if policy.network_mode != NetworkMode.PUBLIC:
                raise ValueError(
                    f"network_mode={policy.network_mode.value!r} requested but "
                    "HARBOR_K8S_EGRESS_ISOLATION=0 (sidecar disabled)."
                )
            return
        script = self._build_egress_iptables(policy)
        try:
            result = await self.exec(
                script,
                container=self._NETADMIN_SIDECAR_NAME,
                command_shell="sh",
                timeout_sec=60,
            )
            if result.return_code != 0:
                self.logger.warning(
                    "[egress] apply_network_policy(%s) rc=%s in %s: %s",
                    policy.network_mode.value,
                    result.return_code,
                    self.pod_name,
                    (result.stderr or "")[:300],
                )
            else:
                self.logger.info(
                    "[egress] applied network_mode=%s in %s",
                    policy.network_mode.value,
                    self.pod_name,
                )
        except Exception as e:
            # Never fail a rollout because egress hardening could not be applied.
            self.logger.warning(
                f"[egress] could not apply network policy in {self.pod_name}: {e}"
            )

    async def _apply_default_egress_policy(self):
        """RL-path hook (setup() bypasses the trial.py per-phase machinery): apply
        the agent-phase policy once, after env setup / before inference. Default =
        ALLOWLIST (private-LAN only) unless the task explicitly declared
        [environment] network_mode = public (opt-out)."""
        if not self._egress_isolation_enabled:
            return
        raw_mode = getattr(self.task_env_config, "network_mode", None)
        if raw_mode == NetworkMode.PUBLIC:
            return  # task opted out
        allowed = list(getattr(self.task_env_config, "allowed_hosts", []) or [])
        mode = raw_mode or NetworkMode.ALLOWLIST
        policy = NetworkPolicy(
            network_mode=mode,
            allowed_hosts=allowed if mode == NetworkMode.ALLOWLIST else [],
        )
        await self.apply_network_policy(policy)

    async def stop(self, delete: bool):
        """Stop/delete the pod."""
        if self._client_manager is None:
            return

        pod_deleted_cleanly = False
        try:
            if delete:
                try:
                    await asyncio.to_thread(
                        self._api.delete_namespaced_pod,
                        name=self.pod_name,
                        namespace=self.namespace,
                        body=k8s_client.V1DeleteOptions(
                            grace_period_seconds=5,
                            propagation_policy="Foreground",
                        ),
                    )
                    # Wait for pod to be deleted
                    for _ in range(60):
                        try:
                            await asyncio.to_thread(
                                self._api.read_namespaced_pod,
                                name=self.pod_name,
                                namespace=self.namespace,
                            )
                            await asyncio.sleep(1)
                        except ApiException as e:
                            if e.status == 404:
                                pod_deleted_cleanly = True
                                break
                    else:
                        self.logger.warning(
                            f"Pod {self.pod_name} did not terminate within 60 seconds."
                        )
                except ApiException as e:
                    if e.status == 404:
                        pod_deleted_cleanly = True
                    else:
                        raise
            else:
                # Caller asked us to leave the pod running — drop our
                # ownership so the emergency cleanup does not race with them.
                pod_deleted_cleanly = True
        finally:
            # Only drop the registry entry if we are confident the pod is
            # gone (or explicitly abandoned). Otherwise leave it registered
            # so atexit / signal handlers can retry the deletion.
            if pod_deleted_cleanly and self._client_manager is not None:
                self._client_manager.unregister_pod(self.pod_name)

            # Release the client reference (actual cleanup happens at program exit)
            if self._client_manager:
                try:
                    await self._client_manager.release_client()
                except Exception as e:
                    self.logger.error(f"Error releasing Kubernetes client: {e}")
                finally:
                    self._client_manager = None
                    self._core_api = None

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
        container: str | None = None,
        command_shell: str = "bash",
    ) -> ExecResult:
        """Execute command in pod using kubectl exec equivalent.

        container=None targets the default (main) container; pass a name to exec
        into a specific container (e.g. the netadmin egress sidecar).
        command_shell selects the in-container shell used to run ``command``."""
        # No implicit deadline: timeout_sec=None means "no timeout" (read until the command
        # returns) and falls into the no-deadline branch below. Callers that need a bound pass
        # timeout_sec explicitly (e.g. the agent loop passes its trajectory budget); the trial
        # layer's asyncio.wait_for remains the outer backstop.
        user = self._resolve_user(user)
        env = self._merge_env(env)

        await self._ensure_client()

        if command_shell not in ("bash", "sh"):
            raise ValueError(f"Unsupported command_shell={command_shell!r}")

        full_command = f"{command_shell} -c {shlex.quote(command)}"

        if env:
            for key, value in env.items():
                full_command = f"{key}={shlex.quote(value)} {full_command}"

        if cwd:
            full_command = f"cd {cwd} && {full_command}"

        if user is not None:
            # su requires a username; resolve numeric UIDs via getent
            if isinstance(user, int):
                user_arg = f"$(getent passwd {user} | cut -d: -f1)"
            else:
                user_arg = shlex.quote(user)
            # Use su (not su -) to preserve the working directory
            full_command = f"su {user_arg} -s /bin/{command_shell} -c {shlex.quote(full_command)}"

        exec_command = ["sh", "-c", full_command]
        # print(f'[DEBUG] k8s exec command: {exec_command}')

        resp = None
        try:
            resp = await asyncio.to_thread(
                stream,
                self._api.connect_get_namespaced_pod_exec,
                self.pod_name,
                self.namespace,
                command=exec_command,
                container=(container or self._MAIN_CONTAINER_NAME),
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False,
                _preload_content=False,
            )

            if timeout_sec and timeout_sec > 0:
                # Pass the deadline INTO _read_exec_output as well: wait_for cancels the await
                # on timeout but cannot kill the worker thread, so the reader must self-terminate
                # (close the stream + raise) — otherwise a hung command leaks a thread that
                # spins forever. The outer wait_for is a small-margin backstop.
                stdout, stderr = await asyncio.wait_for(
                    asyncio.to_thread(self._read_exec_output, resp, timeout_sec),
                    timeout=timeout_sec + 30,
                )
            else:
                stdout, stderr = await asyncio.to_thread(self._read_exec_output, resp)

            resp.run_forever(timeout=0)
            return_code = resp.returncode if resp.returncode is not None else 0

            return ExecResult(
                stdout=stdout,
                stderr=stderr,
                return_code=return_code,
            )

        except asyncio.TimeoutError:
            self.logger.warning(
                "exec command exceeded %.0fs timeout in pod %s; killing. command=%r",
                timeout_sec, self.pod_name, command[:200],
            )
            return ExecResult(
                stdout=None,
                stderr=f"Command timed out after {timeout_sec} seconds",
                return_code=124,
            )
        except ApiException as e:
            if e.status == 404:
                return ExecResult(
                    stdout=None,
                    stderr=f"Pod {self.pod_name} not found (404).",
                    return_code=1,
                )
            elif e.status == 500:
                error_body = str(e.body) if hasattr(e, "body") else str(e)
                if "No agent available" in error_body:
                    return ExecResult(
                        stdout=None,
                        stderr=f"Pod {self.pod_name} unavailable: No agent available.",
                        return_code=1,
                    )
                return ExecResult(
                    stdout=None,
                    stderr=f"Internal server error on pod {self.pod_name}: {e.reason}",
                    return_code=1,
                )
            else:
                return ExecResult(
                    stdout=None,
                    stderr=f"API error ({e.status}) on pod {self.pod_name}: {e.reason}",
                    return_code=1,
                )
        except Exception as e:
            return ExecResult(
                stdout=None,
                stderr=str(e),
                return_code=1,
            )
        finally:
            if resp is not None:
                try:
                    resp.close()
                except Exception:
                    pass

    def _read_exec_output(self, resp, deadline_sec=None):
        """Read output from exec stream.

        ``resp.update(timeout=1)`` bounds each poll, but the ``while resp.is_open()`` loop has
        no total bound — a command that never finishes (the exec stream stays open) spins here
        forever. When ``deadline_sec`` is set, stop after that many seconds: close the stream and
        raise ``TimeoutError`` so this worker thread actually exits (the outer ``asyncio.wait_for``
        can cancel the await but cannot kill this thread, so the bound must live here too).
        """
        stdout = ""
        stderr = ""
        end = (time.monotonic() + deadline_sec) if deadline_sec and deadline_sec > 0 else None

        while resp.is_open():
            resp.update(timeout=1)
            if resp.peek_stdout():
                stdout += resp.read_stdout()
            if resp.peek_stderr():
                stderr += resp.read_stderr()
            if end is not None and time.monotonic() >= end:
                try:
                    resp.close()
                except Exception:
                    pass
                raise TimeoutError(f"exec command exceeded {deadline_sec:.0f}s deadline")

        return stdout, stderr

    async def _wait_for_container_exec_ready(self, max_attempts: int = 60):
        """Wait for container to be ready for exec operations."""
        for attempt in range(max_attempts):
            try:
                test_command = ["true"]
                resp = await asyncio.to_thread(
                    stream,
                    self._api.connect_get_namespaced_pod_exec,
                    self.pod_name,
                    self.namespace,
                    container=self._MAIN_CONTAINER_NAME,
                    command=test_command,
                    stderr=False,
                    stdin=False,
                    stdout=True,
                    tty=False,
                    _preload_content=False,
                )
                resp.close()
                return
            except ApiException as e:
                if "container not found" in str(e) or e.status == 500:
                    if attempt % 10 == 0:
                        self.logger.debug(
                            f"Container not ready, attempt {attempt + 1}/{max_attempts}"
                        )
                    await asyncio.sleep(3)
                    continue
                else:
                    raise
            except Exception as e:
                if attempt < max_attempts - 1:
                    if attempt % 10 == 0:
                        self.logger.debug(f"Error checking container readiness: {e}")
                    await asyncio.sleep(3)
                    continue
                else:
                    raise

        raise RuntimeError(
            f"Container not ready for exec after {max_attempts} attempts"
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def upload_file(self, source_path: Path | str, target_path: str):
        """Upload file using kubectl cp equivalent."""
        await self._ensure_client()

        await self._wait_for_container_exec_ready()

        source_path = Path(source_path)

        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            tar.add(str(source_path), arcname=Path(target_path).name)
        tar_buffer.seek(0)

        target_dir = str(Path(target_path).parent)
        await self.exec(f"mkdir -p {target_dir}", user="root")

        exec_command = ["tar", "xf", "-", "-C", target_dir]

        resp = await asyncio.to_thread(
            stream,
            self._api.connect_get_namespaced_pod_exec,
            self.pod_name,
            self.namespace,
            container=self._MAIN_CONTAINER_NAME,
            command=exec_command,
            stderr=True,
            stdin=True,
            stdout=True,
            tty=False,
            _preload_content=False,
        )

        resp.write_stdin(tar_buffer.read())
        resp.run_forever(timeout=1)
        resp.close()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        """Upload directory using kubectl cp equivalent."""
        await self._ensure_client()

        await self._wait_for_container_exec_ready()

        source_dir = Path(source_dir)

        files_to_upload = []
        for item in source_dir.rglob("*"):
            if item.is_file():
                arcname = str(item.relative_to(source_dir))
                files_to_upload.append(arcname)

        if not files_to_upload:
            self.logger.warning(f"No files to upload from {source_dir}")
            return

        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            for item in source_dir.rglob("*"):
                if item.is_file():
                    arcname = str(item.relative_to(source_dir))
                    tar.add(str(item), arcname=arcname)
        tar_buffer.seek(0)
        tar_size = len(tar_buffer.getvalue())

        mkdir_result = await self.exec(f"mkdir -p {target_dir}", user="root")
        if mkdir_result.return_code != 0:
            raise RuntimeError(
                f"Failed to create target directory {target_dir}: {mkdir_result.stderr}"
            )

        exec_command = ["tar", "xf", "-", "-C", target_dir]

        try:
            resp = await asyncio.to_thread(
                stream,
                self._api.connect_get_namespaced_pod_exec,
                self.pod_name,
                self.namespace,
                container=self._MAIN_CONTAINER_NAME,
                command=exec_command,
                stderr=True,
                stdin=True,
                stdout=True,
                tty=False,
                _preload_content=False,
            )
        except ApiException as e:
            if e.status == 500:
                raise RuntimeError(
                    f"Pod {self.pod_name} returned 500 error during upload."
                )
            raise

        try:
            resp.write_stdin(tar_buffer.read())
        except Exception as e:
            raise RuntimeError(f"Failed to write tar data to pod {self.pod_name}: {e}")

        resp.run_forever(timeout=1)
        resp.close()
        self.logger.debug(
            f"Successfully uploaded {len(files_to_upload)} files ({tar_size} bytes) to {target_dir}"
        )

    def _read_exec_stream(
        self, resp, *, drain_attempts_after_close: int = 2
    ) -> tuple[bytes, str]:
        """Read all stdout/stderr from a pod exec stream, including trailing chunks.

        Kubernetes exec websocket streams can close immediately after the final data
        frame is delivered. Continue draining for a couple of iterations after the
        connection reports closed so we do not drop the tail of tar streams.
        """
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[str] = []
        close_drains_remaining = drain_attempts_after_close

        try:
            while True:
                resp.update(timeout=1)

                saw_output = False
                while resp.peek_stdout():
                    data = resp.read_stdout()
                    if isinstance(data, str):
                        data = data.encode("utf-8", errors="surrogateescape")
                    stdout_chunks.append(data)
                    saw_output = True

                while resp.peek_stderr():
                    stderr_chunks.append(resp.read_stderr())
                    saw_output = True

                if resp.is_open():
                    close_drains_remaining = drain_attempts_after_close
                    continue

                if saw_output:
                    close_drains_remaining = drain_attempts_after_close
                    continue

                close_drains_remaining -= 1
                if close_drains_remaining <= 0:
                    break
        finally:
            resp.close()

        return b"".join(stdout_chunks), "".join(stderr_chunks)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def download_file(self, source_path: str, target_path: Path | str):
        """Download file from pod."""
        await self._ensure_client()

        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        exec_command = ["tar", "cf", "-", source_path]

        resp = await asyncio.to_thread(
            stream,
            self._api.connect_get_namespaced_pod_exec,
            self.pod_name,
            self.namespace,
            container=self._MAIN_CONTAINER_NAME,
            command=exec_command,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
            _preload_content=False,
        )

        tar_data, stderr_data = self._read_exec_stream(resp)

        if stderr_data and "Cannot stat" in stderr_data:
            raise RuntimeError(
                f"Failed to access file {source_path} in pod {self.pod_name}: {stderr_data.strip()}"
            )
        if not tar_data:
            raise RuntimeError(
                f"No data received when downloading {source_path} from pod {self.pod_name}."
            )

        tar_buffer = io.BytesIO(tar_data)
        with tarfile.open(fileobj=tar_buffer, mode="r") as tar:
            for member in tar.getmembers():
                if member.name == source_path or member.name.startswith(
                    source_path.lstrip("/")
                ):
                    member.name = target_path.name
                    tar.extract(member, path=str(target_path.parent))
                    break

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def download_dir(self, source_dir: str, target_dir: Path | str):
        """Download directory from pod."""
        await self._ensure_client()

        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        exec_command = ["sh", "-c", f"cd {source_dir} && tar cf - ."]

        try:
            resp = await asyncio.to_thread(
                stream,
                self._api.connect_get_namespaced_pod_exec,
                self.pod_name,
                self.namespace,
                container=self._MAIN_CONTAINER_NAME,
                command=exec_command,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False,
                _preload_content=False,
            )
        except ApiException as e:
            if e.status == 404:
                raise RuntimeError(f"Pod {self.pod_name} not found (404).")
            elif e.status == 500:
                raise RuntimeError(f"Pod {self.pod_name} is in an error state (500).")
            raise

        tar_data, stderr_data = self._read_exec_stream(resp)

        if stderr_data and (
            "No such file or directory" in stderr_data or "cannot cd" in stderr_data
        ):
            raise RuntimeError(
                f"Failed to access directory {source_dir} in pod {self.pod_name}: {stderr_data.strip()}"
            )

        if not tar_data:
            raise RuntimeError(
                f"No data received when downloading {source_dir} from pod {self.pod_name}."
            )

        tar_buffer = io.BytesIO(tar_data)
        try:
            with tarfile.open(fileobj=tar_buffer, mode="r") as tar:
                tar.extractall(path=str(target_dir))
        except tarfile.TarError as e:
            raise RuntimeError(
                f"Failed to extract directory {source_dir} from pod {self.pod_name}: {e}"
            )

    async def _wait_for_pod_ready(self, timeout_sec: int = 600):
        """Wait for pod to be ready."""
        self.logger.debug(f"Waiting for pod {self.pod_name} to be ready...")

        # Transient apiserver 5xx (watch-cache hiccups, load spikes) during the
        # poll loop must NOT kill the trial: read_namespaced_pod is a read of an
        # already-created pod, so a 500 here is almost always momentary and the
        # next poll succeeds. Previously ANY non-404 ApiException raised
        # immediately, turning ~one transient 500 per ~5 trials into a hard env
        # failure (-> dummy trajectory -> degenerate batch). Tolerate up to
        # _MAX_TRANSIENT_5XX consecutive 5xx, reset on any success, and surface
        # e.body (the apiserver's real error message) when we finally give up.
        _MAX_TRANSIENT_5XX = 30
        consecutive_5xx = 0

        start_time = time.monotonic()
        deadline = start_time + timeout_sec
        backoff_delays = (1, 2, 5, 10, 30, 60)
        attempt = 0

        while time.monotonic() < deadline:
            try:
                pod = await asyncio.to_thread(
                    self._api.read_namespaced_pod,
                    name=self.pod_name,
                    namespace=self.namespace,
                )
                consecutive_5xx = 0

                if pod.status.phase == "Running":
                    if pod.status.container_statuses:
                        if all(c.ready for c in pod.status.container_statuses):
                            self.logger.debug(f"Pod {self.pod_name} is ready!")
                            return

                elif pod.status.phase in ["Failed", "Unknown", "Error"]:
                    error_details = self._get_pod_failure_summary(pod)
                    raise RuntimeError(f"Pod failed to start: {error_details}")

                elif pod.status.phase == "Pending":
                    # Check for image pull errors
                    if pod.status.container_statuses:
                        for c in pod.status.container_statuses:
                            if c.state.waiting:
                                if (
                                    "ImagePullBackOff" in c.state.waiting.reason
                                    or "ErrImagePull" in c.state.waiting.reason
                                ):
                                    raise RuntimeError(
                                        f"Failed to pull image: {c.state.waiting.message or c.state.waiting.reason}"
                                    )

                if attempt % 10 == 0:
                    elapsed = int(time.monotonic() - start_time)
                    self.logger.debug(
                        f"Pod status: {pod.status.phase} ({elapsed}s elapsed)"
                    )

            except ApiException as e:
                # 404: pod not yet visible in the apiserver cache — keep polling.
                if e.status == 404:
                    pass
                # 5xx: transient server-side error — tolerate a run of them,
                # only give up after _MAX_TRANSIENT_5XX in a row.
                elif 500 <= (e.status or 0) < 600:
                    consecutive_5xx += 1
                    self.logger.warning(
                        "Transient apiserver %s while polling pod %s (%d/%d consecutive); "
                        "reason=%s body=%s",
                        e.status, self.pod_name, consecutive_5xx, _MAX_TRANSIENT_5XX,
                        e.reason, (e.body or "")[:300],
                    )
                    if consecutive_5xx >= _MAX_TRANSIENT_5XX:
                        raise RuntimeError(
                            f"Kubernetes API error: {e.status} - {e.reason} "
                            f"(persisted {consecutive_5xx} consecutive polls); body={(e.body or "")[:500]}"
                        )
                # Other 4xx (401/403/409/...): a real client error, fail fast WITH body.
                else:
                    raise RuntimeError(
                        f"Kubernetes API error: {e.status} - {e.reason}; body={(e.body or "")[:500]}"
                    )

            delay = backoff_delays[min(attempt, len(backoff_delays) - 1)]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(delay, remaining))
            attempt += 1

        raise RuntimeError(f"Pod not ready after {timeout_sec} seconds")

    def _get_pod_failure_summary(self, pod) -> str:
        """Get a summary of pod failure reasons."""
        reasons = []

        if pod.status.reason:
            reasons.append(f"Reason: {pod.status.reason}")
        if pod.status.message:
            reasons.append(f"Message: {pod.status.message}")

        if pod.status.container_statuses:
            for c in pod.status.container_statuses:
                if c.state.waiting:
                    reasons.append(
                        f"Container {c.name} waiting: {c.state.waiting.reason}"
                    )
                elif c.state.terminated:
                    reasons.append(
                        f"Container {c.name} terminated: {c.state.terminated.reason} "
                        f"(exit code {c.state.terminated.exit_code})"
                    )

        return "; ".join(reasons) if reasons else "Unknown error"
