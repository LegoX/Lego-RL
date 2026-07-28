"""
RemoteDockerEnvironment — thin wrapper around DockerEnvironment for remote
Docker daemons accessed via DOCKER_HOST (tcp:// or ssh://).

Differs from local DockerEnvironment:

1. ``is_mounted = False`` — bind-mount log volumes land on the *remote* host,
   not the training host. The trial runner uses ``docker compose cp`` to pull
   logs back (already implemented in DockerEnvironment).

2. ``_DOCKER_COMPOSE_BASE_PATH`` points to a compose fragment *without*
   host-volume mounts, avoiding orphan directories on the remote host.

3. ``start()`` creates /logs/{agent,verifier,artifacts} inside the container
   (remote compose base has no bind mounts to create them implicitly).

4. **Agent runtime image mount** — Docker has no ImageVolume (K8s 1.31+).
   When ``agent_runtime_image`` is passed (via environment.kwargs in the
   agent loop config), the runtime image is extracted once to a staging
   directory on the remote Docker host, then bind-mounted read-only into
   every task container via compose volume override.

Usage (agent_loop_config yaml):
    environment:
      import_path: harbor_patch.environments.remote_docker:RemoteDockerEnvironment
      kwargs:
        agent_runtime_image: docker.io/jierun/c-cc-2.1.118:v0.1
        agent_runtime_mount_path: /opt/custom-agent-runtime/claude-code
        agent_runtime_image_subpath: opt/custom-agent-runtime/claude-code
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.trial.paths import EnvironmentPaths

_DIR = Path(__file__).parent
_COMPOSE_BASE_REMOTE = _DIR / "docker-compose-base-remote.yaml"

logger = logging.getLogger(__name__)


def _force_real_docker_cli() -> None:
    """Put the REAL docker CLI ahead of the proot-dind shim on PATH.

    On proot-dind dev pods, `docker` resolves to a `docker-proot` wrapper that
    passes pull/load/info/images through to the real client but intercepts
    `docker compose build/up` with a LOCAL proot build. That proot build is a
    half-implementation: it ignores COPY and its RUN steps silently fail, so the
    task image comes out empty/broken and every trial dies (KeyError / empty
    trajectory / position_ids crash downstream).

    RemoteDockerEnvironment always targets a real remote daemon via DOCKER_HOST,
    where a normal `docker build` handles COPY/RUN/caching correctly. So every
    docker call from this backend must use the real binary. Prepend a dir that
    symlinks it; harbor's compose subprocesses inherit this PATH.
    """
    real = "/usr/bin/docker"
    if not os.path.exists(real):
        return
    bindir = "/tmp/harbor-real-docker-bin"
    try:
        os.makedirs(bindir, exist_ok=True)
        link = os.path.join(bindir, "docker")
        if not os.path.lexists(link):
            os.symlink(real, link)
        parts = os.environ.get("PATH", "").split(":")
        if bindir not in parts:
            os.environ["PATH"] = bindir + ":" + os.environ.get("PATH", "")
        logger.info("RemoteDocker: real docker CLI (%s) fronted ahead of proot shim", real)
    except OSError as e:
        logger.warning("RemoteDocker: could not front real docker CLI: %s", e)


_force_real_docker_cli()

_RUNTIME_STAGE_BASE = "/opt/harbor-agent-runtime-cache"

_runtime_staged: dict[str, str] = {}
_runtime_stage_lock = asyncio.Lock()


class RemoteDockerEnvironment(DockerEnvironment):
    _DOCKER_COMPOSE_BASE_PATH = _COMPOSE_BASE_REMOTE

    def __init__(
        self,
        *args,
        agent_runtime_image: str | None = None,
        agent_runtime_mount_path: str = "/opt/custom-agent-runtime/claude-code",
        agent_runtime_image_subpath: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._agent_runtime_image = (agent_runtime_image or "").strip() or None
        self._agent_runtime_mount_path = agent_runtime_mount_path
        self._agent_runtime_image_subpath = (
            (agent_runtime_image_subpath or "").strip()
            or agent_runtime_mount_path.lstrip("/")
        )

    @property
    def is_mounted(self) -> bool:
        return False

    @staticmethod
    async def _run_cmd(cmd: str) -> tuple[int, str]:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
        return proc.returncode, (out or b"").decode(errors="replace").strip()

    async def _ensure_runtime_staged(self) -> str | None:
        """Extract the runtime image to a directory on the remote Docker host
        (once per image, cached across trials). Returns the remote path."""
        if not self._agent_runtime_image:
            return None

        image = self._agent_runtime_image
        subpath = self._agent_runtime_image_subpath

        async with _runtime_stage_lock:
            cached = _runtime_staged.get(image)
            if cached:
                logger.info("Agent runtime already staged: %s", cached)
                return cached

            safe_tag = image.replace("/", "_").replace(":", "_")
            host_dir = f"{_RUNTIME_STAGE_BASE}/{safe_tag}"

            rc, out = await self._run_cmd(f"docker pull -q {image}")
            if rc != 0:
                logger.warning("runtime image pull failed (%d): %s", rc, out)
                return None

            # Already staged on the remote host (possibly by another worker process)?
            # The marker + all path ops go through the remote daemon (host_dir lives on
            # the remote host, not here), using the runtime image itself as the toolbox.
            rc, _ = await self._run_cmd(
                f"docker run --rm -v {_RUNTIME_STAGE_BASE}:/base {image} "
                f"sh -c 'test -f /base/{safe_tag}/.staged_ok'"
            )
            if rc == 0:
                _runtime_staged[image] = host_dir
                logger.info("Agent runtime already staged on remote: %s", host_dir)
                return host_dir

            # Stage via tar, NOT `cp -a`: cp fails to recreate symlinks on some remote
            # filesystems (243's nydus snapshotter -> "cannot create symbolic link
            # libpython3.12.so"); tar preserves them. Each worker stages into a private,
            # pid-scoped tmp dir and atomically mv's it into place, so concurrent stagers
            # never collide (the old fixed-name `docker create` clashed under concurrency).
            uniq = f"{safe_tag}.tmp.{os.getpid()}"
            rc, out = await self._run_cmd(
                f"docker run --rm -v {_RUNTIME_STAGE_BASE}:/base {image} sh -c '"
                f"rm -rf /base/{uniq}; mkdir -p /base/{uniq} && "
                f"cd /{subpath} && tar cf - . | tar xf - -C /base/{uniq} && "
                f"touch /base/{uniq}/.staged_ok && "
                f"rm -rf /base/{safe_tag} && mv /base/{uniq} /base/{safe_tag}'"
            )
            if rc != 0:
                logger.warning("runtime stage via tar failed (%d): %s", rc, out)
                await self._run_cmd(
                    f"docker run --rm -v {_RUNTIME_STAGE_BASE}:/base {image} "
                    f"sh -c 'rm -rf /base/{uniq}'"
                )
                return None

            _runtime_staged[image] = host_dir
            logger.info("Agent runtime staged at remote %s (via tar)", host_dir)
            return host_dir

    async def start(self, force_build: bool) -> None:
        host_dir = await self._ensure_runtime_staged()
        if host_dir:
            mount_spec = f"{host_dir}:{self._agent_runtime_mount_path}:ro"
            if self._mounts_json is None:
                self._mounts_json = []
            self._mounts_json.append(mount_spec)
            logger.info("Added runtime volume: %s", mount_spec)

        await super().start(force_build)
        await self.exec(
            f"mkdir -p {EnvironmentPaths.agent_dir} "
            f"{EnvironmentPaths.verifier_dir} "
            f"{EnvironmentPaths.artifacts_dir}"
        )

    def _write_mounts_compose_file(self) -> Path:
        """Override the base writer: alongside the bind-mounted runtime volume, inject
        CUSTOM_AGENT_RUNTIME_ROOT / CUSTOM_AGENT_PYTHON into the task container's env so
        the mounted-runtime agent (image_mounted_openhands_ai etc.) can locate the runtime.
        Docker has no ImageVolume, so — unlike KubernetesEnvironment, which injects these
        via V1EnvVar — the base class never sets them; without them the agent can't find
        its runtime and every trial fails (KeyError before any real trajectory)."""
        service: dict[str, object] = {"volumes": list(self._mounts_json or [])}
        if self._agent_runtime_image:
            mount = self._agent_runtime_mount_path.rstrip("/")
            service["environment"] = {
                "CUSTOM_AGENT_RUNTIME_ROOT": mount,
                "CUSTOM_AGENT_PYTHON": f"{mount}/bin/python",
            }
        compose: dict[str, object] = {"services": {"main": service}}
        path = self.trial_paths.trial_dir / "docker-compose-mounts.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(compose, indent=2))
        return path
