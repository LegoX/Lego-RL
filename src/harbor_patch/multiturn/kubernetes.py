"""Benchmark build contexts without requiring CPFS mounts on CPU nodes.

Only selected by the multi-turn factory. The normal Kubernetes backend stays
unchanged. Preserve empty directories and Docker's directory-COPY semantics.
"""

import tarfile
import tempfile
from pathlib import Path

from kubernetes import client

from harbor_patch.environments.kubernetes.kubernetes import KubernetesEnvironment


class MultiTurnKubernetesEnvironment(KubernetesEnvironment):
    def __init__(self, *args, workspace_dir="/app", **kwargs):
        self._workspace_dir = workspace_dir
        super().__init__(*args, **kwargs)

    _multiturn_build_active = False
    _multiturn_runtime_ready = False

    def _build_pod_spec(self):
        pod = super()._build_pod_spec()
        for volume in pod.spec.volumes:
            if volume.name == "build-context":
                volume.host_path = None
                volume.empty_dir = client.V1EmptyDirVolumeSource()
        for container in pod.spec.containers:
            for mount in container.volume_mounts or []:
                if mount.name == "build-context":
                    mount.read_only = False
        if self._inline_build:
            main = next(c for c in pod.spec.containers if c.name == "main")
            # ENV must reach later Agent/verifier execs, not just Dockerfile RUN.
            # This benchmark uses literal NODE_PATH and browser-install controls.
            env_vars = {entry.name: entry for entry in main.env or []}
            for kind, payload in self._parse_dockerfile_for_inline_build(
                self._environment_definition_path
            ):
                if kind == "ENV":
                    for name, value in self._parse_dockerfile_env(payload):
                        env_vars[name] = client.V1EnvVar(name=name, value=value)
            main.env = list(env_vars.values())
            # scipy-notebook's base USER is jovyan; these benchmark Dockerfiles
            # explicitly switch to root before preparing /app.
            if any(
                line.strip() in {"USER root", "USER 0"}
                for line in self._environment_definition_path.read_text().splitlines()
            ):
                main.security_context = (
                    main.security_context or client.V1SecurityContext()
                )
                main.security_context.run_as_user = 0
        return pod

    def _parse_dockerfile_copy(self, payload):
        sources, destination = super()._parse_dockerfile_copy(payload)
        # Docker COPY ./code /app copies CONTENTS, including an empty code/.
        # The inherited shell implementation otherwise creates /app/code.
        sources = [
            source.rstrip("/") + "/."
            if (self.environment_dir / source).is_dir()
            else source
            for source in sources
        ]
        return sources, destination

    async def _apply_dockerfile_inline_build(self, dockerfile):
        with tempfile.TemporaryDirectory(prefix="lego-multiturn-build-") as directory:
            archive = Path(directory) / "context.tar"
            with tarfile.open(archive, "w") as stream:
                stream.add(self.environment_dir, arcname=".")
            await self.upload_file(archive, "/tmp/multiturn-build-context.tar")
        self._multiturn_build_active = True
        try:
            await self.exec(
                "tar -C /build-context -xf /tmp/multiturn-build-context.tar && "
                "rm -f /tmp/multiturn-build-context.tar",
                user="root",
            )
            await super()._apply_dockerfile_inline_build(dockerfile)
            self._multiturn_runtime_ready = True
        finally:
            self._multiturn_build_active = False

    async def exec(self, *args, **kwargs):
        if (
            (self._multiturn_runtime_ready or not getattr(self, "_inline_build", True))
            and not self._multiturn_build_active
            and len(args) < 2
            and kwargs.get("cwd") is None
            and kwargs.get("container") in (None, self._MAIN_CONTAINER_NAME)
        ):
            # Preserve the selected task workspace after inline Dockerfile setup.
            kwargs["cwd"] = getattr(self, "_workspace_dir", "/app")
        result = await super().exec(*args, **kwargs)
        if self._multiturn_build_active and result.return_code != 0:
            raise RuntimeError(
                f"Multi-turn Dockerfile setup failed (exit {result.return_code})"
            )
        return result
