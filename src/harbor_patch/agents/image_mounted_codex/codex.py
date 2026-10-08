"""Codex CLI runtime with a per-trial HTTP Responses provider.

Use a pinned mounted binary when available; otherwise use Harbor's installer.
Training trajectories come from BuiltinSWEAgentLoop's proxy recorder.
"""

import asyncio
import os
import json
import shlex

from harbor.agents.installed.codex import Codex as _BaseCodex
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

_RUNTIME_ROOT_ENV = "CUSTOM_AGENT_RUNTIME_ROOT"
_DEFAULT_RUNTIME_ROOT = "/opt/custom-agent-runtime/codex"

# K8s ImageVolume (beta in 1.31+) can report Pod=Ready before the volume's
# contents are fully extracted on disk. With ~96 AgentLoopWorkers spawning pods
# concurrently, a single probe sees a dangling path, falls back to the npm
# route, and every trial then burns the agent setup timeout. Retry to ride out
# the extraction window — same failure mode documented in
# image_mounted_opencode._detect_mounted_runtime.
_DETECT_RETRY_ATTEMPTS = int(os.environ.get("HARBOR_CX_DETECT_RETRY_ATTEMPTS", 30))
_DETECT_RETRY_SLEEP_SEC = float(os.environ.get("HARBOR_CX_DETECT_RETRY_SLEEP_SEC", 2.0))


class Codex(_BaseCodex):
    """harbor Codex + mounted-runtime detection."""

    def _runtime_root(self) -> str:
        return (
            self._get_env(_RUNTIME_ROOT_ENV)
            or os.environ.get(_RUNTIME_ROOT_ENV)
            or _DEFAULT_RUNTIME_ROOT
        )

    async def _detect_mounted_runtime(self, environment: BaseEnvironment) -> str | None:
        """Return the mounted codex binary path, or None to fall back to npm."""
        root = self._runtime_root()
        binary = f"{root}/bin/codex"
        for attempt in range(1, _DETECT_RETRY_ATTEMPTS + 1):
            try:
                probe = await environment.exec(
                    command=f"test -x {shlex.quote(binary)} && echo MOUNTED || echo MISSING"
                )
                if "MOUNTED" in (probe.stdout or ""):
                    if attempt > 1:
                        self.logger.warning(
                            "[CX-DETECT] mount visible on attempt=%d/%d",
                            attempt, _DETECT_RETRY_ATTEMPTS,
                        )
                    return binary
            except Exception as exc:
                self.logger.warning(
                    "[CX-DETECT] probe failed attempt=%d/%d: %r",
                    attempt, _DETECT_RETRY_ATTEMPTS, exc,
                )
            if attempt < _DETECT_RETRY_ATTEMPTS:
                await asyncio.sleep(_DETECT_RETRY_SLEEP_SEC)
        return None

    async def install(self, environment: BaseEnvironment) -> None:
        binary = await self._detect_mounted_runtime(environment)
        if binary is None:
            self.logger.warning(
                "[CX-DETECT] no mounted runtime under %s; falling back to the in-pod "
                "nvm+npm install (needs pod egress, does not scale to 96 workers)",
                self._runtime_root(),
            )
            await super().install(environment)
            await self._write_provider_config(environment)
            return

        # The parent's run() invokes a bare `codex`, so putting the mounted
        # binary on PATH is all that is needed — no wrapper, no node.
        await self.exec_as_root(
            environment, command=f"ln -sf {shlex.quote(binary)} /usr/local/bin/codex"
        )

        result = await environment.exec(command="codex --version")
        reported = (result.stdout or "").strip().splitlines()
        reported = reported[0].removeprefix("codex-cli").strip() if reported else ""
        expected = str(self._version or "").strip()
        if expected and reported != expected:
            # A silently drifted runtime image trains against a different
            # harness than the config claims. Fail the trial instead.
            raise RuntimeError(
                f"mounted codex version mismatch: image has {reported!r}, "
                f"config pins {expected!r} ({binary})"
            )
        self._version = reported or self._version
        self.logger.info("[CX-DETECT] using mounted codex %s at %s", reported, binary)

        await self._write_provider_config(environment)

    async def _write_provider_config(self, environment: BaseEnvironment) -> None:
        """Use the proxy's HTTP Responses endpoint without WebSocket retries."""
        base_url = self._get_env("OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        if not base_url:
            raise ValueError("Codex requires a per-trial OPENAI_BASE_URL")

        codex_home = EnvironmentPaths.agent_dir.as_posix()
        toml = "\n".join(
            [
                'model_provider = "vllm-proxy"',
                "",
                "[model_providers.vllm-proxy]",
                'name = "Lego-RL in-process vLLM proxy"',
                f"base_url = {json.dumps(base_url)}",
                # codex-cli >=0.122.0 rejects wire_api="chat"; the proxy serves
                # /sess/{id}/v1/responses precisely for this.
                'wire_api = "responses"',
                'env_key = "OPENAI_API_KEY"',
                "requires_openai_auth = false",
                "supports_websockets = false",
                "",
            ]
        )
        await environment.exec(
            command=f"mkdir -p {codex_home} && echo {shlex.quote(toml)} > {codex_home}/config.toml"
        )
        self.logger.info(
            "[CX-CONFIG] wrote %s/config.toml: provider=vllm-proxy wire_api=responses "
            "supports_websockets=false base_url=%s",
            codex_home, base_url,
        )

    def _build_register_mcp_servers_command(self) -> str | None:
        # Harbor normally overwrites config.toml here, which would erase the
        # per-trial provider installed above. Append its MCP tables instead.
        command = super()._build_register_mcp_servers_command()
        if command is None:
            return None
        suffix = ' > "$CODEX_HOME/config.toml"'
        if not command.endswith(suffix):
            raise RuntimeError("Unsupported Harbor MCP config writer")
        return command.removesuffix(suffix) + ' >> "$CODEX_HOME/config.toml"'

    def populate_context_post_run(self, context: AgentContext) -> None:
        super().populate_context_post_run(context)

        if context.metadata is None:
            context.metadata = {}
        context.metadata.setdefault("all_messages", [])
        context.metadata.setdefault("tools", [])
