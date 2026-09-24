"""Tests for the Ray-wide Kubernetes pod-create scheduler.

These tests model the production topology: each simulated AgentLoopWorker is
a separate Ray actor and submits at most one create request.  The Kubernetes
API call itself is mocked with an async sleep, so the kubeconfig is loaded for
configuration validation but no Pod is created in the cluster.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from pathlib import Path

import pytest

ray = pytest.importorskip("ray")

from harbor_patch.environments.kubernetes.kubernetes import (  # noqa: E402
    _GenericK8sClientManager,
)


def _test_kubeconfig() -> Path:
    """Return the kubeconfig supplied by the test environment.

    Keep site/user-specific paths out of the repository.  CI or a developer
    can set HARBOR_TEST_KUBECONFIG explicitly; KUBECONFIG is accepted as the
    conventional fallback.
    """
    raw = os.environ.get("HARBOR_TEST_KUBECONFIG") or os.environ.get("KUBECONFIG")
    if not raw:
        pytest.skip(
            "Set HARBOR_TEST_KUBECONFIG (or KUBECONFIG) to run the Kubernetes "
            "scheduler integration simulation"
        )
    path = Path(raw).expanduser()
    if not path.is_file():
        pytest.skip(f"Test kubeconfig does not exist: {path}")
    return path


def test_kubeconfig_is_loadable_and_scheduler_defaults(monkeypatch):
    """Use the requested kubeconfig without making a Kubernetes API call."""

    async def run():
        kubeconfig = _test_kubeconfig()
        monkeypatch.setenv("KUBECONFIG", str(kubeconfig))
        monkeypatch.delenv("HARBOR_K8S_POD_CREATE_CONCURRENCY", raising=False)
        monkeypatch.delenv("HARBOR_K8S_POD_CREATE_INTERVAL_SEC", raising=False)

        manager = _GenericK8sClientManager()
        await asyncio.to_thread(manager._load_kubeconfig, str(kubeconfig))

        assert manager._pod_create_concurrency == 16
        assert manager._pod_create_interval_sec == 20.0

    asyncio.run(run())


def test_multiple_ray_workers_share_global_create_gate(monkeypatch):
    """Separate Ray actors are globally paced by one detached scheduler."""

    async def run():
        kubeconfig = _test_kubeconfig()
        scheduler_name = f"test-harbor-k8s-{uuid.uuid4().hex}"
        monkeypatch.setenv("KUBECONFIG", str(kubeconfig))
        # Keep the test fast while preserving the production scheduling semantics.
        monkeypatch.setenv("HARBOR_K8S_POD_CREATE_CONCURRENCY", "2")
        monkeypatch.setenv("HARBOR_K8S_POD_CREATE_INTERVAL_SEC", "0.05")
        monkeypatch.setenv("HARBOR_K8S_POD_CREATE_LEASE_SEC", "30")
        monkeypatch.setenv("HARBOR_K8S_POD_CREATE_SCHEDULER_NAME", scheduler_name)

        if ray.is_initialized():
            ray.shutdown()
        ray.init(num_cpus=3, include_dashboard=False, log_to_driver=False)

        @ray.remote(num_cpus=1)
        class SimulatedAgentLoopWorker:
            async def create_one_pod(self, pod_name: str, kubeconfig: str):
                # Every actor has its own process-local manager, matching the
                # production AgentLoopWorker topology.  The kubeconfig path is
                # carried through the simulated request just as the real
                # KubernetesEnvironment carries it; the first test validates the
                # file is loadable without contacting K8s here.
                manager = await _GenericK8sClientManager.get_instance()
                assert Path(kubeconfig).is_file()

                def fake_create_namespaced_pod():
                    # The production path invokes this synchronous Kubernetes
                    # client method via asyncio.to_thread().
                    time.sleep(0.02)

                async with manager.pod_creation_slot(pod_name):
                    create_started = time.monotonic()
                    await asyncio.to_thread(fake_create_namespaced_pod)
                    create_finished = time.monotonic()
                return create_started, create_finished

        workers = [SimulatedAgentLoopWorker.remote() for _ in range(3)]
        try:
            intervals = await asyncio.gather(
                *(
                    worker.create_one_pod.remote(
                        f"simulated-pod-{idx}", str(kubeconfig)
                    )
                    for idx, worker in enumerate(workers)
                )
            )
            intervals.sort()

            # The scheduler's 50ms global interval applies across actors, not just
            # within one actor process.
            starts = [start for start, _ in intervals]
            gaps = [b - a for a, b in zip(starts, starts[1:])]
            assert min(gaps) >= 0.035, gaps

            # Each simulated worker submitted exactly one request and the slot was
            # released after the mock API call, so all three eventually completed.
            assert len(intervals) == 3
            assert all(finished > started for started, finished in intervals)
        finally:
            ray.shutdown()

    asyncio.run(run())
