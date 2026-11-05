import tarfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from harbor_patch.multiturn.kubernetes import (
    KubernetesEnvironment,
    MultiTurnKubernetesEnvironment,
)
from kubernetes import client


def test_uploaded_build_context_has_no_host_mount(monkeypatch):
    pod = client.V1Pod(
        spec=client.V1PodSpec(
            containers=[
                client.V1Container(
                    name="main",
                    volume_mounts=[
                        client.V1VolumeMount(
                            name="build-context",
                            mount_path="/build-context",
                            read_only=True,
                        )
                    ],
                )
            ],
            volumes=[
                client.V1Volume(
                    name="build-context",
                    host_path=client.V1HostPathVolumeSource(path="/unshared/cpfs"),
                ),
                client.V1Volume(name="logs", empty_dir=client.V1EmptyDirVolumeSource()),
            ],
        )
    )
    monkeypatch.setattr(KubernetesEnvironment, "_build_pod_spec", lambda self: pod)
    env = object.__new__(MultiTurnKubernetesEnvironment)
    env._inline_build = False
    output = env._build_pod_spec()
    assert output.spec.volumes[0].host_path is None
    assert output.spec.volumes[0].empty_dir is not None
    assert not output.spec.containers[0].volume_mounts[0].read_only


@pytest.mark.asyncio
async def test_build_upload_preserves_empty_code_directory_and_copy_contents(
    tmp_path, monkeypatch
):
    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "code").mkdir()
    dockerfile = environment / "Dockerfile"
    dockerfile.write_text("FROM test\nWORKDIR /app\nCOPY ./code /app\n")
    env = object.__new__(MultiTurnKubernetesEnvironment)
    env.environment_dir = environment
    seen = []

    async def upload(archive, remote):
        with tarfile.open(archive) as stream:
            seen.extend(stream.getnames())
        assert remote == "/tmp/multiturn-build-context.tar"

    env.upload_file = upload
    env.exec = AsyncMock()
    inline = AsyncMock()
    monkeypatch.setattr(KubernetesEnvironment, "_apply_dockerfile_inline_build", inline)
    await env._apply_dockerfile_inline_build(dockerfile)
    assert "./code" in seen
    assert env._parse_dockerfile_copy("./code /app") == (["./code/."], "/app")
    inline.assert_awaited_once_with(dockerfile)
    assert not env._multiturn_build_active


@pytest.mark.asyncio
async def test_build_failures_are_fatal_but_agent_command_failures_are_returned(
    monkeypatch,
):
    env = object.__new__(MultiTurnKubernetesEnvironment)
    result = SimpleNamespace(return_code=1)
    monkeypatch.setattr(KubernetesEnvironment, "exec", AsyncMock(return_value=result))
    assert await env.exec("false") is result
    env._multiturn_build_active = True
    with pytest.raises(RuntimeError, match="Dockerfile setup failed"):
        await env.exec("false")


def test_benchmark_runtime_env_and_root_user_survive_inline_build(
    tmp_path, monkeypatch
):
    (tmp_path / "Dockerfile").write_text(
        "FROM test\nUSER root\nWORKDIR /app\nENV NODE_PATH=/usr/lib/node_modules\n"
    )
    main = client.V1Container(
        name="main", env=[client.V1EnvVar(name="PATH", value="/bin")]
    )
    pod = client.V1Pod(spec=client.V1PodSpec(containers=[main], volumes=[]))
    monkeypatch.setattr(KubernetesEnvironment, "_build_pod_spec", lambda self: pod)
    env = object.__new__(MultiTurnKubernetesEnvironment)
    env.environment_dir = tmp_path
    env._inline_build = True
    result = env._build_pod_spec().spec.containers[0]
    assert {v.name: v.value for v in result.env} == {
        "PATH": "/bin",
        "NODE_PATH": "/usr/lib/node_modules",
    }
    assert result.security_context.run_as_user == 0


@pytest.mark.asyncio
async def test_agent_exec_defaults_to_app_but_sidecar_and_explicit_cwd_do_not(
    monkeypatch,
):
    parent_exec = AsyncMock(return_value=SimpleNamespace(return_code=0))
    monkeypatch.setattr(KubernetesEnvironment, "exec", parent_exec)
    env = object.__new__(MultiTurnKubernetesEnvironment)
    await env.exec("before-build")
    assert "cwd" not in parent_exec.call_args.kwargs
    env._multiturn_runtime_ready = True
    await env.exec("agent-command")
    assert parent_exec.call_args.kwargs["cwd"] == "/app"
    await env.exec("custom-command", cwd="/tmp")
    assert parent_exec.call_args.kwargs["cwd"] == "/tmp"
    await env.exec("sidecar-command", container="netadmin")
    assert "cwd" not in parent_exec.call_args.kwargs


@pytest.mark.asyncio
async def test_prebuilt_openswe_commands_use_testbed_and_explicit_cwd_wins(monkeypatch):
    env = object.__new__(MultiTurnKubernetesEnvironment)
    env._inline_build = False
    env._workspace_dir = "/testbed"
    base = AsyncMock(return_value=SimpleNamespace(return_code=0))
    monkeypatch.setattr(KubernetesEnvironment, "exec", base)
    await env.exec("pwd")
    assert base.call_args.kwargs["cwd"] == "/testbed"
    await env.exec("pwd", cwd="/tmp")
    assert base.call_args.kwargs["cwd"] == "/tmp"
