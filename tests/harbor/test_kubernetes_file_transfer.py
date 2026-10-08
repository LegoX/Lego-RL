"""Exercise actual download methods with a binary-sensitive websocket substitute."""
import asyncio
import io
import logging
import tarfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from harbor_patch.environments.kubernetes import kubernetes as module


class ExecStream:
    def __init__(self, payload, binary):
        self.payload = payload if binary else payload.decode("utf-8", errors="replace")
        self.stderr = b"tar: removing leading slash\n" if binary else "tar: removing leading slash\n"
        self.closed = False

    def update(self, **kwargs):
        pass

    def peek_stdout(self):
        return bool(self.payload)

    def read_stdout(self):
        data, self.payload = self.payload, b""
        return data

    def peek_stderr(self):
        return bool(self.stderr)

    def read_stderr(self):
        data, self.stderr = self.stderr, b""
        return data

    def is_open(self):
        return False  # Final buffers must still be drained after close.

    def close(self):
        self.closed = True


def make_tar(name, data):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


@pytest.mark.parametrize("directory", [False, True])
def test_download_preserves_all_byte_values(monkeypatch, tmp_path, directory):
    env = object.__new__(module.KubernetesEnvironment)
    env._ensure_client = AsyncMock()
    env._core_api = SimpleNamespace(connect_get_namespaced_pod_exec=object())
    env.pod_name = "test-pod"
    env.namespace = "test"
    env.logger = logging.getLogger("test")
    data = bytes(range(256)) * 100
    tar = make_tar("payload.bin" if directory else "tmp/payload.bin", data)
    streams = []
    def stream(*args, **kwargs):
        obj = ExecStream(tar, kwargs.get("binary", False))
        streams.append(obj)
        return obj
    monkeypatch.setattr(module, "stream", stream)
    if directory:
        asyncio.run(env.download_dir("/tmp", tmp_path / "out"))
        result = tmp_path / "out/payload.bin"
    else:
        result = tmp_path / "payload.bin"
        asyncio.run(env.download_file("/tmp/payload.bin", result))
    assert result.read_bytes() == data
    assert all(s.closed for s in streams)


def test_stderr_bytes_are_decoded_and_stream_is_closed():
    env = object.__new__(module.KubernetesEnvironment)
    stream = ExecStream(b"abc\x00\xff", True)
    stream.stderr = b"warning: \xff"
    stdout, stderr = env._read_exec_stream(stream)
    assert stdout == b"abc\x00\xff"
    assert stderr == "warning: \ufffd"
    assert stream.closed
