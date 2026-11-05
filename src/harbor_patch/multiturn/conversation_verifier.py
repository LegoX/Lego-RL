"""Run a user-round verifier in a disposable pod, never in the live agent pod."""
import asyncio
import base64
import json
import re
import shlex
import time
from uuid import uuid4

from kubernetes import client
from kubernetes.stream import stream
from .native_swe import NativeEpisode, SandboxExecTransportError, _native_failure_category
from .test_progress import summarize_tests, disable_failfast


class SnapshotVerifier(NativeEpisode):
    def configure_sandbox(self, pod):
        spec = pod['spec']
        spec['containers'] = [c for c in spec['containers'] if c['name'] == 'main']
        spec['containers'][0].pop('volumeMounts', None)
        spec['containers'][0]['command'] = ['bash', '-lc', 'mkdir -p /logs/verifier /logs/artifacts; exec sleep infinity']
        spec.pop('initContainers', None)
        spec.pop('volumes', None)
        spec['hostAliases'] = [{'ip': '127.0.0.1', 'hostnames': ['main']}]

    def upload_snapshot(self, encoded):
        # Stream the archive through stdin: a workspace is too large for argv.
        api = self.api()
        code = ("import sys,base64; from pathlib import Path; "
                f"data=sys.stdin.buffer.read({len(encoded)}); "
                f"assert len(data)=={len(encoded)}; "
                "Path('/tmp/round-workspace.tar.gz').write_bytes(base64.b64decode(data,validate=True))")
        ws = None
        deadline = time.monotonic() + 240
        try:
            ws = stream(api.connect_get_namespaced_pod_exec, self.name, self.namespace,
                        container='main', command=['python3', '-c', code],
                        stderr=True, stdin=True, stdout=True, tty=False, _preload_content=False)
            for offset in range(0, len(encoded), 65536):
                if time.monotonic() > deadline:
                    raise TimeoutError('Round archive upload timed out')
                ws.write_stdin(encoded[offset:offset + 65536])
            while ws.is_open():
                if time.monotonic() > deadline:
                    raise TimeoutError('Round archive upload timed out')
                ws.update(timeout=1)
            try:
                status = ws.returncode
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                raise SandboxExecTransportError('Round archive upload lost exit status') from exc
            if status is None:
                raise SandboxExecTransportError('Round archive upload lost exit status')
            if status != 0:
                raise RuntimeError('Round archive upload failed')
        finally:
            try:
                if ws is not None:
                    ws.close()
            finally:
                api.api_client.close()

    async def cleanup(self):
        for kind in reversed(self.created):
            api = (client.NetworkingV1Api(client.ApiClient(self.api_config))
                   if kind == 'policy' else self.api())
            try:
                if kind == 'pod':
                    await asyncio.to_thread(api.delete_namespaced_pod, self.name, self.namespace, grace_period_seconds=0)
                elif kind == 'policy':
                    await asyncio.to_thread(api.delete_namespaced_network_policy, self.name, self.namespace)
                else:
                    await asyncio.to_thread(api.delete_namespaced_secret, self.name, self.namespace)
            except client.ApiException as exc:
                if exc.status != 404:
                    self.summary.setdefault('cleanup_errors', []).append(f'{kind}:{exc.status}')
            except Exception as exc:
                self.summary.setdefault('cleanup_errors', []).append(f'{kind}:{type(exc).__name__}')
            finally:
                api.api_client.close()


def parse_verdict(exit_code, read_code, reward_text, guard_code, output=''):
    # Shell status 139/134 can be a crashing program UNDER TEST, not a
    # crashed verifier container. Require the canonical test's failure evidence.
    signalled_test = (exit_code in (134, 139) and bool(re.search(
        r'/tests/canonical_test\.sh:.*(?:Segmentation fault|Aborted)', output)))
    failed_build = (exit_code == 127 and bool(re.search(
        r'undefined reference to|collect2: error: ld returned|ninja: build stopped|fatal error:.*No such file', output)))
    canonical_failure = reward_text.strip() == '0' and (signalled_test or failed_build)
    if read_code or type(exit_code) is not int or not (0 <= exit_code < 124 or canonical_failure):
        raise RuntimeError(f'Round verifier failed to execute ({exit_code}, {read_code})')
    if guard_code:
        raise RuntimeError('Round verifier setup failed')
    reward = float(reward_text.strip())
    if reward not in (0.0, 1.0):
        raise ValueError('Round verifier must produce a binary reward')
    return {'reward': reward, 'exit_code': exit_code}


async def verify_round_snapshot(episode, round_id):
    for attempt in range(3):
        try:
            return await _verify_round_snapshot_once(episode, round_id, attempt)
        except Exception as exc:
            category, retryable = _native_failure_category(exc)
            if not retryable or attempt == 2:
                raise
            episode.summary['round_verifier_transport_retries'] = episode.summary.get('round_verifier_transport_retries', 0) + 1
            await asyncio.sleep(attempt + 1)


def archive_command(path, arcname, scratch):
    # File-backed compression and bounded reads avoid several full-workspace
    # copies inside the policy pod (large Rust/C++ build trees can exceed RAM).
    code = ("import tarfile,base64,os,sys\n"
            f"p={scratch!r} + str(os.getpid())\n"
            "try:\n"
            " with tarfile.open(p, 'w:gz') as t:\n"
            f"  t.add({path!r},arcname={arcname!r})\n"
            " with open(p,'rb') as f:\n"
            "  while True:\n"
            "   b=f.read(3*1024*1024)\n"
            "   if not b: break\n"
            "   sys.stdout.write(base64.b64encode(b).decode())\n"
            "finally:\n"
            " if os.path.exists(p): os.unlink(p)\n")
    return 'python3 -c ' + shlex.quote(code)


async def _verify_round_snapshot_once(episode, round_id, attempt=0):
    episode.summary['failure_stage'] = 'verifier'
    directory = episode.run_dir / f'round-{round_id:03d}' / 'verification'
    if attempt:
        directory = directory / f'attempt-{attempt + 1:03d}'
    directory.mkdir(parents=True, exist_ok=True)
    verifier = object.__new__(SnapshotVerifier)
    # Copy connection/configuration only, never live execution callbacks or state.
    for name in ('api_config', 'namespace', 'row', 'options', 'host', 'gateway_port',
                 'proxy', 'token', 'export_token', 'task'):
        if hasattr(episode, name):
            setattr(verifier, name, getattr(episode, name))
    verifier.name = episode.name[:40] + '-v-' + uuid4().hex[:12]
    verifier.created = []
    verifier.summary = {}
    verifier.run_dir = directory
    workspace = shlex.quote(episode.row['workdir'])
    try:
        # Capture tracked AND untracked files, permissions, and Git metadata.
        # No hidden tests or verifier output are installed in the policy pod.
        command = archive_command(episode.row['workdir'], 'workspace',
                                  '/tmp/round-snapshot-' + uuid4().hex + '.tar.gz')
        rc, encoded = await asyncio.to_thread(episode.execute, command, timeout=240, retry_read=True)
        if rc:
            raise RuntimeError('Round workspace snapshot failed')
        encoded = encoded.strip()
        await verifier.create_sandbox()
        await asyncio.to_thread(verifier.upload_snapshot, encoded)
        rc, _ = await asyncio.to_thread(verifier.execute,
            f'test -d {workspace} && test ! -L {workspace} && '
            f'find {workspace} -mindepth 1 -maxdepth 1 -exec rm -rf -- {{}} + && '
            f'tar -xzf /tmp/round-workspace.tar.gz -C {workspace} --strip-components=1', timeout=240)
        if rc:
            raise RuntimeError('Round workspace restore failed')
        test_files = episode.test_files('02_final_verify')
        canonical_key = 'tests/canonical_test.sh' if 'tests/canonical_test.sh' in test_files else 'tests/test.sh'
        canonical_bytes, canonical_mode = test_files.get(canonical_key, (b'', 0o644))
        canonical = canonical_bytes.decode(errors='replace')
        prepared, early_stop_disabled = disable_failfast(canonical)
        if early_stop_disabled:
            test_files[canonical_key] = (prepared.encode(), canonical_mode)
        await asyncio.to_thread(verifier.upload, test_files)
        exit_code, output = await asyncio.to_thread(verifier.execute,
            'timeout -s TERM 600 bash /tests/test.sh', timeout=630, log=directory / 'console.log')
        read_code, text = await asyncio.to_thread(verifier.execute, 'cat /logs/verifier/reward.txt', retry_read=True)
        guard = """from pathlib import Path
root=Path('/logs/verifier')
p=root/'test-stdout.txt'
text=p.read_text() if p.exists() else ''
prefixes=('repository workdir is missing:', 'base commit record is missing:',
 'base commit is invalid:', 'failed to capture complete model.patch',
 'failed to move HEAD to base commit', 'failed to normalize the Git index',
 'failed to restore verifier-owned tests')
raise SystemExit(1 if (root/'verifier-guard-failure.json').exists() or text.startswith(prefixes) else 0)
"""
        guard_code, _ = await asyncio.to_thread(verifier.execute, 'python3 -c ' + shlex.quote(guard), retry_read=True)
        (directory / 'execution-status.json').write_text(json.dumps({
            'exit_code': exit_code, 'reward_read_code': read_code, 'guard_code': guard_code}))
        result = parse_verdict(exit_code, read_code, text, guard_code, output)
        result['test_progress'] = summarize_tests(output, prepared, exit_code)
        result['test_progress']['early_stop_disabled'] = early_stop_disabled
        (directory / 'result.json').write_text(json.dumps({'round_id': round_id, **result}, indent=2))
        return result
    except Exception as exc:
        verifier.summary['failure_type'] = type(exc).__name__
        if 'pod' in verifier.created:
            try:
                verifier.summary['sandbox'] = await asyncio.to_thread(verifier.sandbox_diagnostics)
            except Exception as diagnostic_error:
                verifier.summary['diagnostic_error_type'] = type(diagnostic_error).__name__
        raise
    finally:
        try:
            if 'pod' in verifier.created:
                command = archive_command('/logs', 'logs', '/tmp/verifier-logs-' + uuid4().hex + '.tar.gz')
                rc, data = await asyncio.to_thread(verifier.execute, command, timeout=120, retry_read=True)
                if rc == 0:
                    (directory / 'artifacts.tar.gz').write_bytes(base64.b64decode(data.strip(), validate=True))
        except Exception as exc:
            verifier.summary['artifact_error'] = type(exc).__name__
        finally:
            await verifier.cleanup()
            if verifier.summary:
                (directory / 'collection-status.json').write_text(json.dumps(verifier.summary, indent=2))
