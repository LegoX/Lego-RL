"""Opt-in native two-step SWE-Interact adapter for the existing RL token proxy.

One isolated sandbox and native CC session per rollout. No task rewriting, no
reference solutions, no fallback rewards for infrastructure failures. The same
adapter serves training and held-out validation, with terminal verifier rewards.
"""
import asyncio
import base64
from functools import lru_cache
import io
import ipaddress
import json
import logging
import os
from pathlib import Path
import secrets
import shlex
import tarfile
import time
from uuid import uuid4

import aiohttp
from aiohttp import web
from kubernetes import client, config
from kubernetes.stream import stream
from websocket import WebSocketException

from .native_budget import TurnBudget

LOG = logging.getLogger(__name__)
RUNTIME = '/opt/swe-interact-rl'
IMAGE_RUNTIME = '/runtime-image/opt/custom-agent-runtime/claude-code'


@lru_cache(maxsize=4)
def load_manifest(path):
    records = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task = row['task_id']
        if task in records or not row['images_ready'] or row['protocol'] != 'two-step-text@3.6':
            raise ValueError(f'Invalid native task manifest row: {task}')
        if set(row['images']) != {'main', 'user-server'} or any('@sha256:' not in x for x in row['images'].values()):
            raise ValueError(f'Unpinned native task image: {task}')
        records[task] = row
    return records


def classify_result(native, approval, final_reward, verifier_rc, context_overflow, exit_code):
    """Keep legitimate budget failures; reject API/setup/verifier failures."""
    if approval not in (0, 1) or final_reward not in (0, 1):
        raise ValueError('Native verifier must return a binary reward')
    if approval == 1:
        if native.get('exit_status') != 'SimulatorApproved' or exit_code != 0 or verifier_rc not in (0, 1):
            raise RuntimeError('Approval/final-verifier execution is inconsistent')
        return 'agent_completed'
    if final_reward != 0 or verifier_rc is not None:
        raise RuntimeError('Unapproved task cannot run final verifier')
    if context_overflow:
        return 'overlong'
    error = str(native.get('error', ''))
    if any(text in error for text in ('Interaction turn limit exhausted before simulator approval',
                                     'Claude CLI iteration budget exhausted before approval',
                                     'error_max_turns', 'Model failed to hand off within the round budget')):
        return 'max_turns_reached'
    if exit_code in (124, 137):
        return 'timeout'
    raise RuntimeError('Native controller failed before a valid terminal verdict: ' + error[:300])


def pod_creation_scheduler():
    from harbor_patch.environments.kubernetes.kubernetes import _RayPodCreationScheduler
    import ray
    if not ray.is_initialized():
        return None
    # Keep the cluster-wide actor alive between rollout waves, independently
    # of this function-local handle. Ray shutdown cleans it up.
    return _RayPodCreationScheduler.options(
        name='native_swe_pod_creation', lifetime='detached',
        get_if_exists=True, max_restarts=3, max_task_retries=3).remote(
        int(os.environ.get('HARBOR_K8S_POD_CREATE_CONCURRENCY', '16')),
        float(os.environ.get('HARBOR_K8S_POD_CREATE_INTERVAL_SEC', '0.25')), 300.0)


class SandboxExecTransportError(RuntimeError):
    """The exec transport did not deliver a trustworthy process exit status."""


class NativeEpisode:
    def __init__(self, proxy, session_id, row, options, run_dir, sampling_params):
        self.proxy, self.session_id, self.row = proxy, session_id, row
        self.options, self.run_dir = options, Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.task = Path(row['task_path'])
        self.name = 'swe-interact-rl-' + uuid4().hex[:12]
        self.namespace = os.environ.get('K8S_NAMESPACE', 'default')
        address = ipaddress.ip_address(proxy._advertise_host)
        if address.is_loopback or address.is_unspecified:
            raise ValueError('Native sandbox requires a routable proxy advertise address')
        self.host = str(address)
        self.budget = TurnBudget(int(os.environ.get('HARBOR_SWE_FIRST_ROUND_CALLS', '3')),
                                 int(os.environ.get('HARBOR_SWE_ROUND_CALLS', '10')))
        self.key = os.environ[options.simulator_api_key_env]
        self.sampling_params = dict(sampling_params)
        self.token = secrets.token_urlsafe(24)
        self.export_token = secrets.token_urlsafe(24)
        self.created = []
        self.summary = {'enabled': True, 'valid': False, 'schema_version': 1,
                        'task_profile': 'swe_interact', 'reward_mode': 'terminal',
                        'max_rounds': options.max_rounds, 'failure_stage': 'environment',
                        'simulator_model': options.simulator_model, 'task_id': row['task_id'],
                        'stop_reason': 'error', 'round_rewards': [], 'images': row['images']}
        # Never mutate the Kubernetes module's global configuration in workers.
        with config.new_client_from_config(os.environ['K8S_KUBECONFIG']) as api:
            self.api_config = api.configuration

    def api(self):
        return client.CoreV1Api(client.ApiClient(self.api_config))

    def execute(self, command, *, timeout=180, log=None, retry_read=False):
        # An interrupted command may still be running. Never replay agent or
        # verifier commands in place; only explicitly read-only calls may retry.
        deadline = time.monotonic() + timeout
        for attempt in range(3 if retry_read else 1):
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Native sandbox command exceeded its deadline')
                return self._execute_once(command, timeout=remaining, log=log)
            except Exception as exc:
                category, retryable = _native_failure_category(exc)
                if not retry_read or not retryable or attempt == 2:
                    raise
                self.summary['exec_read_retries'] = self.summary.get('exec_read_retries', 0) + 1
                time.sleep(min(attempt + 1, max(0, deadline - time.monotonic())))

    def _execute_once(self, command, *, timeout=180, log=None):
        # stream() temporarily changes ApiClient's transport: use a fresh client.
        api = self.api()
        ws = None
        output = []
        deadline = time.monotonic() + timeout
        try:
            for opening_attempt in range(3):
                try:
                    ws = stream(api.connect_get_namespaced_pod_exec, self.name, self.namespace,
                                container='main', command=['bash', '-lc', command],
                                stderr=True, stdin=False, stdout=True, tty=False, _preload_content=False)
                    break
                except Exception as exc:
                    # A rejected exec handshake explicitly reporting no container
                    # has NOT launched the command. Only that narrow case is safe
                    # to retry for writes; a lost established stream never is.
                    missing = any(t in str(exc).lower() for t in ('container not found', 'no running task found'))
                    if not missing or opening_attempt == 2:
                        raise
                    diagnostics = self.sandbox_diagnostics()
                    self.summary['last_exec_diagnostics'] = diagnostics
                    if diagnostics['phase'] in ('Failed', 'Succeeded') or any(
                        c.get('state', {}).get('exit_code') is not None for c in diagnostics['containers'] if c['name'] == 'main'):
                        raise
                    if time.monotonic() + 1 >= deadline:
                        raise
                    time.sleep(1)
            while ws.is_open():
                if time.monotonic() > deadline:
                    raise TimeoutError('Native sandbox command exceeded its deadline')
                ws.update(timeout=1)
                for read, peek in ((ws.read_stdout, ws.peek_stdout), (ws.read_stderr, ws.peek_stderr)):
                    if peek():
                        chunk = read()
                        output.append(chunk)
                        if log:
                            with Path(log).open('a') as f:
                                f.write(chunk)
            try:
                returncode = ws.returncode
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                # Kubernetes can put an OCI/runtime error string where its
                # Python client expects an integer ExitCode. This is not a
                # command failure (or a successful empty command).
                raise SandboxExecTransportError('Kubernetes exec returned an invalid exit status') from exc
            if not isinstance(returncode, int):
                raise SandboxExecTransportError('Kubernetes exec closed without a process exit status')
            return returncode, ''.join(output)
        finally:
            try:
                if ws is not None:
                    ws.close()
            finally:
                api.api_client.close()

    def sandbox_diagnostics(self):
        """Capture pod health before cleanup; never collect env or credentials."""
        api = self.api()
        try:
            pod = api.read_namespaced_pod(self.name, self.namespace, _request_timeout=(3, 7))
            containers = []
            for status in pod.status.container_statuses or []:
                item = {'name': status.name, 'ready': status.ready, 'restarts': status.restart_count}
                for label, state in [('state', status.state), ('last_state', status.last_state)]:
                    if state and state.terminated:
                        t = state.terminated
                        item[label] = {'reason': t.reason, 'exit_code': t.exit_code, 'signal': t.signal}
                    elif state and state.waiting:
                        item[label] = {'reason': state.waiting.reason}
                containers.append(item)
            return {'pod': self.name, 'node': pod.spec.node_name, 'phase': pod.status.phase,
                    'reason': pod.status.reason, 'containers': containers}
        finally:
            api.api_client.close()

    def upload(self, files):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode='w:gz') as archive:
            for name, (data, mode) in files.items():
                item = tarfile.TarInfo(name.lstrip('/'))
                item.size, item.mode = len(data), mode
                archive.addfile(item, io.BytesIO(data))
        encoded = base64.b64encode(buf.getvalue()).decode()
        code = 'import tarfile,io,base64;tarfile.open(fileobj=io.BytesIO(base64.b64decode(' + repr(encoded) + ")),mode='r:gz').extractall('/')"
        rc, out = self.execute('python3 -c ' + shlex.quote(code))
        if rc:
            raise RuntimeError('Native upload failed: ' + out[-500:])

    def test_files(self, step):
        files = {}
        for root in (self.task / 'tests', self.task / 'steps' / step / 'tests'):
            for path in root.rglob('*'):
                if path.is_file():
                    files['tests/' + str(path.relative_to(root))] = (path.read_bytes(), path.stat().st_mode & 0o777)
        return files

    async def prepare(self, messages, tools, body):
        messages, tools, handoff = self.budget.prepare(messages, tools)
        if handoff:
            await self.proxy.begin_tool_handoff(self.session_id)
            body = {**body, 'tool_choice': 'none'}
        # CC sends its own max_tokens but omits sampling fields. Preserve the
        # actual train/validation sampling distribution supplied by verl.
        body = {**body, **self.sampling_params}
        body['max_tokens'] = min(int(body.get('max_tokens', 4096)), 4096)
        return messages, tools, body

    async def gateway(self):
        async def simulator(request):
            if request.headers.get('Authorization') != 'Bearer ' + self.token:
                raise web.HTTPUnauthorized()
            body = await request.json()
            body['model'] = self.options.simulator_model.removeprefix('openai/')
            body['stream'] = False
            endpoint = self.options.simulator_api_base.rstrip('/') + '/chat/completions'
            async with self.http.post(endpoint, json=body, headers={'Authorization': 'Bearer ' + self.key}) as response:
                return web.Response(body=await response.read(), status=response.status, content_type='application/json')

        async def begin(request):
            if request.match_info['token'] != self.token:
                raise web.HTTPUnauthorized()
            number = int((await request.json())['round'])
            self.budget.begin(number)
            await self.proxy.begin_user_round(self.session_id, number)
            self.summary['num_user_rounds'] = number
            LOG.info('Native task=%s round=%s', self.row['task_id'], number)
            return web.json_response({'ok': True})

        app = web.Application()
        app.router.add_get('/health', lambda request: web.json_response({'ok': True}))
        app.router.add_post('/v1/chat/completions', simulator)
        app.router.add_post('/round/{token}', begin)
        self.gateway_runner = web.AppRunner(app, access_log=None)
        await self.gateway_runner.setup()
        site = web.TCPSite(self.gateway_runner, '0.0.0.0', 0)
        await site.start()
        self.gateway_port = site._server.sockets[0].getsockname()[1]

    async def create_sandbox(self):
        labels = {'app': self.name, 'lego-native-swe': 'true'}
        secret = {'apiVersion': 'v1', 'kind': 'Secret', 'metadata': {'name': self.name, 'labels': labels},
                  'stringData': {'export-token': self.export_token, 'gateway-token': self.token}}
        # Reserve cleanup ownership before create, including ambiguous API timeouts.
        self.created.append('secret')
        await asyncio.to_thread(self.api().create_namespaced_secret, self.namespace, secret)
        def env(values):
            return [{'name': k, 'value': str(v)} for k, v in values.items()]
        def secret_env(name, key):
            return {'name': name, 'valueFrom': {'secretKeyRef': {'name': self.name, 'key': key}}}
        workdir = self.row['workdir']
        pod = {'apiVersion': 'v1', 'kind': 'Pod', 'metadata': {'name': self.name, 'labels': labels}, 'spec': {
            'restartPolicy': 'Never', 'activeDeadlineSeconds': int(self.options.episode_timeout_sec) + 1800,
            'automountServiceAccountToken': False,
            'hostAliases': [{'ip': '127.0.0.1', 'hostnames': ['main', 'user-server']}],
            'volumes': [{'name': 'runtime', 'emptyDir': {}}],
            'initContainers': [{'name': 'install-runtime', 'image': os.environ['HARBOR_SWE_RUNTIME_IMAGE'],
                'imagePullPolicy': 'IfNotPresent',
                'command': ['sh', '-c', 'mkdir -p /runtime-share/opt/custom-agent-runtime/claude-code && cp -a /opt/custom-agent-runtime/claude-code/. /runtime-share/opt/custom-agent-runtime/claude-code/'],
                'volumeMounts': [{'name': 'runtime', 'mountPath': '/runtime-share'}],
                'resources': {'requests': {'cpu': '250m', 'memory': '256Mi'}, 'limits': {'cpu': '2', 'memory': '2Gi'}}}],
            'containers': [
                {'name': 'main', 'image': self.row['images']['main'], 'imagePullPolicy': 'IfNotPresent', 'workingDir': workdir,
                 'command': ['bash', '-lc', 'mkdir -p /logs/agent /logs/verifier /logs/artifacts; python3 /usr/local/bin/repo_exec_server.py & exec sleep infinity'],
                 'env': env({'REPO_EXEC_ROOT': workdir, 'SWE_INTERACT_REPO_WORKDIR': workdir,
                             'CARGO_BUILD_JOBS': '4', 'CMAKE_BUILD_PARALLEL_LEVEL': '4', 'MAKEFLAGS': '-j4',
                             'GIT_CONFIG_COUNT': '1', 'GIT_CONFIG_KEY_0': 'safe.directory', 'GIT_CONFIG_VALUE_0': workdir}),
                 'volumeMounts': [{'name': 'runtime', 'mountPath': '/runtime-image', 'readOnly': True}],
                 'resources': {'requests': {'cpu': '1', 'memory': '4Gi'}, 'limits': {'cpu': '4', 'memory': os.environ.get('HARBOR_SWE_MAIN_MEMORY_LIMIT', '24Gi')}}},
                {'name': 'user-server', 'image': self.row['images']['user-server'], 'imagePullPolicy': 'IfNotPresent',
                 'env': env({'LITELLM_BASE_URL': f'http://{self.host}:{self.gateway_port}',
                             'SIM_USER_MODEL': self.options.simulator_model, 'SIM_USER_LLM_STREAM': '0',
                             'SIM_USER_LLM_TIMEOUT_SEC': '180', 'SIM_USER_TOOL_MAX_RETRIES': '3',
                             'REPO_EXEC_URL': 'http://127.0.0.1:8765/tool'})
                        + [secret_env('LITELLM_API_KEY', 'gateway-token'), secret_env('SIM_USER_EXPORT_TOKEN', 'export-token')],
                 'readinessProbe': {'tcpSocket': {'port': 8001}, 'periodSeconds': 2},
                 'resources': {'requests': {'cpu': '250m', 'memory': '512Mi'}, 'limits': {'cpu': '2', 'memory': '2Gi'}}}]}}
        self.configure_sandbox(pod)
        policy = {'apiVersion': 'networking.k8s.io/v1', 'kind': 'NetworkPolicy', 'metadata': {'name': self.name}, 'spec': {
            'podSelector': {'matchLabels': labels}, 'policyTypes': ['Egress'], 'egress': [
                {'to': [{'ipBlock': {'cidr': self.host + '/32'}}], 'ports': [
                    {'protocol': 'TCP', 'port': self.proxy._port}, {'protocol': 'TCP', 'port': self.gateway_port}]},
                {'ports': [{'protocol': 'UDP', 'port': 53}, {'protocol': 'TCP', 'port': 53}]}]}}
        self.created.append('policy')
        await asyncio.to_thread(client.NetworkingV1Api(client.ApiClient(self.api_config)).create_namespaced_network_policy,
                                self.namespace, policy)
        scheduler = pod_creation_scheduler()
        lease = None
        if scheduler is not None:
            lease = await scheduler.acquire.remote(self.name)
        try:
            self.created.append('pod')
            await asyncio.to_thread(self.api().create_namespaced_pod, self.namespace, pod)
        finally:
            if lease:
                await scheduler.release.remote(lease)
        for _ in range(180):
            status = await asyncio.to_thread(self.api().read_namespaced_pod, self.name, self.namespace)
            containers = status.status.container_statuses or []
            if len(containers) == len(pod['spec']['containers']) and all(x.ready for x in containers):
                return
            if status.status.phase == 'Failed':
                raise RuntimeError('Native sandbox pod failed')
            await asyncio.sleep(5)
        raise TimeoutError('Native sandbox startup timed out')

    def configure_sandbox(self, pod):
        """Allow opt-in interaction protocols to reuse the pinned task sandbox."""

    async def install_runtime(self):
        harbor = Path(os.environ['HARBOR_SWE_NATIVE_ROOT'])
        sources = {
            'runtime/run_claude_harbor.py': harbor / 'scripts/agent_runtimes/claude-code/2.1.118/runtime/run_claude_harbor.py',
            'runtime/validation_policy.py': harbor / 'src/harbor/agents/custom/openhands_sdk_runner.py',
            'runtime/interaction_protocol.py': harbor / 'scripts/agent_runtimes/openhands-sdk/1.33.0/runtime/run_openhands_harbor.py',
            'runtime/native_cc_protocol.py': Path(__file__).with_name('native_cc_protocol.py'),
        }
        files = {RUNTIME + '/' + dest: (path.read_bytes(), 0o644) for dest, path in sources.items()}
        files[RUNTIME + '/bin/claude'] = (f'#!/bin/sh\nexec {IMAGE_RUNTIME}/bin/claude "$@"\n'.encode(), 0o755)
        files[RUNTIME + '/tools/harbor-validate'] = (f'#!/bin/sh\nexec {RUNTIME}/bin/python {RUNTIME}/runtime/run_claude_harbor.py validate "$@"\n'.encode(), 0o755)
        instrument = '''import os,json,sys,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_claude_harbor as native
from native_cc_protocol import claude_code_turn_prompt
original=native.run_cli
endpoint=os.environ.pop('NATIVE_ROUND_URL')
def query(*args,**kwargs):
 args=(claude_code_turn_prompt(args[0],kwargs['turn']),)+args[1:]
 req=urllib.request.Request(endpoint,data=json.dumps({'round':kwargs['turn']}).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=30) as response: assert response.status==200
 return original(*args,**kwargs)
native.run_cli=query
native.main()
'''
        files[RUNTIME + '/runtime/instrumented_driver.py'] = (instrument.encode(), 0o644)
        files['tmp/native-instruction.txt'] = ((self.task / 'steps/01_interactive_develop/instruction.md').read_bytes(), 0o644)
        files['tmp/native-secrets.json'] = (json.dumps({'interaction_turn_url': f'http://user-server:8001/interaction-turn?token={self.export_token}'}).encode(), 0o600)
        await asyncio.to_thread(self.upload, files)
        rc, out = await asyncio.to_thread(self.execute,
            f'ln -s {IMAGE_RUNTIME}/python-home {RUNTIME}/python-home && '
            f'cp {IMAGE_RUNTIME}/runtime-select-python.sh {RUNTIME}/runtime-select-python.sh && '
            f'cp {IMAGE_RUNTIME}/bin/python {RUNTIME}/bin/python && {RUNTIME}/bin/python --version && python3 --version')
        if rc:
            raise RuntimeError('Self-contained runtime setup failed: ' + out[-500:])
        self.summary['python_versions'] = out.strip().splitlines()

    async def execute_episode(self):
        self.summary['failure_stage'] = 'agent'
        values = {'CUSTOM_AGENT_RUNTIME_ROOT': RUNTIME, 'SWE_INTERACT_RUNTIME_SECRET_FILE': '/tmp/native-secrets.json',
                  'ANTHROPIC_BASE_URL': self.proxy.session_anthropic_base(self.session_id), 'ANTHROPIC_API_KEY': 'local-policy',
                  'ANTHROPIC_CUSTOM_HEADERS': '', 'CC_MODEL': os.environ.get('SERVED_MODEL_NAME', 'vllm_model'),
                  'CC_MAX_OUTPUT': '4096', 'CC_REASONING_EFFORT': 'high',
                  'SWE_INTERACT_MAX_TURNS': str(self.options.max_rounds),
                  'MAX_ITERATIONS': str(2 * self.options.max_rounds * self.budget.later_round_calls + 20),
                  'BASH_DEFAULT_TIMEOUT_MS': '30000', 'BASH_MAX_TIMEOUT_MS': '60000',
                  'SWE_INTERACT_TURN_TIMEOUT_SEC': '900', 'CLAUDE_CODE_DISABLE_AUTO_MEMORY': '1',
                  'DISABLE_AUTO_COMPACT': '1', 'DISABLE_COMPACT': '1',
                  'CLAUDE_CODE_MAX_CONTEXT_TOKENS': str(int(os.environ.get('ROLLOUT_MAX_MODEL_LEN', '64000'))),
                  'NATIVE_ROUND_URL': f'http://{self.host}:{self.gateway_port}/round/{self.token}'}
        prefix = ' '.join(k + '=' + shlex.quote(v) for k, v in values.items())
        command = (f'cd {shlex.quote(self.row["workdir"])} && env {prefix} timeout -k 15 -s TERM '
                   f'{int(self.options.episode_timeout_sec)} {RUNTIME}/bin/python {RUNTIME}/runtime/instrumented_driver.py '
                   '--instruction "$(cat /tmp/native-instruction.txt)" --logs-dir /logs/agent')
        rc, _ = await asyncio.to_thread(self.execute, command, timeout=self.options.episode_timeout_sec + 40,
                                       log=self.run_dir / 'agent-console.log')
        _, text = await asyncio.to_thread(self.execute, 'cat /logs/agent/interaction-result.json')
        native = json.loads(text)
        self.summary['native_result'] = native
        self.summary['failure_stage'] = 'verifier'
        await asyncio.to_thread(self.upload, self.test_files('01_interactive_develop'))
        gate_rc, _ = await asyncio.to_thread(self.execute, 'SIM_USER_EXPORT_TOKEN=' + shlex.quote(self.export_token) + ' bash /tests/test.sh',
                                             timeout=180, log=self.run_dir / 'approval-console.log')
        if gate_rc not in (0, 1):
            raise RuntimeError(f'Approval verifier execution failed ({gate_rc})')
        _, text = await asyncio.to_thread(self.execute, 'cat /logs/verifier/reward.txt')
        approval = float(text.strip())
        await asyncio.to_thread(self.execute, 'cp -a /logs/verifier /logs/verifier-step1 && rm -f /logs/verifier/reward.txt')
        final_rc, reward = None, 0.0
        if approval == 1:
            await asyncio.to_thread(self.upload, self.test_files('02_final_verify'))
            final_rc, _ = await asyncio.to_thread(self.execute, 'SIM_USER_EXPORT_TOKEN=' + shlex.quote(self.export_token) + ' timeout -s TERM 600 bash /tests/test.sh',
                                                  timeout=630, log=self.run_dir / 'final-console.log')
            _, text = await asyncio.to_thread(self.execute, 'cat /logs/verifier/reward.txt')
            reward = float(text.strip())
        reason = classify_result(native, approval, reward, final_rc,
                                 bool(self.proxy._sessions[self.session_id].get('context_overflow')), rc)
        self.summary.update(valid=True, failure_stage=None, final_reward=reward, approval_reward=approval,
                            final_verifier_exit_code=final_rc, agent_exit_code=rc, termination_reason=reason,
                            stop_reason=('verifier_success' if reward == 1 else reason))
        return reward, reason

    async def run(self):
        started = time.monotonic()
        self.gateway_runner = None
        self.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600), trust_env=True)
        try:
            await self.gateway()
            await self.create_sandbox()
            self.summary['failure_stage'] = 'runtime'
            await self.install_runtime()
            return await self.execute_episode()
        except Exception as exc:
            # Persist the safe retry category before finally writes multiturn.json.
            category, retryable = _native_failure_category(exc)
            self.summary.update(valid=False, error_type=type(exc).__name__,
                                retry_category=category, retryable=retryable)
            try:
                self.summary['sandbox_diagnostics'] = await asyncio.to_thread(self.sandbox_diagnostics)
            except Exception as diagnostic_error:
                self.summary['sandbox_diagnostics'] = {'error_type': type(diagnostic_error).__name__}
            raise
        finally:
            if 'pod' in self.created:
                try:
                    _, data = await asyncio.to_thread(self.execute,
                        "python3 -c \"import io,tarfile,base64; b=io.BytesIO(); t=tarfile.open(fileobj=b,mode='w:gz'); t.add('/logs',arcname='logs'); t.close(); print(base64.b64encode(b.getvalue()).decode())\"", timeout=120)
                    archive = base64.b64decode(data.strip())
                    (self.run_dir / 'artifacts.tar.gz').write_bytes(archive)
                    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                        tar.extractall(self.run_dir / 'artifacts', filter='data')
                except Exception as exc:
                    self.summary['artifact_error'] = type(exc).__name__
            for kind in reversed(self.created):
                try:
                    if kind == 'pod':
                        await asyncio.to_thread(self.api().delete_namespaced_pod, self.name, self.namespace, grace_period_seconds=5)
                    elif kind == 'policy':
                        await asyncio.to_thread(client.NetworkingV1Api(client.ApiClient(self.api_config)).delete_namespaced_network_policy, self.name, self.namespace)
                    else:
                        await asyncio.to_thread(self.api().delete_namespaced_secret, self.name, self.namespace)
                except client.ApiException as exc:
                    if exc.status != 404:
                        self.summary.setdefault('cleanup_errors', []).append(kind + ':' + str(exc.status))
                except Exception as exc:
                    self.summary.setdefault('cleanup_errors', []).append(kind + ':' + type(exc).__name__)
            if self.gateway_runner:
                await self.gateway_runner.cleanup()
            await self.http.close()
            self.key = ''
            self.summary['duration_sec'] = time.monotonic() - started
            (self.run_dir / 'multiturn.json').write_text(json.dumps(self.summary, indent=2) + '\n')


def _native_failure_category(exc):
    """Return a fixed, credential-free category for recoverable infrastructure errors."""
    text = str(exc).lower()
    if isinstance(exc, (WebSocketException, SandboxExecTransportError)):
        return 'sandbox_transport', True
    # A fresh sandbox does not fix a missing test binary or the CLI output cap.
    if 'final verifier failed to execute (127,' in text:
        return 'verifier_launch_error', False
    if 'response exceeded' in text and 'output token' in text:
        return 'agent_output_limit', False
    if 'nonconsecutive conversation turn' in text or 'http error 500' in text or 'conversation turn failed' in text:
        return 'conversation_gateway', True
    if type(exc).__name__ == 'SimulatorUnavailable' or 'simulator unavailable' in text:
        return 'simulator_unavailable', True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return 'transport_error', True
    if isinstance(exc, client.ApiException):
        status = getattr(exc, 'status', None)
        return 'kubernetes_api', status in (None, 0, 404, 408, 409, 429) or (isinstance(status, int) and status >= 500)
    if any(part in text for part in ('oci runtime exec failed', 'failed to exec in container',
                                    'container not found', 'connection reset', 'connection refused',
                                    'connection closed', 'timed out', 'http error 502', 'http error 504')):
        return 'sandbox_transport', True
    return 'unclassified_error', False


async def run_native_trial(loop, proxy, task_path, global_steps, sampling_params, metrics, is_val):
    """Retry transient infrastructure failures, with a fresh sandbox/session per attempt.

    Three retries means four total attempts. A valid reward of zero is final and
    is never resampled. Only the selected attempt contributes trajectory metrics.
    """
    max_retries = int(os.environ.get('HARBOR_SWE_TRIAL_MAX_RETRIES', '3'))
    if not 0 <= max_retries <= 3:
        raise ValueError('HARBOR_SWE_TRIAL_MAX_RETRIES must be between 0 and 3')
    retry_id = uuid4().hex
    record_path = Path(loop.base_trials_dir) / '_retries' / (retry_id + '.jsonl')
    for attempt in range(max_retries + 1):
        attempt_metrics = {}
        reward, reason, meta = await _run_native_trial_once(
            loop, proxy, task_path, global_steps, sampling_params, attempt_metrics, is_val)
        valid = bool(meta.get('multiturn', {}).get('valid'))
        failure = meta.get('native_retry', {})
        retry = not valid and failure.get('retryable', False) and attempt < max_retries
        record = {'task': Path(task_path).name, 'global_steps': global_steps,
                  'validate': is_val, 'attempt': attempt + 1, 'max_retries': max_retries,
                  'valid': valid, 'will_retry': retry,
                  'category': failure.get('category'), 'session_id': failure.get('session_id')}
        if attempt or retry:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            with record_path.open('a') as output:
                output.write(json.dumps(record) + '\n')
        if not retry:
            metrics.update(attempt_metrics)
            metrics['native_retry/retries'] = float(attempt)
            metrics['native_retry/exhausted'] = float(not valid and failure.get('retryable', False) and attempt == max_retries)
            meta['native_retry'] = {**failure, 'attempts': attempt + 1, 'max_retries': max_retries,
                                    'retry_id': retry_id}
            return reward, reason, meta
        LOG.warning('Retrying native trial task=%s attempt=%s/%s category=%s',
                    Path(task_path).name, attempt + 1, max_retries + 1, failure.get('category'))
        await asyncio.sleep(min(2 ** attempt, 8))


async def _run_native_trial_once(loop, proxy, task_path, global_steps, sampling_params, metrics, is_val):
    row = load_manifest(os.environ['HARBOR_SWE_INTERACT_MANIFEST'])[Path(task_path).name]
    if row['split'] != ('val' if is_val else 'train'):
        raise ValueError('Native dataset split does not match train/validation mode')
    if Path(task_path).resolve() != Path(row['task_path']).resolve():
        raise ValueError('Native task path differs from frozen manifest')
    session_id = 'swe-' + Path(task_path).name[:48] + '-' + uuid4().hex[:10]
    run_dir = Path(loop.base_trials_dir) / f'step_{global_steps:04d}' / session_id
    await proxy.open_session(session_id, trajectory_dir=run_dir, harness_name='claude_code')
    episode = None
    meta = {}
    failure = {'category': None, 'retryable': False, 'session_id': session_id}
    mode = os.environ.get('HARBOR_SWE_INTERACTION_MODE', 'native_review')
    try:
        if mode == 'conversation':
            from .swe_conversation import ConversationEpisode
            episode_type = ConversationEpisode
        elif mode == 'native_review':
            episode_type = NativeEpisode
        else:
            raise ValueError('Unknown SWE interaction mode: ' + mode)
        episode = episode_type(proxy, session_id, row, loop.multiturn, run_dir, sampling_params)
        proxy._sessions[session_id]['native_round_controller'] = episode
        proxy._sessions[session_id]['sampling_defaults'] = dict(sampling_params)
        reward, reason = await episode.run()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        LOG.exception('Invalid native SWE-Interact trajectory task=%s', row['task_id'])
        category, retryable = _native_failure_category(exc)
        failure.update(category=category, retryable=retryable)
        if episode:
            episode.summary.update(valid=False, error_type=type(exc).__name__,
                                   retry_category=category, retryable=retryable)
        reward, reason = 0.0, 'env_setup_failed'
    finally:
        proxy._sessions.get(session_id, {}).pop('native_round_controller', None)
        meta = await proxy.pop_session(session_id)
        loop._assign_proxy_metrics_from_session(metrics, meta)
    meta['native_retry'] = failure
    meta['multiturn'] = episode.summary if episode else {'enabled': True, 'valid': False, 'task_profile': 'swe_interact'}
    if not meta['multiturn'].get('valid'):
        meta['disable_proxy_trajectory'] = True
    elif mode == 'conversation' and loop.multiturn.judge_enabled:
        # The verifier has produced T. Judge failure or abstention retains T.
        from .swe_process_judge import blend_reward, judge
        terminal_reward = float(reward)
        try:
            context = {'task_id': row['task_id'], **episode.context}
            judgment = await judge(run_dir, context, loop.multiturn)
            (run_dir / 'judge_result.json').write_text(json.dumps(judgment, ensure_ascii=False, indent=2))
            episode.summary.update(judge_status=judgment['status'],
                                   judge_model=judgment.get('model'),
                                   judge_usage=judgment.get('usage'),
                                   judge_score=judgment.get('judge_score'))
            reward = blend_reward(terminal_reward, judgment,
                                  loop.multiturn.test_weight, loop.multiturn.judge_weight)
        except Exception as exc:
            LOG.warning('SWE judge unavailable task=%s error=%s', row['task_id'], type(exc).__name__)
            episode.summary.update(judge_status='error', judge_error_type=type(exc).__name__)
        episode.summary.update(terminal_reward=terminal_reward, training_reward=float(reward))
        (run_dir / 'multiturn.json').write_text(json.dumps(episode.summary, indent=2) + '\n')
    return reward, reason, meta
