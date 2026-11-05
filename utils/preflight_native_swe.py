#!/usr/bin/env python3
"""Fail before Ray/GPU startup when native tasks, images or cluster are unavailable."""
import concurrent.futures
import json
import os
from pathlib import Path
import urllib.request

def validate_resume(env):
    """Keep acceptance runs fresh; allow explicit, checked production resumes."""
    mode = env.get('TRAINER_RESUME_MODE', 'disable')
    require_base = env.get('HARBOR_SWE_REQUIRE_BASE_WEIGHTS', 'true').lower()
    if require_base not in ('true', 'false'):
        raise ValueError('HARBOR_SWE_REQUIRE_BASE_WEIGHTS must be true or false')
    if require_base == 'true' and mode != 'disable':
        raise ValueError('Acceptance run must start from base weights')
    if mode == 'disable':
        return {'resume_mode': mode}
    if mode != 'resume_path':
        raise ValueError('Native SWE continuation requires explicit resume_path mode')
    path = Path(env.get('TRAINER_RESUME_FROM_PATH', ''))
    if not path.is_absolute() or not path.name.startswith('global_step_') or not path.name[12:].isdigit():
        raise ValueError('TRAINER_RESUME_FROM_PATH must be an absolute global_step_N directory')
    if not (path / 'actor').is_dir():
        raise ValueError(f'Missing actor checkpoint: {path / "actor"}')
    if env.get('CRITIC_ENABLE', 'false').lower() == 'true' and not (path / 'critic').is_dir():
        raise ValueError(f'Missing critic checkpoint: {path / "critic"}')
    data = path / 'data.pt'
    if not data.is_file() or data.stat().st_size == 0:
        raise ValueError(f'Missing dataloader checkpoint: {data}')
    return {'resume_mode': mode, 'resume_from_path': str(path)}


def main():
    resume = validate_resume(os.environ)
    import pandas as pd
    from kubernetes import client, config
    from harbor_patch.multiturn.native_swe import load_manifest

    manifest = load_manifest(os.environ['HARBOR_SWE_INTERACT_MANIFEST'])
    mode = os.environ.get('HARBOR_SWE_INTERACTION_MODE', 'native_review')
    assert mode in ('native_review', 'conversation'), 'Unknown SWE interaction mode'
    if mode == 'conversation':
        from harbor_patch.multiturn.swe_conversation import task_context
        assert int(os.environ.get('HARBOR_SWE_EPISODE_MAX_ITERATIONS', '500')) > 0
    counts = {}
    repositories = {}
    for split, env in [('train', 'TRAIN_FILES'), ('val', 'VAL_FILES')]:
        frame = pd.read_parquet(os.environ[env])
        ids = {x['instance_id'] for x in frame.extra_info}
        selected = {k for k, r in manifest.items() if r['split'] == split}
        assert ids == selected and len(ids) == len(frame), 'Index/manifest mismatch'
        counts[split] = len(ids)
        repositories[split] = {manifest[k]['repository'] for k in ids}
        for key in ids:
            row = manifest[key]
            root = Path(row['task_path'])
            if mode == 'conversation':
                task_context(root)
            for relative in ['task.toml', 'steps/01_interactive_develop/instruction.md',
                             'steps/01_interactive_develop/tests/test.sh', 'steps/02_final_verify/tests/test.sh']:
                assert (root / relative).is_file(), f'Missing native task file: {root / relative}'
    assert not repositories['train'] & repositories['val'], 'Repository overlap'
    native_root = Path(os.environ['HARBOR_SWE_NATIVE_ROOT'])
    for relative in ['scripts/agent_runtimes/claude-code/2.1.118/runtime/run_claude_harbor.py',
                     'scripts/agent_runtimes/openhands-sdk/1.33.0/runtime/run_openhands_harbor.py',
                     'src/harbor/agents/custom/openhands_sdk_runner.py']:
        assert (native_root / relative).is_file(), 'Missing native runtime source: ' + relative
    api_client = config.new_client_from_config(os.environ['K8S_KUBECONFIG'])
    nodes = client.CoreV1Api(api_client).list_node(_request_timeout=15).items
    assert any(any(c.type == 'Ready' and c.status == 'True' for c in n.status.conditions) for n in nodes), 'No ready sandbox node'
    images = {image for row in manifest.values() for image in row['images'].values()}
    images.add(os.environ['HARBOR_SWE_RUNTIME_IMAGE'])
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    accept = 'application/vnd.oci.image.index.v1+json,application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.docker.distribution.manifest.v2+json'
    def check(image):
        host, suffix = image.split('/', 1)
        repo, digest = suffix.split('@', 1)
        request = urllib.request.Request(f'http://{host}/v2/{repo}/manifests/{digest}', method='HEAD', headers={'Accept': accept})
        with opener.open(request, timeout=15) as response:
            assert response.headers.get('Docker-Content-Digest') == digest, 'Image digest mismatch'
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(check, images))
    print(json.dumps({'native_preflight': 'passed', 'counts': counts, 'verified_images': len(images),
                      'sandbox_nodes': len(nodes), **resume, 'interaction_mode': mode,
                      'approval_required': mode == 'native_review'}))


if __name__ == '__main__':
    main()
