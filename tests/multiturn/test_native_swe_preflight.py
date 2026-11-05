import importlib.util
from pathlib import Path

import pytest

root = Path(__file__).resolve().parents[2]
source = root / 'utils/preflight_native_swe.py'
if not source.exists():
    source = Path(__file__).with_name('preflight_native_swe.py')
spec = importlib.util.spec_from_file_location('native_swe_preflight', source)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


@pytest.fixture
def resume_env(tmp_path):
    path = tmp_path / 'global_step_10'
    (path / 'actor').mkdir(parents=True)
    (path / 'critic').mkdir()
    (path / 'data.pt').write_bytes(b'checkpoint-test-fixture')
    return {
        'TRAINER_RESUME_MODE': 'resume_path',
        'TRAINER_RESUME_FROM_PATH': str(path),
        'HARBOR_SWE_REQUIRE_BASE_WEIGHTS': 'false',
        'CRITIC_ENABLE': 'True',
    }


def test_acceptance_defaults_to_base_weights():
    assert preflight.validate_resume({'TRAINER_RESUME_MODE': 'disable'}) == {'resume_mode': 'disable'}


def test_acceptance_still_rejects_resume(resume_env):
    resume_env.pop('HARBOR_SWE_REQUIRE_BASE_WEIGHTS')
    with pytest.raises(ValueError, match='Acceptance run'):
        preflight.validate_resume(resume_env)


def test_explicit_production_resume_reports_real_mode_and_path(resume_env):
    assert preflight.validate_resume(resume_env) == {
        'resume_mode': 'resume_path', 'resume_from_path': resume_env['TRAINER_RESUME_FROM_PATH'],
    }


@pytest.mark.parametrize('path', ['', 'global_step_10', '/tmp/global_step_bad', '/tmp/actor'])
def test_invalid_resume_path_is_rejected(resume_env, path):
    resume_env['TRAINER_RESUME_FROM_PATH'] = path
    with pytest.raises(ValueError, match='absolute global_step_N'):
        preflight.validate_resume(resume_env)


@pytest.mark.parametrize('part', ['actor', 'critic', 'data.pt'])
def test_incomplete_checkpoint_is_rejected(resume_env, part):
    path = Path(resume_env['TRAINER_RESUME_FROM_PATH']) / part
    path.rmdir() if path.is_dir() else path.unlink()
    with pytest.raises(ValueError, match='Missing'):
        preflight.validate_resume(resume_env)


def test_auto_resume_cannot_silently_fall_back_to_fresh_training(resume_env):
    resume_env['TRAINER_RESUME_MODE'] = 'auto'
    with pytest.raises(ValueError, match='explicit resume_path'):
        preflight.validate_resume(resume_env)
