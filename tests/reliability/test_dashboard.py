import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("dashboard_under_test", ROOT / "webui/server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


def test_duplicate_log_paths_are_scanned_once(tmp_path):
    folder = tmp_path / "harbor_trials" / "run" / "experiment"
    folder.mkdir(parents=True)
    log = tmp_path / "train.log"
    log.write_text(f"trials={folder}/step_0001\ntrials=//{folder}/step_0002\n")
    assert server._exp_dirs_from_log(str(log)) == [str(folder)]


def test_same_swe_task_across_harnesses_has_one_identity():
    names = [f"{h}-django__django-11099-abcdef12" for h in ["oc", "cc", "ohsdk", "cx", "codex"]]
    assert {server._strip_trial_hash(n) for n in names} == {"django__django-11099"}
    assert server._strip_trial_hash("cc-custom-task-abcdef12") == "cc-custom-task"
    assert server._strip_trial_hash("django__django-11099-abcdef12") == "django__django-11099"


def test_hyphenated_repository_owner_keeps_one_task_identity():
    assert server._strip_trial_hash("cc-scikit-learn__scikit-learn-123-abcdef12") == "scikit-learn__scikit-learn-123"
