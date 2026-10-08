import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]


def check(**updates):
    env = dict(os.environ, MODEL_PATH="/tmp", SCAFFOLD="oc", ENABLE_R3="False",
               LR_SCHEDULER="constant", ADV_ESTIMATOR="grpo", MODEL_ENGINE="veomni",
               PF_STRUCTURE_ONLY="1", PYTHON_BIN=sys.executable, PREFLIGHT_EMBEDDED="1", IMAGE_REGISTRY="registry.example", VAL_TIMEOUT="60")
    env.pop("CONFIG", None)
    env.update(updates)
    return subprocess.run(["bash", str(ROOT / "scripts/lib/preflight.sh")], env=env,
                          text=True, capture_output=True)


@pytest.mark.parametrize("engine", ["fsdp", "fsdp2", "veomni"])
def test_supported_episode_configs(engine):
    result = check(HARBOR_TRAJECTORY_SELECTION="all", MODEL_ENGINE=engine)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("updates,diagnostic", [
    ({"ADV_ESTIMATOR": "gae", "CRITIC_ENABLE": "True"}, "requires ADV_ESTIMATOR=grpo"),
    ({"USE_KL_IN_REWARD": "True"}, "requires USE_KL_IN_REWARD=False"),
    ({"MODEL_ENGINE": "megatron"}, "requires FSDP/FSDP2 or VeOmni"),
    ({"HARBOR_TRAJECTORY_SELECTION": "frist"}, "must be longest or all"),
])
def test_invalid_episode_configs_fail_before_launch(updates, diagnostic):
    result = check(**({"HARBOR_TRAJECTORY_SELECTION": "all"} | updates))
    assert result.returncode != 0
    assert diagnostic in result.stderr


def test_longest_and_eval_keep_existing_behavior():
    assert check(HARBOR_TRAJECTORY_SELECTION="longest", MODEL_ENGINE="megatron").returncode == 0
    assert check(PF_KIND="eval", HARBOR_TRAJECTORY_SELECTION="all", MODEL_ENGINE="megatron").returncode == 0


@pytest.mark.parametrize("valid,code,message", [(8, 0, "task paths resolve"), (0, 1, "task paths do NOT exist"), (4, 0, "partially missing")])
def test_real_parquet_task_paths(tmp_path, valid, code, message):
    paths = []
    for i in range(8):
        p = tmp_path / f"task{i}"
        p.mkdir()
        (p / "instruction.md").write_text("fix the bug")
        if i < valid:
            (p / "task.toml").write_text("")
        paths.append({"harbor_task_path": str(p)})
    index = tmp_path / "index.parquet"
    pd.DataFrame({"extra_info": paths}).to_parquet(index)
    result = check(PF_STRUCTURE_ONLY="0", TRAIN_INDEX=str(index), VAL_INDEX=str(index))
    assert result.returncode == code, result.stdout + result.stderr
    assert message in result.stdout + result.stderr


def test_non_harbor_index_and_structure_mode_skip(tmp_path):
    index = tmp_path / "index.parquet"
    pd.DataFrame({"extra_info": [{"other": "value"}]}).to_parquet(index)
    result = check(PF_STRUCTURE_ONLY="0", TRAIN_INDEX=str(index))
    assert result.returncode == 0
    assert "no harbor_task_path" in result.stdout
    result = check(TRAIN_INDEX="/missing/index.parquet")
    assert result.returncode == 0
    assert "task path resolution: structure-only" in result.stdout


@pytest.mark.parametrize("selection,expected", [("all", True), ("longest", False)])
def test_worker_override_is_scoped_to_all(selection, expected):
    # Execute the real argument builder; unrelated unset settings are irrelevant here.
    script = "source scripts/train/lib/hydra_args.sh; var_is_set() { return 1; }; init_hydra_args; append_common_hydra_args; printf '%s\n' \"${hydra_args[@]}\""
    result = subprocess.run(["bash", "-c", script], cwd=ROOT, env=dict(os.environ,
                            HARBOR_TRAJECTORY_SELECTION=selection, MODEL_ENGINE="fsdp"),
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert ("++trainer.use_legacy_worker_impl=disable" in result.stdout.splitlines()) == expected


def test_empty_or_unreadable_index_is_not_reported_as_valid(tmp_path):
    index = tmp_path / "index.parquet"
    pd.DataFrame({"extra_info": []}).to_parquet(index)
    result = check(PF_STRUCTURE_ONLY="0", TRAIN_INDEX=str(index))
    assert result.returncode == 0
    assert "has 0 rows" in result.stderr
    index.write_text("not parquet")
    result = check(PF_STRUCTURE_ONLY="0", TRAIN_INDEX=str(index))
    assert result.returncode == 0
    assert "unreadable" in result.stdout
    assert "task paths resolve" not in result.stdout
