#!/usr/bin/env python3
"""
Purge INFRA/verifier-failed trials so the inference script's resume logic
(get_completed_trials, which only skips status=="completed") will RE-RUN them.

Background: infra-failed trials are still written with status=="completed" and a
non-null `error` (or a verifier that never really ran the model's patch), so a
plain re-run skips them and bakes false reward=0 into the dataset.  This deletes
those trials (both <inst>/trial_<i>.json and <inst>/harbor_trials/<inst>_trial_<i>/)
while KEEPING genuine model failures and any reward>0.

Usage:
    # dry-run (default): classify + write manifest, delete nothing
    python3 utils/purge_infra_failed_trials.py [RESULTS_DIR]

    # actually delete
    APPLY=1 python3 utils/purge_infra_failed_trials.py [RESULTS_DIR]

    # also purge the ambiguous "agent cmd exit1" bucket (off by default)
    INCLUDE_EXIT1=1 APPLY=1 python3 utils/purge_infra_failed_trials.py [RESULTS_DIR]
"""
import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

DEFAULT_RD = "/path/to/shared/test_data_bcd/infer_openswe_filtered_qwen3_30b_ohsdk_d"
RD = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RD)
APPLY = os.environ.get("APPLY", "") not in ("", "0", "false", "False")
INCLUDE_EXIT1 = os.environ.get("INCLUDE_EXIT1", "") not in ("", "0", "false", "False")


def read_text(p: Path) -> str:
    try:
        return p.read_text(errors="ignore")
    except Exception:
        return ""


def classify(trial_json: dict, test_stdout: str) -> tuple[str, bool]:
    """Return (category, is_infra). is_infra=True -> candidate for deletion."""
    r = trial_json.get("reward")
    if r and r > 0:
        return "KEEP: reward>0", False

    e = trial_json.get("error")
    if e:
        s = str(e)
        if "Handshake" in s or "ApiException" in s:
            return "INFRA: k8s exec handshake 4xx", True
        if "exit 137" in s:
            return "INFRA: OOM (exit137)", True
        if "AgentTimeout" in s:
            return "KEEP: agent timeout 2400s (model)", False
        if "exit 1)" in s:
            return "INFRA?: agent cmd exit1", INCLUDE_EXIT1
        if "Verifier" in s or "TestsDir" in s or "RewardFileNotFound" in s:
            return "INFRA: verifier setup", True
        return f"INFRA: other({s.split(':')[0][:24]})", True

    # NO_ERROR -> look at the verifier output
    to = test_stdout
    m = re.search(r"OPENSWE_EXIT_CODE=(\d+)", to)
    ec = m.group(1) if m else None
    if "could not apply test patch" in to or "patch does not apply" in to.lower():
        return "INFRA: gold test_patch won't apply", True
    if ec == "2" or "during collection" in to:
        return "INFRA: pytest collection/import error", True
    if ec == "127" or "command not found" in to:
        return "INFRA: command not found (env)", True
    if ec == "1":
        return "KEEP: tests ran & failed (model)", False
    if ec in ("3", "4", "5"):
        return "INFRA: pytest usage/env (exit3-5)", True
    if ec in ("134", "139"):
        return "INFRA: verifier crash (segv/abort)", True
    if not to.strip():
        return "INFRA: no verifier output", True
    return "KEEP?: ran, no exitcode marker (kept)", False


def main():
    cat = Counter()
    cat_infra = {}  # category -> bool (is this category being deleted)
    to_delete = []  # (trial_json_path, harbor_trial_dir or None)
    n_trials = 0

    for d in sorted(RD.iterdir()):
        if not d.is_dir():
            continue
        for tf in d.glob("trial_*.json"):
            try:
                tj = json.loads(tf.read_text())
            except Exception:
                continue
            n_trials += 1
            tname = f"{d.name}_trial_{tj['trial_idx']}"
            ht = d / "harbor_trials" / tname
            tout = ht / "verifier" / "test-stdout.txt"
            category, is_infra = classify(tj, read_text(tout) if tout.exists() else "")
            cat[category] += 1
            cat_infra[category] = is_infra
            if is_infra:
                to_delete.append((tf, ht if ht.exists() else None))

    print(f"RESULTS_DIR = {RD}")
    print(f"APPLY = {APPLY}   INCLUDE_EXIT1 = {INCLUDE_EXIT1}")
    print(f"scanned trials = {n_trials}\n")
    print("classification:")
    for c, v in cat.most_common():
        mark = "DELETE" if cat_infra.get(c) else " keep "
        print(f"  {mark}  {c:42s} {v:6d}")
    print(f"\n>>> trials to delete: {len(to_delete)}")

    manifest = RD / "_purge_manifest.txt"
    with manifest.open("w") as f:
        for tf, ht in to_delete:
            f.write(f"{tf}\t{ht or ''}\n")
    print(f">>> manifest written: {manifest}")

    if not APPLY:
        print("\nDRY-RUN: nothing deleted. Re-run with APPLY=1 to delete.")
        return

    deleted_json = deleted_dir = 0
    for tf, ht in to_delete:
        try:
            tf.unlink()
            deleted_json += 1
        except FileNotFoundError:
            pass
        if ht is not None:
            shutil.rmtree(ht, ignore_errors=True)
            deleted_dir += 1
    print(f"\nDELETED {deleted_json} trial_*.json and {deleted_dir} harbor_trials/ dirs.")
    print("Re-run the inference script; resume will now regenerate these trials.")


if __name__ == "__main__":
    main()
