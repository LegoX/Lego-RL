#!/usr/bin/env python3
"""Offline SWE-bench grader (Verified + Multilingual), run by the task's tests/test.sh.

Equivalent to ``swebench.harness.grading.get_logs_eval`` + ``get_eval_tests_report`` +
``get_resolution_status`` but without ``make_test_spec`` (which needs the repo/version
constants and, for some versions, network) and without ``MAP_REPO_VERSION_TO_SPECS``.
Only the pieces the log parsers really use are provided (``instance_id``, ``repo``,
``version``, ``FAIL_TO_PASS``, ``PASS_TO_PASS``).

Runs under the self-contained interpreter shipped next to it
(``/opt/grading/python311/bin/python3``, python-build-standalone + ``swebench==4.1.0``),
so it works inside images that have no usable Python at all (Go / Rust / Java / ...).

Inputs (same contract as the original parser_offline.py):
  /tests/config.json   the upstream SWE-bench record (needs repo, version, instance_id,
                       FAIL_TO_PASS, PASS_TO_PASS; ``log_parser`` is used when present)
  $LOG_FILE            captured verifier output

Outputs:
  /logs/verifier/report.json
  stdout: "SWEBench results starts here" / PASSED|FAILED / "SWEBench results ends here"
  exit 0 = resolved, 1 = not resolved, 3 = parser could not run (test.sh then falls
  back to the online ``uv run`` parser).  The END marker is only printed on a valid
  verdict, so a crash is never mistaken for reward=0.
"""
import json
import os
import re
import sys

CONFIG_PATH = os.environ.get("SWEBENCH_CONFIG_JSON", "/tests/config.json")
REPORT_DIR = os.environ.get("SWEBENCH_REPORT_DIR", "/logs/verifier")

try:
    from swebench.harness.constants import (
        APPLY_PATCH_FAIL,
        END_TEST_OUTPUT,
        FAIL_ONLY_REPOS,
        FAIL_TO_PASS,
        KEY_INSTANCE_ID,
        PASS_TO_PASS,
        RESET_FAILED,
        START_TEST_OUTPUT,
        TESTS_ERROR,
        TESTS_TIMEOUT,
        EvalType,
        ResolvedStatus,
    )
    from swebench.harness.grading import get_eval_tests_report, get_resolution_status
    import swebench.harness.log_parsers as log_parsers_mod
    from swebench.harness.log_parsers import MAP_REPO_TO_PARSER

    class MiniSpec:
        def __init__(self, repo, version, instance_id, f2p, p2p):
            self.repo = repo
            self.version = str(version)
            self.instance_id = instance_id
            self.FAIL_TO_PASS = f2p
            self.PASS_TO_PASS = p2p

    def aslist(x):
        if isinstance(x, str):
            try:
                return json.loads(x)
            except json.JSONDecodeError:
                import ast

                return ast.literal_eval(x)
        return list(x)

    datum = json.load(open(CONFIG_PATH))
    iid = datum.get(KEY_INSTANCE_ID) or datum["instance_id"]
    repo = datum["repo"]
    spec = MiniSpec(repo, datum.get("version", ""), iid,
                    aslist(datum[FAIL_TO_PASS]), aslist(datum[PASS_TO_PASS]))

    # Pick the parser the official harness would use for this repo. Newer
    # self-contained datasets (Multilingual) also name it explicitly.
    parser = MAP_REPO_TO_PARSER.get(repo)
    if parser is None and datum.get("log_parser"):
        parser = getattr(log_parsers_mod, datum["log_parser"], None)
    if parser is None:
        raise KeyError(f"no log parser for repo {repo!r} (log_parser={datum.get('log_parser')!r})")

    log_fp = os.environ["LOG_FILE"]
    with open(log_fp, "r+", errors="replace") as f:
        content = f.read()
        # The eval script emits the markers as `: '>>>>> Start Test Output'`; under
        # `set -x` bash traces them as `+ : '>>>>> Start Test Output'`. Normalise so
        # the official marker split below works.
        content = re.sub(r"^\++\s*:\s*'(>>>>>[^']*)'\s*$", r"\1", content, flags=re.MULTILINE)
        if START_TEST_OUTPUT not in content or END_TEST_OUTPUT not in content:
            content = f"{START_TEST_OUTPUT}\n{content}\n{END_TEST_OUTPUT}"
        f.seek(0)
        f.write(content)
        f.truncate()

    # --- get_logs_eval, minus the make_test_spec dependency ------------------------
    bad_codes = [c for c in (APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT) if c in content]
    if bad_codes:
        status_map, found = {}, False
    else:
        test_content = content.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
        status_map = parser(test_content, spec)
        if not status_map:
            status_map = parser(content, spec)
        found = True

    report_map = {iid: {"patch_is_None": False, "patch_exists": True,
                        "patch_successfully_applied": bool(found), "resolved": False,
                        "bad_codes": bad_codes}}
    if found:
        ref = {KEY_INSTANCE_ID: spec.instance_id, FAIL_TO_PASS: spec.FAIL_TO_PASS,
               PASS_TO_PASS: spec.PASS_TO_PASS}
        et = EvalType.FAIL_ONLY if spec.repo in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL
        report = get_eval_tests_report(status_map, ref, eval_type=et)
        if get_resolution_status(report) == ResolvedStatus.FULL.value:
            report_map[iid]["resolved"] = True
        report_map[iid]["tests_status"] = report

    os.makedirs(REPORT_DIR, exist_ok=True)
    json.dump(report_map, open(os.path.join(REPORT_DIR, "report.json"), "w"), indent=2)
    print("SWEBench results starts here")
    print("PASSED" if report_map[iid]["resolved"] else "FAILED")
    print("SWEBench results ends here")  # only reached on a valid verdict
    sys.exit(0 if report_map[iid]["resolved"] else 1)
except SystemExit:
    raise
except Exception as e:  # noqa: BLE001
    import traceback

    traceback.print_exc()
    print(f"OFFLINE_PARSER_ERROR: {e!r}")  # no END marker -> test.sh falls back online
    sys.exit(3)
