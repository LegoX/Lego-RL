# Benchmark controls from LegoX/harbor-internal 91c4902 (zf_multiturn).
from pydantic import BaseModel, Field


class MultiTurnBehaviorConfig(BaseModel):
    id: str
    type: str
    instruction: str = ""


class MultiTurnRoundResult(BaseModel):
    round_id: int
    instruction: str
    instruction_path: str = ""
    artifacts: dict[str, str] = Field(default_factory=dict)


CATEGORY = {
    "behavior_ids": [
        "debug-followup",
        "add-tests",
        "failure-report",
        "correct-requirement",
    ],
    "description": "Prioritize follow-ups that help the coding agent satisfy benchmark test "
    "cases and recover from failing verification.",
    "jobs_subdir": "higher-test-case-pass-rate",
    "round_verifier_feedback": {
        "enabled": True,
        "max_test_file_chars": 12000,
        "max_verifier_output_chars": 12000,
    },
    "simulator_guidance": "Optimize the next user message for improving final verifier pass "
    "rate. Prefer concrete failure evidence, targeted debugging, "
    "missing edge cases, focused regression tests or verification "
    "commands, and requirement clarifications that keep the agent "
    "aligned with the benchmark task. Avoid broad design reviews, Git "
    "workflow requests, or repository-tour questions unless they "
    "directly unblock a test failure. When round_verifier_feedback is "
    "present, use its sanitized outcome to choose the next follow-up "
    "without inventing or disclosing specific hidden tests, verifier "
    "output, or reward values.",
}

BEHAVIORS = {
    "add-tests": {
        "instruction": "Simulate a user who wants confidence that the fix will "
        "not regress. Ask for focused regression tests or targeted "
        "verification commands that cover the original issue and "
        "any newly discussed edge cases. The user should prefer "
        "minimal, relevant tests over broad unrelated test churn, "
        "and should expect the agent to explain what was tested.",
        "type": "test",
    },
    "correct-requirement": {
        "instruction": "Simulate a user correction or requirement "
        "change. Point out that the previous "
        "interpretation was slightly wrong, incomplete, "
        "or too narrow, then provide a clarified "
        "requirement or adjacent case that must also be "
        "handled. The user should expect the agent to "
        "adapt the existing solution with minimal code "
        "changes and verify that the original behavior "
        "still works.",
        "type": "correct_requirement",
    },
    "debug-followup": {
        "instruction": "Simulate a user who is not fully convinced by the "
        "previous answer. Ask the agent to inspect whether "
        "any important failure mode, edge case, or incomplete "
        "part remains. The user should reference the previous "
        "patch or behavior at a high level, request concrete "
        "diagnosis, ask the agent to diagnose from evidence, "
        "and expect the agent to read relevant code before "
        "making further changes.",
        "type": "debug",
    },
    "failure-report": {
        "instruction": "Simulate a user who tried the agent's solution and "
        "found that it still fails. Provide a concise failure "
        "report, such as an error message, failing test name, "
        "unexpected output, or reproduction hint. The user "
        "should ask the agent to investigate the failure from "
        "evidence, avoid guessing, and update the fix only "
        "after identifying the likely root cause.",
        "type": "failure_report",
    },
}

PROFILE = {
    "persona": "balanced maintainer",
    "style": "Concise and evidence-seeking. Ask for practical clarification, focused tests, "
    "and minimal fixes when appropriate.",
    "task_stratum": "bugfix",
}
