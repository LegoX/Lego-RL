"""Render native SWE-Interact control guidance for the Claude Code tool API."""

VALIDATION_GUIDANCE = """
[Claude Code runtime guidance]
The maintainer's generic terminal-validation instructions map to this Claude Code
runtime as follows: Bash has no validation=true parameter. To provide trustworthy
validation evidence, call Bash with a single command beginning with:
  harbor-validate --cwd . -- <your focused test/build command>
For example, use `harbor-validate --cwd . -- python -m pytest path/to/your_test.py`.
Choose the actual relevant check; the example is not a task-specific test path.
Inspect your diff BEFORE this check, then make this check your LAST tool action
before your plain-text reply. Do not use pipes, redirection, shell wrappers, or
extra commands. Ordinary pytest through Bash does not attach approval evidence.
If validation fails, fix the problem or report it honestly. Only the maintainer
can approve completion; this guidance does not change the task requirements.
""".strip()


def claude_code_turn_prompt(prompt: str, turn: int) -> str:
    """Preserve maintainer feedback and clarify its runtime-specific invocation."""
    if turn < 1:
        raise ValueError("User round numbers start at 1")
    return prompt if turn == 1 else prompt + "\n\n" + VALIDATION_GUIDANCE
