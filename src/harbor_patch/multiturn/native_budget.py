"""Bound coding turns without inventing a solver answer or approval evidence."""

from copy import deepcopy


class TurnBudgetError(RuntimeError):
    status_code = 400


class TurnBudget:
    def __init__(self, first_round_calls=3, later_round_calls=10):
        if min(first_round_calls, later_round_calls) < 2:
            raise ValueError("Reserve at least one coding call and one handoff call")
        self.first_round_calls = first_round_calls
        self.later_round_calls = later_round_calls
        self.round = 0
        self.calls = 0

    def begin(self, number):
        if number != self.round + 1:
            raise ValueError("User rounds must be consecutive")
        self.round, self.calls = number, 0

    def prepare(self, messages, tools):
        if not self.round:
            raise RuntimeError("No active user round")
        limit = self.first_round_calls if self.round == 1 else self.later_round_calls
        if self.calls >= limit:
            raise TurnBudgetError("Model failed to hand off within the round budget")
        self.calls += 1
        handoff = self.calls == limit
        copied = deepcopy(messages)
        if handoff:
            copied.append({
                "role": "user",
                "content": (
                    "[Harness turn budget] This coding turn has reached its tool-call "
                    "budget. Tools are unavailable for this response. Reply to the "
                    "maintainer now in ordinary text: give a concise progress report, "
                    "an honest account of any validation already run, and any question "
                    "or blocker. Do not claim approval or passing tests without evidence. "
                    "This ends only this user round; you can continue working after "
                    "the maintainer replies. Do not emit tool calls."
                ),
            })
            tools = None
        else:
            guidance = (
                "\n[Harness turn budget] Each maintainer round has a limited coding "
                "budget. Inspect and ask a concise question early; hand off progress "
                "rather than repeatedly retrying a failing approach. Before a completed "
                "implementation handoff, inspect the diff and run harbor-validate as "
                "your last tool. Only the maintainer can approve the task."
            )
            if copied and copied[0].get("role") == "system" and isinstance(copied[0].get("content"), str):
                copied[0]["content"] += guidance
            else:
                copied.insert(0, {"role": "system", "content": guidance})
        return copied, tools, handoff
