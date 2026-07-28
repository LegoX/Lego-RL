# Env-only override shim (no repo source edits).
# Auto-imported by CPython at interpreter startup when this directory is on
# PYTHONPATH. Bumps harbor's hard-coded agent-setup timeout from an env var.
#
# Usage (set before launching the infer run):
#   export PYTHONPATH=/path/to/SWE-Lego-RL/.env-shims:$PYTHONPATH
#   export HARBOR_AGENT_SETUP_TIMEOUT_SEC=1200
import os


def _apply() -> None:
    val = os.environ.get("HARBOR_AGENT_SETUP_TIMEOUT_SEC")
    if not val:
        return
    try:
        secs = float(val)
    except ValueError:
        return
    try:
        from harbor.trial.trial import Trial
    except Exception:
        return
    Trial._AGENT_SETUP_TIMEOUT_SEC = secs


_apply()
