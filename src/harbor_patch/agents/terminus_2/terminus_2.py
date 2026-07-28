"""harbor_patch Terminus2 — exposes chat history to verl.

Upstream ``harbor.agents.terminus_2.Terminus2`` only writes
``context.metadata["all_messages"]`` when ``store_all_messages=True`` (see
terminus_2.py:1633-1639). Without it, verl's ``BuiltinOHAgentLoop`` sees an
empty trial -> reward=0. We force the flag on so the trainer can always read
the full chat history.

Architecture (same as installed_openhands_sdk):

    Terminus2 (in trainer process, BaseAgent -> LiteLLM)
      -- OpenAI Chat Completions on api_base injected per-trial by
         BuiltinOHAgentLoop._run_harbor_trial -->
        in-process vLLM Chat Completions proxy
          -- forwards to vLLM (session-pinned for KV reuse)
"""

from harbor.agents.terminus_2.terminus_2 import Terminus2 as _BaseTerminus2


class Terminus2(_BaseTerminus2):
    """Upstream Terminus2 with ``store_all_messages`` forced on."""

    def __init__(self, *args, **kwargs):
        kwargs["store_all_messages"] = True
        super().__init__(*args, **kwargs)
