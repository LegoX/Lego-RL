#!/bin/sh
# Sourced by the harbor agent before invoking the mounted Codex CLI.
#
# Codex needs no interpreter and no libc shims -- the binary under bin/ is
# static-pie linked against musl -- so this only has to make it findable and
# give it a writable CODEX_HOME (the mount itself is read-only).
CUSTOM_AGENT_RUNTIME_ROOT="${CUSTOM_AGENT_RUNTIME_ROOT:-/opt/custom-agent-runtime/codex}"
export CUSTOM_AGENT_RUNTIME_ROOT
export CUSTOM_AGENT_CODEX="${CUSTOM_AGENT_CODEX:-$CUSTOM_AGENT_RUNTIME_ROOT/bin/codex}"
export PATH="$CUSTOM_AGENT_RUNTIME_ROOT/bin:$PATH"
