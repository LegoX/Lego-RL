#!/usr/bin/env bash
# lib/model_traits.sh — read facts about a checkpoint straight from its config.json,
# so defaults that only make sense for one model family stop being hand-maintained.
#
# Sourced by the runner libs before any template is sourced, so templates can write
#   : "${SOME_KNOB:=$(model_is_moe && echo True || echo False)}"
# and a config that sets the knob explicitly still wins.
#
# Everything here is best-effort and never fatal: an unreadable config.json is
# reported as "unknown", and the caller decides. preflight is where a wrong
# combination is actually blocked.

# Set by model_is_moe: a human-readable reason for its verdict. Declared here so
# a `set -u` caller can read it before the first call.
MODEL_MOE_REASON=""

# model_is_moe [model_path]
#   0 (true)  → the checkpoint is a Mixture-of-Experts model
#   1 (false) → dense, OR the config could not be read (check model_moe_reason)
#
# Two independent signals, because neither alone covers every family:
#   - config.model_type ending in "_moe" (qwen3_moe, qwen3_5_moe, mixtral has its own
#     name so the key check below catches it)
#   - an expert-count key anywhere in the config or its text_config sub-config
#     (multimodal checkpoints hide the language config one level down)
model_is_moe() {
    local path="${1:-${MODEL_PATH:-}}"
    MODEL_MOE_REASON=""
    [ -n "$path" ] || { MODEL_MOE_REASON="MODEL_PATH is empty"; return 1; }
    local cfg="$path/config.json"
    [ -f "$cfg" ] || { MODEL_MOE_REASON="no config.json under $path"; return 1; }

    local out
    out="$(
        "${PYTHON_BIN:-python3}" - "$cfg" <<'PY' 2>/dev/null
import json, sys

EXPERT_KEYS = ("num_experts", "n_routed_experts", "num_local_experts",
               "moe_intermediate_size", "num_experts_per_tok")

try:
    cfg = json.load(open(sys.argv[1]))
except Exception as exc:
    print(f"unknown\tcannot parse config.json: {type(exc).__name__}")
    raise SystemExit(0)

# A multimodal checkpoint keeps the language model's fields under text_config.
scopes = [cfg] + [v for v in (cfg.get("text_config"), cfg.get("llm_config")) if isinstance(v, dict)]
model_type = next((s["model_type"] for s in scopes if s.get("model_type")), None)

by_type = bool(model_type) and model_type.endswith("_moe")
hit_key = next((k for s in scopes for k in EXPERT_KEYS if k in s), None)

if by_type or hit_key:
    why = f"model_type={model_type!r}" if by_type else f"config has {hit_key!r}"
    print(f"moe\t{why}")
else:
    print(f"dense\tmodel_type={model_type!r}, no expert-count key")
PY
    )"

    case "${out%%$'\t'*}" in
        moe)   MODEL_MOE_REASON="${out#*$'\t'}"; return 0 ;;
        dense) MODEL_MOE_REASON="${out#*$'\t'}"; return 1 ;;
        *)     MODEL_MOE_REASON="${out#*$'\t'}"; [ -n "$MODEL_MOE_REASON" ] || MODEL_MOE_REASON="could not read $cfg"; return 1 ;;
    esac
}

# model_moe_verdict [model_path] → prints moe / dense / unknown.
# "unknown" is the case a caller must not silently treat as dense.
#
# NOTE: capturing this with $(...) runs it in a subshell, so MODEL_MOE_REASON will
# NOT be visible afterwards. A caller that wants the reason too should call
# model_is_moe directly and branch on its exit status, the way preflight does.
model_moe_verdict() {
    if model_is_moe "$@"; then
        printf 'moe'
    elif case "$MODEL_MOE_REASON" in "model_type="*) true ;; *) false ;; esac; then
        printf 'dense'
    else
        printf 'unknown'
    fi
}
