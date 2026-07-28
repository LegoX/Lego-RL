#!/usr/bin/env bash
# lib/model_presets.sh — model-slot presets (swap models by changing only MODEL_PRESET, no new scripts / no hand-copied knobs)
# Usage: set MODEL_PRESET + SCAFFOLD first, then call apply_model_preset. Only fills values "not explicitly set",
#       anything already set in config is respected, never overridden. Models not in the table: rely entirely on explicit MODEL_PATH/TOOL_PARSER/...
# Key pitfall: tool_parser is not a pure function of the model — 30B uses qwen3_coder for cc, hermes for ohsdk/oh.
#         qwen3.5/3.6 chat_template is XML → must use qwen3_coder. See troubleshooting §tool-parser.
apply_model_preset() {
    local p sc root
    p="$(printf '%s' "${MODEL_PRESET:-}" | tr '[:upper:]' '[:lower:]')"
    sc="$(printf '%s' "${SCAFFOLD:-ohsdk}" | tr '[:upper:]' '[:lower:]')"
    root="${MODEL_ROOT:-/path/to/models}"   # model root dir (site.env can change it to your path)
    case "$p" in
      qwen35a3b)          # Qwen3.5-35B-A3B: hybrid GDN linear-attention MoE, requires veomni, XML tools
        : "${MODEL_PATH:=$root/Qwen3.5-35B-A3B}"
        : "${TOOL_PARSER:=qwen3_coder}"; : "${ENGINE:=veomni}"
        : "${SP_SIZE:=4}"; : "${MAX_PROMPT:=40000}"; : "${MAX_RESP:=91072}" ;;
      qwen3-30b-a3b)      # Qwen3-30B-A3B-Instruct-2507: standard-attention MoE, veomni/fsdp both work
        : "${MODEL_PATH:=$root/Qwen3-30B-A3B-Instruct-2507}"
        case "$sc" in cc) : "${TOOL_PARSER:=qwen3_coder}" ;; *) : "${TOOL_PARSER:=hermes}" ;; esac
        : "${ENGINE:=veomni}"; : "${SP_SIZE:=4}"; : "${MAX_PROMPT:=40000}"; : "${MAX_RESP:=91072}" ;;
      qwen36-27b)         # Qwen3.6-27B: dense hybrid-mamba, mainly for eval/infer, XML tools, 200k window
        : "${MODEL_PATH:=$root/Qwen3.6-27B}"
        : "${TOOL_PARSER:=qwen3_coder}"; : "${ENGINE:=fsdp}"
        : "${SP_SIZE:=1}"; : "${MAX_PROMPT:=167232}"; : "${MAX_RESP:=32768}" ;;
      "")   : ;;          # no preset: rely entirely on explicit
      *)    echo "[model_presets][WARN] unknown MODEL_PRESET='$p', relying entirely on explicit settings." >&2 ;;
    esac
}
