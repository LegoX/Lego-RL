#!/usr/bin/env bash
# =============================================================================
# scripts/lib/preflight.sh  —  pre-launch config check (fail-fast before burning a cluster)
# =============================================================================
# Usage:
#   1) inside runner:  source scripts/lib/preflight.sh   (reads currently exported config vars)
#   2) standalone check:   CONFIG=scripts/configs/xxx.env bash scripts/lib/preflight.sh
#
# Reads config only, never touches training logic. On FATAL it exits immediately (non-0), turning
# "training crashes midway / silently runs bad" into "a clear error one second before starting the
# cluster". Every rule is a real pitfall we hit (see MEMORY refs in comments); confirmed wrong=block, suspicious=WARN.
# =============================================================================
# Available env vars (config should export them; if missing, fall back to checks here):
#   MODEL_PATH SCAFFOLD TOOL_PARSER AGENT_NAME ENGINE PROFILE TRAINING_MODE
#   NNODES N_NODES_TRAIN N_NODES_ROLLOUT NGPUS_PER_NODE SP_SIZE
#   TRAINER_N_GPUS_PER_NODE ROLLOUT_N_GPUS_PER_NODE ALLOW_SINGLE_NODE_ASYNC_SPLIT
#   MAX_PROMPT MAX_RESP VERL_HAS_ROUTER_REPLAY ENABLE_R3
#   FUSED_KERNELS ACTIVATION_OFFLOAD LR_SCHEDULER VAL_TIMEOUT ROLLOUT_IS
#   IMAGE_REGISTRY INLINE_BUILD NYDUS_MIRROR K8S_KUBECONFIG
#   TRAIN_INDEX VAL_INDEX
# =============================================================================

# Allow standalone run: if CONFIG is given, source it first.
# When called embedded from the runner, set PREFLIGHT_EMBEDDED=1 to avoid a second source overwriting the values the runner already mapped.
if [ -z "${PREFLIGHT_EMBEDDED:-}" ] && [ -n "${CONFIG:-}" ] && [ -f "${CONFIG}" ]; then
    # shellcheck disable=SC1090
    set -a; source "${CONFIG}"; set +a
fi

_PF_ERR=0
_PF_WARN=0
# Structure-only mode: run every check that reads config values alone, and skip the
# ones that need the checkpoint, the indexes or the kubeconfig to be on this disk.
# Lets a fresh clone prove the software path is wired up before any data exists.
_PF_STRUCT="${PF_STRUCTURE_ONLY:-0}"
_pf_fatal() { printf '  \033[31m✗ FATAL\033[0m  %s\n' "$*" >&2; _PF_ERR=$((_PF_ERR+1)); }
_pf_warn()  { printf '  \033[33m⚠ WARN \033[0m  %s\n' "$*" >&2; _PF_WARN=$((_PF_WARN+1)); }
_pf_ok()    { printf '  \033[32m✓ OK   \033[0m  %s\n' "$*"; }
_pf_skip()  { printf '  \033[36m⊘ SKIP \033[0m  %s\n' "$*"; }
_lc() { printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]'; }
_pf_probe_verl_has_router_replay() {
    local py="${PYTHON_BIN:-python3}" origin root
    origin="$("$py" -c 'import importlib.util as u; s=u.find_spec("verl"); print(s.origin or "")' 2>/dev/null || true)"
    [ -n "$origin" ] || return 1
    root="$(dirname "$origin")"
    [ -f "$root/utils/veomni/router_replay.py" ]
}

echo "───────────────────────── preflight config check ─────────────────────────"

# --- 1. tool_parser × (model × scaffold) match ── §4.1 highest risk ──────────────────
# Pitfall: tool_parser is not a pure function of the model. qwen3.5's chat_template is qwen3_coder XML;
# the same 30B-Instruct uses qwen3_coder under CC, hermes under ohsdk/oh. Set wrong → 100%
# tool-parse failure → too few turns / 26% don't finish / fake reward 0. Related: openswe-toolcall-json-parse-failure
_model_lc="$(_lc "${MODEL_PATH:-}")"; _tp="$(_lc "${TOOL_PARSER:-}")"; _sc="$(_lc "${SCAFFOLD:-}")"
if [ "$_sc" = oc ]; then
    _pf_ok "scaffold=oc: tool_parser uses in-pod opencode config, check skipped"
elif [ -z "${TOOL_PARSER:-}" ]; then
    _pf_warn "TOOL_PARSER unset, using harbor default — please confirm it matches the model chat_template"
elif printf '%s' "$_model_lc" | grep -qE '3[._]?[56]|qwen3_[56]|35b'; then
    # qwen3.5 / qwen3.6 family → must be qwen3_coder (XML chat_template)
    if [ "$_tp" = "qwen3_coder" ]; then _pf_ok "tool_parser=qwen3_coder ✓ matches qwen3.5/3.6 (XML template)"
    else _pf_fatal "model looks like qwen3.5/3.6 but TOOL_PARSER=$TOOL_PARSER; their template is XML, must be qwen3_coder, otherwise 100% tool-parse failure"; fi
elif printf '%s' "$_model_lc" | grep -qE 'qwen3-30b|qwen3_30b|30b-a3b|30b_a3b'; then
    # 30B-Instruct-2507 → depends on scaffold
    case "$_sc" in
        cc)        [ "$_tp" = "qwen3_coder" ] && _pf_ok "tool_parser=qwen3_coder ✓ (30B + CC)" || _pf_fatal "30B + CC should be qwen3_coder, current=$TOOL_PARSER" ;;
        oh|ohsdk)  [ "$_tp" = "hermes" ]      && _pf_ok "tool_parser=hermes ✓ (30B + $SCAFFOLD)"  || _pf_fatal "30B + $SCAFFOLD should be hermes, current=$TOOL_PARSER (30B uses hermes on the ohsdk path, not qwen3_coder)" ;;
        *)         _pf_warn "30B model but SCAFFOLD='$SCAFFOLD' unknown, cannot check tool_parser=$TOOL_PARSER, verify manually" ;;
    esac
else
    _pf_warn "model '$MODEL_PATH' not in known rules, tool_parser=$TOOL_PARSER verify manually"
fi

# --- 2. topology consistency: NNODES == train + rollout ── most easily missed ─────────────────────
if [ -n "${NNODES:-}" ] && [ -n "${N_NODES_TRAIN:-}" ] && [ -n "${N_NODES_ROLLOUT:-}" ]; then
    _sum=$((N_NODES_TRAIN + N_NODES_ROLLOUT))
    if [ "$NNODES" -eq "$_sum" ]; then
        _pf_ok "topology consistent: NNODES=$NNODES = train $N_NODES_TRAIN + rollout $N_NODES_ROLLOUT"
    elif [ "${ALLOW_SINGLE_NODE_ASYNC_SPLIT:-1}" = "1" ] \
        && [ "${TRAINING_MODE:-}" = "async" ] \
        && [ "$NNODES" -eq 1 ] \
        && [ "$N_NODES_TRAIN" -eq 1 ] \
        && [ "$N_NODES_ROLLOUT" -eq 1 ]; then
        _trainer_gpn="${TRAINER_N_GPUS_PER_NODE:-${NGPUS_PER_NODE:-8}}"
        _rollout_gpn="${ROLLOUT_N_GPUS_PER_NODE:-${NGPUS_PER_NODE:-8}}"
        _split_gpus=$((_trainer_gpn + _rollout_gpn))
        _node_gpus="${NGPUS_PER_NODE:-8}"
        if [ "$_split_gpus" -le "$_node_gpus" ]; then
            _pf_ok "single-node async split allowed: NNODES=1 with train=${_trainer_gpn} GPU + rollout=${_rollout_gpn} GPU (capacity ${_split_gpus}/${_node_gpus})"
        else
            _pf_fatal "single-node async split requests ${_split_gpus} GPUs (train=${_trainer_gpn} + rollout=${_rollout_gpn}) but NGPUS_PER_NODE=${_node_gpus}; reduce per-role GPU counts or use separate nodes"
        fi
    else
        _pf_fatal "topology inconsistent: NNODES=$NNODES ≠ train($N_NODES_TRAIN)+rollout($N_NODES_ROLLOUT)=$_sum. ray cluster node count must equal the sum of logical roles, otherwise placement hangs/fails"
    fi
fi

# --- 3. SP × dp device-mesh + VRAM ── §4.3 OOM/assertion high risk ──────────────────────
# Device mesh is built from the trainer pool, not the node's total physical
# capacity.  In a single-node async split the latter also includes rollout GPUs.
_gpn="${TRAINER_N_GPUS_PER_NODE:-${NGPUS_PER_NODE:-8}}"
if [ -n "${N_NODES_TRAIN:-}" ] && [ -n "${SP_SIZE:-}" ]; then
    _train_world=$((N_NODES_TRAIN * _gpn))
    if [ $((_train_world % SP_SIZE)) -ne 0 ]; then
        _pf_fatal "SP_SIZE=$SP_SIZE does not evenly divide train_world=$_train_world(=${N_NODES_TRAIN}×${_gpn}) → veomni device-mesh AssertionError"
    elif [ "$SP_SIZE" -gt "$_train_world" ]; then
        _pf_fatal "SP_SIZE=$SP_SIZE > train_world=$_train_world → device-mesh crash"
    else
        _dp=$((_train_world / SP_SIZE))
        _pf_ok "device-mesh valid: dp=$_dp (train_world $_train_world / sp $SP_SIZE)"
    fi
    # 35B VRAM: 1 train node @ full 131k window OOMs (measured 149GB>139.8); needs 2 train nodes
    _win=$(( ${MAX_PROMPT:-40000} + ${MAX_RESP:-91072} ))
    if printf '%s' "$_model_lc" | grep -qE '3[._]?5|35b'; then
        if [ "${N_NODES_TRAIN}" -le 1 ] && [ "$_win" -ge 120000 ]; then
            _pf_warn "35B + 1 train node + window${_win} → peak ~149GB will backward OOM (measured 2026-07-06). Recommend N_NODES_TRAIN=2 (peak ~122GB) or MAX_RESP<=58304"
        else
            _pf_ok "35B VRAM: train_node=${N_NODES_TRAIN} window=${_win} in safe zone"
        fi
    fi
fi

# --- 4. veomni engine constraints ── §4.4 storage-of-size-0 crash ───────────────────────
if [ "$(_lc "${ENGINE:-}")" = "veomni" ]; then
    [ "$(_lc "${FUSED_KERNELS:-false}")" = "false" ]      && _pf_ok "veomni: fused_kernels=False ✓" \
        || _pf_fatal "veomni requires FUSED_KERNELS=False, otherwise backward after FSDP2 reshard crashes with 'storage of size 0'"
    [ "$(_lc "${ACTIVATION_OFFLOAD:-false}")" = "false" ] && _pf_ok "veomni: activation_offload=False ✓" \
        || _pf_fatal "veomni requires ACTIVATION_OFFLOAD=False (same as above, FSDP1-era monkey patch crashes)"
fi

# --- 5. R3 × engine × imported verl ── §4.6 silent low pearson ─────────────────────
# 5a. R3 × model family. R3 replays MoE expert routing, so it is meaningless on a
# dense checkpoint — and not harmlessly so. veomni raises "router replay is not
# wired for model_type=..." from validate_model_for_replay inside engine init,
# which happens *after* vLLM has loaded weights and captured cuda graphs; fsdp
# patches 0 router gates and silently trains without replay while the rollout
# still passes --enable-return-routed-experts to vLLM. Block the first, warn on
# the second, and warn when a MoE run is leaving R3 on the table.
if [ "$_PF_STRUCT" = "1" ]; then
    _pf_skip "R3 × MoE check: needs the checkpoint's config.json on disk"
elif [ -n "${MODEL_PATH:-}" ] && command -v model_is_moe >/dev/null 2>&1; then
    # Called directly, not via $(model_moe_verdict) — a command substitution would
    # run in a subshell and MODEL_MOE_REASON would not survive it.
    if model_is_moe "$MODEL_PATH"; then
        _pf_moe=moe
    elif case "${MODEL_MOE_REASON:-}" in "model_type="*) true ;; *) false ;; esac; then
        _pf_moe=dense
    else
        _pf_moe=unknown
    fi
    _pf_r3_on=0
    [[ "$(_lc "${ENABLE_R3:-true}")" =~ ^(1|true)$ ]] && _pf_r3_on=1
    case "$_pf_moe:$_pf_r3_on" in
        moe:1)     _pf_ok "R3 on × MoE model ✓ (${MODEL_MOE_REASON:-})" ;;
        dense:0)   _pf_ok "R3 off × dense model ✓ (${MODEL_MOE_REASON:-})" ;;
        dense:1)   _pf_fatal "ENABLE_R3=on but $MODEL_PATH is DENSE (${MODEL_MOE_REASON:-}). veomni will raise 'router replay is not wired for model_type=...' inside engine init (~10-20min in, after vLLM warmup); fsdp will patch 0 gates and silently skip replay → set ENABLE_R3=False" ;;
        moe:0)     _pf_warn "MoE model but ENABLE_R3=off (${MODEL_MOE_REASON:-}) — training will not replay rollout routing (coverage ~24% instead of 100%). Intentional? Otherwise set ENABLE_R3=True" ;;
        unknown:*) _pf_warn "cannot tell whether $MODEL_PATH is MoE (${MODEL_MOE_REASON:-}) — ENABLE_R3=${ENABLE_R3:-unset} was not verified against the checkpoint; set it explicitly" ;;
    esac
fi

if [[ "$(_lc "${ENABLE_R3:-true}")" =~ ^(1|true)$ ]]; then
    if [ -z "${VERL_HAS_ROUTER_REPLAY+x}" ]; then
        if _pf_probe_verl_has_router_replay; then
            VERL_HAS_ROUTER_REPLAY=1
        else
            VERL_HAS_ROUTER_REPLAY=0
        fi
    fi
    [ "${VERL_HAS_ROUTER_REPLAY:-0}" = "1" ] && _pf_ok "R3 on + imported verl has router_replay ✓" \
        || _pf_fatal "ENABLE_R3=on but imported verl lacks router_replay.py -> R3 will FATAL"
    # engine-specific R3 key: veomni→veomni.router_replay; fsdp→fsdp_config.router_replay and forces SP=1
    if [ "$(_lc "${ENGINE:-}")" = "veomni" ]; then
        _pf_ok "R3 key uses veomni.router_replay (matches engine) — verify pearson~0.999 at first step"
    elif [ "$(_lc "${ENGINE:-}")" = "fsdp" ]; then
        if [ "${SP_SIZE:-1}" -gt 1 ] 2>/dev/null; then
            _pf_fatal "FSDP + R3 only supports SP=1 (SP resharding of routed_experts not handled in the fsdp path), current SP_SIZE=$SP_SIZE → disable R3 or SP_SIZE=1"
        else
            _pf_ok "R3 key uses fsdp_config.router_replay + SP=1 ✓ (fsdp path)"
        fi
    fi
fi

# --- 6. AGENT_NAME × scaffold ── §4.2 env_setup avalanche / factory dispatch ───────────
_an="$(_lc "${AGENT_NAME:-}")"
case "$_sc" in
    ohsdk|oh)   # must be null → uses import_path, no in-pod venv triggered (no-egress cluster will crash)
        { [ -z "$_an" ] || [ "$_an" = null ]; } && _pf_ok "AGENT_NAME=null ✓ ($_sc uses import_path)" \
            || _pf_fatal "SCAFFOLD=$_sc but AGENT_NAME='$AGENT_NAME' non-null → uses in-pod venv install, no-egress cluster will fail → env_setup avalanche" ;;
    cc)         [ "$_an" = claude-code ] && _pf_ok "AGENT_NAME=claude-code ✓ (cc)" \
            || _pf_warn "SCAFFOLD=cc normally AGENT_NAME=claude-code, current='$AGENT_NAME'" ;;
    oc)         [ "$_an" = opencode ] && _pf_ok "AGENT_NAME=opencode ✓ (oc)" \
            || _pf_warn "SCAFFOLD=oc normally AGENT_NAME=opencode, current='$AGENT_NAME'" ;;
    "")         : ;;
    *)          _pf_warn "unknown SCAFFOLD='$_sc' (supports ohsdk|oh|cc|oc)" ;;
esac
# backend=docker: currently only oh_docker loop config, cc/oc on docker not wired up
if [ "$(_lc "${BACKEND:-k8s}")" = docker ]; then
    case "$_sc" in
        oh|ohsdk) _pf_ok "backend=docker + $_sc ✓ (RemoteDockerEnvironment + oh_docker loop)" ;;
        *)        _pf_warn "backend=docker currently only agent_loop_config_oh_docker.yaml; SCAFFOLD=$_sc on docker not wired up, recommend k8s or SCAFFOLD=oh" ;;
    esac
fi

# --- 7. image source / kubeconfig ── §4.5 val fake 0 / duplicate build (k8s backend only) ──────
if [ "$(_lc "${BACKEND:-k8s}")" = k8s ]; then
    _reg="${IMAGE_REGISTRY:-}"; _inline="$(_lc "${INLINE_BUILD:-false}")"
    if [ -n "$_reg" ]; then
        [ "$_inline" = "false" ] && _pf_ok "image: registry=$_reg + inline_build=false ✓ (pull prebuilt images to speed up)" \
            || _pf_fatal "registry non-empty ($_reg) but INLINE_BUILD=true → rebuilds every trial, crushes node I/O. Pick one"
    elif [ "$_inline" = "true" ]; then
        _pf_ok "image: registry empty + inline_build=true ✓ (vanilla build-on-demand, slow but portable)"
    else
        _pf_fatal "IMAGE_REGISTRY empty and INLINE_BUILD=false → no image source, pod won't start → env_setup_failed all crash"
    fi
    if [ "$_PF_STRUCT" = "1" ]; then
        _pf_skip "K8S_KUBECONFIG existence: needs the cluster's kubeconfig on disk"
    elif [ -n "${K8S_KUBECONFIG:-}" ] && [ ! -f "${K8S_KUBECONFIG}" ]; then
        _pf_fatal "K8S_KUBECONFIG file does not exist: ${K8S_KUBECONFIG} (cluster path should be set in scripts/lib/site.env)"
    fi
    [ -z "${NYDUS_MIRROR:-}" ] && _pf_warn "NYDUS_MIRROR empty: if val uses official swebench images (swebench/sweb.eval.*) and pod has no Docker Hub egress → val reward=0 (related val-zero-swebench-image-unpullable-221)"
else
    _pf_ok "backend=docker: task images built on-demand on remote daemon (FORCE_BUILD), skipping registry/inline/kubeconfig check"
fi

# --- 8. lr_scheduler / val timeout / rollout_is ── other silent pitfalls ───────────────────
# Training-only: eval/infer have no optimizer, no val loop and no IS correction.
if [ "${PF_KIND:-train}" = train ]; then
    # Never default this: an unset LR_SCHEDULER used to fall back to "constant" here and
    # print a green ✓ while the runner actually launched with cosine (sync mode), i.e. the
    # check validated its own default instead of the value that reached the trainer.
    # A check that lies is worse than no check, so unset is now a WARN, not an OK.
    if [ -z "${LR_SCHEDULER:-}" ]; then
        _pf_warn "LR_SCHEDULER not passed by the runner: cannot verify the scheduler that will reach the trainer (do not assume constant)"
    elif [ "$(_lc "$LR_SCHEDULER")" = "constant" ]; then
        _pf_ok "lr_scheduler=constant ✓ (avoids the total_training_steps=-1 → lr=0 collapse)"
    else
        _pf_warn "LR_SCHEDULER=${LR_SCHEDULER}: with trainer.total_training_steps=-1 (ray_trainer.py:438 lets the -1 sentinel override the computed value) a cosine/decaying schedule collapses to lr=0 and the model never updates — hit on sync 20260722, not just fully-async (related fully-async-veomni-lr-zero-bug)"
    fi
    [ -n "${VAL_TIMEOUT:-}" ] && _pf_ok "val-specific timeout=${VAL_TIMEOUT}s ✓" \
        || _pf_warn "no HARBOR_VAL_AGENT_MAX_TIMEOUT_SEC: long val tasks get cut off by the train timeout (2400s), only ~170/500 evaluated each time (related fully-async-val-coverage-170of500)"
    if [ "$(_lc "${ROLLOUT_IS:-null}")" = "sequence" ]; then
        _pf_warn "ROLLOUT_IS=sequence: on long responses (~54k tok) sequence-TIS gives ESS~0.06 gradient starvation (related seq-tis-ess-collapse-long-response). Unless certain, use null/token"
    fi
fi

# --- 9. critical path existence ──────────────────────────────────────────────────────
# Which paths matter depends on the kind. An empty value is only worth a WARN where
# the runner requires it (train); for eval/infer an empty INSTANCES_FILE/DATASET_PATH
# is a legitimate "whole index / dataset by name" choice.
case "${PF_KIND:-train}" in
    eval)  _pf_path_vars="MODEL_PATH DATASET_PATH" ;;
    infer) _pf_path_vars="MODEL_PATH TRAIN_INDEX INSTANCES_FILE" ;;
    *)     _pf_path_vars="MODEL_PATH TRAIN_INDEX VAL_INDEX" ;;
esac
if [ "$_PF_STRUCT" = "1" ]; then
    _pf_skip "path existence ($_pf_path_vars): structure-only run"
    _pf_path_vars=""
fi
for _k in $_pf_path_vars; do
    eval "_v=\"\${$_k:-}\""
    if [ -z "$_v" ]; then
        [ "${PF_KIND:-train}" = train ] && _pf_warn "$_k unset"
        continue
    fi
    [ -e "$_v" ] && _pf_ok "$_k exists: $_v" || _pf_fatal "$_k path does not exist: $_v"
done

echo "──────────────────────────────────────────────────────────────────────"
if [ "$_PF_ERR" -gt 0 ]; then
    printf '\033[31m[preflight] %d FATAL, %d WARN — refusing to start. Fix the ✗ items above.\033[0m\n' "$_PF_ERR" "$_PF_WARN" >&2
    return 1 2>/dev/null || exit 1
else
    if [ "$_PF_STRUCT" = "1" ]; then
        printf '\033[32m[preflight] structure-only: all config checks passed (%d WARN). Filesystem checks skipped.\033[0m\n' "$_PF_WARN"
    else
        printf '\033[32m[preflight] all fatal checks passed (%d WARN).\033[0m\n' "$_PF_WARN"
    fi
    return 0 2>/dev/null || exit 0
fi
