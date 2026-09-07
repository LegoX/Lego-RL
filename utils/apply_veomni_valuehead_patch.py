#!/usr/bin/env python3
"""Make VeOmni able to build a value model (critic) at all.

veomni is installed from a wheel (requirements.txt), not a managed checkout, so
this edits site-packages and is LOST on every `pip install`/venv rebuild/machine
switch. scripts/setup_env.sh runs it once after installing, and
scripts/train/train.sh re-runs it before any CRITIC_ENABLE=True veomni launch. The optional verl half patches ONLY the
`VeOmniEngineWithValueHead` module that `import verl` resolves to (or an explicit
`VERL_TRANSFORMER_IMPL` / `VERL_DIR`). It will not walk a hardcoded checkout, and a
missing verl is a warning, not a failed launch -- worker nodes often have veomni
but not a verl tree. Run after any veomni (re)install before a critic.enable=True
config.

THE BUG. verl's VeOmniEngineWithValueHead does the right thing: it rewrites the HF
config to `architectures=["Qwen3MoeForTokenClassification"]`, num_labels=1, dropout 0
(verl/workers/engine/veomni/transformer_impl.py:1029-1047). VeOmni then resolves that
name through `MODELING_REGISTRY[model_type](arch_name)` (veomni/models/loader.py:142),
and the per-family registration function dispatched on substrings:

    if   "ForCausalLM"          in architecture: return Qwen3MoeForCausalLM
    elif "ForQuestionAnswering" in architecture: return Qwen3MoeForQuestionAnswering
    elif "Model"                in architecture: return Qwen3MoeModel
    else:                                        return Qwen3MoeForCausalLM

"Qwen3MoeForTokenClassification" matches NONE of the three -- note "Moe" is not
"Model" -- so it fell through to the else and the critic was silently built as a
LANGUAGE MODEL. No error, no warning. The class Qwen3MoeForTokenClassification exists
and is exported by the same patched module; it was simply never imported here.

HOW IT SURFACED. 3-node SAO smoke, 2026-08-11 11:17, first training step:
    core_algos.py:306 in _gae
      delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
    RuntimeError: The size of tensor a (8) must match the size of tensor b (151936)
                  at non-singleton dimension 1
151936 is Qwen3's vocab size: the "values" were lm_head logits. Had GAE happened to
broadcast, this would instead have been a run that trains happily on a garbage critic.

WHAT IT PATCHES:
  1. qwen3_moe/__init__.py -- add the ForTokenClassification import + dispatch branch,
     AND add the class to the _create_checkpoint_tensor_converter tuple. The second
     half is load-bearing and easy to miss: that converter carries the MoE expert
     tensor layout (keyed on config.num_experts), and the value model has the exact
     same MoE body as the CausalLM -- without it, Qwen3-30B-A3B expert weights load
     wrong into the critic.
  2. qwen3/__init__.py -- same dispatch fix for the dense family. No converter there.

Ordering follows veomni's own seed_oss/__init__.py, the one family upstream got right:
specific For* branches before the generic "Model" branch.

Qwen3.5-MoE has no generated ForTokenClassification class. This script copies
src/verl_patch/models/qwen3_5_moe_token_classification.py next to the family and
dispatches to it. It also patches VeOmniEngineWithValueHead so a missing
AutoModel mapping falls back to rewriting ForConditionalGeneration, rather than
raising (or, worse, building a language model).

Idempotent: skipped if the family already dispatches ForTokenClassification.
Run with `--check` to report status without writing.

Usage:
    python utils/apply_veomni_valuehead_patch.py            # apply
    python utils/apply_veomni_valuehead_patch.py --check    # report only
"""

from __future__ import annotations

import os
import sys


def _veomni_dir() -> str:
    try:
        import veomni  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        sys.exit(f"[veomni-valuehead] cannot import veomni: {e!r}")
    return os.path.dirname(os.path.abspath(veomni.__file__))


QWEN3_MOE = "models/transformers/qwen3_moe/__init__.py"
QWEN3 = "models/transformers/qwen3/__init__.py"
QWEN3_5_MOE = "models/transformers/qwen3_5_moe/__init__.py"
QWEN3_5_MOE_SIDECAR = "models/transformers/qwen3_5_moe/token_classification.py"

_MOE_FIND = '''    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_qwen3_moe_npu import (
            Qwen3MoeForCausalLM,
            Qwen3MoeForQuestionAnswering,
            Qwen3MoeModel,
        )
    else:
        from .generated.patched_modeling_qwen3_moe_gpu import (
            Qwen3MoeForCausalLM,
            Qwen3MoeForQuestionAnswering,
            Qwen3MoeModel,
        )

    for model_cls in (Qwen3MoeForCausalLM, Qwen3MoeForQuestionAnswering, Qwen3MoeModel):
        model_cls._create_checkpoint_tensor_converter = staticmethod(create_qwen3_moe_checkpoint_tensor_converter)

    if "ForCausalLM" in architecture:
        return Qwen3MoeForCausalLM
    elif "ForQuestionAnswering" in architecture:
        return Qwen3MoeForQuestionAnswering
    elif "Model" in architecture:
        return Qwen3MoeModel
    else:
        return Qwen3MoeForCausalLM'''

_MOE_REPLACE = '''    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_qwen3_moe_npu import (
            Qwen3MoeForCausalLM,
            Qwen3MoeForQuestionAnswering,
            Qwen3MoeForTokenClassification,
            Qwen3MoeModel,
        )
    else:
        from .generated.patched_modeling_qwen3_moe_gpu import (
            Qwen3MoeForCausalLM,
            Qwen3MoeForQuestionAnswering,
            Qwen3MoeForTokenClassification,
            Qwen3MoeModel,
        )

    # Qwen3MoeForTokenClassification shares the MoE body, so it needs the expert-layout
    # converter exactly as much as the CausalLM does.
    for model_cls in (
        Qwen3MoeForCausalLM,
        Qwen3MoeForQuestionAnswering,
        Qwen3MoeForTokenClassification,
        Qwen3MoeModel,
    ):
        model_cls._create_checkpoint_tensor_converter = staticmethod(create_qwen3_moe_checkpoint_tensor_converter)

    if "ForCausalLM" in architecture:
        return Qwen3MoeForCausalLM
    elif "ForQuestionAnswering" in architecture:
        return Qwen3MoeForQuestionAnswering
    elif "ForTokenClassification" in architecture:
        return Qwen3MoeForTokenClassification
    elif "Model" in architecture:
        return Qwen3MoeModel
    else:
        return Qwen3MoeForCausalLM'''

_DENSE_FIND = '''    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_qwen3_npu import (
            Qwen3ForCausalLM,
            Qwen3ForSequenceClassification,
            Qwen3Model,
        )
    else:
        from .generated.patched_modeling_qwen3_gpu import (
            Qwen3ForCausalLM,
            Qwen3ForSequenceClassification,
            Qwen3Model,
        )

    if "ForCausalLM" in architecture:
        return Qwen3ForCausalLM
    elif "ForSequenceClassification" in architecture:
        return Qwen3ForSequenceClassification
    elif "Model" in architecture:
        return Qwen3Model
    else:
        return Qwen3ForCausalLM'''

_DENSE_REPLACE = '''    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_qwen3_npu import (
            Qwen3ForCausalLM,
            Qwen3ForSequenceClassification,
            Qwen3ForTokenClassification,
            Qwen3Model,
        )
    else:
        from .generated.patched_modeling_qwen3_gpu import (
            Qwen3ForCausalLM,
            Qwen3ForSequenceClassification,
            Qwen3ForTokenClassification,
            Qwen3Model,
        )

    if "ForCausalLM" in architecture:
        return Qwen3ForCausalLM
    elif "ForSequenceClassification" in architecture:
        return Qwen3ForSequenceClassification
    elif "ForTokenClassification" in architecture:
        return Qwen3ForTokenClassification
    elif "Model" in architecture:
        return Qwen3Model
    else:
        return Qwen3ForCausalLM'''

_Q35_MOE_FIND = '''    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_qwen3_5_moe_npu import (
            Qwen3_5MoeForCausalLM,
            Qwen3_5MoeForConditionalGeneration,
        )
    else:
        from .generated.patched_modeling_qwen3_5_moe_gpu import (
            Qwen3_5MoeForCausalLM,
            Qwen3_5MoeForConditionalGeneration,
        )

    if "ForCausalLM" in architecture:
        return Qwen3_5MoeForCausalLM
    elif "ForConditionalGeneration" in architecture:
        return Qwen3_5MoeForConditionalGeneration
    else:
        return Qwen3_5MoeForCausalLM'''

_Q35_MOE_REPLACE = '''    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_qwen3_5_moe_npu import (
            Qwen3_5MoeForCausalLM,
            Qwen3_5MoeForConditionalGeneration,
        )
    else:
        from .generated.patched_modeling_qwen3_5_moe_gpu import (
            Qwen3_5MoeForCausalLM,
            Qwen3_5MoeForConditionalGeneration,
        )

    from .token_classification import Qwen3_5MoeForTokenClassification

    if "ForTokenClassification" in architecture:
        return Qwen3_5MoeForTokenClassification
    elif "ForCausalLM" in architecture:
        return Qwen3_5MoeForCausalLM
    elif "ForConditionalGeneration" in architecture:
        return Qwen3_5MoeForConditionalGeneration
    else:
        return Qwen3_5MoeForCausalLM'''

_Q35_TEXT_FIND = '''    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_qwen3_5_moe_npu import Qwen3_5MoeForCausalLM
    else:
        from .generated.patched_modeling_qwen3_5_moe_gpu import Qwen3_5MoeForCausalLM

    return Qwen3_5MoeForCausalLM'''

_Q35_TEXT_REPLACE = '''    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_qwen3_5_moe_npu import Qwen3_5MoeForCausalLM
    else:
        from .generated.patched_modeling_qwen3_5_moe_gpu import Qwen3_5MoeForCausalLM

    # qwen3_5_moe_text refuses TokenClassification (needs Qwen3_5MoeConfig)
    if "ForTokenClassification" in architecture:
        raise ValueError(
            "qwen3_5_moe_text has no value-head class; use model_type=qwen3_5_moe"
        )
    return Qwen3_5MoeForCausalLM'''

# Already-applied text-family dispatch from the first Qwen3.5 revision. Upgraded
# in place so a re-run does not keep returning the multimodal class for a
# text-only config (that class reads config.text_config and would AttributeError).
_Q35_TEXT_UPGRADE_FIND = '''    from .token_classification import Qwen3_5MoeForTokenClassification

    # qwen3_5_moe_text ForTokenClassification
    if "ForTokenClassification" in architecture:
        return Qwen3_5MoeForTokenClassification
    return Qwen3_5MoeForCausalLM'''

_VERL_FIND = '''        token_cls = AutoModelForTokenClassification._model_mapping.get(type(config), None)
        if token_cls is None:
            raise ValueError(f"No ForTokenClassification class in transformers for {type(config).__name__}.")
        config.architectures = [token_cls.__name__]
        return config'''

_VERL_REPLACE = '''        token_cls = AutoModelForTokenClassification._model_mapping.get(type(config), None)
        if token_cls is None:
            # Qwen3.5-MoE is ForConditionalGeneration and has no HF TokenClassification
            # mapping. Rewrite the suffix so VeOmni's MODELING_REGISTRY can dispatch
            # to the sidecar class instead of raising (or falling through to CausalLM).
            # A checkpoint that is already *ForTokenClassification (e.g. a merged
            # warm-start critic) passes through unchanged.
            arch = (getattr(config, "architectures", None) or [type(config).__name__])[0]
            if not arch.endswith("ForTokenClassification"):
                for suffix in ("ForConditionalGeneration", "ForCausalLM"):
                    if suffix in arch:
                        arch = arch.replace(suffix, "ForTokenClassification")
                        break
                else:
                    raise ValueError(
                        f"No ForTokenClassification class in transformers for {type(config).__name__}."
                    )
            config.architectures = [arch]
            return config
        config.architectures = [token_cls.__name__]
        return config'''

# (name, marker, find, replace). marker -> already-applied detection.
# "ForTokenClassification" appears nowhere in either unpatched qwen3 file, so it is
# an unambiguous marker there. Qwen3.5 uses more specific markers because both
# register functions live in one file.
PATCHES: dict[str, list[tuple[str, str, str, str]]] = {
    QWEN3_MOE: [
        (
            "ForTokenClassification dispatch + expert-layout converter",
            "ForTokenClassification",
            _MOE_FIND,
            _MOE_REPLACE,
        ),
    ],
    QWEN3: [
        (
            "ForTokenClassification dispatch",
            "ForTokenClassification",
            _DENSE_FIND,
            _DENSE_REPLACE,
        ),
    ],
    QWEN3_5_MOE: [
        (
            "qwen3_5_moe ForTokenClassification dispatch",
            "return Qwen3_5MoeForTokenClassification",
            _Q35_MOE_FIND,
            _Q35_MOE_REPLACE,
        ),
    ],
}


def _patch_q35_text_family(veomni_root: str, check_only: bool) -> int:
    """Keep qwen3_5_moe_text from returning the multimodal value-head class."""
    path = os.path.join(veomni_root, QWEN3_5_MOE)
    if not os.path.isfile(path):
        print(f"  [MISSING FILE] {QWEN3_5_MOE}")
        return 1
    with open(path, encoding="utf-8") as f:
        content = f.read()
    if "qwen3_5_moe_text refuses TokenClassification" in content:
        print(f"  [already]  {QWEN3_5_MOE} :: qwen3_5_moe_text refuses TokenClassification")
        return 0
    if _Q35_TEXT_UPGRADE_FIND in content:
        find = _Q35_TEXT_UPGRADE_FIND
    elif _Q35_TEXT_FIND in content:
        find = _Q35_TEXT_FIND
    else:
        print(f"  [!! NOT FOUND] {QWEN3_5_MOE} :: qwen3_5_moe_text TokenClassification guard")
        return 1
    print(f"  [{'WOULD' if check_only else 'APPLY'}]  {QWEN3_5_MOE} :: qwen3_5_moe_text refuses TokenClassification")
    if check_only:
        return 1
    content = content.replace(find, _Q35_TEXT_REPLACE, 1)
    tmp = f"{path}.vhtmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)
    return 0


def _repo_sidecar_path() -> str:
    # utils/apply_veomni_valuehead_patch.py -> repo root
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "src/verl_patch/models/qwen3_5_moe_token_classification.py")


def _install_q35_sidecar(veomni_root: str, check_only: bool) -> int:
    src = _repo_sidecar_path()
    dst = os.path.join(veomni_root, QWEN3_5_MOE_SIDECAR)
    if not os.path.isfile(src):
        print(f"  [!! NOT FOUND] sidecar source {src}")
        return 1
    if os.path.isfile(dst) and open(dst, encoding="utf-8").read() == open(src, encoding="utf-8").read():
        print(f"  [already]  {QWEN3_5_MOE_SIDECAR}")
        return 0
    print(f"  [{'WOULD' if check_only else 'APPLY'}]  {QWEN3_5_MOE_SIDECAR}")
    if check_only:
        return 1
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = f"{dst}.vhtmp.{os.getpid()}"
    with open(src, encoding="utf-8") as f:
        content = f.read()
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, dst)
    return 0


def _verl_transformer_impl_path() -> str | None:
    """Only the verl this process would actually import, plus explicit overrides.

    Do not fall back to a machine-local checkout: that can patch a tree the
    trainer never loads, while workers without that path would fail launch.
    """
    try:
        import verl.workers.engine.veomni.transformer_impl as impl  # noqa: PLC0415

        return os.path.abspath(impl.__file__)
    except Exception:
        pass
    explicit = os.environ.get("VERL_TRANSFORMER_IMPL", "")
    if explicit and os.path.isfile(explicit):
        return explicit
    verl_dir = os.environ.get("VERL_DIR", "")
    if verl_dir:
        candidate = os.path.join(verl_dir, "verl/workers/engine/veomni/transformer_impl.py")
        if os.path.isfile(candidate):
            return candidate
    return None


def _patch_verl_valuehead(check_only: bool) -> int:
    path = _verl_transformer_impl_path()
    if path is None:
        print(
            "  [skip] verl not importable -- not patching a checkout. "
            "Qwen3.5 critic needs the ForConditionalGeneration fallback in the "
            "verl the trainer imports; export PYTHONPATH/VERL_DIR if this process "
            "will be the one that builds the critic."
        )
        return 0
    rel = path
    with open(path, encoding="utf-8") as f:
        content = f.read()
    if "ForConditionalGeneration" in content and "No ForTokenClassification class" in content:
        # already has the fallback (the raise string remains as the else branch)
        if "for suffix in (\"ForConditionalGeneration\", \"ForCausalLM\")" in content:
            print(f"  [already]  {rel} :: ForConditionalGeneration value-head fallback")
            return 0
    if _VERL_FIND in content:
        print(f"  [{'WOULD' if check_only else 'APPLY'}]  {rel} :: ForConditionalGeneration value-head fallback")
        if check_only:
            return 1
        content = content.replace(_VERL_FIND, _VERL_REPLACE, 1)
        tmp = f"{path}.vhtmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)
        return 0
    if "for suffix in (\"ForConditionalGeneration\", \"ForCausalLM\")" in content:
        print(f"  [already]  {rel} :: ForConditionalGeneration value-head fallback")
        return 0
    print(f"  [!! NOT FOUND] {rel} :: VeOmniEngineWithValueHead lookup (verl version mismatch?)")
    return 1


def _verify(root: str) -> int:
    """Import the registry and assert every architecture resolves to its own class.

    Substring dispatch is exactly the kind of thing that looks right and is not, so
    check the previously-working branches too, not just the new one.
    """
    try:
        from veomni.models.loader import MODELING_REGISTRY  # noqa: PLC0415
    except Exception as e:
        print(f"[veomni-valuehead] verify SKIPPED (cannot import registry: {e!r})")
        return 0

    cases = [
        ("qwen3_moe", "Qwen3MoeForTokenClassification"),
        ("qwen3_moe", "Qwen3MoeForCausalLM"),
        ("qwen3_moe", "Qwen3MoeForQuestionAnswering"),
        ("qwen3_moe", "Qwen3MoeModel"),
        ("qwen3", "Qwen3ForTokenClassification"),
        ("qwen3", "Qwen3ForCausalLM"),
        ("qwen3", "Qwen3ForSequenceClassification"),
        ("qwen3", "Qwen3Model"),
        ("qwen3_5_moe", "Qwen3_5MoeForTokenClassification"),
        ("qwen3_5_moe", "Qwen3_5MoeForCausalLM"),
        ("qwen3_5_moe", "Qwen3_5MoeForConditionalGeneration"),
        ("qwen3_5_moe_text", "Qwen3_5MoeForCausalLM"),
    ]
    bad = 0
    for family, arch in cases:
        try:
            got = MODELING_REGISTRY[family](arch).__name__
        except Exception as e:
            print(f"  [ERROR] {family:10} {arch:32} -> {e!r}")
            bad += 1
            continue
        ok = got == arch
        bad += not ok
        print(f"  [{'ok' if ok else 'FAIL'}] {family:10} {arch:32} -> {got}")

    try:
        MODELING_REGISTRY["qwen3_5_moe_text"]("Qwen3_5MoeForTokenClassification")
        print("  [FAIL] qwen3_5_moe_text Qwen3_5MoeForTokenClassification should raise")
        bad += 1
    except ValueError:
        print("  [ok] qwen3_5_moe_text Qwen3_5MoeForTokenClassification raises")
    except Exception as e:
        print(f"  [ERROR] qwen3_5_moe_text Qwen3_5MoeForTokenClassification -> {e!r}")
        bad += 1

    try:
        from veomni.models.transformers.qwen3_moe.generated.patched_modeling_qwen3_moe_gpu import (  # noqa: PLC0415
            Qwen3MoeForTokenClassification,
        )

        has_conv = hasattr(Qwen3MoeForTokenClassification, "_create_checkpoint_tensor_converter")
        bad += not has_conv
        print(f"  [{'ok' if has_conv else 'FAIL'}] expert-layout converter attached to the MoE value head: {has_conv}")
    except Exception as e:
        print(f"  [ERROR] converter check: {e!r}")
        bad += 1
    return bad


def main() -> int:
    check_only = "--check" in sys.argv
    root = _veomni_dir()
    print(f"[veomni-valuehead] veomni at: {root}")
    total_applied = total_skipped = total_missing = 0
    total_missing += _install_q35_sidecar(root, check_only)
    verl_status = _patch_verl_valuehead(check_only)
    if verl_status:
        total_missing += verl_status

    for rel, patches in PATCHES.items():
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            print(f"  [MISSING FILE] {rel}")
            total_missing += len(patches)
            continue
        with open(path, encoding="utf-8") as f:
            content = f.read()
        changed = False
        for name, marker, find, replace in patches:
            if marker in content:
                print(f"  [already]  {rel} :: {name}")
                total_skipped += 1
            elif find in content:
                content = content.replace(find, replace, 1)
                changed = True
                total_applied += 1
                print(f"  [{'WOULD' if check_only else 'APPLY'}]  {rel} :: {name}")
            else:
                total_missing += 1
                print(f"  [!! NOT FOUND] {rel} :: {name} (veomni version mismatch? patch the find-string)")
        if changed and not check_only:
            # Atomic write, same reason as the R3 script: a shared venv can be patched
            # from several nodes at once and must never be observed half-written.
            tmp = f"{path}.vhtmp.{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp, path)

    total_missing += _patch_q35_text_family(root, check_only)

    print(
        f"[veomni-valuehead] applied={total_applied} already={total_skipped} "
        f"not_found={total_missing}{' (check only)' if check_only else ''}"
    )
    if total_missing:
        print(
            "[veomni-valuehead] WARNING: some patches did not match -- a veomni critic "
            "will be silently built as a CausalLM. Update the find-strings in this script."
        )
        return 2

    if not check_only:
        print("[veomni-valuehead] verifying dispatch:")
        if _verify(root):
            print("[veomni-valuehead] VERIFY FAILED -- do not run a veomni critic.")
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
