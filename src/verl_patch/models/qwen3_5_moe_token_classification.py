"""Qwen3.5-MoE value head. HuggingFace and VeOmni ship no ForTokenClassification.

Mirrors ``Qwen3_5MoeForConditionalGeneration`` so a critic pointed at the same
HF directory as the actor loads ``model.language_model.*`` / ``model.visual.*``
and only the new ``score`` head is randomly initialised. ``lm_head`` is ignored.

The class is built on VeOmni's patched modeling (GDN + sequence-parallel), not
the stock transformers module -- an FSDP/HF-only subclass would drop those
kernels and silently train a different body.
"""

from __future__ import annotations

import sys
from typing import Any

import torch
from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.utils import can_return_tuple

try:
    from veomni.utils.device import IS_NPU_AVAILABLE
except Exception:  # pragma: no cover - veomni missing in some util processes
    IS_NPU_AVAILABLE = False

if IS_NPU_AVAILABLE:
    from veomni.models.transformers.qwen3_5_moe.generated.patched_modeling_qwen3_5_moe_npu import (
        Qwen3_5MoeModel,
        Qwen3_5MoePreTrainedModel,
    )
else:
    from veomni.models.transformers.qwen3_5_moe.generated.patched_modeling_qwen3_5_moe_gpu import (
        Qwen3_5MoeModel,
        Qwen3_5MoePreTrainedModel,
    )

# Re-export every OpSlot of the patched modeling module at this module's top
# level. ``build_foundation_model`` resolves kernels by scanning ``dir()`` of
# the module that defines ``model_cls`` -- for the critic that is THIS module,
# and without these names ``_bind_veomni_ops`` finds no slots and silently
# skips binding, so ``Qwen3_5MoeGatedDeltaNet.__init__`` freezes
# ``causal_conv1d_fn = None`` and the first varlen forward raises
# NotImplementedError. The slots are shared instances; binding them via this
# module binds the modeling module's kernels. The set differs between the GPU
# and NPU variants, so mirror it dynamically instead of naming slots here.
from veomni.ops.dispatch import OpSlot as _OpSlot

_modeling_module = sys.modules[Qwen3_5MoeModel.__module__]
_op_slot_names = [
    _name
    for _name in dir(_modeling_module)
    if isinstance(getattr(_modeling_module, _name, None), _OpSlot)
]
if not _op_slot_names:
    raise ImportError(
        f"No OpSlot instances found in {Qwen3_5MoeModel.__module__}; "
        "the critic would be built with unbound (eager) kernels and fail on varlen input."
    )
for _name in _op_slot_names:
    globals()[_name] = getattr(_modeling_module, _name)


def rewrite_token_classification_architecture(architecture: str) -> str:
    """``*ForConditionalGeneration`` / ``*ForCausalLM`` -> ``*ForTokenClassification``."""
    for suffix in ("ForConditionalGeneration", "ForCausalLM"):
        if suffix in architecture:
            return architecture.replace(suffix, "ForTokenClassification")
    return architecture


class Qwen3_5MoeForTokenClassification(Qwen3_5MoePreTrainedModel):
    """Per-token value model for Qwen3.5-35B-A3B SAO critics."""

    _keys_to_ignore_on_load_unexpected = [r"^lm_head.*", r"^mtp.*"]
    accepts_loss_kwargs = False
    _no_split_modules = ["Qwen3_5MoeDecoderLayer", "Qwen3_5MoeVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = int(getattr(config, "num_labels", 1) or 1)
        self.model = Qwen3_5MoeModel(config)
        text_cfg = getattr(config, "text_config", None)
        hidden_size = (
            text_cfg.hidden_size if text_cfg is not None else config.hidden_size
        )
        dropout = getattr(config, "classifier_dropout", None)
        if dropout is None:
            dropout = getattr(config, "hidden_dropout", 0.0) or 0.0
        try:
            dropout = float(dropout)
        except (TypeError, ValueError):
            dropout = 0.0
        self.dropout = nn.Dropout(dropout)
        self.score = nn.Linear(hidden_size, self.num_labels)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> TokenClassifierOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.score(sequence_output)
        loss = None
        if labels is not None and hasattr(self, "loss_function"):
            loss = self.loss_function(logits, labels, self.config)
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )


def register_qwen3_5_moe_token_classification() -> None:
    """Make ``AutoModelForTokenClassification`` resolve this family.

    VeOmniEngineWithValueHead looks up the class by config type before it
    builds the model. In-memory only -- every process that builds a critic
    must call this (the veomni dispatch patch also imports this module).
    """
    try:
        from transformers import AutoModelForTokenClassification
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
            Qwen3_5MoeConfig,
        )
    except Exception:
        return

    AutoModelForTokenClassification.register(
        Qwen3_5MoeConfig, Qwen3_5MoeForTokenClassification
    )


register_qwen3_5_moe_token_classification()
