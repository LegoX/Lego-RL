"""Model-family sidecars that HuggingFace / VeOmni do not ship."""

from .qwen3_5_moe_token_classification import (  # noqa: F401
    Qwen3_5MoeForTokenClassification,
    register_qwen3_5_moe_token_classification,
    rewrite_token_classification_architecture,
)

register_qwen3_5_moe_token_classification()
