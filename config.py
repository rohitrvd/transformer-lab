"""Central configuration for the transformer implementation.

Every architectural choice that a component needs (dimensions, number of
layers, which attention implementation to use, pre/post layer-norm
placement, activation function, ...) lives on `TransformerConfig`. Model
assembly code (see `transformer.py`) reads this config and wires up the
concrete modules, so experimenting with a new component is a config change,
not a surgical edit across multiple files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Type, TYPE_CHECKING

import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from attention import BaseAttention


# ---------------------------------------------------------------------------
# Attention registry
# ---------------------------------------------------------------------------
# Maps a string key (used in TransformerConfig.attention_type /
# cross_attention_type) to a concrete BaseAttention subclass. New attention
# variants register themselves here so they can be selected purely by name
# in the config, without encoder.py / decoder.py ever importing them.
#
# Populated at the bottom of attention.py (after MultiHeadAttention is
# defined) to avoid a circular import between config.py and attention.py.
ATTENTION_REGISTRY: Dict[str, Type["BaseAttention"]] = {}


def register_attention(name: str, cls: Type["BaseAttention"]) -> None:
    """Register an attention implementation under a string key.

    Conceptually this is the plug point for new attention variants: call
    `register_attention("sliding_window", SlidingWindowAttention)` once,
    anywhere, and `TransformerConfig(attention_type="sliding_window")`
    becomes a valid config.
    """
    ATTENTION_REGISTRY[name] = cls


def resolve_attention(name: str) -> Type["BaseAttention"]:
    """Look up a registered attention class by its config string key."""
    if name not in ATTENTION_REGISTRY:
        available = ", ".join(sorted(ATTENTION_REGISTRY)) or "<none registered>"
        raise KeyError(
            f"Unknown attention_type '{name}'. Available: {available}"
        )
    return ATTENTION_REGISTRY[name]


# ---------------------------------------------------------------------------
# Activation registry
# ---------------------------------------------------------------------------
ACTIVATION_REGISTRY: Dict[str, Callable[[Tensor], Tensor]] = {
    "relu": F.relu,
    "gelu": F.gelu,
}


def resolve_activation(name: str) -> Callable[[Tensor], Tensor]:
    """Look up an activation function by its config string key."""
    if name not in ACTIVATION_REGISTRY:
        available = ", ".join(sorted(ACTIVATION_REGISTRY))
        raise KeyError(f"Unknown activation '{name}'. Available: {available}")
    return ACTIVATION_REGISTRY[name]


@dataclass
class TransformerConfig:
    """Hyperparameters and architectural switches for the transformer.

    This is the single source of truth model-assembly code reads from.
    Nothing in the model files should hardcode a dimension, layer count,
    or component choice that belongs here.
    """

    # --- vocabulary / sequence ---
    vocab_size: int
    max_seq_len: int = 512
    pad_token_id: int = 0

    # --- core dimensions ---
    d_model: int = 512
    num_heads: int = 8
    d_ff: int = 2048

    # --- depth ---
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6

    # --- regularization ---
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    # --- pluggable components (selected by string key, see registries above) ---
    activation: str = "relu"  # "relu" | "gelu"
    norm_placement: str = "pre"  # "pre" | "post"
    attention_type: str = "multi_head"
    cross_attention_type: Optional[str] = None  # defaults to attention_type

    # --- misc ---
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        """Validate cross-field invariants once all fields are set."""
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.norm_placement not in ("pre", "post"):
            raise ValueError(
                f"norm_placement must be 'pre' or 'post', got "
                f"'{self.norm_placement}'"
            )
        if self.cross_attention_type is None:
            self.cross_attention_type = self.attention_type

    @property
    def d_k(self) -> int:
        """Per-head key/query/value dimension: d_model split across heads."""
        return self.d_model // self.num_heads
