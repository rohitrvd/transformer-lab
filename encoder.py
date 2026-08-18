"""Transformer encoder: a stack of self-attention + feed-forward layers.

`Encoder` and `EncoderLayer` never import a concrete attention class.
Instead they receive `attention_cls: Type[BaseAttention]` as a constructor
argument and instantiate it themselves. This is what lets a new attention
variant (sliding-window, grouped-query, ...) plug into the encoder without
any edit to this file — only `config.attention_type` and the class passed
in at model-build time change.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Type

from torch import Tensor, nn

from attention import BaseAttention
from config import TransformerConfig
from feedforward import PositionwiseFeedForward
from layer_norm import LayerNorm, SublayerConnection


class EncoderLayer(nn.Module):
    """One encoder block: self-attention sublayer + feed-forward sublayer.

    Each sublayer is wrapped in a residual connection and layer norm via
    `SublayerConnection`, whose pre/post-norm ordering is config-driven.
    """

    def __init__(self, config: TransformerConfig, attention_cls: Type[BaseAttention]) -> None:
        super().__init__()
        self.self_attn = attention_cls(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.self_attn_sublayer = SublayerConnection(config)
        self.feed_forward_sublayer = SublayerConnection(config)
        self.last_attn_weights: Optional[Tensor] = None

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Run self-attention over `x`, then the feed-forward block.

        x: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        mask: broadcastable to (batch, num_heads, seq_len, seq_len); used to
              hide padding positions from attention (encoders are
              non-causal, so no future-masking is applied here).
        """

        def self_attention(x: Tensor) -> Tensor:
            out, attn_weights = self.self_attn(x, x, x, mask=mask)
            self.last_attn_weights = attn_weights  # (batch, num_heads, seq_len, seq_len), kept for inspection/tests
            return out

        x = self.self_attn_sublayer(x, self_attention)  # (batch, seq_len, d_model)
        x = self.feed_forward_sublayer(x, self.feed_forward)  # (batch, seq_len, d_model)
        return x


class Encoder(nn.Module):
    """A stack of `num_encoder_layers` identical `EncoderLayer`s.

    Applies a final layer norm after the stack (standard practice,
    particularly important for pre-norm architectures where the residual
    stream is otherwise never explicitly normalized at the output).
    """

    def __init__(self, config: TransformerConfig, attention_cls: Type[BaseAttention]) -> None:
        super().__init__()
        self.layers: List[EncoderLayer] = nn.ModuleList(
            [EncoderLayer(config, attention_cls) for _ in range(config.num_encoder_layers)]
        )
        self.norm = LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Run `x` through every encoder layer in sequence, then a final norm.

        x: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask=mask)  # (batch, seq_len, d_model)
        return self.norm(x)  # (batch, seq_len, d_model)
