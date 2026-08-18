"""Transformer decoder: self-attention (causal) + cross-attention + feed-forward.

Like `encoder.py`, `Decoder`/`DecoderLayer` take attention classes as
constructor arguments rather than importing a concrete implementation, so
attention variants plug in via config without touching this file. Self-
attention and cross-attention may use different attention implementations
(`attention_cls` vs `cross_attention_cls`) since, e.g., a windowed variant
may only make sense for self-attention over the target sequence.
"""

from __future__ import annotations

from typing import List, Optional, Type

from torch import Tensor, nn

from attention import BaseAttention
from config import TransformerConfig
from feedforward import PositionwiseFeedForward
from layer_norm import LayerNorm, SublayerConnection


class DecoderLayer(nn.Module):
    """One decoder block: causal self-attention, cross-attention, feed-forward.

    Self-attention lets each target position attend to earlier target
    positions (masked to prevent looking ahead). Cross-attention lets each
    target position attend to the full encoder output. The feed-forward
    block then transforms each position independently, as in the encoder.
    """

    def __init__(
        self,
        config: TransformerConfig,
        attention_cls: Type[BaseAttention],
        cross_attention_cls: Type[BaseAttention],
    ) -> None:
        super().__init__()
        self.self_attn = attention_cls(config)
        self.cross_attn = cross_attention_cls(config)
        self.feed_forward = PositionwiseFeedForward(config)

        self.self_attn_sublayer = SublayerConnection(config)
        self.cross_attn_sublayer = SublayerConnection(config)
        self.feed_forward_sublayer = SublayerConnection(config)

        self.last_self_attn_weights: Optional[Tensor] = None
        self.last_cross_attn_weights: Optional[Tensor] = None

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        self_attn_mask: Optional[Tensor] = None,
        cross_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Run causal self-attention, then cross-attention to `memory`, then feed-forward.

        x: (batch, tgt_len, d_model) -> (batch, tgt_len, d_model)
        memory: (batch, src_len, d_model), the encoder output being attended to.
        self_attn_mask: broadcastable to (batch, num_heads, tgt_len, tgt_len);
            should combine causal + target-padding masking.
        cross_attn_mask: broadcastable to (batch, num_heads, tgt_len, src_len);
            should mask source padding.
        """

        def self_attention(x: Tensor) -> Tensor:
            out, attn_weights = self.self_attn(x, x, x, mask=self_attn_mask)
            self.last_self_attn_weights = attn_weights  # (batch, num_heads, tgt_len, tgt_len)
            return out

        def cross_attention(x: Tensor) -> Tensor:
            out, attn_weights = self.cross_attn(x, memory, memory, mask=cross_attn_mask)
            self.last_cross_attn_weights = attn_weights  # (batch, num_heads, tgt_len, src_len)
            return out

        x = self.self_attn_sublayer(x, self_attention)  # (batch, tgt_len, d_model)
        x = self.cross_attn_sublayer(x, cross_attention)  # (batch, tgt_len, d_model)
        x = self.feed_forward_sublayer(x, self.feed_forward)  # (batch, tgt_len, d_model)
        return x


class Decoder(nn.Module):
    """A stack of `num_decoder_layers` identical `DecoderLayer`s."""

    def __init__(
        self,
        config: TransformerConfig,
        attention_cls: Type[BaseAttention],
        cross_attention_cls: Type[BaseAttention],
    ) -> None:
        super().__init__()
        self.layers: List[DecoderLayer] = nn.ModuleList(
            [
                DecoderLayer(config, attention_cls, cross_attention_cls)
                for _ in range(config.num_decoder_layers)
            ]
        )
        self.norm = LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        self_attn_mask: Optional[Tensor] = None,
        cross_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Run `x` through every decoder layer, attending to `memory`, then a final norm.

        x: (batch, tgt_len, d_model) -> (batch, tgt_len, d_model)
        memory: (batch, src_len, d_model)
        """
        for layer in self.layers:
            x = layer(
                x, memory, self_attn_mask=self_attn_mask, cross_attn_mask=cross_attn_mask
            )  # (batch, tgt_len, d_model)
        return self.norm(x)  # (batch, tgt_len, d_model)
