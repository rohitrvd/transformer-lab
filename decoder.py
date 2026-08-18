"""Transformer decoder: self-attention (causal) + cross-attention + feed-forward.

Like `encoder.py`, `Decoder`/`DecoderLayer` take attention classes as
constructor arguments rather than importing a concrete implementation, so
attention variants plug in via config without touching this file. Self-
attention and cross-attention may use different attention implementations
(`attention_cls` vs `cross_attention_cls`) since, e.g., a windowed variant
may only make sense for self-attention over the target sequence.

`DecoderLayer`/`Decoder` also accept an optional `kv_cache` (see
`inference/kv_cache.py`), threaded only to self-attention — cross-attention
always recomputes over the full (fixed) encoder `memory`, since memory
doesn't grow across decode steps the way the target sequence does. Each
layer is assigned a `layer_idx` (its slot in the cache) when the stack is
built. Passing no cache (the default) leaves training/no-cache behavior
unchanged.
"""

from __future__ import annotations

from typing import List, Optional, Type, TYPE_CHECKING

from torch import Tensor, nn

from attention import BaseAttention
from config import TransformerConfig
from feedforward import PositionwiseFeedForward
from layer_norm import LayerNorm, SublayerConnection

if TYPE_CHECKING:
    from inference.kv_cache import KVCache


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
        self.layer_idx: Optional[int] = None  # assigned by Decoder.__init__ / DecoderOnlyModel.__init__

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        self_attn_mask: Optional[Tensor] = None,
        cross_attn_mask: Optional[Tensor] = None,
        kv_cache: Optional["KVCache"] = None,
    ) -> Tensor:
        """Run causal self-attention, then cross-attention to `memory`, then feed-forward.

        x: (batch, tgt_len, d_model) -> (batch, tgt_len, d_model)
        memory: (batch, src_len, d_model), the encoder output being attended to.
        self_attn_mask: broadcastable to (batch, num_heads, tgt_len, tgt_len);
            should combine causal + target-padding masking.
        cross_attn_mask: broadcastable to (batch, num_heads, tgt_len, src_len);
            should mask source padding.
        kv_cache: optional cache for self-attention only (see module docstring).
        """

        def self_attention(x: Tensor) -> Tensor:
            if kv_cache is not None:
                out, attn_weights = self.self_attn(
                    x, x, x, mask=self_attn_mask, kv_cache=kv_cache, layer_idx=self.layer_idx
                )
            else:
                out, attn_weights = self.self_attn(x, x, x, mask=self_attn_mask)
            self.last_self_attn_weights = attn_weights.detach()  # (batch, num_heads, tgt_len, tgt_len); detached,
            # see encoder.py's EncoderLayer.forward for why
            return out

        def cross_attention(x: Tensor) -> Tensor:
            out, attn_weights = self.cross_attn(x, memory, memory, mask=cross_attn_mask)
            self.last_cross_attn_weights = attn_weights.detach()  # (batch, num_heads, tgt_len, src_len); detached
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
        for idx, layer in enumerate(self.layers):
            layer.layer_idx = idx  # this layer's slot in a KVCache, see inference/kv_cache.py
        self.norm = LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        self_attn_mask: Optional[Tensor] = None,
        cross_attn_mask: Optional[Tensor] = None,
        kv_cache: Optional["KVCache"] = None,
    ) -> Tensor:
        """Run `x` through every decoder layer, attending to `memory`, then a final norm.

        x: (batch, tgt_len, d_model) -> (batch, tgt_len, d_model)
        memory: (batch, src_len, d_model)
        kv_cache: optional cache for self-attention only, shared across all layers
            (each layer reads/writes its own slot via its assigned `layer_idx`).
        """
        for layer in self.layers:
            x = layer(
                x, memory, self_attn_mask=self_attn_mask, cross_attn_mask=cross_attn_mask,
                kv_cache=kv_cache,
            )  # (batch, tgt_len, d_model)
        return self.norm(x)  # (batch, tgt_len, d_model)
