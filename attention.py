"""Pluggable attention mechanisms.

Defines `BaseAttention`, the abstract interface every attention variant
must implement, and `MultiHeadAttention`, the standard scaled dot-product
multi-head implementation from "Attention Is All You Need".

To add a new variant (sliding-window, grouped-query, a custom experiment):
    1. Subclass `BaseAttention`.
    2. Implement `__init__(self, config)` and
       `forward(query, key, value, mask=None) -> (output, attn_weights)`.
    3. Call `register_attention("your_name", YourClass)` at import time.
    4. Set `TransformerConfig(attention_type="your_name")` (and/or
       `cross_attention_type`) to use it.

`encoder.py` and `decoder.py` never import a concrete attention class —
they receive an `attention_cls: Type[BaseAttention]` resolved from the
config's string key and instantiate it themselves. This is what makes
attention swaps a config change instead of an edit to the surrounding
model.

`MultiHeadAttention.forward` additionally accepts optional `kv_cache` /
`layer_idx` keyword arguments (see `inference/kv_cache.py`), used only
during autoregressive decoding. They default to `None`, so every existing
call site (training, the no-cache forward pass) is unaffected. Custom
`BaseAttention` subclasses are not required to support them — only variants
that opt in by handling these kwargs work with the cached generation path
in `inference/generation.py`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple, TYPE_CHECKING

import torch
from torch import Tensor, nn

from config import TransformerConfig, register_attention

if TYPE_CHECKING:
    from inference.kv_cache import KVCache


class BaseAttention(nn.Module, ABC):
    """Abstract interface for all attention implementations.

    Every implementation takes a config (so construction is uniform
    regardless of variant) and maps (query, key, value, mask) to
    (output, attention_weights). `query`/`key`/`value` may have different
    sequence lengths (e.g. decoder cross-attention: query from the decoder,
    key/value from the encoder), but must share `batch` and `d_model`.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
        kv_cache: Optional["KVCache"] = None,
        layer_idx: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute attention.

        query: (batch, q_len, d_model)
        key:   (batch, kv_len, d_model)
        value: (batch, kv_len, d_model)
        mask:  broadcastable to (batch, num_heads, q_len, kv_len); positions
               with mask == 0 are excluded from attention.
        kv_cache: optional KVCache (see inference/kv_cache.py); when given,
               `key`/`value` are the newly computed tokens only, and prior
               K/V are read from (and the new K/V appended to) the cache at
               `layer_idx` instead of being recomputed from scratch.
        layer_idx: which layer's slot to use in `kv_cache`; required
               together with `kv_cache`, ignored otherwise.
        Returns:
            output: (batch, q_len, d_model)
            attn_weights: (batch, num_heads, q_len, kv_len)
        """
        raise NotImplementedError


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> Tuple[Tensor, Tensor]:
    """Compute softmax(QK^T / sqrt(d_k)) V for already-split heads.

    This is the core attention operation, factored out as a free function
    so alternative attention modules can reuse it on differently shaped
    (e.g. windowed) Q/K/V without reimplementing the math.

    query: (batch, num_heads, q_len, d_k)
    key:   (batch, num_heads, kv_len, d_k)
    value: (batch, num_heads, kv_len, d_k)
    mask:  broadcastable to (batch, num_heads, q_len, kv_len)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # (batch, num_heads, q_len, d_k) @ (batch, num_heads, d_k, kv_len) -> (batch, num_heads, q_len, kv_len)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))  # (batch, num_heads, q_len, kv_len)

    attn_weights = torch.softmax(scores, dim=-1)  # (batch, num_heads, q_len, kv_len)

    if dropout is not None:
        attn_weights = dropout(attn_weights)  # (batch, num_heads, q_len, kv_len)

    output = torch.matmul(attn_weights, value)
    # (batch, num_heads, q_len, kv_len) @ (batch, num_heads, kv_len, d_k) -> (batch, num_heads, q_len, d_k)

    return output, attn_weights


class MultiHeadAttention(BaseAttention):
    """Standard multi-head scaled dot-product attention.

    Projects query/key/value into `num_heads` parallel lower-dimensional
    subspaces, applies scaled dot-product attention independently in each
    subspace, concatenates the results, and projects back to `d_model`.
    Running several smaller attention operations in parallel lets the model
    attend to information from different representation subspaces at
    different positions simultaneously.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_k = config.d_k  # d_model // num_heads, dimension of each head

        self.w_q = nn.Linear(config.d_model, config.d_model)
        self.w_k = nn.Linear(config.d_model, config.d_model)
        self.w_v = nn.Linear(config.d_model, config.d_model)
        self.w_o = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        """Reshape the last dimension into (num_heads, d_k) and move heads up.

        x: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        """
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.d_k)  # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k)
        return x.permute(0, 2, 1, 3)  # (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """Inverse of `_split_heads`: recombine heads into a single d_model vector.

        x: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        """
        batch, num_heads, seq_len, d_k = x.shape
        x = x.permute(0, 2, 1, 3)  # (batch, num_heads, seq_len, d_k) -> (batch, seq_len, num_heads, d_k)
        return x.contiguous().view(batch, seq_len, num_heads * d_k)  # -> (batch, seq_len, d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
        kv_cache: Optional["KVCache"] = None,
        layer_idx: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Project to Q/K/V, split into heads, attend, merge, project out.

        query: (batch, q_len, d_model)
        key:   (batch, kv_len, d_model) — or, when `kv_cache` is given, just
               the newly-seen tokens (kv_len = new_len, e.g. 1 during decode)
        value: (batch, kv_len, d_model) — same shape convention as `key`
        mask:  broadcastable to (batch, num_heads, q_len, kv_len)
        kv_cache: optional cache; see `BaseAttention.forward` for semantics
        layer_idx: this layer's slot in `kv_cache`, required if it's given
        """
        q = self.w_q(query)  # (batch, q_len, d_model) -> (batch, q_len, d_model)
        q = self._split_heads(q)  # (batch, q_len, d_model) -> (batch, num_heads, q_len, d_k)

        if kv_cache is not None:
            new_k = self._split_heads(self.w_k(key))  # (batch, new_len, d_model) -> (batch, num_heads, new_len, d_k)
            new_v = self._split_heads(self.w_v(value))  # (batch, new_len, d_model) -> (batch, num_heads, new_len, d_k)
            k, v = kv_cache.update(layer_idx, new_k, new_v)  # (batch, num_heads, total_len, d_k) each
        else:
            k = self._split_heads(self.w_k(key))  # (batch, kv_len, d_model) -> (batch, num_heads, kv_len, d_k)
            v = self._split_heads(self.w_v(value))  # (batch, kv_len, d_model) -> (batch, num_heads, kv_len, d_k)

        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (batch, q_len, kv_len) -> (batch, 1, q_len, kv_len), broadcasts over heads

        attn_output, attn_weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout
        )  # attn_output: (batch, num_heads, q_len, d_k)

        merged = self._merge_heads(attn_output)  # (batch, num_heads, q_len, d_k) -> (batch, q_len, d_model)
        output = self.w_o(merged)  # (batch, q_len, d_model) -> (batch, q_len, d_model)

        return output, attn_weights


# Register the built-in implementation so it's selectable via
# TransformerConfig(attention_type="multi_head") — the default.
register_attention("multi_head", MultiHeadAttention)
