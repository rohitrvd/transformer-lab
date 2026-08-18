"""Hand-rolled layer normalization and the residual/norm wrapper.

Implements LayerNorm from first principles (no `nn.LayerNorm`) so the
mean/variance/scale/shift math is visible and editable, plus a
`SublayerConnection` module that applies the residual connection either
before or after normalizing, controlled by `config.norm_placement`.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn

from config import TransformerConfig


class LayerNorm(nn.Module):
    """Normalizes activations across the last (feature) dimension.

    For each position in the batch/sequence, subtracts the mean and divides
    by the standard deviation computed over the `d_model` features, then
    applies a learned elementwise scale (`gamma`) and shift (`beta`). This
    stabilizes the distribution of activations flowing between sublayers.
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))  # (d_model,)
        self.beta = nn.Parameter(torch.zeros(d_model))  # (d_model,)

    def forward(self, x: Tensor) -> Tensor:
        """Apply layer normalization over the last dimension of `x`.

        x: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        """
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, d_model) -> (batch, seq_len, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # (batch, seq_len, d_model) -> (batch, seq_len, 1)
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)  # (batch, seq_len, d_model)
        out = self.gamma * x_normalized + self.beta  # (batch, seq_len, d_model)
        return out


class SublayerConnection(nn.Module):
    """Wraps a sublayer (attention or feed-forward) with a residual connection.

    The relative order of "normalize" and "add residual" is a well-known
    architectural fork: post-norm (the original Transformer paper,
    `Norm(x + Sublayer(x))`) versus pre-norm (`x + Sublayer(Norm(x))`,
    generally easier to train at depth). `config.norm_placement` selects
    between them so this is a config change, not a code change.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.norm = LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        self.norm_placement = config.norm_placement

    def forward(self, x: Tensor, sublayer: Callable[[Tensor], Tensor]) -> Tensor:
        """Apply `sublayer` to `x` with a residual connection and norm.

        x: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        `sublayer` is any function/module mapping (batch, seq_len, d_model)
        to a tensor of the same shape (e.g. self-attention or the
        feed-forward block).
        """
        if self.norm_placement == "pre":
            # x + Sublayer(Norm(x))
            return x + self.dropout(sublayer(self.norm(x)))  # (batch, seq_len, d_model)
        else:
            # Norm(x + Sublayer(x))
            return self.norm(x + self.dropout(sublayer(x)))  # (batch, seq_len, d_model)
