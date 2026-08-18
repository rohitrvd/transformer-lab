"""Position-wise feed-forward network.

The second sublayer in each encoder/decoder layer: a two-layer MLP applied
identically (with shared weights) to every position in the sequence,
expanding to `d_ff` and back down to `d_model`.
"""

from __future__ import annotations

from torch import Tensor, nn

from config import TransformerConfig, resolve_activation


class PositionwiseFeedForward(nn.Module):
    """Two linear layers with a nonlinearity, applied per-position.

    Expands each position's `d_model`-dim vector to `d_ff` dimensions,
    applies a nonlinearity (selected via `config.activation`), then
    projects back to `d_model`. Because the same weights are applied at
    every sequence position independently, this is where the model does
    per-token nonlinear feature transformation, complementing attention's
    cross-token mixing.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(config.d_model, config.d_ff)
        self.linear_2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = resolve_activation(config.activation)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the expand -> activate -> dropout -> project sequence.

        x: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        """
        x = self.linear_1(x)  # (batch, seq_len, d_model) -> (batch, seq_len, d_ff)
        x = self.activation(x)  # (batch, seq_len, d_ff) -> (batch, seq_len, d_ff)
        x = self.dropout(x)  # (batch, seq_len, d_ff) -> (batch, seq_len, d_ff)
        x = self.linear_2(x)  # (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return x
