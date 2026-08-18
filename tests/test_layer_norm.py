"""Tests for the hand-rolled LayerNorm and pre/post-norm SublayerConnection."""

from __future__ import annotations

import torch

from config import TransformerConfig
from layer_norm import LayerNorm, SublayerConnection


def test_layer_norm_output_has_zero_mean_unit_variance():
    norm = LayerNorm(d_model=16)
    x = torch.randn(4, 5, 16) * 10 + 3  # arbitrary scale/shift

    out = norm(x)

    mean = out.mean(dim=-1)
    var = out.var(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-3)


def test_layer_norm_preserves_shape():
    norm = LayerNorm(d_model=16)
    x = torch.randn(2, 3, 16)

    assert norm(x).shape == x.shape


def test_sublayer_connection_preserves_shape_pre_and_post():
    for placement in ("pre", "post"):
        config = TransformerConfig(
            vocab_size=10, d_model=16, num_heads=2, d_ff=32, dropout=0.0,
            norm_placement=placement,
        )
        sublayer_conn = SublayerConnection(config)
        x = torch.randn(2, 5, 16)

        out = sublayer_conn(x, lambda t: t * 2)  # trivial sublayer

        assert out.shape == x.shape


def test_sublayer_connection_pre_norm_normalizes_before_sublayer():
    """In pre-norm mode, a zero sublayer should leave the residual `x` unchanged."""
    config = TransformerConfig(
        vocab_size=10, d_model=16, num_heads=2, d_ff=32, dropout=0.0, norm_placement="pre",
    )
    sublayer_conn = SublayerConnection(config)
    x = torch.randn(2, 5, 16)

    out = sublayer_conn(x, lambda t: torch.zeros_like(t))

    assert torch.allclose(out, x)
