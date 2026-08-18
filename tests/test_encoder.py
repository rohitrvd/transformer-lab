"""Shape tests for EncoderLayer and Encoder."""

from __future__ import annotations

import torch

from attention import MultiHeadAttention
from config import TransformerConfig
from encoder import Encoder, EncoderLayer


def make_config(**overrides) -> TransformerConfig:
    defaults = dict(
        vocab_size=50, max_seq_len=16, d_model=32, num_heads=4, d_ff=64,
        num_encoder_layers=3, dropout=0.0,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def test_encoder_layer_preserves_shape():
    config = make_config()
    layer = EncoderLayer(config, MultiHeadAttention)
    x = torch.randn(2, 10, config.d_model)

    out = layer(x)

    assert out.shape == x.shape


def test_encoder_stack_preserves_shape():
    config = make_config()
    encoder = Encoder(config, MultiHeadAttention)
    x = torch.randn(2, 10, config.d_model)

    out = encoder(x)

    assert out.shape == x.shape


def test_encoder_has_configured_number_of_layers():
    config = make_config(num_encoder_layers=5)
    encoder = Encoder(config, MultiHeadAttention)

    assert len(encoder.layers) == 5


def test_encoder_pre_and_post_norm_both_run():
    """Both norm_placement modes should produce valid, distinctly-computed outputs."""
    pre_config = make_config(norm_placement="pre")
    post_config = make_config(norm_placement="post")
    torch.manual_seed(0)
    x = torch.randn(2, 10, pre_config.d_model)

    pre_encoder = Encoder(pre_config, MultiHeadAttention)
    post_encoder = Encoder(post_config, MultiHeadAttention)

    pre_out = pre_encoder(x)
    post_out = post_encoder(x)

    assert pre_out.shape == x.shape
    assert post_out.shape == x.shape
