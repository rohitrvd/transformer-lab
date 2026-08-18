"""Shape tests for DecoderLayer and Decoder, including cross-attention to memory."""

from __future__ import annotations

import torch

from attention import MultiHeadAttention
from config import TransformerConfig
from decoder import Decoder, DecoderLayer
from masks import make_causal_mask


def make_config(**overrides) -> TransformerConfig:
    defaults = dict(
        vocab_size=50, max_seq_len=16, d_model=32, num_heads=4, d_ff=64,
        num_decoder_layers=3, dropout=0.0,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def test_decoder_layer_preserves_target_shape():
    config = make_config()
    layer = DecoderLayer(config, MultiHeadAttention, MultiHeadAttention)
    x = torch.randn(2, 6, config.d_model)  # target
    memory = torch.randn(2, 9, config.d_model)  # source, different length

    out = layer(x, memory)

    assert out.shape == x.shape


def test_decoder_stack_preserves_target_shape():
    config = make_config()
    decoder = Decoder(config, MultiHeadAttention, MultiHeadAttention)
    x = torch.randn(2, 6, config.d_model)
    memory = torch.randn(2, 9, config.d_model)

    out = decoder(x, memory)

    assert out.shape == x.shape


def test_decoder_causal_self_attention_ignores_future_targets():
    """Changing a future target token must not change earlier decoder outputs."""
    config = make_config(dropout=0.0)
    decoder = Decoder(config, MultiHeadAttention, MultiHeadAttention)
    decoder.eval()

    seq_len = 6
    x = torch.randn(1, seq_len, config.d_model)
    memory = torch.randn(1, 9, config.d_model)
    causal_mask = make_causal_mask(seq_len, x.device)

    x_modified = x.clone()
    x_modified[:, -1, :] = torch.randn(config.d_model)  # perturb only the last position

    out_original = decoder(x, memory, self_attn_mask=causal_mask)
    out_modified = decoder(x_modified, memory, self_attn_mask=causal_mask)

    # All positions except the last must be unaffected by the perturbation.
    assert torch.allclose(out_original[:, :-1, :], out_modified[:, :-1, :], atol=1e-5)
