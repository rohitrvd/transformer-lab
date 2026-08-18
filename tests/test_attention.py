"""Tests for MultiHeadAttention: shapes, softmax normalization, causal masking."""

from __future__ import annotations

import torch

from attention import MultiHeadAttention
from config import TransformerConfig
from masks import make_causal_mask, make_padding_mask


def make_config(**overrides) -> TransformerConfig:
    defaults = dict(vocab_size=50, max_seq_len=16, d_model=32, num_heads=4, d_ff=64, dropout=0.0)
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def test_multihead_attention_output_shape():
    config = make_config()
    attn = MultiHeadAttention(config)
    x = torch.randn(2, 10, config.d_model)

    out, weights = attn(x, x, x)

    assert out.shape == (2, 10, config.d_model)
    assert weights.shape == (2, config.num_heads, 10, 10)


def test_attention_weights_sum_to_one():
    """Each query position's attention distribution must sum to 1 over keys."""
    config = make_config()
    attn = MultiHeadAttention(config)
    x = torch.randn(3, 6, config.d_model)

    _, weights = attn(x, x, x)

    sums = weights.sum(dim=-1)  # (batch, num_heads, q_len, kv_len) -> (batch, num_heads, q_len)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_causal_mask_zeroes_future_attention_weights():
    """With a causal mask, attention weight to any future key must be exactly 0."""
    config = make_config()
    attn = MultiHeadAttention(config)
    seq_len = 6
    x = torch.randn(2, seq_len, config.d_model)
    mask = make_causal_mask(seq_len, x.device)  # (1, 1, seq_len, seq_len)

    _, weights = attn(x, x, x, mask=mask)  # (batch, num_heads, seq_len, seq_len)

    # weights[..., i, j] must be 0 whenever j > i (future key position).
    future_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    future_weights = weights[..., future_mask]
    assert torch.allclose(future_weights, torch.zeros_like(future_weights), atol=1e-6)

    # Rows must still sum to 1 (mass redistributed over allowed positions).
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_padding_mask_zeroes_attention_to_pad_positions():
    config = make_config(pad_token_id=0)
    attn = MultiHeadAttention(config)
    batch, seq_len = 2, 5
    x = torch.randn(batch, seq_len, config.d_model)

    token_ids = torch.ones(batch, seq_len, dtype=torch.long)
    token_ids[:, -2:] = config.pad_token_id  # last two positions are padding
    mask = make_padding_mask(token_ids, config.pad_token_id)  # (batch, 1, 1, seq_len)

    _, weights = attn(x, x, x, mask=mask)  # (batch, num_heads, seq_len, seq_len)

    pad_weights = weights[..., -2:]
    assert torch.allclose(pad_weights, torch.zeros_like(pad_weights), atol=1e-6)


def test_cross_attention_supports_different_query_and_key_lengths():
    """Decoder cross-attention: query length != key/value length is valid."""
    config = make_config()
    attn = MultiHeadAttention(config)
    query = torch.randn(2, 4, config.d_model)  # tgt_len = 4
    key_value = torch.randn(2, 9, config.d_model)  # src_len = 9

    out, weights = attn(query, key_value, key_value)

    assert out.shape == (2, 4, config.d_model)
    assert weights.shape == (2, config.num_heads, 4, 9)
