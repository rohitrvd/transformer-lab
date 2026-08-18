"""Shape and value-sanity tests for embeddings and positional encoding."""

from __future__ import annotations

import torch

from config import TransformerConfig
from embeddings import SinusoidalPositionalEncoding, TokenEmbedding, TransformerEmbedding


def make_config(**overrides) -> TransformerConfig:
    defaults = dict(vocab_size=50, max_seq_len=16, d_model=32, num_heads=4, d_ff=64)
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def test_token_embedding_shape():
    config = make_config()
    embed = TokenEmbedding(config)
    token_ids = torch.randint(0, config.vocab_size, (2, 10))

    out = embed(token_ids)

    assert out.shape == (2, 10, config.d_model)


def test_positional_encoding_shape_and_determinism():
    config = make_config()
    pe = SinusoidalPositionalEncoding(config)
    pe.eval()  # disable dropout for a deterministic comparison
    x = torch.zeros(2, 10, config.d_model)

    out1 = pe(x)
    out2 = pe(x)

    assert out1.shape == (2, 10, config.d_model)
    assert torch.allclose(out1, out2)


def test_positional_encoding_differs_across_positions():
    """Different positions must receive different positional signals."""
    config = make_config()
    pe = SinusoidalPositionalEncoding(config)
    pe.eval()
    x = torch.zeros(1, 5, config.d_model)

    out = pe(x)

    for i in range(4):
        assert not torch.allclose(out[0, i], out[0, i + 1])


def test_transformer_embedding_end_to_end_shape():
    config = make_config()
    embedding = TransformerEmbedding(config)
    token_ids = torch.randint(0, config.vocab_size, (3, 7))

    out = embedding(token_ids)

    assert out.shape == (3, 7, config.d_model)
