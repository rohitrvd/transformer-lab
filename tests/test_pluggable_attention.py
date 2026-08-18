"""Verifies the pluggable-attention interface.

A dummy alternate attention implementation can be swapped in via
config/constructor args and the model still runs end-to-end, with no
changes to encoder.py/decoder.py/transformer.py. This is the template to
copy when validating a real new attention variant.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from attention import BaseAttention, scaled_dot_product_attention
from config import TransformerConfig, register_attention, resolve_attention
from decoder import Decoder, DecoderLayer
from encoder import Encoder, EncoderLayer
from transformer import build_encoder_decoder


class DummyUniformAttention(BaseAttention):
    """A deliberately trivial attention variant used only to test pluggability.

    Ignores Q/K content entirely and attends uniformly over all (unmasked)
    key positions, still respecting the supplied mask. This is intentionally
    a bad attention mechanism — the point is only to prove that *any*
    `BaseAttention` subclass can be dropped into the encoder/decoder without
    touching those files.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        self.w_v = nn.Linear(config.d_model, config.d_model)
        self.w_o = nn.Linear(config.d_model, config.d_model)
        self.num_heads = config.num_heads
        self.d_k = config.d_k

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        batch, q_len, _ = query.shape
        kv_len = key.size(1)

        # Uniform (pre-softmax-zero) scores -> softmax makes every allowed
        # position equally weighted, still masked where required.
        scores = torch.zeros(batch, self.num_heads, q_len, kv_len, device=query.device)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)  # (batch, num_heads, q_len, kv_len)

        v = self.w_v(value)  # (batch, kv_len, d_model) -> (batch, kv_len, d_model)
        v = v.view(batch, kv_len, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        # (batch, kv_len, d_model) -> (batch, num_heads, kv_len, d_k)

        out = torch.matmul(attn_weights, v)  # (batch, num_heads, q_len, d_k)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch, q_len, self.num_heads * self.d_k)
        # (batch, num_heads, q_len, d_k) -> (batch, q_len, d_model)

        return self.w_o(out), attn_weights


def make_config(**overrides) -> TransformerConfig:
    defaults = dict(
        vocab_size=40, max_seq_len=16, d_model=32, num_heads=4, d_ff=64,
        num_encoder_layers=2, num_decoder_layers=2, dropout=0.0,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def test_dummy_attention_satisfies_base_interface():
    config = make_config()
    attn = DummyUniformAttention(config)
    x = torch.randn(2, 5, config.d_model)

    out, weights = attn(x, x, x)

    assert isinstance(attn, BaseAttention)
    assert out.shape == (2, 5, config.d_model)
    assert weights.shape == (2, config.num_heads, 5, 5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, config.num_heads, 5), atol=1e-5)


def test_dummy_attention_plugs_into_encoder_and_decoder_directly():
    """Encoder/Decoder accept any BaseAttention subclass via constructor arg."""
    config = make_config()

    encoder_layer = EncoderLayer(config, DummyUniformAttention)
    x = torch.randn(2, 6, config.d_model)
    assert encoder_layer(x).shape == x.shape

    encoder = Encoder(config, DummyUniformAttention)
    assert encoder(x).shape == x.shape

    decoder_layer = DecoderLayer(config, DummyUniformAttention, DummyUniformAttention)
    memory = torch.randn(2, 8, config.d_model)
    assert decoder_layer(x, memory).shape == x.shape

    decoder = Decoder(config, DummyUniformAttention, DummyUniformAttention)
    assert decoder(x, memory).shape == x.shape


def test_dummy_attention_plugs_in_via_config_registry():
    """The registry + config path: register once, select by string, build a full model."""
    register_attention("dummy_uniform", DummyUniformAttention)
    assert resolve_attention("dummy_uniform") is DummyUniformAttention

    config = make_config(attention_type="dummy_uniform", cross_attention_type="dummy_uniform")
    model = build_encoder_decoder(config)

    src_ids = torch.randint(1, config.vocab_size, (2, 7))
    tgt_ids = torch.randint(1, config.vocab_size, (2, 5))

    logits = model(src_ids, tgt_ids)

    assert logits.shape == (2, 5, config.vocab_size)
    assert torch.isfinite(logits).all()

    # End-to-end backward pass also works with the swapped-in attention.
    # DummyUniformAttention intentionally ignores its `query` input (that is
    # what "uniform attention" means), so in pre-norm mode the norm feeding
    # the cross-attention query legitimately receives no gradient. Check a
    # representative set of parameters that must always receive gradient
    # regardless of which attention variant is plugged in, rather than
    # requiring literally every parameter to be touched.
    logits.sum().backward()
    always_trained = [
        model.src_embedding.token_embedding.embedding.weight,
        model.tgt_embedding.token_embedding.embedding.weight,
        model.output_projection.weight,
        model.encoder.layers[0].feed_forward.linear_1.weight,
        model.decoder.layers[0].feed_forward.linear_1.weight,
        model.decoder.layers[0].cross_attn.w_v.weight,
        model.decoder.layers[0].cross_attn.w_o.weight,
    ]
    assert all(p.grad is not None for p in always_trained)


def test_scaled_dot_product_attention_helper_reusable_by_variants():
    """Alternate attention modules can reuse the core SDPA math directly."""
    q = torch.randn(2, 4, 5, 8)
    k = torch.randn(2, 4, 5, 8)
    v = torch.randn(2, 4, 5, 8)

    out, weights = scaled_dot_product_attention(q, k, v)

    assert out.shape == (2, 4, 5, 8)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4, 5), atol=1e-5)
