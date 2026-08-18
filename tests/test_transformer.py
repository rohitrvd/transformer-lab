"""End-to-end forward-pass tests for all three top-level model variants."""

from __future__ import annotations

import torch

from config import TransformerConfig
from transformer import build_decoder_only, build_encoder_decoder, build_encoder_only


def make_config(**overrides) -> TransformerConfig:
    defaults = dict(
        vocab_size=40, max_seq_len=16, d_model=32, num_heads=4, d_ff=64,
        num_encoder_layers=2, num_decoder_layers=2, dropout=0.0,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def test_encoder_only_forward_shape():
    config = make_config()
    model = build_encoder_only(config)
    input_ids = torch.randint(1, config.vocab_size, (2, 7))

    logits = model(input_ids)

    assert logits.shape == (2, 7, config.vocab_size)


def test_decoder_only_forward_shape():
    config = make_config()
    model = build_decoder_only(config)
    input_ids = torch.randint(1, config.vocab_size, (2, 7))

    logits = model(input_ids)

    assert logits.shape == (2, 7, config.vocab_size)


def test_decoder_only_is_causal():
    """Perturbing the last input token must not change earlier-position logits."""
    config = make_config(dropout=0.0)
    model = build_decoder_only(config)
    model.eval()

    input_ids = torch.randint(1, config.vocab_size, (1, 6))
    input_ids_modified = input_ids.clone()
    input_ids_modified[0, -1] = (input_ids[0, -1] + 1) % config.vocab_size

    with torch.no_grad():
        logits_original = model(input_ids)
        logits_modified = model(input_ids_modified)

    assert torch.allclose(logits_original[:, :-1, :], logits_modified[:, :-1, :], atol=1e-4)


def test_encoder_decoder_forward_shape():
    config = make_config()
    model = build_encoder_decoder(config)
    src_ids = torch.randint(1, config.vocab_size, (2, 9))
    tgt_ids = torch.randint(1, config.vocab_size, (2, 5))

    logits = model(src_ids, tgt_ids)

    assert logits.shape == (2, 5, config.vocab_size)


def test_encoder_decoder_backward_pass_runs():
    """A full forward + backward pass should run without shape/gradient errors."""
    config = make_config()
    model = build_encoder_decoder(config)
    src_ids = torch.randint(1, config.vocab_size, (2, 6))
    tgt_ids = torch.randint(1, config.vocab_size, (2, 4))

    logits = model(src_ids, tgt_ids)
    loss = logits.sum()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)


def test_pre_and_post_norm_configs_both_produce_valid_models():
    for placement in ("pre", "post"):
        config = make_config(norm_placement=placement)
        model = build_encoder_decoder(config)
        src_ids = torch.randint(1, config.vocab_size, (2, 5))
        tgt_ids = torch.randint(1, config.vocab_size, (2, 4))

        logits = model(src_ids, tgt_ids)

        assert logits.shape == (2, 4, config.vocab_size)
        assert torch.isfinite(logits).all()


def test_gelu_activation_config_produces_valid_model():
    config = make_config(activation="gelu")
    model = build_encoder_decoder(config)
    src_ids = torch.randint(1, config.vocab_size, (2, 5))
    tgt_ids = torch.randint(1, config.vocab_size, (2, 4))

    logits = model(src_ids, tgt_ids)

    assert torch.isfinite(logits).all()
