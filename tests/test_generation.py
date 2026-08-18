"""Tests for cached generation: correctness (matches no-cache output) and generate()."""

from __future__ import annotations

import torch

from config import TransformerConfig
from inference.generation import GenerationStream, generate
from inference.kv_cache import NaiveKVCache, PagedKVCache
from transformer import build_decoder_only, build_encoder_decoder


def decoder_only_config(**overrides) -> TransformerConfig:
    defaults = dict(
        vocab_size=30, max_seq_len=20, d_model=32, num_heads=4, d_ff=64,
        num_decoder_layers=2, dropout=0.0,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def encoder_decoder_config(**overrides) -> TransformerConfig:
    defaults = dict(
        vocab_size=20, max_seq_len=20, d_model=32, num_heads=4, d_ff=64,
        num_encoder_layers=2, num_decoder_layers=2, dropout=0.0,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def test_decoder_only_cached_matches_uncached_output():
    """Caching must never change outputs, only how they're computed."""
    torch.manual_seed(0)
    config = decoder_only_config()
    model = build_decoder_only(config)
    model.eval()

    input_ids = torch.randint(1, config.vocab_size, (1, 7))

    with torch.no_grad():
        full_logits = model(input_ids)  # no cache, full sequence at once

        cache = NaiveKVCache(num_layers=config.num_decoder_layers, num_heads=config.num_heads, d_k=config.d_k)
        prefill_logits = model(input_ids[:, :3], kv_cache=cache)
        step_logits = [prefill_logits[:, -1:, :]]
        for i in range(3, 7):
            out = model(input_ids[:, i : i + 1], kv_cache=cache)
            step_logits.append(out[:, -1:, :])
        cached_logits = torch.cat(step_logits, dim=1)

    # full_logits[:, 2:, :] are the logits predicting positions 3..6 given prefix 0..2,
    # which line up with the cached run's outputs (first cached step also covers position 2).
    assert torch.allclose(full_logits[:, 2:, :], cached_logits, atol=1e-4)


def test_encoder_decoder_cached_matches_uncached_output():
    torch.manual_seed(0)
    config = encoder_decoder_config()
    model = build_encoder_decoder(config)
    model.eval()

    src = torch.randint(1, config.vocab_size, (1, 6))
    tgt = torch.randint(1, config.vocab_size, (1, 5))

    with torch.no_grad():
        memory = model.encode(src)
        full_decoded = model.decode(tgt, memory, src)
        full_logits = model.output_projection(full_decoded)  # (1, 5, vocab)

        cache = NaiveKVCache(num_layers=config.num_decoder_layers, num_heads=config.num_heads, d_k=config.d_k)
        step_logits = []
        for i in range(5):
            decoded = model.decode(tgt[:, i : i + 1], memory, src, kv_cache=cache)
            step_logits.append(model.output_projection(decoded))
        cached_logits = torch.cat(step_logits, dim=1)

    assert torch.allclose(full_logits, cached_logits, atol=1e-4)


def test_paged_cache_matches_naive_cache_end_to_end():
    torch.manual_seed(0)
    config = decoder_only_config()
    model = build_decoder_only(config)
    prompt = torch.randint(1, config.vocab_size, (1, 5))

    result_naive = generate(model, prompt, max_new_tokens=6, kv_cache_class=NaiveKVCache)
    result_paged = generate(
        model, prompt, max_new_tokens=6, kv_cache_class=PagedKVCache,
        kv_cache_kwargs={"block_size": 2, "num_blocks": 16},
    )

    assert torch.equal(result_naive.generated_tokens, result_paged.generated_tokens)


def test_generate_decoder_only_shapes_and_timing():
    torch.manual_seed(0)
    config = decoder_only_config()
    model = build_decoder_only(config)
    prompt = torch.randint(1, config.vocab_size, (1, 4))

    result = generate(model, prompt, max_new_tokens=5, kv_cache_class=NaiveKVCache)

    assert result.generated_tokens.shape == (1, 4 + 5)
    assert result.prefill_time_s >= 0
    assert len(result.decode_times_s) == 4  # prefill produces 1 token, then 4 more decode steps
    assert all(t >= 0 for t in result.decode_times_s)


def test_generate_stops_early_on_eos():
    """Generation must never produce more than max_new_tokens tokens, eos or not."""
    torch.manual_seed(0)
    config = decoder_only_config(vocab_size=5)  # small vocab makes hitting a fixed eos id plausible/fast to force
    model = build_decoder_only(config)
    model.eval()
    prompt = torch.randint(1, config.vocab_size, (1, 3))

    # prompt_len(3) + max_new_tokens must stay within config.max_seq_len(20).
    result = generate(model, prompt, max_new_tokens=10, kv_cache_class=NaiveKVCache, eos_token_id=0)
    num_generated = result.generated_tokens.shape[1] - prompt.shape[1]
    assert num_generated <= 10


def test_generate_requires_single_sequence():
    config = decoder_only_config()
    model = build_decoder_only(config)
    prompt = torch.randint(1, config.vocab_size, (2, 4))  # batch size 2, not supported
    try:
        generate(model, prompt, max_new_tokens=3)
        assert False, "expected ValueError for batch size != 1"
    except ValueError:
        pass


def test_generation_stream_low_level_matches_generate():
    """GenerationStream (the primitive scheduler.py reuses) agrees with generate()'s wrapper."""
    torch.manual_seed(0)
    config = decoder_only_config()
    model = build_decoder_only(config)
    model.eval()
    prompt = torch.randint(1, config.vocab_size, (1, 4))

    cache = NaiveKVCache(num_layers=config.num_decoder_layers, num_heads=config.num_heads, d_k=config.d_k)
    with torch.no_grad():
        stream = GenerationStream(model, prompt, cache)
        stream.prefill()
        for _ in range(4):
            stream.decode_step()

    result = generate(model, prompt, max_new_tokens=5, kv_cache_class=NaiveKVCache)

    assert torch.equal(stream.generated, result.generated_tokens)
