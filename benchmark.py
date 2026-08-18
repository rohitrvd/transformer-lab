"""Compare KV cache strategies on the sequence-reversal generation task.

Runs `inference.generation.generate()` across several prompts with
`NaiveKVCache` vs `PagedKVCache` and reports mean prefill time, mean
per-token decode time, and peak memory — so cache strategies can be
compared empirically rather than just conceptually. See the README's
inference section for what each number is a simplified stand-in for.

Usage:
    python benchmark.py
"""

from __future__ import annotations

import tracemalloc
from dataclasses import dataclass
from typing import Callable, List, Optional, Type

import torch
from torch import Tensor

from inference.generation import generate
from inference.kv_cache import KVCache, NaiveKVCache, PagedKVCache
from train_example import BOS_ID, EOS_ID, VOCAB_SIZE, make_default_config, train_model
from transformer import EncoderDecoderTransformer


def measure_peak_memory_bytes(fn: Callable[[], None]) -> int:
    """Run `fn()` and return the peak memory it allocated, in bytes.

    Uses CUDA's own peak-allocation counter when a GPU is available (exact,
    device-level); falls back to `tracemalloc` (Python-level allocation
    tracking) on CPU, which is coarser but requires no extra dependencies
    and is good enough to compare cache strategies against each other.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        fn()
        return torch.cuda.max_memory_allocated()
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


@dataclass
class CacheBenchmarkResult:
    """Aggregated timing/memory for one KV cache strategy across several prompts."""

    cache_name: str
    mean_prefill_time_s: float
    mean_decode_time_s: float
    peak_memory_bytes: int


def run_cache_benchmark(
    model: EncoderDecoderTransformer,
    prompts: List[Tensor],
    max_new_tokens: int,
    kv_cache_class: Type[KVCache],
    kv_cache_kwargs: Optional[dict] = None,
) -> CacheBenchmarkResult:
    """Run `generate()` over every prompt with one cache strategy, aggregate the results."""
    prefill_times: List[float] = []
    decode_times: List[float] = []

    def _run_all() -> None:
        for prompt in prompts:
            result = generate(
                model,
                prompt,
                max_new_tokens=max_new_tokens,
                kv_cache_class=kv_cache_class,
                kv_cache_kwargs=kv_cache_kwargs,
                eos_token_id=EOS_ID,
                decoder_start_token_id=BOS_ID,
            )
            prefill_times.append(result.prefill_time_s)
            decode_times.extend(result.decode_times_s)

    peak_bytes = measure_peak_memory_bytes(_run_all)

    return CacheBenchmarkResult(
        cache_name=kv_cache_class.__name__,
        mean_prefill_time_s=sum(prefill_times) / len(prefill_times),
        mean_decode_time_s=(sum(decode_times) / len(decode_times)) if decode_times else 0.0,
        peak_memory_bytes=peak_bytes,
    )


def print_comparison(results: List[CacheBenchmarkResult]) -> None:
    """Print an aligned table comparing cache strategies."""
    header = f"{'cache':<16}{'mean prefill (ms)':<20}{'mean decode/token (ms)':<24}{'peak memory (KB)':<18}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.cache_name:<16}"
            f"{r.mean_prefill_time_s * 1000:<20.3f}"
            f"{r.mean_decode_time_s * 1000:<24.3f}"
            f"{r.peak_memory_bytes / 1024:<18.1f}"
        )


def main() -> None:
    """Train the toy model, then compare NaiveKVCache vs PagedKVCache on generation."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    config = make_default_config()

    print("Training the toy sequence-reversal model...")
    model = train_model(config, num_steps=1500, device=device, log_every=None)
    model.eval()

    prompt_len = 8
    max_new_tokens = 20  # long enough to amplify per-step differences between cache strategies
    num_prompts = 15
    prompts = [torch.randint(3, VOCAB_SIZE, (1, prompt_len), device=device) for _ in range(num_prompts)]

    results = [
        run_cache_benchmark(model, prompts, max_new_tokens, NaiveKVCache),
        run_cache_benchmark(
            model, prompts, max_new_tokens, PagedKVCache, kv_cache_kwargs={"block_size": 8, "num_blocks": 32}
        ),
    ]

    print(f"\nGeneration benchmark: {num_prompts} prompts, prompt_len={prompt_len}, max_new_tokens={max_new_tokens}\n")
    print_comparison(results)


if __name__ == "__main__":
    main()
