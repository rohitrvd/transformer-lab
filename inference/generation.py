"""Autoregressive generation: prefill + decode, timed per phase.

`GenerationStream` is the reusable core: one in-progress generation
sequence that owns its own `KVCache` (and, for encoder-decoder models, its
encoder memory) and knows how to advance by exactly one token. `generate()`
is a thin single-stream driver built on top of it; `scheduler.py` reuses the
same class directly to advance many concurrent streams, one token each, per
scheduling step — so the "one token at a time" mechanics are defined once.

Both are duck-typed over the two autoregressive model variants from
`transformer.py`:
  - `DecoderOnlyModel` (GPT-style): `prompt_tokens` can be arbitrary length;
    prefill processes the whole prompt in one forward pass, genuinely
    demonstrating "prefill is one shot, decode is one token at a time".
  - `EncoderDecoderTransformer` (what `train_example.py` trains): the
    encoder side (`prompt_tokens` as `src`) is processed once, non-
    autoregressively, with no cache needed; the decoder side always starts
    from a single `decoder_start_token_id` (there's no multi-token decoder
    prompt in this repo's toy task), so "prefill" here is the encoder pass
    plus a length-1 decoder seed rather than a multi-token prefill. This is
    a property of the task, not a bug — see the README's inference section.

Both branches populate the same `KVCache` interface, so `kv_cache_class`
(e.g. `NaiveKVCache` vs `PagedKVCache`) is swappable without touching this
file — that's what `benchmark.py` exploits to compare strategies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Type, Union

import torch
from torch import Tensor

from inference.kv_cache import KVCache, NaiveKVCache
from transformer import DecoderOnlyModel, EncoderDecoderTransformer

GenerativeModel = Union[DecoderOnlyModel, EncoderDecoderTransformer]


class GenerationStream:
    """One in-progress autoregressive generation sequence.

    Owns a single `KVCache` instance and (for encoder-decoder models) the
    encoder `memory` computed once at prefill time. Exposes `prefill()` and
    `decode_step()`, each advancing the sequence and returning the wall-
    clock time taken — `generate()` calls these in a loop for one sequence;
    `scheduler.py` calls `decode_step()` across many `GenerationStream`
    instances per scheduling step to simulate concurrent in-flight requests.
    """

    def __init__(
        self,
        model: GenerativeModel,
        prompt_tokens: Tensor,
        kv_cache: KVCache,
        decoder_start_token_id: Optional[int] = None,
    ) -> None:
        """Set up the stream and its cache; does not run any forward pass yet.

        prompt_tokens: (1, prompt_len)
        kv_cache: a fresh (or pooled-and-evicted) cache for this sequence alone.
        decoder_start_token_id: required if `model` is an EncoderDecoderTransformer.
        """
        if prompt_tokens.size(0) != 1:
            raise ValueError("GenerationStream supports a single sequence (batch size 1)")

        self.model = model
        self.kv_cache = kv_cache
        self.is_seq2seq = isinstance(model, EncoderDecoderTransformer)

        if self.is_seq2seq:
            if decoder_start_token_id is None:
                raise ValueError("decoder_start_token_id is required for EncoderDecoderTransformer generation")
            self.memory: Optional[Tensor] = model.encode(prompt_tokens)  # (1, src_len, d_model)
            self.src_tokens: Optional[Tensor] = prompt_tokens
            self._prefill_input = torch.tensor(
                [[decoder_start_token_id]], device=prompt_tokens.device, dtype=torch.long
            )  # (1, 1)
        elif isinstance(model, DecoderOnlyModel):
            self.memory = None
            self.src_tokens = None
            self._prefill_input = prompt_tokens  # (1, prompt_len)
        else:
            raise TypeError(
                f"GenerationStream supports DecoderOnlyModel or EncoderDecoderTransformer, "
                f"got {type(model).__name__}"
            )

        self.generated: Optional[Tensor] = None  # populated by prefill()
        self.next_token: Optional[Tensor] = None  # (1, 1), the token to feed into the next step

    def _compute_logits(self, token: Tensor) -> Tensor:
        """Run one forward step (prefill or decode) through the cache-aware model path.

        token: (1, new_len) -> (1, new_len, vocab_size)
        """
        if self.is_seq2seq:
            decoded = self.model.decode(token, self.memory, self.src_tokens, kv_cache=self.kv_cache)
            return self.model.output_projection(decoded)  # (1, new_len, vocab_size)
        return self.model(token, kv_cache=self.kv_cache)  # (1, new_len, vocab_size)

    def prefill(self) -> float:
        """Run the one-shot prefill step, populating the cache. Returns elapsed seconds."""
        t0 = time.perf_counter()
        logits = self._compute_logits(self._prefill_input)  # (1, prefill_len, vocab_size)
        next_token = logits[:, -1:, :].argmax(dim=-1)  # (1, 1)
        self.generated = torch.cat([self._prefill_input, next_token], dim=1)  # (1, prefill_len + 1)
        self.next_token = next_token
        return time.perf_counter() - t0

    def decode_step(self) -> float:
        """Advance the sequence by exactly one token. Returns elapsed seconds."""
        t0 = time.perf_counter()
        logits = self._compute_logits(self.next_token)  # (1, 1, vocab_size)
        next_token = logits[:, -1:, :].argmax(dim=-1)  # (1, 1)
        self.generated = torch.cat([self.generated, next_token], dim=1)  # (1, cur_len + 1)
        self.next_token = next_token
        return time.perf_counter() - t0

    @property
    def last_token_id(self) -> int:
        """The most recently generated token, as a plain int (for EOS checks)."""
        return self.next_token.item()


@dataclass
class GenerationResult:
    """Output tokens plus per-phase timing, for comparing cache strategies.

    generated_tokens: (1, prompt_len_or_1 + num_generated) — for
        `DecoderOnlyModel` this includes the original prompt; for
        `EncoderDecoderTransformer` it's the decoder-side sequence only
        (starting from `decoder_start_token_id`), since the encoder's
        `prompt_tokens` are an input, not part of the generated output.
    prefill_time_s: wall-clock time for the one-shot prefill phase.
    decode_times_s: wall-clock time for each individual decode step, in
        generation order — inspect this list directly to see per-token cost
        (e.g. `NaiveKVCache`'s cost growing with sequence length vs.
        `PagedKVCache`'s roughly-constant per-step cost).
    """

    generated_tokens: Tensor
    prefill_time_s: float
    decode_times_s: List[float] = field(default_factory=list)

    @property
    def total_time_s(self) -> float:
        """Total wall-clock time across prefill and every decode step."""
        return self.prefill_time_s + sum(self.decode_times_s)

    @property
    def mean_decode_time_s(self) -> float:
        """Average per-token decode latency (0.0 if only prefill ran)."""
        return sum(self.decode_times_s) / len(self.decode_times_s) if self.decode_times_s else 0.0


@torch.no_grad()
def generate(
    model: GenerativeModel,
    prompt_tokens: Tensor,
    max_new_tokens: int,
    kv_cache_class: Type[KVCache] = NaiveKVCache,
    eos_token_id: Optional[int] = None,
    decoder_start_token_id: Optional[int] = None,
    kv_cache_kwargs: Optional[dict] = None,
) -> GenerationResult:
    """Generate up to `max_new_tokens` tokens via prefill + a cached decode loop.

    prompt_tokens: (1, prompt_len) — batch size must be 1; `scheduler.py`
        provides concurrency across multiple generation streams by giving
        each its own `GenerationStream`/cache rather than batching tensors
        here.
    kv_cache_class: which `KVCache` implementation to populate and reuse
        across every decode step (see `inference/kv_cache.py`).
    eos_token_id: if given, generation stops early the first time this
        token is produced.
    decoder_start_token_id: required when `model` is an
        `EncoderDecoderTransformer` (the decoder's first input token, e.g.
        BOS); ignored for `DecoderOnlyModel`.
    kv_cache_kwargs: extra constructor kwargs for `kv_cache_class` (e.g.
        `block_size`/`num_blocks` for `PagedKVCache`).
    """
    was_training = model.training
    model.eval()
    try:
        cache = kv_cache_class(
            num_layers=model.config.num_decoder_layers,
            num_heads=model.config.num_heads,
            d_k=model.config.d_k,
            device=prompt_tokens.device,
            **(kv_cache_kwargs or {}),
        )
        stream = GenerationStream(model, prompt_tokens, cache, decoder_start_token_id=decoder_start_token_id)

        prefill_time_s = stream.prefill()

        decode_times_s: List[float] = []
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and stream.last_token_id == eos_token_id:
                break
            decode_times_s.append(stream.decode_step())

        return GenerationResult(
            generated_tokens=stream.generated, prefill_time_s=prefill_time_s, decode_times_s=decode_times_s
        )
    finally:
        model.train(was_training)
