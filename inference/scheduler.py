"""Toy continuous-batching scheduler: sequences join and leave mid-batch.

Simulates the scheduling side of continuous batching: at each step, waiting
requests may be admitted into the active batch (up to a capacity limit, per
a pluggable `admission_policy`), every active sequence advances by exactly
one token, and any sequence that finishes (EOS or `max_new_tokens` reached)
leaves the batch and frees its cache.

Simplification, stated up front: each `SequenceState` owns an independent
`GenerationStream` / `KVCache`, and `step()` advances active sequences with
a plain Python loop of independent forward calls rather than fusing their
single-token decode steps into one batched tensor operation. Real
continuous-batching engines (e.g. vLLM) get their throughput win precisely
from that fusion — sharing one physical KV-cache pool across sequences and
running one batched/paged attention kernel per step. This scheduler makes
the *scheduling decisions* (who's admitted, who's evicted, what happens
mid-batch) visible and swappable, at the cost of not demonstrating that
throughput win. See the README's inference section for the fuller
comparison.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, List, Optional, Type

from torch import Tensor

from inference.generation import GenerationStream, GenerativeModel
from inference.kv_cache import KVCache, NaiveKVCache


@dataclass
class GenerationRequest:
    """A pending request waiting to be admitted into the active batch."""

    request_id: int
    prompt_tokens: Tensor  # (1, prompt_len)
    max_new_tokens: int
    eos_token_id: Optional[int] = None
    decoder_start_token_id: Optional[int] = None  # required if the scheduler's model is an EncoderDecoderTransformer


@dataclass
class SequenceState:
    """One admitted, in-flight (or just-finished) generation sequence."""

    request_id: int
    stream: GenerationStream
    max_new_tokens: int
    eos_token_id: Optional[int]
    tokens_generated: int = 0
    admitted_at_step: int = 0
    finished_at_step: Optional[int] = None

    @property
    def finished(self) -> bool:
        return self.finished_at_step is not None

    @property
    def generated_tokens(self) -> Tensor:
        """The sequence generated so far: (1, tokens_generated + 1)."""
        return self.stream.generated


@dataclass
class StepReport:
    """What happened during one `scheduler.step()` call, for inspection/logging."""

    step: int
    admitted_ids: List[int] = field(default_factory=list)
    decoded_ids: List[int] = field(default_factory=list)
    finished_ids: List[int] = field(default_factory=list)
    active_ids: List[int] = field(default_factory=list)
    waiting_ids: List[int] = field(default_factory=list)


AdmissionPolicy = Callable[[Deque[GenerationRequest], List[SequenceState], int], List[GenerationRequest]]


def fcfs_admission_policy(
    waiting: Deque[GenerationRequest], active: List[SequenceState], max_batch_size: int
) -> List[GenerationRequest]:
    """Default admission policy: first-come-first-served, fill up to `max_batch_size`.

    Admits waiting requests, oldest first, until the active batch is full.
    This is the pluggable extension point for scheduling experiments: pass
    a different callable with the same signature (e.g. shortest-prompt-
    first, priority-based, or one that reserves headroom instead of always
    filling to capacity) as `admission_policy` to `ContinuousBatchingScheduler`
    without changing the scheduler itself.
    """
    free_slots = max_batch_size - len(active)
    admitted = []
    for _ in range(max(0, free_slots)):
        if not waiting:
            break
        admitted.append(waiting.popleft())
    return admitted


class ContinuousBatchingScheduler:
    """Toy simulator for continuous batching over multiple in-flight sequences.

    Tracks three sets of requests: `waiting` (not yet admitted), `active`
    (currently generating), `finished` (done). `step()` admits new requests
    per `admission_policy`, advances every active sequence by one token, and
    retires finished ones. `run_until_empty()` drives the simulation to
    completion.

    A newly admitted request is prefilled during the same step it joins
    (its first generated token counts as that step's advancement for it),
    matching real "iteration-level scheduling" where prefill and decode
    work are interleaved rather than run in strictly separate phases.
    """

    def __init__(
        self,
        model: GenerativeModel,
        kv_cache_class: Type[KVCache] = NaiveKVCache,
        max_batch_size: int = 4,
        admission_policy: AdmissionPolicy = fcfs_admission_policy,
        kv_cache_kwargs: Optional[dict] = None,
    ) -> None:
        self.model = model
        self.kv_cache_class = kv_cache_class
        self.kv_cache_kwargs = kv_cache_kwargs or {}
        self.max_batch_size = max_batch_size
        self.admission_policy = admission_policy

        self.waiting: Deque[GenerationRequest] = deque()
        self.active: List[SequenceState] = []
        self.finished: List[SequenceState] = []

        self._next_request_id = itertools.count()
        self._current_step = 0

    def add_request(
        self,
        prompt_tokens: Tensor,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
    ) -> int:
        """Enqueue a new generation request; returns its `request_id`."""
        request_id = next(self._next_request_id)
        self.waiting.append(
            GenerationRequest(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                decoder_start_token_id=decoder_start_token_id,
            )
        )
        return request_id

    def _new_kv_cache(self, device) -> KVCache:
        """Build a fresh cache for a newly admitted sequence.

        A real system would draw a `KVCache` back out of a pool here (using
        `KVCache.evict()`'s release, on the finished side, to make objects
        available for reuse) instead of allocating fresh tensors every time
        a sequence is admitted. This toy scheduler still calls `evict()`
        when a sequence finishes (see `step()`) so the eviction moment is
        visible, but does not implement the pooling/reuse itself.
        """
        return self.kv_cache_class(
            num_layers=self.model.config.num_decoder_layers,
            num_heads=self.model.config.num_heads,
            d_k=self.model.config.d_k,
            device=device,
            **self.kv_cache_kwargs,
        )

    def step(self) -> StepReport:
        """Advance the simulation by one scheduling step.

        1. Admit waiting requests per `admission_policy`, up to capacity;
           each admitted request is prefilled immediately.
        2. Decode one token for every other active sequence.
        3. Move any sequence that hit EOS or `max_new_tokens` into
           `finished`, evicting its cache.
        """
        self._current_step += 1
        report = StepReport(step=self._current_step)

        admitted_requests = self.admission_policy(self.waiting, self.active, self.max_batch_size)
        newly_admitted_ids = set()
        for request in admitted_requests:
            cache = self._new_kv_cache(device=request.prompt_tokens.device)
            stream = GenerationStream(
                self.model, request.prompt_tokens, cache, decoder_start_token_id=request.decoder_start_token_id
            )
            stream.prefill()  # this step's advancement for a newly admitted sequence
            seq = SequenceState(
                request_id=request.request_id,
                stream=stream,
                max_new_tokens=request.max_new_tokens,
                eos_token_id=request.eos_token_id,
                tokens_generated=1,
                admitted_at_step=self._current_step,
            )
            self.active.append(seq)
            newly_admitted_ids.add(seq.request_id)
            report.admitted_ids.append(request.request_id)

        for seq in self.active:
            if seq.request_id in newly_admitted_ids:
                continue  # already advanced by prefill() above, this same step
            seq.stream.decode_step()
            seq.tokens_generated += 1
            report.decoded_ids.append(seq.request_id)

        still_active: List[SequenceState] = []
        for seq in self.active:
            hit_eos = seq.eos_token_id is not None and seq.stream.last_token_id == seq.eos_token_id
            hit_max_len = seq.tokens_generated >= seq.max_new_tokens
            if hit_eos or hit_max_len:
                seq.stream.kv_cache.evict()
                seq.finished_at_step = self._current_step
                self.finished.append(seq)
                report.finished_ids.append(seq.request_id)
            else:
                still_active.append(seq)
        self.active = still_active

        report.active_ids = [seq.request_id for seq in self.active]
        report.waiting_ids = [request.request_id for request in self.waiting]
        return report

    def run_until_empty(self, max_steps: int = 10_000) -> List[SequenceState]:
        """Call `step()` until both `waiting` and `active` are empty. Returns `finished`."""
        steps = 0
        while self.waiting or self.active:
            self.step()
            steps += 1
            if steps >= max_steps:
                raise RuntimeError(f"run_until_empty exceeded max_steps={max_steps}; a sequence may be stuck")
        return self.finished
