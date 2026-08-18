"""Tests for the continuous-batching scheduler: admission, mid-batch join/leave, eviction."""

from __future__ import annotations

from collections import deque
from typing import Deque, List

import torch

from config import TransformerConfig
from inference.kv_cache import NaiveKVCache
from inference.scheduler import ContinuousBatchingScheduler, GenerationRequest, SequenceState, fcfs_admission_policy
from transformer import build_decoder_only


def make_model():
    torch.manual_seed(0)
    config = TransformerConfig(
        vocab_size=30, max_seq_len=20, d_model=16, num_heads=2, d_ff=32, num_decoder_layers=2, dropout=0.0
    )
    model = build_decoder_only(config)
    model.eval()
    return model


def prompt(vocab_size=30, length=4):
    return torch.randint(1, vocab_size, (1, length))


def test_scheduler_respects_max_batch_size():
    model = make_model()
    scheduler = ContinuousBatchingScheduler(model, kv_cache_class=NaiveKVCache, max_batch_size=2)
    scheduler.add_request(prompt(), max_new_tokens=10)
    scheduler.add_request(prompt(), max_new_tokens=10)
    scheduler.add_request(prompt(), max_new_tokens=10)  # should stay waiting

    report = scheduler.step()
    assert len(scheduler.active) == 2
    assert len(scheduler.waiting) == 1
    assert report.admitted_ids == [0, 1]
    assert report.waiting_ids == [2]


def test_finished_sequence_frees_a_slot_for_a_waiting_request():
    model = make_model()
    scheduler = ContinuousBatchingScheduler(model, kv_cache_class=NaiveKVCache, max_batch_size=1)
    short_id = scheduler.add_request(prompt(), max_new_tokens=1)  # finishes after its prefill step
    long_id = scheduler.add_request(prompt(), max_new_tokens=10)

    step1 = scheduler.step()
    assert step1.admitted_ids == [short_id]
    assert step1.finished_ids == [short_id]  # max_new_tokens=1 reached immediately after prefill
    assert step1.waiting_ids == [long_id]

    step2 = scheduler.step()
    assert step2.admitted_ids == [long_id]  # slot freed by short_id's completion
    active_ids = [seq.request_id for seq in scheduler.active]
    assert long_id in active_ids or step2.finished_ids == [long_id]


def test_sequences_join_mid_run_when_capacity_frees_up():
    model = make_model()
    scheduler = ContinuousBatchingScheduler(model, kv_cache_class=NaiveKVCache, max_batch_size=2)
    a = scheduler.add_request(prompt(), max_new_tokens=2)
    b = scheduler.add_request(prompt(), max_new_tokens=5)
    scheduler.step()  # admits a, b
    scheduler.step()  # decodes both; a now has tokens_generated=2 -> finishes this step

    report_after_a_finishes = scheduler.step()
    # a should have finished by now (max_new_tokens=2), freeing a slot
    assert a in [seq.request_id for seq in scheduler.finished]

    c = scheduler.add_request(prompt(), max_new_tokens=3)
    report = scheduler.step()
    assert c in report.admitted_ids  # the newly freed slot admits the new request mid-run


def test_all_requests_eventually_finish_and_leave_active():
    model = make_model()
    scheduler = ContinuousBatchingScheduler(model, kv_cache_class=NaiveKVCache, max_batch_size=2)
    ids = [scheduler.add_request(prompt(), max_new_tokens=n) for n in (2, 3, 1, 4)]

    finished = scheduler.run_until_empty()

    assert sorted(seq.request_id for seq in finished) == sorted(ids)
    assert scheduler.active == []
    assert len(scheduler.waiting) == 0
    prompt_len = 4  # length used by the prompt() helper above
    for seq in finished:
        assert seq.generated_tokens.shape[1] == prompt_len + seq.tokens_generated


def test_finished_sequence_cache_is_evicted():
    model = make_model()
    scheduler = ContinuousBatchingScheduler(model, kv_cache_class=NaiveKVCache, max_batch_size=1)
    scheduler.add_request(prompt(), max_new_tokens=1)
    scheduler.step()

    finished_seq = scheduler.finished[0]
    assert finished_seq.stream.kv_cache.seq_len == 0  # evict() cleared it


def test_custom_admission_policy_is_used():
    """A different admission_policy callable should override the default FCFS behavior."""

    def admit_none(waiting: Deque[GenerationRequest], active: List[SequenceState], max_batch_size: int):
        return []  # never admit anything

    model = make_model()
    scheduler = ContinuousBatchingScheduler(
        model, kv_cache_class=NaiveKVCache, max_batch_size=4, admission_policy=admit_none
    )
    scheduler.add_request(prompt(), max_new_tokens=3)

    report = scheduler.step()
    assert report.admitted_ids == []
    assert len(scheduler.waiting) == 1
    assert len(scheduler.active) == 0


def test_fcfs_admission_policy_orders_by_arrival():
    waiting = deque(
        [
            GenerationRequest(request_id=i, prompt_tokens=prompt(), max_new_tokens=1)
            for i in range(5)
        ]
    )
    admitted = fcfs_admission_policy(waiting, active=[], max_batch_size=3)
    assert [r.request_id for r in admitted] == [0, 1, 2]
    assert [r.request_id for r in waiting] == [3, 4]
