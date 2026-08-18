"""Tests for NaiveKVCache and PagedKVCache: correctness and block-table mapping."""

from __future__ import annotations

import torch

from inference.kv_cache import KVCache, NaiveKVCache, PagedKVCache


def test_naive_cache_accumulates_across_updates():
    cache = NaiveKVCache(num_layers=2, num_heads=2, d_k=4)
    k1 = torch.randn(1, 2, 3, 4)
    v1 = torch.randn(1, 2, 3, 4)
    k2 = torch.randn(1, 2, 1, 4)
    v2 = torch.randn(1, 2, 1, 4)

    out_k1, out_v1 = cache.update(0, k1, v1)
    assert torch.equal(out_k1, k1)
    assert cache.seq_len == 3

    out_k2, out_v2 = cache.update(0, k2, v2)
    assert out_k2.shape == (1, 2, 4, 4)
    assert torch.equal(out_k2[:, :, :3, :], k1)
    assert torch.equal(out_k2[:, :, 3:, :], k2)
    assert cache.seq_len == 4


def test_naive_cache_get_without_modifying():
    cache = NaiveKVCache(num_layers=1, num_heads=1, d_k=2)
    assert cache.get(0) == (None, None)
    k = torch.randn(1, 1, 2, 2)
    v = torch.randn(1, 1, 2, 2)
    cache.update(0, k, v)
    got_k, got_v = cache.get(0)
    assert torch.equal(got_k, k)
    assert cache.seq_len == 2  # get() must not have appended anything


def test_naive_cache_evict_clears_layer_or_all():
    cache = NaiveKVCache(num_layers=2, num_heads=1, d_k=2)
    cache.update(0, torch.randn(1, 1, 2, 2), torch.randn(1, 1, 2, 2))
    cache.update(1, torch.randn(1, 1, 2, 2), torch.randn(1, 1, 2, 2))

    cache.evict(layer_idx=0)
    assert cache.get(0) == (None, None)
    assert cache.get(1) != (None, None)

    cache.evict()
    assert cache.get(1) == (None, None)
    assert cache.seq_len == 0


def test_paged_cache_maps_logical_blocks_to_physical_blocks():
    """Directly verify the block_table mechanism with a small, inspectable block_size."""
    block_size = 2
    cache = PagedKVCache(num_layers=1, num_heads=1, d_k=2, block_size=block_size, num_blocks=4)

    # Write 5 tokens one at a time; with block_size=2 this needs ceil(5/2)=3 physical blocks.
    tokens_k = [torch.full((1, 1, 1, 2), float(i)) for i in range(5)]
    tokens_v = [torch.full((1, 1, 1, 2), float(i) * 10) for i in range(5)]
    for k, v in zip(tokens_k, tokens_v):
        cache.update(0, k, v)

    assert len(cache.block_table) == 3
    assert cache.seq_len == 5
    # All physical block indices must be distinct (no aliasing) and drawn from the pool.
    assert len(set(cache.block_table)) == 3
    assert all(0 <= b < 4 for b in cache.block_table)

    # Reading back must reconstruct the tokens in original logical order, regardless of
    # which physical blocks they landed in.
    gathered_k, gathered_v = cache.get(0)
    expected_k = torch.cat(tokens_k, dim=2)
    expected_v = torch.cat(tokens_v, dim=2)
    assert torch.equal(gathered_k, expected_k)
    assert torch.equal(gathered_v, expected_v)

    # Directly inspect physical storage: logical position 2 (third token) lives in logical
    # block 1 (since block_size=2), offset 0, at whatever physical block block_table[1] is.
    physical_block = cache.block_table[1]
    stored = cache._k_blocks[0][physical_block, :, 0, :]
    assert torch.equal(stored, tokens_k[2][0, :, 0, :])


def test_paged_cache_evict_returns_blocks_to_free_pool():
    cache = PagedKVCache(num_layers=1, num_heads=1, d_k=2, block_size=2, num_blocks=4)
    cache.update(0, torch.randn(1, 1, 3, 2), torch.randn(1, 1, 3, 2))
    assert len(cache.block_table) == 2
    assert len(cache._free_blocks) == 2

    cache.evict()
    assert cache.block_table == []
    assert len(cache._free_blocks) == 4
    assert cache.seq_len == 0


def test_paged_cache_raises_when_pool_exhausted():
    cache = PagedKVCache(num_layers=1, num_heads=1, d_k=2, block_size=2, num_blocks=1)
    cache.update(0, torch.randn(1, 1, 2, 2), torch.randn(1, 1, 2, 2))  # fills the only block
    try:
        cache.update(0, torch.randn(1, 1, 1, 2), torch.randn(1, 1, 1, 2))
        assert False, "expected RuntimeError when the block pool is exhausted"
    except RuntimeError:
        pass


def test_naive_and_paged_caches_agree_on_gathered_kv():
    """Both cache strategies must reconstruct identical K/V for the same input sequence."""
    torch.manual_seed(0)
    naive = NaiveKVCache(num_layers=1, num_heads=2, d_k=4)
    paged = PagedKVCache(num_layers=1, num_heads=2, d_k=4, block_size=3, num_blocks=8)

    chunks = [torch.randn(1, 2, n, 4) for n in (2, 1, 3, 1)]
    v_chunks = [torch.randn(1, 2, n, 4) for n in (2, 1, 3, 1)]

    for k, v in zip(chunks, v_chunks):
        naive_k, naive_v = naive.update(0, k, v)
        paged_k, paged_v = paged.update(0, k, v)
        assert torch.equal(naive_k, paged_k)
        assert torch.equal(naive_v, paged_v)


def test_kv_cache_is_abstract():
    try:
        KVCache()
        assert False, "KVCache should not be directly instantiable"
    except TypeError:
        pass
