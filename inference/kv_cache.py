"""Pluggable KV-cache strategies for autoregressive decoding.

Mirrors the `BaseAttention` pluggability pattern in `attention.py`: a single
abstract interface (`KVCache`), multiple interchangeable implementations, and
a uniform constructor so `generation.py`/`scheduler.py` can swap strategies
without knowing which concrete class they're holding.

Every implementation here is scoped to a single generation stream (batch
size 1). `scheduler.py` achieves multi-sequence concurrency by giving each
in-flight sequence its own `KVCache` instance rather than sharing one
batched/paged tensor across sequences — a real simplification versus systems
like vLLM, documented in the README's inference section.

To add a third strategy (e.g. a prefix-sharing radix-tree cache): subclass
`KVCache`, implement `update`/`get`/`evict`/`seq_len` with the same uniform
constructor signature `(num_layers, num_heads, d_k, device=None, dtype=...)`,
and pass the class as `kv_cache_class` to `generate()` or the scheduler. See
the README's "Adding a new KV cache strategy" section for the full pattern,
including how a radix-tree cache would differ (sharing physical storage
across sequences with identical prompt prefixes instead of allocating a
private block table per sequence).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
from torch import Tensor


class KVCache(ABC):
    """Abstract interface for key/value cache storage strategies.

    A `KVCache` instance holds the accumulated self-attention key/value
    tensors for every decoder layer of one generation stream. During
    autoregressive decoding, `MultiHeadAttention.forward()` reads/writes
    through this interface instead of recomputing attention over the full
    sequence at every step: only the newly produced K/V for the current
    token(s) need to be projected; the rest is retrieved from the cache.
    """

    @abstractmethod
    def update(self, layer_idx: int, new_k: Tensor, new_v: Tensor) -> Tuple[Tensor, Tensor]:
        """Append newly computed K/V for one layer and return the full cache so far.

        new_k, new_v: (batch, num_heads, new_len, d_k) — the keys/values just
            computed for the tokens processed this call (the whole prompt
            during prefill, or a single token during decode).
        Returns:
            (k, v), each (batch, num_heads, total_len, d_k): the full
            accumulated cache for this layer, ready to pass directly as the
            key/value of `scaled_dot_product_attention`.
        """
        raise NotImplementedError

    @abstractmethod
    def get(self, layer_idx: int) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Return the currently cached (k, v) for a layer without appending anything.

        Returns (None, None) if nothing has been cached for this layer yet.
        """
        raise NotImplementedError

    @abstractmethod
    def evict(self, layer_idx: Optional[int] = None) -> None:
        """Release cached entries, freeing them for reuse.

        `layer_idx=None` releases the entire cache (all layers) — the
        expected usage when a sequence finishes generating and its `KVCache`
        object is returned to a pool for reuse by a newly admitted sequence
        (see `scheduler.py`). Passing a specific `layer_idx` clears only that
        layer's entries.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def seq_len(self) -> int:
        """Number of tokens currently cached (assumed uniform across layers)."""
        raise NotImplementedError


class NaiveKVCache(KVCache):
    """Baseline KV cache: one growing tensor per layer, no memory management.

    Every `update()` call reallocates a bigger tensor via `torch.cat`. This
    is the simplest possible correct implementation and the natural
    "before" against which more sophisticated strategies (like
    `PagedKVCache`) are compared — it does no block reuse, no eviction under
    memory pressure, and its cost per decode step grows with the sequence
    length already cached (a fresh copy of everything seen so far).
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_k: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_k = d_k
        self.device = device
        self.dtype = dtype
        self._k: List[Optional[Tensor]] = [None] * num_layers
        self._v: List[Optional[Tensor]] = [None] * num_layers

    def update(self, layer_idx: int, new_k: Tensor, new_v: Tensor) -> Tuple[Tensor, Tensor]:
        """Concatenate the new K/V onto whatever is already cached for this layer.

        new_k, new_v: (batch, num_heads, new_len, d_k)
        """
        if self._k[layer_idx] is None:
            self._k[layer_idx] = new_k  # (batch, num_heads, new_len, d_k)
            self._v[layer_idx] = new_v
        else:
            self._k[layer_idx] = torch.cat([self._k[layer_idx], new_k], dim=2)
            # (batch, num_heads, cached_len, d_k) + (batch, num_heads, new_len, d_k) -> (batch, num_heads, cached_len + new_len, d_k)
            self._v[layer_idx] = torch.cat([self._v[layer_idx], new_v], dim=2)
        return self._k[layer_idx], self._v[layer_idx]

    def get(self, layer_idx: int) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Return whatever has been cached for this layer so far, unmodified."""
        return self._k[layer_idx], self._v[layer_idx]

    def evict(self, layer_idx: Optional[int] = None) -> None:
        """Drop cached tensors for one layer, or all layers if `layer_idx` is None."""
        if layer_idx is None:
            self._k = [None] * self.num_layers
            self._v = [None] * self.num_layers
        else:
            self._k[layer_idx] = None
            self._v[layer_idx] = None

    @property
    def seq_len(self) -> int:
        """Cached sequence length, read from layer 0 (all layers stay in sync)."""
        first = self._k[0]
        return 0 if first is None else first.size(2)


class PagedKVCache(KVCache):
    """Toy block/page-based KV cache: fixed-size physical blocks + a logical block table.

    Physical storage is preallocated as a fixed pool of `num_blocks` blocks,
    each holding `block_size` tokens' worth of K/V per layer. A per-sequence
    `block_table` maps logical block index -> physical block index; new
    tokens are written into whichever physical block their logical position
    falls into, allocating a fresh physical block from the shared free list
    only when the current one fills up. Reading K/V back for attention
    gathers the (possibly non-contiguous) physical blocks into one tensor
    via the table.

    This is a simplified stand-in for PagedAttention's core idea — storage
    doesn't need to be one contiguous growing allocation, so memory can be
    reused/recycled block-by-block — without any of its CUDA-kernel-level
    memory-coalescing optimizations, and without sharing physical blocks
    across sequences (see the module docstring's note on prefix sharing).
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_k: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        block_size: int = 16,
        num_blocks: int = 64,
        **kwargs,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_k = d_k
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Physical storage pool, one tensor per layer: (num_blocks, num_heads, block_size, d_k).
        # A given physical block index reserves space in every layer's pool simultaneously,
        # so a single shared block_table (below) is valid across all layers.
        self._k_blocks: List[Tensor] = [
            torch.zeros(num_blocks, num_heads, block_size, d_k, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self._v_blocks: List[Tensor] = [
            torch.zeros(num_blocks, num_heads, block_size, d_k, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]

        self._free_blocks: List[int] = list(range(num_blocks))
        self.block_table: List[int] = []  # logical block index -> physical block index
        self._layer_lengths: List[int] = [0] * num_layers  # tokens written so far, per layer

    def _ensure_capacity(self, total_len: int) -> None:
        """Allocate physical blocks from the free list until `total_len` tokens fit.

        Idempotent: called once per layer per step with the same `total_len`
        (since every layer processes the same number of new tokens per
        forward pass), so only the first call in a step actually allocates.
        """
        needed_blocks = math.ceil(total_len / self.block_size) if total_len > 0 else 0
        while len(self.block_table) < needed_blocks:
            if not self._free_blocks:
                raise RuntimeError(
                    f"PagedKVCache exhausted its {self.num_blocks}-block pool "
                    f"(block_size={self.block_size}); increase num_blocks."
                )
            self.block_table.append(self._free_blocks.pop())

    def update(self, layer_idx: int, new_k: Tensor, new_v: Tensor) -> Tuple[Tensor, Tensor]:
        """Write new K/V into physical blocks per the logical block table, then gather.

        new_k, new_v: (batch, num_heads, new_len, d_k); batch must be 1 (this
            toy implementation caches one sequence per instance).
        """
        batch, _, new_len, _ = new_k.shape
        if batch != 1:
            raise ValueError("PagedKVCache supports batch size 1 per cache instance")

        start = self._layer_lengths[layer_idx]
        total = start + new_len
        self._ensure_capacity(total)

        for i in range(new_len):
            pos = start + i
            logical_block, offset = divmod(pos, self.block_size)
            physical_block = self.block_table[logical_block]
            self._k_blocks[layer_idx][physical_block, :, offset, :] = new_k[0, :, i, :]
            # (num_heads, d_k) written into one token-slot of one physical block
            self._v_blocks[layer_idx][physical_block, :, offset, :] = new_v[0, :, i, :]

        self._layer_lengths[layer_idx] = total
        return self._gather(layer_idx, total)

    def _gather(self, layer_idx: int, total_len: int) -> Tuple[Tensor, Tensor]:
        """Reconstruct a contiguous (1, num_heads, total_len, d_k) tensor from scattered blocks."""
        k_chunks, v_chunks = [], []
        remaining = total_len
        for physical_block in self.block_table:
            if remaining <= 0:
                break
            take = min(self.block_size, remaining)
            k_chunks.append(self._k_blocks[layer_idx][physical_block, :, :take, :])  # (num_heads, take, d_k)
            v_chunks.append(self._v_blocks[layer_idx][physical_block, :, :take, :])
            remaining -= take

        k = torch.cat(k_chunks, dim=1).unsqueeze(0)  # (num_heads, total_len, d_k) -> (1, num_heads, total_len, d_k)
        v = torch.cat(v_chunks, dim=1).unsqueeze(0)
        return k, v

    def get(self, layer_idx: int) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Return the currently cached (k, v) for a layer without writing anything."""
        total = self._layer_lengths[layer_idx]
        if total == 0:
            return None, None
        return self._gather(layer_idx, total)

    def evict(self, layer_idx: Optional[int] = None) -> None:
        """Release physical blocks back to the free pool.

        `layer_idx=None` (the expected usage) frees this sequence's entire
        block table, since the table is shared across all layers — this is
        what lets a finished sequence's cache object be handed to a new
        sequence in `scheduler.py` without reallocating tensors. A specific
        `layer_idx` only resets that layer's logical length, leaving its
        physical blocks reserved (they're still needed by the other layers).
        """
        if layer_idx is None:
            self._free_blocks.extend(self.block_table)
            self.block_table = []
            self._layer_lengths = [0] * self.num_layers
        else:
            self._layer_lengths[layer_idx] = 0

    @property
    def seq_len(self) -> int:
        """Cached sequence length, read from layer 0 (all layers stay in sync)."""
        return self._layer_lengths[0] if self._layer_lengths else 0
