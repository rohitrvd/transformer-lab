"""Mask construction helpers shared by the encoder, decoder, and top-level models.

Two kinds of masking are used throughout:
  - Padding masks: hide `pad_token_id` positions so attention never
    attends to padding.
  - Causal (subsequent) masks: hide future positions so a decoder position
    can only attend to itself and earlier positions.
"""

from __future__ import annotations

import torch
from torch import Tensor


def make_padding_mask(token_ids: Tensor, pad_token_id: int) -> Tensor:
    """Build a mask that is 0 at padding positions and 1 elsewhere.

    token_ids: (batch, seq_len) -> (batch, 1, 1, seq_len)
    The (1, 1, ...) dims broadcast over the num_heads and query-length axes
    of attention scores shaped (batch, num_heads, q_len, kv_len).
    """
    mask = (token_ids != pad_token_id).unsqueeze(1).unsqueeze(2)  # (batch, seq_len) -> (batch, 1, 1, seq_len)
    return mask


def make_causal_mask(seq_len: int, device: torch.device) -> Tensor:
    """Build a lower-triangular mask preventing attention to future positions.

    Returns a (1, 1, seq_len, seq_len) tensor that is 1 where position `i`
    is allowed to attend to position `j` (i.e. j <= i), and 0 otherwise.
    Broadcasts over batch and num_heads dims of attention scores.
    """
    causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))  # (seq_len, seq_len)
    return causal.unsqueeze(0).unsqueeze(0)  # (seq_len, seq_len) -> (1, 1, seq_len, seq_len)


def combine_masks(*masks: Tensor) -> Tensor:
    """Elementwise-AND multiple broadcastable boolean masks into one.

    Used to combine, e.g., a decoder's padding mask and causal mask into a
    single mask passed to self-attention.
    """
    combined = masks[0]
    for m in masks[1:]:
        combined = combined & m
    return combined
