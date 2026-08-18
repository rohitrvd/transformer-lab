"""Tests for mask construction helpers."""

from __future__ import annotations

import torch

from masks import combine_masks, make_causal_mask, make_padding_mask


def test_causal_mask_is_lower_triangular():
    mask = make_causal_mask(4, torch.device("cpu")).squeeze(0).squeeze(0)  # (4, 4)

    expected = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    assert torch.equal(mask, expected)


def test_padding_mask_marks_pad_positions_false():
    token_ids = torch.tensor([[1, 2, 0, 0], [1, 1, 1, 0]])
    mask = make_padding_mask(token_ids, pad_token_id=0).squeeze(1).squeeze(1)  # (batch, seq_len)

    expected = torch.tensor([[True, True, False, False], [True, True, True, False]])
    assert torch.equal(mask, expected)


def test_combine_masks_is_logical_and():
    a = torch.tensor([[True, True, False]])
    b = torch.tensor([[True, False, False]])

    combined = combine_masks(a, b)

    assert torch.equal(combined, torch.tensor([[True, False, False]]))
