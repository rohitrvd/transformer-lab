"""Token embeddings and sinusoidal positional encoding.

Implements both pieces from first principles: a learned token embedding
table (scaled by sqrt(d_model), as in the original Transformer paper) and
a fixed (non-learned) sinusoidal positional encoding computed directly from
sine/cosine functions of position and dimension index.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from config import TransformerConfig


class TokenEmbedding(nn.Module):
    """Maps input token ids to dense vectors.

    A standard lookup-table embedding, scaled by sqrt(d_model). The scaling
    (from "Attention Is All You Need") balances the relative magnitude of
    the embedding against the positional encoding it will be added to.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        self.d_model = config.d_model

    def forward(self, token_ids: Tensor) -> Tensor:
        """Embed a batch of token id sequences.

        token_ids: (batch, seq_len) -> (batch, seq_len, d_model)
        """
        embedded = self.embedding(token_ids)  # (batch, seq_len) -> (batch, seq_len, d_model)
        return embedded * math.sqrt(self.d_model)  # (batch, seq_len, d_model)


class SinusoidalPositionalEncoding(nn.Module):
    """Injects position information using fixed sine/cosine signals.

    Since self-attention has no inherent notion of token order, a
    deterministic positional signal is added to each token embedding. For
    position `pos` and dimension `i`:
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    This is precomputed once (not learned) and registered as a buffer so it
    moves with the module across devices but is excluded from gradients.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)

        d_model = config.d_model
        max_seq_len = config.max_seq_len

        position = torch.arange(0, max_seq_len).unsqueeze(1).float()  # (max_seq_len,) -> (max_seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model / 2,)

        pe = torch.zeros(max_seq_len, d_model)  # (max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims: (max_seq_len, d_model / 2)
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims: (max_seq_len, d_model / 2)
        pe = pe.unsqueeze(0)  # (max_seq_len, d_model) -> (1, max_seq_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor, position_offset: int = 0) -> Tensor:
        """Add positional encodings to a batch of token embeddings.

        x: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        position_offset: index of `x`'s first position in the full sequence.
            0 for a normal forward pass (or prefill). During cached
            autoregressive decoding, `x` holds only the newest token(s), so
            the caller passes `position_offset=kv_cache.seq_len` (the number
            of positions already cached) to fetch the correct slice of `pe`
            instead of always starting from position 0.
        """
        seq_len = x.size(1)
        max_seq_len = self.pe.size(1)
        if position_offset + seq_len > max_seq_len:
            raise ValueError(
                f"position {position_offset + seq_len} exceeds max_seq_len={max_seq_len}; "
                "generation (or input) has run past the length TransformerConfig.max_seq_len "
                "was built for. Without this check, slicing past the end of `pe` would silently "
                "return an empty tensor instead of a clear error."
            )
        x = x + self.pe[:, position_offset : position_offset + seq_len, :]
        # (batch, seq_len, d_model) + (1, seq_len, d_model)
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    """Combines token embedding and positional encoding into one module.

    Convenience wrapper so encoder/decoder input handling is a single call:
    token ids in, position-aware embeddings out.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.token_embedding = TokenEmbedding(config)
        self.positional_encoding = SinusoidalPositionalEncoding(config)

    def forward(self, token_ids: Tensor, position_offset: int = 0) -> Tensor:
        """Embed token ids and add positional information.

        token_ids: (batch, seq_len) -> (batch, seq_len, d_model)
        position_offset: see `SinusoidalPositionalEncoding.forward`; 0 for a
            normal/prefill pass, `kv_cache.seq_len` during cached decoding.
        """
        x = self.token_embedding(token_ids)  # (batch, seq_len) -> (batch, seq_len, d_model)
        return self.positional_encoding(x, position_offset=position_offset)  # (batch, seq_len, d_model)
