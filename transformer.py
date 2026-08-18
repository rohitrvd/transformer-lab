"""Top-level model assembly: encoder-only, decoder-only, and encoder-decoder.

This is the single place a `TransformerConfig` becomes concrete `nn.Module`
objects. The `build_*` factory functions resolve `config.attention_type`
and `config.cross_attention_type` from the attention registry and wire the
resulting classes into `Encoder`/`Decoder`, so choosing a different
attention implementation is done entirely through the config passed here.
"""

from __future__ import annotations

from typing import Optional, Type

import torch
from torch import Tensor, nn

from attention import BaseAttention
from config import TransformerConfig, resolve_attention
from decoder import Decoder
from embeddings import TransformerEmbedding
from encoder import Encoder
from feedforward import PositionwiseFeedForward
from layer_norm import LayerNorm, SublayerConnection
from masks import combine_masks, make_causal_mask, make_padding_mask


class EncoderOnlyModel(nn.Module):
    """Encoder-only transformer (BERT-style): bidirectional self-attention.

    Maps input token ids to per-position contextual representations, plus
    an optional projection to vocabulary logits for masked-language-model
    style pretraining/experimentation.
    """

    def __init__(self, config: TransformerConfig, attention_cls: Type[BaseAttention]) -> None:
        super().__init__()
        self.config = config
        self.embedding = TransformerEmbedding(config)
        self.encoder = Encoder(config, attention_cls)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        if config.tie_embeddings:
            self.output_projection.weight = self.embedding.token_embedding.embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        """Embed, encode, and project to vocabulary logits.

        input_ids: (batch, seq_len) -> (batch, seq_len, vocab_size)
        """
        mask = make_padding_mask(input_ids, self.config.pad_token_id)  # (batch, 1, 1, seq_len)
        x = self.embedding(input_ids)  # (batch, seq_len) -> (batch, seq_len, d_model)
        x = self.encoder(x, mask=mask)  # (batch, seq_len, d_model)
        return self.output_projection(x)  # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)


class DecoderOnlyModel(nn.Module):
    """Decoder-only transformer (GPT-style): causal self-attention, no cross-attention.

    `decoder.py`'s `DecoderLayer` always includes cross-attention to an
    encoder's memory, which a decoder-only model has none of. Rather than
    contort that class with an optional-cross-attention branch, this model
    uses a dedicated `_DecoderOnlyLayer` stack: self-attention (causally
    masked) + feed-forward only.
    """

    def __init__(self, config: TransformerConfig, attention_cls: Type[BaseAttention]) -> None:
        super().__init__()
        self.config = config
        self.embedding = TransformerEmbedding(config)
        self.layers = nn.ModuleList(
            [_DecoderOnlyLayer(config, attention_cls) for _ in range(config.num_decoder_layers)]
        )
        self.norm = LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        if config.tie_embeddings:
            self.output_projection.weight = self.embedding.token_embedding.embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        """Embed, run causally-masked self-attention layers, project to logits.

        input_ids: (batch, seq_len) -> (batch, seq_len, vocab_size)
        """
        seq_len = input_ids.size(1)
        padding_mask = make_padding_mask(input_ids, self.config.pad_token_id)  # (batch, 1, 1, seq_len)
        causal_mask = make_causal_mask(seq_len, input_ids.device)  # (1, 1, seq_len, seq_len)
        mask = combine_masks(padding_mask, causal_mask)  # (batch, 1, seq_len, seq_len)

        x = self.embedding(input_ids)  # (batch, seq_len) -> (batch, seq_len, d_model)
        for layer in self.layers:
            x = layer(x, mask=mask)  # (batch, seq_len, d_model)
        x = self.norm(x)  # (batch, seq_len, d_model)
        return self.output_projection(x)  # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)


class _DecoderOnlyLayer(nn.Module):
    """Self-attention + feed-forward block with no cross-attention (GPT-style).

    Structurally identical to `EncoderLayer`, kept as a distinct class so
    `decoder.py`'s `DecoderLayer` (which always has cross-attention) is not
    overloaded with an "is this encoder-decoder or decoder-only" branch.
    """

    def __init__(self, config: TransformerConfig, attention_cls: Type[BaseAttention]) -> None:
        super().__init__()
        self.self_attn = attention_cls(config)
        self.feed_forward = PositionwiseFeedForward(config)
        self.self_attn_sublayer = SublayerConnection(config)
        self.feed_forward_sublayer = SublayerConnection(config)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Causal self-attention followed by feed-forward, each with residual+norm.

        x: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        """

        def self_attention(x: Tensor) -> Tensor:
            out, _ = self.self_attn(x, x, x, mask=mask)
            return out

        x = self.self_attn_sublayer(x, self_attention)  # (batch, seq_len, d_model)
        x = self.feed_forward_sublayer(x, self.feed_forward)  # (batch, seq_len, d_model)
        return x


class EncoderDecoderTransformer(nn.Module):
    """Full encoder-decoder transformer, as in "Attention Is All You Need".

    The encoder builds a contextual representation of the source sequence;
    the decoder generates the target sequence autoregressively, attending
    to both its own (causally-masked) prefix and the full encoder output.
    """

    def __init__(
        self,
        config: TransformerConfig,
        attention_cls: Type[BaseAttention],
        cross_attention_cls: Type[BaseAttention],
    ) -> None:
        super().__init__()
        self.config = config
        self.src_embedding = TransformerEmbedding(config)
        self.tgt_embedding = TransformerEmbedding(config)
        self.encoder = Encoder(config, attention_cls)
        self.decoder = Decoder(config, attention_cls, cross_attention_cls)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        if config.tie_embeddings:
            self.output_projection.weight = self.tgt_embedding.token_embedding.embedding.weight

    def encode(self, src_ids: Tensor) -> Tensor:
        """Embed and encode the source sequence.

        src_ids: (batch, src_len) -> (batch, src_len, d_model)
        """
        src_mask = make_padding_mask(src_ids, self.config.pad_token_id)  # (batch, 1, 1, src_len)
        x = self.src_embedding(src_ids)  # (batch, src_len) -> (batch, src_len, d_model)
        return self.encoder(x, mask=src_mask)  # (batch, src_len, d_model)

    def decode(self, tgt_ids: Tensor, memory: Tensor, src_ids: Tensor) -> Tensor:
        """Embed the target prefix and decode against encoder memory.

        tgt_ids: (batch, tgt_len) -> (batch, tgt_len, d_model)
        memory: (batch, src_len, d_model), the encoder output.
        src_ids: (batch, src_len), used to rebuild the source padding mask
            for cross-attention.
        """
        tgt_len = tgt_ids.size(1)
        tgt_padding_mask = make_padding_mask(tgt_ids, self.config.pad_token_id)  # (batch, 1, 1, tgt_len)
        causal_mask = make_causal_mask(tgt_len, tgt_ids.device)  # (1, 1, tgt_len, tgt_len)
        self_attn_mask = combine_masks(tgt_padding_mask, causal_mask)  # (batch, 1, tgt_len, tgt_len)

        cross_attn_mask = make_padding_mask(src_ids, self.config.pad_token_id)  # (batch, 1, 1, src_len)

        x = self.tgt_embedding(tgt_ids)  # (batch, tgt_len) -> (batch, tgt_len, d_model)
        return self.decoder(
            x, memory, self_attn_mask=self_attn_mask, cross_attn_mask=cross_attn_mask
        )  # (batch, tgt_len, d_model)

    def forward(self, src_ids: Tensor, tgt_ids: Tensor) -> Tensor:
        """Encode the source, decode the target, project to vocabulary logits.

        src_ids: (batch, src_len)
        tgt_ids: (batch, tgt_len)
        returns: (batch, tgt_len, vocab_size)
        """
        memory = self.encode(src_ids)  # (batch, src_len, d_model)
        decoded = self.decode(tgt_ids, memory, src_ids)  # (batch, tgt_len, d_model)
        return self.output_projection(decoded)  # (batch, tgt_len, d_model) -> (batch, tgt_len, vocab_size)


# ---------------------------------------------------------------------------
# Config-driven factory functions
# ---------------------------------------------------------------------------
# These are the intended entry points for building a model: pass a
# TransformerConfig, get back a wired-up model. Attention variants are
# selected purely through config.attention_type / config.cross_attention_type.


def build_encoder_only(config: TransformerConfig) -> EncoderOnlyModel:
    """Build a BERT-style encoder-only model from `config`."""
    attention_cls = resolve_attention(config.attention_type)
    return EncoderOnlyModel(config, attention_cls)


def build_decoder_only(config: TransformerConfig) -> DecoderOnlyModel:
    """Build a GPT-style decoder-only model from `config`."""
    attention_cls = resolve_attention(config.attention_type)
    return DecoderOnlyModel(config, attention_cls)


def build_encoder_decoder(config: TransformerConfig) -> EncoderDecoderTransformer:
    """Build a full encoder-decoder model from `config`."""
    attention_cls = resolve_attention(config.attention_type)
    cross_attention_cls = resolve_attention(config.cross_attention_type)
    return EncoderDecoderTransformer(config, attention_cls, cross_attention_cls)
