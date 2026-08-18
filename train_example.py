"""Toy training example: sequence reversal, used as a regression check.

Trains the full encoder-decoder transformer to reverse a sequence of
random digit tokens (e.g. [3, 7, 1, 9] -> [9, 1, 7, 3]). This exercises
embeddings, positional encoding, encoder self-attention, decoder causal
self-attention, decoder cross-attention, and the feed-forward blocks end
to end.

Run after modifying any component (e.g. swapping in a new attention
variant) to confirm the model still trains: loss should drop and greedy-
decoded accuracy on held-out sequences should approach 1.0.

Usage:
    python train_example.py
"""

from __future__ import annotations

import random

import torch
from torch import Tensor, nn

from config import TransformerConfig
from transformer import EncoderDecoderTransformer, build_encoder_decoder

# --- toy vocabulary ---
# 0: PAD, 1: BOS, 2: EOS, 3..12: digits 0-9
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
DIGIT_OFFSET = 3
VOCAB_SIZE = DIGIT_OFFSET + 10


def make_batch(batch_size: int, seq_len: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """Sample a batch of (source, decoder_input, target) for sequence reversal.

    source: (batch, seq_len) random digit tokens.
    decoder_input: (batch, seq_len + 1) = [BOS] + reversed(source), fed to the
        decoder as teacher-forcing input.
    target: (batch, seq_len + 1) = reversed(source) + [EOS], the labels the
        decoder's output logits are trained to predict at each position.
    """
    src = torch.randint(0, 10, (batch_size, seq_len), device=device) + DIGIT_OFFSET  # (batch, seq_len)
    reversed_src = src.flip(dims=[1])  # (batch, seq_len)

    bos_col = torch.full((batch_size, 1), BOS_ID, device=device, dtype=torch.long)
    eos_col = torch.full((batch_size, 1), EOS_ID, device=device, dtype=torch.long)

    decoder_input = torch.cat([bos_col, reversed_src], dim=1)  # (batch, seq_len + 1)
    target = torch.cat([reversed_src, eos_col], dim=1)  # (batch, seq_len + 1)

    return src, decoder_input, target


@torch.no_grad()
def greedy_decode_accuracy(
    model: EncoderDecoderTransformer,
    seq_len: int,
    num_samples: int,
    device: torch.device,
) -> float:
    """Greedily decode reversed sequences and report exact-match accuracy.

    Runs the encoder once per batch, then autoregressively generates the
    target sequence one token at a time (no teacher forcing), comparing the
    final result against the true reversal.
    """
    model.eval()
    src = torch.randint(0, 10, (num_samples, seq_len), device=device) + DIGIT_OFFSET  # (num_samples, seq_len)
    expected = src.flip(dims=[1])  # (num_samples, seq_len)

    memory = model.encode(src)  # (num_samples, seq_len, d_model)
    generated = torch.full((num_samples, 1), BOS_ID, device=device, dtype=torch.long)  # (num_samples, 1)

    for _ in range(seq_len):
        decoded = model.decode(generated, memory, src)  # (num_samples, cur_len, d_model)
        logits = model.output_projection(decoded)  # (num_samples, cur_len, vocab_size)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (num_samples, 1)
        generated = torch.cat([generated, next_token], dim=1)  # (num_samples, cur_len + 1)

    predicted = generated[:, 1:]  # drop BOS -> (num_samples, seq_len)
    exact_match = (predicted == expected).all(dim=1).float().mean().item()
    model.train()
    return exact_match


def main() -> None:
    """Train an encoder-decoder transformer on sequence reversal and report accuracy."""
    random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = TransformerConfig(
        vocab_size=VOCAB_SIZE,
        max_seq_len=32,
        pad_token_id=PAD_ID,
        d_model=64,
        num_heads=4,
        d_ff=256,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.0,  # this is a fast regression check, not a realistic training run
        attention_type="multi_head",
    )
    model = build_encoder_decoder(config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    seq_len = 8
    batch_size = 64
    num_steps = 2000

    for step in range(1, num_steps + 1):
        src, decoder_input, target = make_batch(batch_size, seq_len, device)

        logits = model(src, decoder_input)  # (batch, seq_len + 1, vocab_size)
        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE), target.reshape(-1)
        )  # flatten (batch, tgt_len, vocab) -> (batch * tgt_len, vocab) for cross-entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0 or step == 1:
            acc = greedy_decode_accuracy(model, seq_len, num_samples=128, device=device)
            print(f"step {step:4d} | loss {loss.item():.4f} | greedy exact-match acc {acc:.3f}")

    final_acc = greedy_decode_accuracy(model, seq_len, num_samples=256, device=device)
    print(f"\nFinal greedy exact-match accuracy over 256 samples: {final_acc:.3f}")
    if final_acc < 0.9:
        print("WARNING: accuracy is lower than expected for a correctly-wired model.")
    else:
        print("Model trains and reverses sequences correctly.")


if __name__ == "__main__":
    main()
