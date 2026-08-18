# Transformer Lab

A from-scratch PyTorch implementation of the transformer architecture, built as an
experimentation base rather than a production library. Every core operation (Q/K/V
projections, scaled dot-product attention, multi-head splitting, sinusoidal positional
encoding, layer normalization) is hand-rolled on top of raw `torch.nn.Module` — no
`nn.MultiheadAttention`, no `nn.Transformer`. The goal is to be able to swap out a single
component (most importantly, the attention mechanism) and see whether the model still
works, without touching the rest of the architecture.

## Repository structure

```
config.py            TransformerConfig dataclass — all hyperparameters and component
                      choices (attention type, activation, norm placement, ...)
embeddings.py         TokenEmbedding, SinusoidalPositionalEncoding, TransformerEmbedding
attention.py          BaseAttention (abstract interface) + MultiHeadAttention
                      + scaled_dot_product_attention (reusable core math)
feedforward.py        PositionwiseFeedForward (config-driven activation)
layer_norm.py         LayerNorm (hand-rolled) + SublayerConnection (pre/post-norm residual)
masks.py              Padding mask / causal mask / mask-combination helpers
encoder.py            EncoderLayer, Encoder — take an attention class as a constructor arg
decoder.py            DecoderLayer, Decoder — self-attention + cross-attention, each
                      pluggable independently
transformer.py         EncoderOnlyModel, DecoderOnlyModel, EncoderDecoderTransformer,
                      and the build_encoder_only / build_decoder_only /
                      build_encoder_decoder factory functions (config -> wired model)
train_example.py      Toy sequence-reversal training script — regression check after
                      any modification
tests/                Unit tests: shapes, attention-weight properties, causal masking,
                      and the pluggable-attention interface
```

## The pluggable-attention pattern

`encoder.py` and `decoder.py` never import a concrete attention class. `EncoderLayer`,
`Encoder`, `DecoderLayer`, and `Decoder` all take an `attention_cls: Type[BaseAttention]`
constructor argument and instantiate it themselves (`self.self_attn = attention_cls(config)`).
`transformer.py`'s `build_*` factory functions are the only place a config's
`attention_type` string is resolved to an actual class, via a small registry:

```python
# config.py
ATTENTION_REGISTRY: Dict[str, Type[BaseAttention]] = {}

def register_attention(name: str, cls: Type[BaseAttention]) -> None: ...
def resolve_attention(name: str) -> Type[BaseAttention]: ...
```

`attention.py` registers the built-in implementation at import time:

```python
register_attention("multi_head", MultiHeadAttention)
```

### Adding a new attention variant

1. Subclass `BaseAttention` in a new file (or in `attention.py` directly):

   ```python
   from attention import BaseAttention
   from config import TransformerConfig, register_attention

   class SlidingWindowAttention(BaseAttention):
       def __init__(self, config: TransformerConfig) -> None:
           super().__init__(config)
           # your Q/K/V projections, window size, etc.

       def forward(self, query, key, value, mask=None):
           # must return (output, attn_weights) with the same shapes
           # BaseAttention documents: output (batch, q_len, d_model),
           # attn_weights (batch, num_heads, q_len, kv_len)
           ...

   register_attention("sliding_window", SlidingWindowAttention)
   ```

2. Select it in a config — no changes to `encoder.py`, `decoder.py`, or `transformer.py`:

   ```python
   config = TransformerConfig(
       vocab_size=1000,
       attention_type="sliding_window",       # used for encoder self-attn and decoder self-attn
       cross_attention_type="multi_head",     # decoder cross-attention can differ
   )
   model = build_encoder_decoder(config)
   ```

   Or bypass the registry entirely and construct components directly with the class:

   ```python
   from encoder import Encoder
   encoder = Encoder(config, SlidingWindowAttention)
   ```

`tests/test_pluggable_attention.py` implements a deliberately trivial
`DummyUniformAttention` variant and exercises both paths (direct constructor injection and
config/registry-driven), as a template for testing your own variant.

## Config-driven assembly

`TransformerConfig` (in `config.py`) is a dataclass covering dimensions
(`d_model`, `num_heads`, `d_ff`), depth (`num_encoder_layers`, `num_decoder_layers`),
regularization (`dropout`, `layer_norm_eps`), and architectural switches:

- `attention_type` / `cross_attention_type` — which registered attention class to use
- `activation` — `"relu"` or `"gelu"` (resolved via `ACTIVATION_REGISTRY` in `config.py`)
- `norm_placement` — `"pre"` (`x + Sublayer(Norm(x))`) or `"post"` (`Norm(x + Sublayer(x))`),
  applied uniformly by `SublayerConnection`

`transformer.py` exposes three build functions that turn a config into a runnable model:

```python
from config import TransformerConfig
from transformer import build_encoder_only, build_decoder_only, build_encoder_decoder

config = TransformerConfig(vocab_size=1000, d_model=256, num_heads=8, d_ff=1024)

encoder_model = build_encoder_only(config)      # BERT-style
decoder_model = build_decoder_only(config)      # GPT-style, causal self-attention only
full_model    = build_encoder_decoder(config)   # encoder + decoder with cross-attention
```

## Running the training example

```
pip install -r requirements.txt
python train_example.py
```

Trains `EncoderDecoderTransformer` on a toy sequence-reversal task (reverse a short
sequence of digit tokens). Prints loss and greedy-decode exact-match accuracy every 50
steps; a correctly-wired model should reach >90% accuracy within 500 steps. Re-run this
after swapping in a new attention variant (or any other component change) as a smoke/
regression check — if accuracy stays near 0, something is wired incorrectly.

## Running the tests

```
pip install -r requirements.txt
pytest
```

Coverage includes:

- **Shape tests** at every stage (embeddings, attention, encoder/decoder layers and
  stacks, all three model variants) — `tests/test_embeddings.py`,
  `tests/test_encoder.py`, `tests/test_decoder.py`, `tests/test_transformer.py`
- **Known properties** — attention weights sum to 1 over keys
  (`test_attention_weights_sum_to_one`), causal masks zero out attention to future
  positions (`test_causal_mask_zeroes_future_attention_weights`,
  `test_decoder_causal_self_attention_ignores_future_targets`,
  `test_decoder_only_is_causal`), padding positions receive zero attention weight
  (`test_padding_mask_zeroes_attention_to_pad_positions`) — `tests/test_attention.py`,
  `tests/test_masks.py`
- **Pluggable-attention interface** — a dummy alternate `BaseAttention` implementation
  is injected directly into `Encoder`/`Decoder` and via the config registry, and the
  full model still runs forward and backward — `tests/test_pluggable_attention.py`

## Type hints and shape comments

All function signatures are type-hinted. Every non-trivial tensor operation (especially
`.view()` / `.permute()` calls in `attention.py`'s head-splitting) has an inline comment
showing the shape transformation, e.g.:

```python
x = x.view(batch, seq_len, self.num_heads, self.d_k)  # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k)
```
