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
inference/            Inference-experimentation layer (KV caching, batching, quantization)
  kv_cache.py            KVCache (abstract interface) + NaiveKVCache + PagedKVCache
  generation.py          GenerationStream (per-sequence prefill/decode primitive) + generate()
  scheduler.py           ContinuousBatchingScheduler — toy multi-sequence simulator
  quantization.py        quantize_model() + fp32-vs-int8 latency/memory benchmark
benchmark.py           Compares NaiveKVCache vs PagedKVCache on the same generation task
tests/                Unit tests: shapes, attention-weight properties, causal masking,
                      the pluggable-attention interface, and the inference layer
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
- **KV cache correctness** — cached generation produces (numerically, up to float
  rounding) identical output to the equivalent no-cache full-sequence forward pass, for
  both `DecoderOnlyModel` and `EncoderDecoderTransformer` — `tests/test_generation.py`
- **Paged cache block mapping** — `PagedKVCache`'s logical→physical block table is
  checked directly (block allocation, gather-back-in-order, eviction returning blocks
  to the free pool) — `tests/test_kv_cache.py`
- **Scheduler behavior** — sequences joining mid-run when capacity frees up, finishing
  and leaving the active batch, cache eviction on completion, and a custom
  `admission_policy` overriding the default — `tests/test_scheduler.py`

## Inference layer

`inference/` is a second, separable layer on top of the architecture files above: it
adds KV caching, a toy continuous-batching scheduler, and quantization experiments,
without changing how `transformer.py`/`encoder.py`/`decoder.py`/`attention.py` behave
when no cache is involved (every new parameter — `kv_cache`, `layer_idx`,
`position_offset` — defaults to `None`/`0` and reproduces the exact training-time
forward pass).

### How prefill/decode work here

Autoregressive generation has two phases:

- **Prefill**: the initial prompt is run through the model in one forward pass, which
  populates a `KVCache` with that prompt's key/value tensors for every decoder layer.
- **Decode**: one new token at a time is fed in; instead of recomputing self-attention
  over the whole sequence so far, `MultiHeadAttention.forward()` (see `attention.py`)
  projects only the new token's K/V, appends them to the cache
  (`kv_cache.update(layer_idx, new_k, new_v)`), and attends over everything the cache
  returns. `SinusoidalPositionalEncoding` (see `embeddings.py`) is given a
  `position_offset` so the new token gets its true position instead of restarting at 0.

`inference/generation.py`'s `GenerationStream` is the primitive that runs both phases
for one sequence; `generate()` is a thin driver around it for a single prompt, and
`ContinuousBatchingScheduler` (`inference/scheduler.py`) reuses `GenerationStream`
directly to advance many sequences concurrently.

Two model shapes are supported, with different prefill semantics:

- `DecoderOnlyModel` (GPT-style): `prompt_tokens` can be arbitrary length, and prefill
  genuinely processes it all in one shot — the clearest illustration of "prefill is O(1)
  forward passes, decode is one token per forward pass."
- `EncoderDecoderTransformer` (what `train_example.py` trains): the encoder side runs
  once, non-autoregressively (no cache needed — nothing about it grows across decode
  steps). The decoder side always starts from a single `decoder_start_token_id` (BOS) —
  there's no multi-token decoder prompt in this toy task — so "prefill" here is the
  encoder pass plus a length-1 decoder seed, not a multi-token prefill. That's a property
  of the task, not a bug.

Cross-attention is deliberately **not** cached: the encoder's `memory` is fixed after
`encode()`, so recomputing cross-attention K/V projections every decode step is wasted
work but not incorrect. Caching is scoped to self-attention — the actually-growing
sequence — since that's the mechanism worth seeing clearly. Caching cross-attention K/V
too is a natural follow-on extension, left undone here.

### Adding a new KV cache strategy

`KVCache` (in `inference/kv_cache.py`) mirrors `BaseAttention`'s pluggability pattern:
one abstract interface, a uniform constructor, multiple interchangeable implementations.

```python
from inference.kv_cache import KVCache

class RadixTreeKVCache(KVCache):
    """Prefix-sharing cache: identical prompt prefixes across sequences share
    physical storage instead of each sequence allocating its own private blocks."""

    def __init__(self, num_layers: int, num_heads: int, d_k: int, device=None, dtype=..., **kwargs):
        ...  # a shared trie of blocks, keyed by token-sequence prefix

    def update(self, layer_idx, new_k, new_v): ...  # walk/extend the trie, return full K/V
    def get(self, layer_idx): ...
    def evict(self, layer_idx=None): ...  # decrement refcounts; free only unshared blocks

    @property
    def seq_len(self) -> int: ...
```

Pass the class directly:

```python
from inference.generation import generate
generate(model, prompt_tokens, max_new_tokens=50, kv_cache_class=RadixTreeKVCache)
```

`NaiveKVCache` and `PagedKVCache` (the two implementations included here) both share
this same constructor signature `(num_layers, num_heads, d_k, device=None, dtype=...)`,
which is exactly what lets `generate()`, `benchmark.py`, and `ContinuousBatchingScheduler`
swap between them without knowing which concrete class they're holding.

### Running the benchmarks

```
python benchmark.py                 # NaiveKVCache vs PagedKVCache: prefill time,
                                     # per-token decode time, peak memory
python -m inference.quantization    # fp32 vs dynamic int8: latency + serialized size
```

Both scripts train a fresh small model via `train_example.train_model()` (the same
architecture/task as `train_example.py`) before benchmarking, so no separate setup step
is needed. `inference/quantization.py` forces CPU, since PyTorch's dynamic quantization
is CPU-only.

Two results worth knowing going in, so they don't look like bugs: at this toy model's
tiny scale, `PagedKVCache` doesn't reliably beat `NaiveKVCache` on latency (Python-loop
and forward-pass overhead dominate over the tensor-`cat` cost `NaiveKVCache` pays each
step — the effect `PagedKVCache` is meant to demonstrate becomes visible at longer
sequences / larger models), and dynamic quantization can be *slower* than fp32 here for
the same reason (per-call activation quantization overhead outweighs the smaller
matmuls at this size). Both are genuine, reproducible measurements, not benchmark bugs —
try increasing `d_model`/`max_new_tokens` to see the effects grow.

### What each piece stands in for

| This repo | Real technique | Simplified away |
|---|---|---|
| `NaiveKVCache` | The "no cache management" baseline every real KV cache improves on | — (this one's meant to be naive) |
| `PagedKVCache` | [PagedAttention](https://arxiv.org/abs/2309.06180) (vLLM) — non-contiguous block storage + a logical→physical block table | No CUDA kernel, no cross-sequence block sharing, single sequence per cache instance |
| `RadixTreeKVCache` (documented, not implemented) | vLLM/SGLang prefix caching — a radix tree of shared blocks across requests with identical prompt prefixes | — |
| `ContinuousBatchingScheduler` | Continuous / iteration-level batching (e.g. vLLM's scheduler) — sequences join and leave the active batch every iteration, not just between fixed-size batches | Each sequence advances via an independent Python-level forward call rather than one fused batched/paged kernel call across all active sequences — the scheduling *decisions* are faithful, the GPU throughput win from batching them together is not reproduced |
| `inference/quantization.py` | Post-training dynamic quantization (a real, standard `torch` API, not hand-rolled) | Only dynamic (not static/QAT) quantization; CPU only; only `nn.Linear` layers |

## Type hints and shape comments

All function signatures are type-hinted. Every non-trivial tensor operation (especially
`.view()` / `.permute()` calls in `attention.py`'s head-splitting) has an inline comment
showing the shape transformation, e.g.:

```python
x = x.view(batch, seq_len, self.num_heads, self.d_k)  # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k)
```
