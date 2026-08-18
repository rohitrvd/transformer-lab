"""Post-training dynamic quantization experiments on the trained toy model.

Applies PyTorch's built-in dynamic quantization (`torch.ao.quantization.quantize_dynamic`)
to every `nn.Linear` layer in the model — the Q/K/V/output projections in
`MultiHeadAttention` and both layers of `PositionwiseFeedForward` — converting
their weights from fp32 to int8 (activations are quantized on the fly, per
call, hence "dynamic"). This is a real, standard PyTorch quantization API,
not a hand-rolled approximation; what's toy-scoped here is only the
benchmark harness around it.

Dynamic quantization in PyTorch is CPU-only, so this module forces CPU
regardless of CUDA availability.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor, nn

from config import TransformerConfig
from inference.generation import generate
from inference.kv_cache import NaiveKVCache
from train_example import BOS_ID, EOS_ID, VOCAB_SIZE, make_default_config, train_model
from transformer import EncoderDecoderTransformer


def quantize_model(model: nn.Module, dtype: torch.dtype = torch.qint8) -> nn.Module:
    """Return a dynamically-quantized copy of `model`'s `nn.Linear` layers.

    Leaves everything else (embeddings, our hand-rolled `LayerNorm`, etc.)
    untouched and in fp32 — only `nn.Linear` weights are converted, with
    activations quantized on the fly at each call. The returned model is
    the same Python class as the input (quantization replaces submodules
    in place on a copy), so it's still usable anywhere the original was,
    including `generate()`.
    """
    model = model.to("cpu").eval()  # dynamic quantization is CPU-only in PyTorch
    return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=dtype)


def model_memory_bytes(model: nn.Module) -> int:
    """Measure a model's serialized state_dict size, in bytes.

    Serializing to an in-memory buffer (rather than trying to sum
    `parameter.numel() * element_size()` directly) sidesteps the internal
    representation of quantized layers, whose weights are stored as packed
    `torch.qint8` tensors behind a `_packed_params` wrapper rather than
    plain `nn.Parameter`s — this way fp32 and quantized models are measured
    the same way.
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes


@dataclass
class QuantizationBenchmarkResult:
    """FP32 vs quantized comparison on the sequence-reversal toy task."""

    fp32_memory_bytes: int
    quantized_memory_bytes: int
    fp32_mean_total_time_s: float
    quantized_mean_total_time_s: float
    fp32_mean_decode_time_s: float
    quantized_mean_decode_time_s: float

    @property
    def memory_reduction_ratio(self) -> float:
        """How many times smaller the quantized model's serialized size is."""
        return self.fp32_memory_bytes / self.quantized_memory_bytes

    def report(self) -> str:
        """Human-readable summary, for printing from a script."""
        lines = [
            "Quantization benchmark (fp32 vs dynamic int8, CPU, sequence-reversal task)",
            f"  memory (serialized state_dict): fp32={self.fp32_memory_bytes:,} B, "
            f"quantized={self.quantized_memory_bytes:,} B "
            f"({self.memory_reduction_ratio:.2f}x smaller)",
            f"  mean total generate() time:     fp32={self.fp32_mean_total_time_s * 1000:.2f} ms, "
            f"quantized={self.quantized_mean_total_time_s * 1000:.2f} ms",
            f"  mean per-token decode time:     fp32={self.fp32_mean_decode_time_s * 1000:.3f} ms, "
            f"quantized={self.quantized_mean_decode_time_s * 1000:.3f} ms",
        ]
        return "\n".join(lines)


def _time_generation_runs(
    model: EncoderDecoderTransformer, prompts: List[Tensor], max_new_tokens: int
) -> tuple[float, float]:
    """Run `generate()` once per prompt, return (mean total time, mean decode time)."""
    total_times, decode_times = [], []
    for prompt in prompts:
        result = generate(
            model,
            prompt,
            max_new_tokens=max_new_tokens,
            kv_cache_class=NaiveKVCache,
            eos_token_id=EOS_ID,
            decoder_start_token_id=BOS_ID,
        )
        total_times.append(result.total_time_s)
        decode_times.append(result.mean_decode_time_s)
    return sum(total_times) / len(total_times), sum(decode_times) / len(decode_times)


def benchmark_quantization(
    config: Optional[TransformerConfig] = None,
    num_train_steps: int = 800,
    num_prompts: int = 20,
    prompt_len: int = 8,
    max_new_tokens: int = 9,
    seed: int = 0,
) -> QuantizationBenchmarkResult:
    """Train the toy sequence-reversal model, quantize it, and compare fp32 vs int8.

    Reuses `train_example.train_model` so this benchmark exercises the same
    architecture/task as `train_example.py`, rather than a separate model.
    """
    torch.manual_seed(seed)
    device = torch.device("cpu")  # dynamic quantization requires CPU
    config = config or make_default_config()

    fp32_model = train_model(config, num_steps=num_train_steps, device=device, log_every=None)
    fp32_model.eval()
    quantized_model = quantize_model(fp32_model)

    prompts = [torch.randint(3, VOCAB_SIZE, (1, prompt_len), device=device) for _ in range(num_prompts)]

    fp32_total, fp32_decode = _time_generation_runs(fp32_model, prompts, max_new_tokens)
    quantized_total, quantized_decode = _time_generation_runs(quantized_model, prompts, max_new_tokens)

    return QuantizationBenchmarkResult(
        fp32_memory_bytes=model_memory_bytes(fp32_model),
        quantized_memory_bytes=model_memory_bytes(quantized_model),
        fp32_mean_total_time_s=fp32_total,
        quantized_mean_total_time_s=quantized_total,
        fp32_mean_decode_time_s=fp32_decode,
        quantized_mean_decode_time_s=quantized_decode,
    )


if __name__ == "__main__":
    result = benchmark_quantization()
    print(result.report())
