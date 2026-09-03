# Benchmarks

Talkie benchmarks separate prompt processing from autoregressive decoding. A
single end-to-end tokens-per-second figure hides the distinction: long prompts
are prefill-bound, while generation is decode-bound.

## Current results

Measured with the `zimengxiong/talkie-1930-13b-it-mlx` bfloat16 checkpoint:

| Hardware            |                 Shape | Median TTFT |      Prefill |     Decode |
| ------------------- | --------------------: | ----------: | -----------: | ---------: |
| Apple M4 Max, 64 GB | 32 prompt / 32 decode |      387 ms |  82.73 tok/s | 6.98 tok/s |
| Apple M4 Max, 64 GB | 512 prompt / 1 decode |     2.373 s | 215.79 tok/s |          — |

Peak Metal allocator memory was 26.16 GiB. The benchmark used MLX 0.32.2,
Python 3.14.2, and macOS 26.5.2. See
[`mlx-m4-max.json`](mlx-m4-max.json) for the raw samples, full model
configuration, checkpoint snapshot, and Talkie revision. The benchmark started
from a clean worktree at commit `912da94`.

## Methodology

The MLX harness reports:

- model object and checkpoint-mapping initialization time;
- time to first token, including prefill and greedy selection;
- prefill throughput, derived from prompt length and time to first token;
- steady-state greedy decode throughput after the first token; and
- peak Metal allocator memory after model initialization.

Each shape is warmed once before measurement. Published latency and throughput
values are medians of three runs. The default cases isolate two regimes:

|     Prompt | Measured decode | Purpose                              |
| ---------: | --------------: | ------------------------------------ |
|  32 tokens |       32 tokens | Interactive, decode-bound generation |
| 512 tokens |         1 token | Longer-context prefill               |

The harness operates directly on token IDs so tokenizer and terminal output do
not distort model timing. Decode includes the backend's current full-vocabulary
logit transfer and greedy NumPy selection, matching the implementation used by
`MLXTalkie` when temperature is zero.

## Reproduce

Run from the repository root with the MLX extra installed:

```bash
uv run --extra mlx python docs/benchmarks/benchmark_mlx.py \
  --model-dir /path/to/talkie-1930-13b-it-mlx \
  --output docs/benchmarks/mlx-m4-max.json
```

Do not compare results unless the model, dtype, prompt/decode lengths, hardware,
OS, Talkie revision, and backend version match. Run on a warm, otherwise idle
machine. Initialization is reported separately and is not included in
throughput. MLX loads lazily, so this setup time should not be interpreted as
the time required to materialize every model weight.

## Scope

These benchmarks measure inference performance, not model quality. Perplexity,
quantization quality, and historical-behavior evaluations belong in separate
evaluation suites. CUDA and PyTorch MPS results should be added only when the
same checkpoint, token shapes, sampling policy, and synchronization boundaries
can be reproduced on those backends.
