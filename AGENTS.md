# AGENTS.md

This file provides guidance to Ai coding tools when working with code in this repository.

## What this is

`talkie` is an inference-only library + CLI for the Talkie 13B language model family
(custom decoder-only transformer, weights hosted on HuggingFace). There is no training
code here — only checkpoint loading, tokenization, sampling, and generation.

Model variants are registered in `src/talkie/config.py` (`MODELS`). Two "styles":
`base` (raw completion) and `it` (instruction-tuned, uses a chat template).

## Commands

```bash
uv sync                                  # install (pulls CUDA torch — see note below)
uv run talkie list                       # list registered models
uv run talkie generate "..." -m talkie-1930-13b-base -t 0.8 -n 256
uv run talkie chat -m talkie-1930-13b-it # interactive REPL (Ctrl-D to quit)
uv run talkie download all               # pre-fetch checkpoints from HF
```

Running anything that loads a model needs a CUDA GPU with ≥28 GB VRAM (bf16) and
downloads 26–50 GB per model from HuggingFace on first use. `device="cpu"` is accepted
but not practically usable at this size.

### Tests

```bash
uv run pytest                 # Torch, tokenizer, download, sampling, and chat tests
uv run --extra mlx pytest     # also exercises Torch↔MLX parity and KV caching
```

The tests use a randomly initialized tiny model and never download checkpoints.
MLX tests skip at collection time when MLX or a Metal device is unavailable.

Compatibility floors are intentional and have focused regression coverage:

- `tiktoken>=0.14.0` is covered by base/IT round-trip and special-token tests in
  `tests/test_tokenizer.py`;
- `huggingface-hub>=1.16.1` retains compatibility with the PyTorch fork's Spin
  0.18 / Click <8.4 environment and is covered by `tests/test_download.py`; and
- the MLX extra uses `safetensors>=0.8.0`, covered by the checkpoint-loading path
  in `tests/test_mlx_parity.py`.

The local PyTorch fork checkout is `/Users/eeaglstun/Documents/dev/pytorch/`.
Its `.venv` is the compatibility environment for validating these dependency
floors; it is not required by CUDA or MLX users.

## Running on Apple Silicon (Mac)

This branch (`mlx-backend`, PR #2) adds an **MLX backend** under `src/talkie/mlx/` with a
separate `talkie-mlx` CLI — no CUDA required. The torch pin is scoped to Linux here (see
below), so `uv sync --extra mlx` resolves a normal macOS MPS wheel. Run against converted
MLX weights:

```bash
uv sync --extra mlx
uv run talkie-mlx --model-dir <converted-mlx-dir> --stream "..."
```

Two ways this model is served on the Mac (full detail in the `talkie-mlx-mac-setup` and
`talkie-on-ollama-arch` Claude memories):

- **Standalone MLX** — the `talkie-mlx` CLI above. Fast on Apple Silicon; reloads weights
  per CLI invocation, so keep a Python session open (`MLXTalkie`) for back-and-forth.
- **Native Ollama** — a custom `TalkieForCausalLM` architecture was ported into an Ollama
  fork (`~/Documents/dev/ollama-v0.30.2`, `x/models/talkie/`) so Ollama serves it as
  `talkie-1930`. The `src/talkie/mlx/model.py` here is the reference that port mirrors.

## Architecture

Request flow: `cli.py` / public API → `generate.Talkie` (the one class users touch) →
`download.get_model_files` (HF) → `tokenizer.build_tokenizer` + `model.load_checkpoint`
→ generation loop calls `model.sample_batch` which uses `sampling.apply_top_k_top_p`.

Key files:

- `generate.py` — `Talkie` orchestrator. `stream`/`generate` (any model), `chat`/
  `chat_stream`/`batch_generate` (IT only, guarded by `_require_it`). IT prompts are
  auto-wrapped in the chat template; base prompts pass through raw.
- `model.py` — the transformer (`TalkieModel`) + `load_checkpoint`.
- `config.py` — model **registry** (`MODELS`, `ModelSpec`). Note: unrelated to the
  architecture config — see naming gotcha below.
- `tokenizer.py`, `chat.py`, `sampling.py`, `download.py` — supporting pieces.

### Things that will bite you if you don't know them

- **Two unrelated "configs."** `model.GPTConfig` is the _architecture_ (layers, heads,
  dims). `config.ModelSpec` / `MODELS` is the _registry_ (HF repo ids, filenames,
  style). They share a word and nothing else.
- **No KV cache.** The generation loop re-runs a full forward over the entire sequence
  every step, and `TalkieModel.forward` returns logits for the **last position only**.
  Generation is O(n²) by design — keep this in mind before "optimizing" the forward
  pass or trying to read all-position logits out of it.
- **Sampling is Gumbel-max, not multinomial.** `sample_batch` adds Gumbel noise to
  logits and takes `argmax` (`sampling.sample_gumbel`). top-k/top-p filter to `-inf`
  _before_ the noise is added.
- **Non-standard transformer.** Beyond the usual RoPE + SwiGLU + RMSNorm: there's an
  **embedding skip connection** (the normalized embedding `e_x` is re-added in every
  block via `embed_skip`, an `ActGain` initialized to 0), per-head gain on queries
  (`HeadGain`), RMSNorm applied to q and k, a weight-gain on the (untied, raw
  `nn.Parameter`) `lm_head`, and RoPE base = 1e6. Don't "simplify" these away — they're
  load-bearing for checkpoint compatibility.
- **IT vocab resize.** Base vocab is 65536; the IT style adds 4 special tokens
  (`IT_VOCAB_SIZE = 65540`). `load_checkpoint` will grow embeddings/`lm_head` if the
  checkpoint is smaller than the target (`resize_model_embeddings`) — usually a no-op
  since IT checkpoints already ship resized, but the path exists.
- **IT streaming buffers output** to strip leaked chat-template markers
  (`chat.truncate_at_stop` + `STOP_WINDOW`); base streaming yields each token directly.
- **`finish_reason` is a heuristic** in `stream`/`generate`: `"stop"` if fewer than
  `max_tokens` were produced, else `"length"`. `batch_generate` tracks it properly.
- **`load_checkpoint` builds on CPU, casts to bf16, then moves to GPU** to avoid a
  transient 2× fp32 memory spike. Don't reorder this. It also strips `_orig_mod.`
  prefixes left by `torch.compile`.

### torch / CUDA pinning

`pyproject.toml` pins torch to the `pytorch-cu128` index (`[tool.uv.sources]`), so
`uv sync` fetches a CUDA 12.8 build regardless of host. Changing CUDA version means
editing that index URL.
