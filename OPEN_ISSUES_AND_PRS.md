# Open Issues & PRs — talkie-lm/talkie

Snapshot taken 2026-06-06. 2 open issues, 7 open PRs.

## Open Issues

### #10 — Will the dataset be made available

- **Author:** kirawi · opened 2026-05-03 · 4 comments
- Request to release the pre-1931 training dataset for experimentation. Discussion in
  comments clarifies the _model_ is already on HuggingFace (claims knowledge cutoff
  ~1939); the ask is specifically the **datasets**. A commenter cites a researcher
  saying they'd publish the dataset or the build scripts once data quality improves.

### #12 — Value contamination through post-training: anachronistic evaluations on Church-State relations

- **Author:** PaulKercadiou · opened 2026-05-07 · 0 comments
- Research-style report (paper: doi.org/10.5281/zenodo.20070239). Argues that the DPO
  post-training injects **modern evaluative dispositions** even where pre-1930 training
  data would produce different ones — distinct from the temporal/factual leakage the
  blog post already discusses. Test case: the model gives period-authentic answers in
  general, but a modern (1960s+) liberal framing on Church reconciliation with the
  principles of 1789. Framed as a constructive contribution, not a bug report.

---

## Open PRs

A cluster of these overlap around two themes: **(a) reducing load/runtime memory so the
model runs on consumer GPUs / macOS**, and **(b) inference features** (KV cache, int8,
seeding, MLX, benchmarking). Worth triaging together.

### Memory / install footprint (overlapping — pick an approach)

| PR  | Title                                                    | Author     | Stats           | State     | Approach                                                                                                                                                                                                                                   |
| --- | -------------------------------------------------------- | ---------- | --------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| #5  | Reduce system memory by mmapping weights + earlier bf16  | richardxia | 1 file +3/-2    | mergeable | Reduces overhead in the existing CPU load/copy/move path (mmap + cast-to-bf16-sooner). Minimal patch.                                                                                                                                      |
| #6  | Reduce checkpoint load peak memory; enable macOS install | metrovoc   | 2 files +36/-21 | mergeable | Different approach: build on `meta`, `load_state_dict(assign=True)`, rebuild RoPE buffers after, preserve dtype on resize. **Also restricts the `pytorch-cu128` uv source to Linux so macOS resolves a default MPS wheel.** Tested on MPS. |

> #5 and #6 solve the same problem two ways. #6 is the more thorough rewrite and also
> unblocks macOS/MPS install. #5 is a 3-line band-aid on the current path.

### Inference features

| PR  | Title                                          | Author      | Stats            | State           | Notes                                                                                                                                                                                                                                                                                                    |
| --- | ---------------------------------------------- | ----------- | ---------------- | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| #7  | Add KV cache for autoregressive generation     | alexop1000  | 2 files +194/-53 | mergeable       | **Biggest perf win.** Decode one token/step with cached K/V instead of re-running the full sequence (the O(n²) issue noted in CLAUDE.md). Claims ~30–100× faster. New `KVCache`, prefill+decode rewrite of `_stream_raw`/`batch_generate`. bf16-exact equivalence smoke test. Independent of loader PRs. |
| #8  | Add INT8 weight-only quantization (closes #4)  | alexop1000  | 4 files +186/-18 | **CONFLICTING** | Pure-PyTorch int8 (no bitsandbytes dep), ~26 GB → ~14 GB, opt-in `quantize="int8"`. Reports near-lossless: ppl −0.03%, top-1 99%, top-5 100%. **Has merge conflicts — needs rebase.**                                                                                                                    |
| #9  | Add perplexity eval + bf16-vs-int8 benchmark   | alexop1000  | 4 files +276/-5  | mergeable       | **Depends on #8** (benchmark CLI uses int8). Adds `Talkie.perplexity()`, `talkie benchmark` CLI, exposes `max_seq_len` constructor param, `return_all_logits`/`last_k` on forward. Includes a context-length sweep suggesting ~2k trained position, cliff past ~5k.                                      |
| #3  | Add seed parameter for reproducible generation | mvanhorn    | 5 files +90/-8   | mergeable       | Threads `seed`/`torch.Generator` through `sample_gumbel` → sampling → `generate/stream/chat/batch_generate` + `--seed` CLI. Motivated by the README's "controlled comparison" goal. **Adds the repo's first test** (`tests/test_seed.py`, CPU-only).                                                     |
| #2  | Add Apple MLX inference backend                | ZimengXiong | 7 files +551/-0  | mergeable       | Optional MLX backend + `talkie-mlx` CLI + checkpoint→MLX safetensors converter for Apple Silicon. Tested full 13B on M4 Max. MLX weights published separately on HF.                                                                                                                                     |

### Suggested triage order

1. **#7 (KV cache)** — largest, independent perf win; lands cleanly.
2. **#6 vs #5** — decide one memory approach (#6 also fixes macOS install).
3. **#3 (seed)** — small, high-value for the project's stated comparison purpose; brings a test harness.
4. **#8 → #9** — merge int8 first (after rebasing the conflict), then the benchmark that depends on it.
5. **#2 (MLX)** — large additive backend; review on its own track.
