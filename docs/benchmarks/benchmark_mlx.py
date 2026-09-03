#!/usr/bin/env python3
"""Benchmark Talkie's MLX prefill and decode paths."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

import mlx.core as mx
import numpy as np

from talkie.mlx.generate import MLXTalkie


DEFAULT_PROMPT = (
    "Write a short note about radio, science, and daily life in the year 1930."
)


def repeated_tokens(talkie: MLXTalkie, text: str, count: int) -> list[int]:
    seed = talkie.tokenizer.encode(text)
    if not seed:
        raise ValueError("benchmark prompt produced no tokens")
    repeats = (count + len(seed) - 1) // len(seed)
    return (seed * repeats)[:count]


def greedy_token(logits: mx.array) -> int:
    return int(np.argmax(np.array(logits[0], dtype=np.float32)))


def run_case(
    talkie: MLXTalkie, prompt_ids: list[int], decode_tokens: int
) -> tuple[float, float]:
    start = time.perf_counter()
    logits, cache = talkie.model(mx.array([prompt_ids], dtype=mx.int32))
    mx.eval(logits, cache)
    token_id = greedy_token(logits)
    time_to_first_token = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(decode_tokens):
        logits, cache = talkie.model(
            mx.array([[token_id]], dtype=mx.int32), cache
        )
        mx.eval(logits, cache)
        token_id = greedy_token(logits)
    decode_seconds = time.perf_counter() - start
    return time_to_first_token, decode_seconds


def hardware_summary() -> dict[str, str]:
    result = subprocess.run(
        ["system_profiler", "SPHardwareDataType", "SPSoftwareDataType"],
        check=False,
        capture_output=True,
        text=True,
    )
    wanted = {
        "Model Name",
        "Model Identifier",
        "Chip",
        "Total Number of Cores",
        "Memory",
        "System Version",
    }
    summary: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.strip().partition(":")
        if separator and key in wanted:
            summary[key] = value.strip()
    return summary


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    return result.stdout.strip() or None


def git_worktree_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    start = time.perf_counter()
    talkie = MLXTalkie(args.model_dir)
    model_init_seconds = time.perf_counter() - start

    cases = [(32, 32), (512, 1)]
    prompts = {
        prompt_tokens: repeated_tokens(talkie, args.prompt, prompt_tokens)
        for prompt_tokens, _ in cases
    }

    for prompt_tokens, decode_tokens in cases:
        run_case(talkie, prompts[prompt_tokens], decode_tokens)

    mx.reset_peak_memory()
    results = []
    for prompt_tokens, decode_tokens in cases:
        ttft_samples = []
        decode_samples = []
        for _ in range(args.runs):
            ttft, decode_seconds = run_case(
                talkie, prompts[prompt_tokens], decode_tokens
            )
            ttft_samples.append(ttft)
            decode_samples.append(decode_seconds)
        median_ttft = statistics.median(ttft_samples)
        median_decode = statistics.median(decode_samples)
        results.append(
            {
                "prompt_tokens": prompt_tokens,
                "decode_tokens": decode_tokens,
                "runs": args.runs,
                "median_time_to_first_token_seconds": median_ttft,
                "median_prefill_tokens_per_second": prompt_tokens / median_ttft,
                "median_decode_seconds": median_decode,
                "median_decode_tokens_per_second": decode_tokens / median_decode,
                "time_to_first_token_samples_seconds": ttft_samples,
                "decode_samples_seconds": decode_samples,
            }
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": "mlx",
        "model_snapshot": args.model_dir.name,
        "model_config": talkie.model.config.to_dict(),
        "model_init_seconds": model_init_seconds,
        "peak_metal_memory_gib": mx.get_peak_memory() / (1024**3),
        "mlx_version": mx.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hardware": hardware_summary(),
        "talkie_revision": git_revision(),
        "talkie_worktree_dirty": git_worktree_dirty(),
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
