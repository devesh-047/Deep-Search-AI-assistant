"""
LLM Benchmark
=============
Measures inference speed for the configured LLM backend.

Supports:
    - OpenVINO GenAI / optimum-intel (OVLLMClient)
    - Ollama (OllamaClient) as fallback

Measures:
    - Average generation time (seconds)
    - Tokens per second (output tokens / elapsed time)
    - Peak RSS memory during generation (MB)
    - Device used

Design notes:
    - A fixed prompt and fixed max_tokens are used across all runs.
    - Warmup run is excluded from reported numbers.
    - time.perf_counter() used for sub-millisecond accuracy.
    - Token count is derived from whitespace splitting of the output,
      which gives a conservative approximation without requiring a
      tokenizer round-trip inside the benchmark loop.
    - If the OpenVINO LLM model is not configured, the benchmark falls
      back to Ollama and reports accordingly.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.benchmark.system_metrics import SystemMetricsSampler, psutil_available, current_rss_mb


# ---------------------------------------------------------------------------
# Default benchmark parameters
# ---------------------------------------------------------------------------
DEFAULT_PROMPT = (
    "Explain in detail how Retrieval-Augmented Generation works and "
    "what advantages it provides over standard language model inference."
)
DEFAULT_MAX_TOKENS = 100
DEFAULT_N_ITERATIONS = 5
DEFAULT_N_WARMUP = 1


# ---------------------------------------------------------------------------
# OpenVINO LLM runner
# ---------------------------------------------------------------------------

def _run_openvino_llm(
    prompt: str,
    max_tokens: int,
    n_iterations: int,
    n_warmup: int,
    model_dir: str,
    device: str,
) -> Dict:
    """Run and time OVLLMClient.generate() calls."""
    if not Path(model_dir).exists():
        return {"error": f"OpenVINO LLM model directory not found: {model_dir}"}

    try:
        from src.llm.openvino_llm import OVLLMClient
    except ImportError as exc:
        return {"error": f"OVLLMClient import failed: {exc}"}

    load_start = time.perf_counter()
    client = OVLLMClient(model_dir=model_dir, device=device)
    load_time_s = time.perf_counter() - load_start

    if not client.is_available():
        return {"error": "OVLLMClient.is_available() returned False after load."}

    # Warmup
    for _ in range(n_warmup):
        client.generate(question=prompt, max_tokens=max_tokens, temperature=0.0)

    # Timed iterations
    times: List[float] = []
    token_counts: List[int] = []
    memory_before = current_rss_mb()

    with SystemMetricsSampler(interval=0.1) as metrics:
        for _ in range(n_iterations):
            t0 = time.perf_counter()
            output = client.generate(question=prompt, max_tokens=max_tokens, temperature=0.0)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            token_counts.append(len(output.split()))

    times_arr = np.array(times)
    tokens_arr = np.array(token_counts)
    memory_after = metrics.peak_rss_mb

    return {
        "backend": f"OpenVINO ({client._backend})",
        "device": device,
        "model_dir": model_dir,
        "prompt_words": len(prompt.split()),
        "max_tokens": max_tokens,
        "n_iterations": n_iterations,
        "n_warmup": n_warmup,
        "load_time_s": round(load_time_s, 3),
        "avg_generation_s": round(float(times_arr.mean()), 3),
        "min_generation_s": round(float(times_arr.min()), 3),
        "max_generation_s": round(float(times_arr.max()), 3),
        "avg_output_tokens": round(float(tokens_arr.mean()), 1),
        "tokens_per_sec": round(float(tokens_arr.mean() / times_arr.mean()), 2),
        "mean_cpu_percent": round(metrics.mean_cpu_percent, 1),
        "peak_cpu_percent": round(metrics.peak_cpu_percent, 1),
        "peak_rss_mb": round(memory_after, 1),
        "memory_delta_mb": round(memory_after - memory_before, 1),
    }


# ---------------------------------------------------------------------------
# Ollama LLM runner
# ---------------------------------------------------------------------------

def _run_ollama_llm(
    prompt: str,
    max_tokens: int,
    n_iterations: int,
    n_warmup: int,
    model: str = "mistral",
    endpoint: str = "http://localhost:11434",
) -> Dict:
    """Run and time OllamaClient.generate() calls."""
    try:
        from src.llm.ollama_client import OllamaClient
    except ImportError as exc:
        return {"error": f"OllamaClient import failed: {exc}"}

    client = OllamaClient(base_url=endpoint, model=model)
    if not client.is_available():
        return {"error": f"Ollama not reachable at {endpoint} or model '{model}' not pulled."}

    # Warmup
    for _ in range(n_warmup):
        client.generate(question=prompt, max_tokens=max_tokens, temperature=0.0)

    # Timed iterations
    times: List[float] = []
    token_counts: List[int] = []
    memory_before = current_rss_mb()

    with SystemMetricsSampler(interval=0.1) as metrics:
        for _ in range(n_iterations):
            t0 = time.perf_counter()
            output = client.generate(question=prompt, max_tokens=max_tokens, temperature=0.0)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            token_counts.append(len(output.split()))

    times_arr = np.array(times)
    tokens_arr = np.array(token_counts)
    memory_after = metrics.peak_rss_mb

    return {
        "backend": "Ollama",
        "device": "CPU (via Ollama)",
        "model": model,
        "prompt_words": len(prompt.split()),
        "max_tokens": max_tokens,
        "n_iterations": n_iterations,
        "n_warmup": n_warmup,
        "avg_generation_s": round(float(times_arr.mean()), 3),
        "min_generation_s": round(float(times_arr.min()), 3),
        "max_generation_s": round(float(times_arr.max()), 3),
        "avg_output_tokens": round(float(tokens_arr.mean()), 1),
        "tokens_per_sec": round(float(tokens_arr.mean() / times_arr.mean()), 2),
        "mean_cpu_percent": round(metrics.mean_cpu_percent, 1),
        "peak_cpu_percent": round(metrics.peak_cpu_percent, 1),
        "peak_rss_mb": round(memory_after, 1),
        "memory_delta_mb": round(memory_after - memory_before, 1),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_llm_benchmark(
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    n_iterations: int = DEFAULT_N_ITERATIONS,
    n_warmup: int = DEFAULT_N_WARMUP,
    model_dir: Optional[str] = None,
    ov_device: str = "CPU",
    ollama_model: str = "mistral",
    ollama_endpoint: str = "http://localhost:11434",
    prefer_openvino: bool = True,
) -> Dict:
    """
    Run the LLM benchmark and return structured results.

    Tries OpenVINO LLM first (if model_dir supplied and prefer_openvino=True),
    then falls back to Ollama.

    Args:
        prompt           : Fixed benchmark prompt.
        max_tokens       : Maximum output tokens per generation call.
        n_iterations     : Timed iterations (warmup excluded).
        n_warmup         : Warmup iterations (not counted).
        model_dir        : Path to OpenVINO IR LLM directory.
        ov_device        : OpenVINO device (CPU, GPU, NPU).
        ollama_model     : Ollama model name.
        ollama_endpoint  : Ollama server URL.
        prefer_openvino  : If True, attempt OpenVINO before Ollama.

    Returns:
        Dict with key "result" holding the benchmark result dict.
    """
    output: Dict = {}

    if prefer_openvino and model_dir and Path(model_dir).exists():
        result = _run_openvino_llm(
            prompt=prompt,
            max_tokens=max_tokens,
            n_iterations=n_iterations,
            n_warmup=n_warmup,
            model_dir=model_dir,
            device=ov_device,
        )
        if "error" not in result:
            output["result"] = result
            return output
        output["openvino_error"] = result["error"]

    # Ollama fallback
    result = _run_ollama_llm(
        prompt=prompt,
        max_tokens=max_tokens,
        n_iterations=n_iterations,
        n_warmup=n_warmup,
        model=ollama_model,
        endpoint=ollama_endpoint,
    )
    output["result"] = result
    return output


def print_llm_results(results: Dict) -> None:
    """Render benchmark results to stdout in a readable report format."""
    sep = "-" * 50

    print()
    print("LLM Benchmark Results")
    print(sep)

    if "openvino_error" in results:
        print(f"\nOpenVINO LLM skipped: {results['openvino_error']}")
        print("Falling back to Ollama.\n")

    r = results.get("result", {})
    if not r:
        print("No results available.")
        return

    if "error" in r:
        print(f"\nError: {r['error']}")
        print(sep)
        return

    print(f"\nBackend      : {r.get('backend', 'unknown')}")
    print(f"Device       : {r.get('device', 'unknown')}")
    model_id = r.get("model_dir") or r.get("model", "unknown")
    print(f"Model        : {model_id}")
    print(f"Prompt words : {r.get('prompt_words', '?')}")
    print(f"Max tokens   : {r.get('max_tokens', '?')}")
    print(f"Iterations   : {r.get('n_iterations', '?')}  (warmup: {r.get('n_warmup', '?')})")
    print()
    print(f"Avg Gen Time : {r['avg_generation_s']} s")
    print(f"Min Gen Time : {r['min_generation_s']} s")
    print(f"Max Gen Time : {r['max_generation_s']} s")
    print(f"Avg Output   : {r['avg_output_tokens']} tokens")
    print(f"Tokens/sec   : {r['tokens_per_sec']}")
    if psutil_available():
        print(f"Mean CPU     : {r['mean_cpu_percent']}%")
        print(f"Peak RSS     : {r['peak_rss_mb']} MB")

    if r.get("load_time_s") is not None:
        print(f"Model load   : {r['load_time_s']} s")

    print(f"\n{sep}")
