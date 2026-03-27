"""
Embedding Benchmark
===================
Compares PyTorch (sentence-transformers) vs OpenVINO IR inference for the
all-MiniLM-L6-v2 embedding model.

Measures:
    - Latency per batch (ms)
    - Throughput (samples / sec)
    - CPU utilization (%)
    - Peak RSS memory (MB)

Design notes:
    - Warmup iterations excluded from all reported numbers.
    - time.perf_counter() used for sub-millisecond accuracy.
    - Model load time is separated from inference time.
    - SystemMetricsSampler collects CPU/memory on a background thread
      so it does not perturb the timed inference loop.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.benchmark.system_metrics import SystemMetricsSampler, psutil_available


# ---------------------------------------------------------------------------
# Fixed benchmark corpus (same across all runs for fair comparison)
# ---------------------------------------------------------------------------
_BENCHMARK_TEXTS: List[str] = [
    "OpenVINO is a toolkit developed by Intel for optimizing deep learning inference.",
    "Retrieval-Augmented Generation combines document search with language model generation.",
    "FAISS is a library for efficient similarity search over dense vector collections.",
    "The all-MiniLM-L6-v2 model produces 384-dimensional embeddings from text.",
    "Neural network inference on CPUs benefits significantly from graph-level optimizations.",
    "Sentence embeddings encode the semantic meaning of entire phrases into fixed-length vectors.",
    "Intel NPUs accelerate AI workloads with lower power consumption than discrete GPUs.",
    "Batch inference amortizes model overhead and increases hardware utilization.",
    "Cosine similarity between normalized vectors is equivalent to their inner product.",
    "Quantization reduces model size and speeds up inference with minimal accuracy loss.",
    "Document chunking splits long texts into overlapping windows for embedding.",
    "Vector databases store and index high-dimensional embeddings for fast nearest-neighbor search.",
    "Transformer models apply self-attention to capture long-range dependencies in text.",
    "Mean pooling of token embeddings produces a single sentence-level representation.",
    "L2 normalization converts raw embeddings so that dot product equals cosine similarity.",
    "The OpenVINO IR format consists of an XML graph definition and a BIN weights file.",
]


def _build_corpus(n_texts: int) -> List[str]:
    """Repeat the base corpus to reach the requested number of texts."""
    base = _BENCHMARK_TEXTS
    repeats = (n_texts // len(base)) + 1
    return (base * repeats)[:n_texts]


# ---------------------------------------------------------------------------
# PyTorch baseline runner
# ---------------------------------------------------------------------------

def _run_pytorch(
    texts: List[str],
    batch_size: int,
    n_iterations: int,
    n_warmup: int,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Dict:
    """
    Benchmark the PyTorch / sentence-transformers encoder.

    Returns a result dict with timing and system metrics.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        return {"error": f"sentence-transformers not installed: {exc}"}

    # --- Load model (not timed as part of inference) ---
    load_start = time.perf_counter()
    model = SentenceTransformer(model_name, device="cpu")
    load_time_s = time.perf_counter() - load_start

    # --- Warmup ---
    for _ in range(n_warmup):
        model.encode(texts[:batch_size], batch_size=batch_size,
                     convert_to_numpy=True, show_progress_bar=False)

    # --- Timed iterations ---
    latencies: List[float] = []
    with SystemMetricsSampler(interval=0.05) as metrics:
        for _ in range(n_iterations):
            t0 = time.perf_counter()
            model.encode(texts, batch_size=batch_size,
                         convert_to_numpy=True, show_progress_bar=False)
            latencies.append(time.perf_counter() - t0)

    latencies_arr = np.array(latencies)
    n_texts = len(texts)

    return {
        "backend": "PyTorch CPU",
        "model": model_name,
        "n_texts": n_texts,
        "batch_size": batch_size,
        "n_iterations": n_iterations,
        "n_warmup": n_warmup,
        "load_time_s": round(load_time_s, 3),
        "avg_latency_ms": round(float(latencies_arr.mean() * 1000), 2),
        "min_latency_ms": round(float(latencies_arr.min() * 1000), 2),
        "max_latency_ms": round(float(latencies_arr.max() * 1000), 2),
        "std_latency_ms": round(float(latencies_arr.std() * 1000), 2),
        "throughput_sps": round(float(n_texts / latencies_arr.mean()), 1),
        "mean_cpu_percent": round(metrics.mean_cpu_percent, 1),
        "peak_cpu_percent": round(metrics.peak_cpu_percent, 1),
        "peak_rss_mb": round(metrics.peak_rss_mb, 1),
    }


# ---------------------------------------------------------------------------
# OpenVINO runner
# ---------------------------------------------------------------------------

def _run_openvino(
    texts: List[str],
    batch_size: int,
    n_iterations: int,
    n_warmup: int,
    model_xml: str,
    device: str = "CPU",
) -> Dict:
    """
    Benchmark the OpenVINO IR encoder.

    Measures both cold-start load time (compile from IR, no cached blob) and
    warm-start load time (load pre-compiled blob from ~/.deepsearch/cache/).
    Reports both values so the cache speedup is immediately visible.
    """
    if not Path(model_xml).exists():
        return {"error": f"OpenVINO IR model not found: {model_xml}"}

    try:
        from src.embeddings.openvino_encoder import OVEmbeddingEncoder, _CACHE_DIR, _get_cache_key
    except ImportError as exc:
        return {"error": f"OVEmbeddingEncoder not importable: {exc}"}

    # ------------------------------------------------------------------
    # Measure COLD load time (force recompile by removing cached blob)
    # ------------------------------------------------------------------
    cache_key  = _get_cache_key(str(Path(model_xml).resolve()), device)
    cache_path = _CACHE_DIR / f"{cache_key}.blob"

    # Remove blob so we always get a reproducible cold measurement
    if cache_path.exists():
        cache_path.unlink()

    cold_start = time.perf_counter()
    encoder_cold = OVEmbeddingEncoder(model_xml=model_xml, device=device)
    cold_load_s = round(time.perf_counter() - cold_start, 3)

    if encoder_cold._compiled_model is None:
        return {"error": "OpenVINO model failed to compile. Check model_xml path."}

    # ------------------------------------------------------------------
    # Measure WARM load time (blob now exists from the cold run above)
    # ------------------------------------------------------------------
    warm_load_s = None
    if cache_path.exists():
        warm_start = time.perf_counter()
        encoder_warm = OVEmbeddingEncoder(model_xml=model_xml, device=device)
        warm_load_s = round(time.perf_counter() - warm_start, 3)
        # Use the warm encoder for the inference benchmark
        encoder = encoder_warm
    else:
        # export_model not supported on this OV version — use cold encoder
        encoder = encoder_cold

    # ------------------------------------------------------------------
    # Inference benchmark (timed iterations, warmup excluded)
    # ------------------------------------------------------------------
    for _ in range(n_warmup):
        encoder.encode(texts[:batch_size], batch_size=batch_size)

    latencies: List[float] = []
    with SystemMetricsSampler(interval=0.05) as metrics:
        for _ in range(n_iterations):
            t0 = time.perf_counter()
            encoder.encode(texts, batch_size=batch_size)
            latencies.append(time.perf_counter() - t0)

    latencies_arr = np.array(latencies)
    n_texts = len(texts)

    result: Dict = {
        "backend": f"OpenVINO {device}",
        "model": model_xml,
        "device": device,
        "n_texts": n_texts,
        "batch_size": batch_size,
        "n_iterations": n_iterations,
        "n_warmup": n_warmup,
        "cold_load_s": cold_load_s,
        "warm_load_s": warm_load_s,
        # Keep load_time_s as the cold-start value for backward compatibility
        "load_time_s": cold_load_s,
        "avg_latency_ms": round(float(latencies_arr.mean() * 1000), 2),
        "min_latency_ms": round(float(latencies_arr.min() * 1000), 2),
        "max_latency_ms": round(float(latencies_arr.max() * 1000), 2),
        "std_latency_ms": round(float(latencies_arr.std() * 1000), 2),
        "throughput_sps": round(float(n_texts / latencies_arr.mean()), 1),
        "mean_cpu_percent": round(metrics.mean_cpu_percent, 1),
        "peak_cpu_percent": round(metrics.peak_cpu_percent, 1),
        "peak_rss_mb": round(metrics.peak_rss_mb, 1),
    }

    if warm_load_s is not None and warm_load_s > 0:
        result["cache_speedup"] = round(cold_load_s / warm_load_s, 2)

    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_embedding_benchmark(
    batch_size: int = 16,
    n_iterations: int = 20,
    n_warmup: int = 3,
    n_texts: int = 64,
    model_xml: Optional[str] = None,
    ov_device: str = "CPU",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    run_pytorch: bool = True,
    run_openvino: bool = True,
) -> Dict:
    """
    Run the full embedding benchmark and return structured results.

    Args:
        batch_size    : Texts per encoder forward pass.
        n_iterations  : Timed iterations (warmup excluded).
        n_warmup      : Warmup iterations (not counted).
        n_texts       : Total corpus size per iteration.
        model_xml     : Path to OpenVINO IR .xml. Required for OpenVINO run.
        ov_device     : OpenVINO device string (CPU, GPU, NPU).
        model_name    : HuggingFace model name for PyTorch baseline.
        run_pytorch   : Whether to include the PyTorch baseline.
        run_openvino  : Whether to include the OpenVINO comparison.

    Returns:
        Dict with keys "pytorch" and/or "openvino", each holding a result
        dict (or {"error": ...} if the run failed), plus "speedup" if both
        runs succeeded.
    """
    texts = _build_corpus(n_texts)
    output: Dict = {}

    if run_pytorch:
        output["pytorch"] = _run_pytorch(
            texts=texts,
            batch_size=batch_size,
            n_iterations=n_iterations,
            n_warmup=n_warmup,
            model_name=model_name,
        )

    if run_openvino and model_xml:
        output["openvino"] = _run_openvino(
            texts=texts,
            batch_size=batch_size,
            n_iterations=n_iterations,
            n_warmup=n_warmup,
            model_xml=model_xml,
            device=ov_device,
        )

    # Compute speedup if both runs are clean
    pt = output.get("pytorch", {})
    ov = output.get("openvino", {})
    if "avg_latency_ms" in pt and "avg_latency_ms" in ov and ov["avg_latency_ms"] > 0:
        output["speedup"] = round(pt["avg_latency_ms"] / ov["avg_latency_ms"], 2)

    return output


def run_cache_benchmark(
    model_xml: str,
    ov_device: str = "CPU",
) -> Dict:
    """
    Run a minimal OpenVINO benchmark to report cold and warm load times.

    Args:
        model_xml : Path to OpenVINO IR .xml.
        ov_device : OpenVINO device string (CPU, GPU, NPU).

    Returns:
        Dict with cold_load_s, warm_load_s, and cache_speedup.
    """
    # Use minimal parameters for inference as we only care about load times
    results = _run_openvino(
        texts=_build_corpus(1),
        batch_size=1,
        n_iterations=1,
        n_warmup=0,
        model_xml=model_xml,
        device=ov_device,
    )
    return {
        "cold_load_s": results.get("cold_load_s"),
        "warm_load_s": results.get("warm_load_s"),
        "cache_speedup": results.get("cache_speedup"),
        "error": results.get("error"),
    }


def print_embedding_results(results: Dict) -> None:
    """Render benchmark results to stdout in a readable report format."""
    sep = "-" * 50

    print()
    print("Embedding Benchmark Results")
    print(sep)

    pt = results.get("pytorch")
    ov = results.get("openvino")

    def _section(label: str, r: Dict) -> None:
        print(f"\n{label}:")
        if "error" in r:
            print(f"  Error: {r['error']}")
            return
        print(f"  Avg Latency  : {r['avg_latency_ms']} ms")
        print(f"  Min Latency  : {r['min_latency_ms']} ms")
        print(f"  Max Latency  : {r['max_latency_ms']} ms")
        print(f"  Throughput   : {r['throughput_sps']} samples/sec")
        if psutil_available():
            print(f"  Mean CPU     : {r['mean_cpu_percent']}%")
            print(f"  Peak RSS     : {r['peak_rss_mb']} MB")

        # Load time — show cold/warm breakdown for OV, plain value for PyTorch
        cold = r.get("cold_load_s")
        warm = r.get("warm_load_s")
        if cold is not None and warm is not None:
            cache_speedup = r.get("cache_speedup", "—")
            print(f"  Model load   : {cold} s  (cold — compile from IR)")
            print(f"  Cached load  : {warm} s  (warm — from blob)  [{cache_speedup}x faster]")
        elif cold is not None:
            print(f"  Model load   : {cold} s  (cold — blob export not supported)")
        else:
            load = r.get("load_time_s", "—")
            print(f"  Model load   : {load} s")

    if pt:
        n_texts = pt.get("n_texts", "?")
        batch_size = pt.get("batch_size", "?")
        n_iter = pt.get("n_iterations", "?")
        n_warm = pt.get("n_warmup", "?")
        print(f"\nCorpus size  : {n_texts} texts")
        print(f"Batch size   : {batch_size}")
        print(f"Iterations   : {n_iter}  (warmup: {n_warm})")
        _section("PyTorch CPU", pt)
    elif ov:
        n_texts = ov.get("n_texts", "?")
        batch_size = ov.get("batch_size", "?")
        n_iter = ov.get("n_iterations", "?")
        n_warm = ov.get("n_warmup", "?")
        print(f"\nCorpus size  : {n_texts} texts")
        print(f"Batch size   : {batch_size}")
        print(f"Iterations   : {n_iter}  (warmup: {n_warm})")

    if ov:
        _section(f"OpenVINO {ov.get('device', 'CPU')}", ov)

    speedup = results.get("speedup")
    if speedup is not None:
        print(f"\nSpeedup      : {speedup}x  (OpenVINO vs PyTorch)")

    print(f"\n{sep}")
