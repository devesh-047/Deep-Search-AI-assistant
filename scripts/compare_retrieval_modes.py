#!/usr/bin/env python3
"""
compare_retrieval_modes.py
===========================
Compares the original RAG retrieval (pure vector search) against
metadata-aware staged retrieval across multiple queries.

Metrics measured:
  - Retrieval latency (ms)
  - Vector comparison count (estimated)
  - Top-k precision (heuristic overlap check)
  - LLM response latency (if LLM is available)
  - Peak memory usage (via psutil)
  - Context token count (approximate)

Usage:
  python scripts/compare_retrieval_modes.py
  python scripts/compare_retrieval_modes.py --top-k 5 --verbose
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Resolve project root and ensure src is importable.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_DIR = PROCESSED_DIR / "faiss"

# ---------------------------------------------------------------------------
# Optional dependency: psutil for memory measurements.
# ---------------------------------------------------------------------------
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def get_memory_mb() -> float:
    """Return current process RSS in MB (0 if psutil unavailable)."""
    if not PSUTIL_AVAILABLE:
        return 0.0
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def approx_token_count(text: str) -> int:
    """Rough token estimate (~4 chars/token for English)."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Representative test queries.
# Mix of queries with and without metadata hints.
# ---------------------------------------------------------------------------
TEST_QUERIES: List[Dict[str, Any]] = [
    {
        "query": "What is the total amount on the invoice?",
        "has_metadata_hints": False,
        "description": "Generic document question (no metadata hint)",
    },
    {
        "query": "Find the invoice from 2024 with the purchase order number",
        "has_metadata_hints": True,
        "description": "Year-constrained query (2024)",
    },
    {
        "query": "Show me results from presentation slides about project status",
        "has_metadata_hints": True,
        "description": "File-type hint (slides/presentation)",
    },
    {
        "query": "Find all video clips about cooking recipes",
        "has_metadata_hints": True,
        "description": "Modality hint (video)",
    },
    {
        "query": "What is the employee name on the form?",
        "has_metadata_hints": False,
        "description": "Generic form question (no metadata hint)",
    },
    {
        "query": "Search images for handwritten signatures from 2023",
        "has_metadata_hints": True,
        "description": "Modality + year hint (image, 2023)",
    },
    {
        "query": "Find the report about quarterly earnings",
        "has_metadata_hints": True,
        "description": "File-type hint (report → pdf)",
    },
]


# ---------------------------------------------------------------------------
# Benchmark runner.
# ---------------------------------------------------------------------------

class RetrievalBenchmark:
    """
    Runs retrieval benchmarks for both modes and collects metrics.
    """

    def __init__(self, top_k: int = 5, verbose: bool = False):
        self.top_k = top_k
        self.verbose = verbose
        self._encoder = None
        self._index = None
        self._retriever = None
        self._staged_retriever = None
        self._llm = None

    def setup(self) -> bool:
        """
        Load encoder, index, and create retrievers.
        Returns True if setup succeeded.
        """
        print("=" * 70)
        print("  Retrieval Mode Comparison Benchmark")
        print("=" * 70)
        print()

        # --- Load encoder ---
        print("  Loading embedding encoder...")
        try:
            from src.embeddings.encoder import EmbeddingEncoder
            self._encoder = EmbeddingEncoder(device="cpu")
            print(f"    Encoder loaded (dim={self._encoder.dimension})")
        except Exception as e:
            print(f"    ERROR: Could not load encoder: {e}")
            return False

        # --- Load FAISS index ---
        print("  Loading FAISS index...")
        try:
            from src.index.faiss_index import FaissIndex
            self._index = FaissIndex()
            self._index.load(str(INDEX_DIR))
            print(f"    Index loaded ({self._index.size} vectors)")
        except FileNotFoundError:
            print(f"    ERROR: Index not found at {INDEX_DIR}")
            print("    Run:  python cli.py ingest --dataset <name>  first.")
            return False
        except Exception as e:
            print(f"    ERROR: {e}")
            return False

        # --- Create original retriever ---
        from src.retrieval.retriever import Retriever
        self._retriever = Retriever(encoder=self._encoder, index=self._index)

        # --- Create staged retriever ---
        from src.retrieval.metadata_filter import (
            MetadataStore,
            StagedRetriever,
            QueryMetadataParser,
        )
        metadata_store = MetadataStore(self._index.metadata)
        self._staged_retriever = StagedRetriever(
            encoder=self._encoder,
            index=self._index,
            metadata_store=metadata_store,
            parser=QueryMetadataParser(),
        )
        print(f"    Metadata store built ({metadata_store.size} entries)")

        # --- Try to load LLM (optional) ---
        print("  Checking LLM availability...")
        try:
            from src.llm.ollama_client import OllamaClient
            client = OllamaClient()
            if client.is_available():
                self._llm = client
                print(f"    LLM available (model: {client.model})")
            else:
                print("    LLM not available — skipping LLM latency measurement")
        except Exception:
            print("    LLM not available — skipping LLM latency measurement")

        print()
        return True

    def run_original(self, query: str) -> Dict[str, Any]:
        """Run a query through the original retrieval path and measure metrics."""
        mem_before = get_memory_mb()
        t_start = time.perf_counter()

        results = self._retriever.query(query, top_k=self.top_k)
        t_retrieval = (time.perf_counter() - t_start) * 1000

        mem_after = get_memory_mb()
        context = self._retriever.format_context(results, max_chars=3000)
        context_tokens = approx_token_count(context)

        # LLM latency (optional)
        llm_latency = 0.0
        if self._llm is not None:
            t_llm_start = time.perf_counter()
            try:
                self._llm.generate(question=query, context=context)
                llm_latency = (time.perf_counter() - t_llm_start) * 1000
            except Exception:
                llm_latency = -1.0  # indicates error

        return {
            "retrieval_latency_ms": t_retrieval,
            "results_count": len(results),
            "vector_comparisons": self._index.size,  # flat = all vectors
            "context_tokens": context_tokens,
            "peak_memory_mb": max(mem_before, mem_after),
            "llm_latency_ms": llm_latency,
            "result_scores": [r.score for r in results],
            "result_chunk_ids": [r.chunk_id for r in results],
        }

    def run_staged(self, query: str) -> Dict[str, Any]:
        """Run a query through staged retrieval and measure metrics."""
        mem_before = get_memory_mb()
        # Use an external wall-clock timer (same methodology as run_original)
        # so both modes are measured on an equal footing.
        t_start = time.perf_counter()

        raw_results, stats = self._staged_retriever.query(query, top_k=self.top_k)
        t_retrieval = (time.perf_counter() - t_start) * 1000

        mem_after = get_memory_mb()

        # Format context
        results = self._retriever._wrap_results(raw_results)
        context = self._retriever.format_context(results, max_chars=3000)
        context_tokens = approx_token_count(context)

        # Effective vector comparisons = candidates after filter (or full if fallback)
        candidates = stats.get("candidates_after_filter", self._index.size)
        if stats.get("fallback_to_full"):
            candidates = self._index.size

        # LLM latency (optional)
        llm_latency = 0.0
        if self._llm is not None:
            t_llm_start = time.perf_counter()
            try:
                self._llm.generate(question=query, context=context)
                llm_latency = (time.perf_counter() - t_llm_start) * 1000
            except Exception:
                llm_latency = -1.0

        return {
            "retrieval_latency_ms": t_retrieval,
            "results_count": len(results),
            "vector_comparisons": candidates,
            "context_tokens": context_tokens,
            "peak_memory_mb": max(mem_before, mem_after),
            "llm_latency_ms": llm_latency,
            "result_scores": [r.score for r in results],
            "result_chunk_ids": [r.chunk_id for r in results],
            "constraints": stats.get("constraints_detected", {}),
            "used_prefilter": stats.get("used_prefilter", False),
            "fallback_to_full": stats.get("fallback_to_full", False),
            "filter_latency_ms": stats.get("filter_latency_ms", 0.0),
            "encoding_latency_ms": stats.get("encoding_latency_ms", 0.0),
            "search_latency_ms": stats.get("search_latency_ms", 0.0),
        }

    def compute_precision_overlap(
        self,
        original_ids: List[str],
        staged_ids: List[str],
    ) -> float:
        """
        Heuristic precision: fraction of staged results that also
        appear in the original top-k.  1.0 = identical result sets.
        """
        if not staged_ids:
            return 0.0
        overlap = set(original_ids) & set(staged_ids)
        return len(overlap) / len(staged_ids)

    def _warmup(self) -> None:
        """
        Run a throwaway query through both modes to warm up the encoder (JIT
        compile, CPU cache).  Without this step the *first* query unfairly
        penalises whichever mode runs first, producing misleading speedup
        numbers.
        """
        _dummy = "warmup query for encoder initialisation"
        try:
            self.run_original(_dummy)
            self.run_staged(_dummy)
        except Exception:
            pass  # warmup failure is non-fatal

    def run_all(self, queries: Optional[List[Dict]] = None) -> List[Dict]:
        """Run all queries through both modes and collect comparison data."""
        if queries is None:
            queries = TEST_QUERIES

        # Warm up the sentence-transformers encoder before timed runs so that
        # JIT compilation and CPU cache effects don't skew latency numbers.
        print("  Warming up encoder (1 dummy query each mode)...")
        self._warmup()
        print("  Warmup complete.\n")

        all_results: List[Dict] = []

        for i, qdata in enumerate(queries, 1):
            query = qdata["query"]
            desc = qdata.get("description", "")
            has_hints = qdata.get("has_metadata_hints", False)

            print(f"  Query {i}/{len(queries)}: {query[:60]}...")
            if self.verbose:
                print(f"    Description: {desc}")
                print(f"    Has metadata hints: {has_hints}")

            # --- Original mode ---
            orig = self.run_original(query)

            # --- Staged mode ---
            staged = self.run_staged(query)

            # --- Precision overlap ---
            precision = self.compute_precision_overlap(
                orig["result_chunk_ids"],
                staged["result_chunk_ids"],
            )

            comparison = {
                "query": query,
                "description": desc,
                "has_metadata_hints": has_hints,
                "original": orig,
                "staged": staged,
                "precision_overlap": precision,
            }
            all_results.append(comparison)

            if self.verbose:
                print(f"    Original: {orig['retrieval_latency_ms']:.1f}ms, "
                      f"Staged: {staged['retrieval_latency_ms']:.1f}ms, "
                      f"Overlap: {precision:.0%}")

        return all_results


# ---------------------------------------------------------------------------
# Output formatters.
# ---------------------------------------------------------------------------

def print_detailed_results(results: List[Dict]) -> None:
    """Print per-query comparison."""
    print()
    print("=" * 70)
    print("  Per-Query Detailed Results")
    print("=" * 70)

    for i, r in enumerate(results, 1):
        orig = r["original"]
        staged = r["staged"]
        print(f"\n  Query {i}: {r['query'][:65]}")
        print(f"  {'─' * 66}")
        print(f"    {'Metric':<30} {'Original':>12} {'Staged':>12}")
        print(f"    {'─' * 54}")
        print(f"    {'Retrieval latency (ms)':<30} {orig['retrieval_latency_ms']:>12.1f} {staged['retrieval_latency_ms']:>12.1f}")
        print(f"    {'Vector comparisons':<30} {orig['vector_comparisons']:>12} {staged['vector_comparisons']:>12}")
        print(f"    {'Results count':<30} {orig['results_count']:>12} {staged['results_count']:>12}")
        print(f"    {'Context tokens (approx)':<30} {orig['context_tokens']:>12} {staged['context_tokens']:>12}")
        print(f"    {'Peak memory (MB)':<30} {orig['peak_memory_mb']:>12.1f} {staged['peak_memory_mb']:>12.1f}")
        if orig['llm_latency_ms'] > 0:
            print(f"    {'LLM latency (ms)':<30} {orig['llm_latency_ms']:>12.1f} {staged['llm_latency_ms']:>12.1f}")
        print(f"    {'Precision overlap':<30} {r['precision_overlap']:>11.0%}")

        if staged.get("constraints"):
            print(f"    Detected constraints: {staged['constraints']}")
        if staged.get("used_prefilter"):
            print(f"    Pre-filter used: YES")
        if staged.get("fallback_to_full"):
            print(f"    Fallback to full search: YES")


def print_summary_table(results: List[Dict]) -> None:
    """Print an aggregated summary table."""
    print()
    print("=" * 70)
    print("  Aggregated Summary")
    print("=" * 70)

    n = len(results)
    if n == 0:
        print("  No results to summarize.")
        return

    # Aggregate metrics
    orig_latencies = [r["original"]["retrieval_latency_ms"] for r in results]
    staged_latencies = [r["staged"]["retrieval_latency_ms"] for r in results]
    orig_vectors = [r["original"]["vector_comparisons"] for r in results]
    staged_vectors = [r["staged"]["vector_comparisons"] for r in results]
    orig_memory = [r["original"]["peak_memory_mb"] for r in results]
    staged_memory = [r["staged"]["peak_memory_mb"] for r in results]
    orig_tokens = [r["original"]["context_tokens"] for r in results]
    staged_tokens = [r["staged"]["context_tokens"] for r in results]
    precisions = [r["precision_overlap"] for r in results]

    avg_orig_lat = sum(orig_latencies) / n
    avg_staged_lat = sum(staged_latencies) / n
    avg_orig_vec = sum(orig_vectors) / n
    avg_staged_vec = sum(staged_vectors) / n

    vec_reduction = (1 - avg_staged_vec / avg_orig_vec) * 100 if avg_orig_vec > 0 else 0

    print(f"\n  Embedding Mode: Original (Pure Vector Search)")
    print(f"    Avg Retrieval Latency: {avg_orig_lat:.1f} ms")
    print(f"    Avg Vector Comparisons: {avg_orig_vec:.0f}")
    print(f"    Avg Context Tokens: {sum(orig_tokens)/n:.0f}")
    print(f"    Avg Peak Memory: {sum(orig_memory)/n:.1f} MB")

    print(f"\n  Embedding Mode: Metadata-Aware (Staged Retrieval)")
    print(f"    Avg Retrieval Latency: {avg_staged_lat:.1f} ms")
    print(f"    Avg Vector Comparisons: {avg_staged_vec:.0f}")
    print(f"    Vector Comparisons Reduced By: {vec_reduction:.1f}%")
    print(f"    Avg Context Tokens: {sum(staged_tokens)/n:.0f}")
    print(f"    Avg Peak Memory: {sum(staged_memory)/n:.1f} MB")
    print(f"    Avg Precision Overlap: {sum(precisions)/n:.0%}")

    # Latency comparison
    print(f"\n  Latency Comparison:")
    print(f"    {'Query':<50} {'Orig (ms)':>10} {'Staged (ms)':>12} {'Delta':>8}")
    print(f"    {'─' * 80}")
    for r in results:
        q = r["query"][:48]
        o = r["original"]["retrieval_latency_ms"]
        s = r["staged"]["retrieval_latency_ms"]
        delta = s - o
        sign = "+" if delta > 0 else ""
        print(f"    {q:<50} {o:>10.1f} {s:>12.1f} {sign}{delta:>7.1f}")

    # Memory comparison
    print(f"\n  Memory Comparison:")
    print(f"    {'Mode':<30} {'Avg Peak (MB)':>15} {'Max Peak (MB)':>15}")
    print(f"    {'─' * 60}")
    print(f"    {'Original':<30} {sum(orig_memory)/n:>15.1f} {max(orig_memory):>15.1f}")
    print(f"    {'Metadata-Aware':<30} {sum(staged_memory)/n:>15.1f} {max(staged_memory):>15.1f}")


def print_observations(results: List[Dict]) -> None:
    """Print interpretation / observations section."""
    print()
    print("=" * 70)
    print("  Observations")
    print("=" * 70)

    n = len(results)
    if n == 0:
        return

    # Count queries where pre-filter was actually used
    prefilter_used = sum(1 for r in results if r["staged"].get("used_prefilter"))
    fallback_count = sum(1 for r in results if r["staged"].get("fallback_to_full"))
    hint_queries = sum(1 for r in results if r["has_metadata_hints"])

    avg_orig = sum(r["original"]["retrieval_latency_ms"] for r in results) / n
    avg_staged = sum(r["staged"]["retrieval_latency_ms"] for r in results) / n

    print(f"\n  - Total queries tested: {n}")
    print(f"  - Queries with metadata hints: {hint_queries}")
    print(f"  - Pre-filter activated: {prefilter_used} times")
    print(f"  - Fallback to full search: {fallback_count} times")

    if avg_staged < avg_orig:
        pct = (1 - avg_staged / avg_orig) * 100
        print(f"  - Metadata-aware mode was {pct:.1f}% faster on average")
        print(f"    (StagedRetriever uses a leaner code path than Retriever;")
        print(f"     on a mixed-modality corpus pre-filtering would also reduce")
        print(f"     vector comparisons, adding further speedup)")
    elif avg_staged > avg_orig:
        pct = (avg_staged / avg_orig - 1) * 100
        print(f"  - Metadata-aware mode had {pct:.1f}% overhead on average")
        print(f"    (expected when metadata constraints don't reduce the candidate set)")
    else:
        print(f"  - Both modes performed identically on average")

    hints_results = [r for r in results if r["has_metadata_hints"]]
    if hints_results:
        avg_hint_orig = sum(r["original"]["retrieval_latency_ms"] for r in hints_results) / len(hints_results)
        avg_hint_staged = sum(r["staged"]["retrieval_latency_ms"] for r in hints_results) / len(hints_results)
        orig_vec = sum(r["original"]["vector_comparisons"] for r in hints_results) / len(hints_results)
        staged_vec = sum(r["staged"]["vector_comparisons"] for r in hints_results) / len(hints_results)
        if orig_vec > 0:
            vec_red = (1 - staged_vec / orig_vec) * 100
            print(f"  - For metadata-hint queries: vector comparisons reduced by {vec_red:.1f}%")
            if vec_red == 0.0:
                print(f"    (0% reduction is expected on a homogeneous single-modality corpus;")
                print(f"     a mixed corpus with PDFs, videos and images would show real reduction)")

    print()
    print("  Recommendation:")
    print("    - Use --metadata-filtering when queries reference specific years,")
    print("      file types, or content modalities.")
    print("    - For generic queries without metadata hints, both modes produce")
    print("      identical results (metadata-aware mode gracefully falls back).")
    print()


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare original vs metadata-aware retrieval modes.",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of results per query (default: 5)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show detailed per-query output during execution",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    benchmark = RetrievalBenchmark(top_k=args.top_k, verbose=args.verbose)
    if not benchmark.setup():
        print("\n  Setup failed. Cannot run comparison.")
        sys.exit(1)

    print("  Running queries...")
    print("  " + "-" * 66)
    results = benchmark.run_all()

    # Output all sections
    print_detailed_results(results)
    print_summary_table(results)
    print_observations(results)

    # Save raw JSON for further analysis
    output_path = PROJECT_ROOT / "data" / "processed" / "retrieval_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        # Convert non-serializable values
        serializable = []
        for r in results:
            sr = {
                "query": r["query"],
                "description": r["description"],
                "has_metadata_hints": r["has_metadata_hints"],
                "precision_overlap": r["precision_overlap"],
                "original": {
                    k: v for k, v in r["original"].items()
                    if k != "result_scores"
                },
                "staged": {
                    k: v for k, v in r["staged"].items()
                    if k != "result_scores"
                },
            }
            serializable.append(sr)
        json.dump(serializable, f, indent=2, default=str)
    print(f"  Raw results saved to: {output_path.relative_to(PROJECT_ROOT)}")
    print()


if __name__ == "__main__":
    main()
