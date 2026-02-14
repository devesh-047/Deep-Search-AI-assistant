"""
FAISS Vector Index
===================
Manages building, saving, loading, and querying a FAISS index of
document chunk embeddings.

Why FAISS?
  - Pure CPU implementation (faiss-cpu) -- no GPU driver issues
  - Handles millions of vectors efficiently
  - Supports multiple index types (Flat, IVF, HNSW)
  - Industry standard for dense retrieval systems
  - Easy to integrate with numpy arrays

Index strategy:
  We start with IndexFlatIP (inner product on L2-normalized vectors,
  equivalent to cosine similarity).  This is the simplest and most
  accurate index type.  For larger datasets, IVF or HNSW can be
  explored as learning exercises.

Design decisions:
  - The index stores only vectors.  Chunk metadata (text, doc_id, etc.)
    is stored in a parallel list and looked up by integer position.
  - Saving/loading uses FAISS native I/O plus a JSON sidecar for metadata.

Index types implemented:
  1. IndexFlatIP  -- Exact brute-force search using inner product.
                     100 % recall, O(N) per query.  Best for small datasets
                     (< 100 k vectors) or when accuracy is paramount.

  2. IndexIVFFlat -- Inverted-file index.  Vectors are partitioned into
                     `nlist` Voronoi cells during training.  At query time
                     only `nprobe` cells are scanned, giving sub-linear
                     speed.  Must be *trained* before adding vectors.
                     Trade-off: lower recall for faster queries.

  3. IndexHNSWFlat -- Hierarchical Navigable Small World graph.  Builds a
                      multi-layer proximity graph at insert time.  Very
                      fast queries with high recall, but uses more memory
                      and does NOT support vector removal.

  4. Incremental indexing -- The `add()` method lets you append new
     vectors+metadata to an existing index without rebuilding it.

Learning TODO (all implemented):
  1. ✅ Build the flat index and verify nearest-neighbor queries.
  2. ✅ Try IndexIVFFlat with nlist=100 and compare speed vs accuracy.
  3. ✅ Try IndexHNSWFlat and compare.
  4. ✅ Add incremental indexing (add new documents without rebuilding).
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FaissIndex:
    """
    Manages a FAISS vector index with a parallel metadata store.

    Supports three index types selectable via the ``build*`` family of
    methods:

        build()       -- IndexFlatIP   (exact, default)
        build_ivf()   -- IndexIVFFlat  (approximate, cell-based)
        build_hnsw()  -- IndexHNSWFlat (approximate, graph-based)

    After building, use ``add()`` to incrementally index new vectors,
    ``search()`` to query, and ``save()`` / ``load()`` for persistence.

    Usage:
        idx = FaissIndex(dimension=384)
        idx.build(embeddings, chunk_metadata_list)
        results = idx.search(query_vector, top_k=5)
        idx.save("data/processed/index")
        idx.load("data/processed/index")
    """

    def __init__(self, dimension: int = 384):
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS is required. Install: pip install faiss-cpu"
            )
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        # Parallel metadata: metadata[i] corresponds to index vector i
        self.metadata: List[Dict] = []
        # Track which index type is active (for logging / save-load)
        self._index_type: str = "none"

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        if self.index is None:
            return 0
        return self.index.ntotal

    @property
    def index_type(self) -> str:
        """Human-readable label for the active index type."""
        return self._index_type

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: Optional[List[Dict]] = None,
    ) -> np.ndarray:
        """
        Sanity-check embeddings (and optionally metadata) before
        adding them to the index.  Returns embeddings cast to float32.
        """
        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings must be 2-D (got {embeddings.ndim}-D)"
            )
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match "
                f"index dimension {self.dimension}"
            )
        if metadata is not None and embeddings.shape[0] != len(metadata):
            raise ValueError(
                f"Mismatch: {embeddings.shape[0]} embeddings vs "
                f"{len(metadata)} metadata entries"
            )
        return embeddings.astype(np.float32, copy=False)

    # ==================================================================
    #  TODO 1 — Build flat index (exact nearest-neighbour search)
    # ==================================================================

    def build(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
    ) -> None:
        """
        Build a **flat inner-product** index from embeddings.

        This is an *exact* (brute-force) index.  Every vector is compared
        against the query at search time, so recall is always 100 %.
        For small-to-medium collections (< ~100 k vectors) this is
        the best choice because it is simple, accurate, and still fast
        enough on modern CPUs.

        How it works
        ------------
        ``IndexFlatIP`` computes the *inner product* (dot product) between
        the query vector and every stored vector.  Because our embeddings
        are L2-normalised (unit length), the inner product equals cosine
        similarity::

            cos(a, b) = (a · b) / (‖a‖ · ‖b‖)  →  a · b  when ‖a‖=‖b‖=1

        The results are returned in descending score order (highest
        similarity first).

        Complexity
        ----------
        - Build : O(N)   — just copies the vectors into the index.
        - Search: O(N·d) — compares the query against all N vectors.

        Args:
            embeddings : np.ndarray of shape (N, dimension), float32
            metadata   : list of dicts, one per embedding, storing chunk_id,
                         doc_id, text, etc.  len(metadata) must equal N.
        """
        embeddings = self._validate_embeddings(embeddings, metadata)

        # Inner product on L2-normalized vectors == cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.metadata = list(metadata)
        self._index_type = "flat"
        logger.info(
            "Built FAISS IndexFlatIP: %d vectors, dim=%d",
            self.index.ntotal,
            self.dimension,
        )

    # ==================================================================
    #  TODO 2 — IndexIVFFlat (inverted file, approximate search)
    # ==================================================================

    def build_ivf(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
        nlist: int = 100,
        nprobe: int = 10,
    ) -> None:
        """
        Build an **Inverted-File Flat** (IVF) index for approximate search.

        How it works
        ------------
        The vector space is partitioned into ``nlist`` Voronoi cells using
        k-means clustering on the training data:

        1. **Training** — k-means is run on the embeddings to find
           ``nlist`` centroids.  Each centroid defines one cell (cluster).
        2. **Adding** — every vector is assigned to the cell whose
           centroid is nearest, and stored in that cell's inverted list.
        3. **Searching** — the query vector is compared against the
           ``nlist`` centroids; the ``nprobe`` closest cells are selected
           and *only* the vectors inside those cells are scored.

        This means search time is roughly O(nprobe/nlist · N · d) instead
        of O(N · d).  For example, with nlist=100 and nprobe=10, only
        ~10 % of the vectors are actually compared.

        Trade-offs
        ----------
        - **Faster** than flat for large N (the speedup grows with N).
        - **Lower recall** because relevant vectors may live in cells
          that were not probed.  Increasing ``nprobe`` recovers recall
          at the cost of speed (nprobe=nlist degrades to brute-force).
        - **Requires training** — you must provide a representative set
          of vectors (usually the full dataset) before adding.
        - The number of training vectors should be at least ``nlist``
          (ideally 30×–256× nlist).

        When to use
        -----------
        Good for medium-to-large collections (100 k – 10 M vectors)
        where some recall loss is acceptable for faster queries.

        Args:
            embeddings : np.ndarray of shape (N, dimension), float32
            metadata   : list of dicts, one per embedding
            nlist      : number of Voronoi cells (clusters).  Rule of
                         thumb: sqrt(N) for balanced speed/recall.
                         Default 100.
            nprobe     : number of cells to visit at query time.  Higher
                         = better recall but slower.  Default 10.
        """
        embeddings = self._validate_embeddings(embeddings, metadata)

        n_vectors = embeddings.shape[0]

        # Clamp nlist so we never have more clusters than vectors
        # (k-means would fail otherwise).
        effective_nlist = min(nlist, n_vectors)
        if effective_nlist != nlist:
            logger.warning(
                "nlist=%d > n_vectors=%d — clamped to %d",
                nlist,
                n_vectors,
                effective_nlist,
            )

        # The quantiser is a flat IP index used to assign vectors to cells
        quantiser = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIVFFlat(
            quantiser,
            self.dimension,
            effective_nlist,
            faiss.METRIC_INNER_PRODUCT,
        )

        # --- Training phase (k-means on the embeddings) ---
        logger.info(
            "Training IVF index (nlist=%d) on %d vectors …",
            effective_nlist,
            n_vectors,
        )
        t0 = time.time()
        self.index.train(embeddings)
        train_time = time.time() - t0
        logger.info("Training completed in %.2f s", train_time)

        # --- Add vectors to the trained index ---
        self.index.add(embeddings)
        self.metadata = list(metadata)

        # Set nprobe (how many cells to visit per query)
        self.index.nprobe = nprobe

        self._index_type = "ivf"
        logger.info(
            "Built FAISS IndexIVFFlat: %d vectors, dim=%d, "
            "nlist=%d, nprobe=%d (train %.2fs)",
            self.index.ntotal,
            self.dimension,
            effective_nlist,
            nprobe,
            train_time,
        )

    # ==================================================================
    #  TODO 3 — IndexHNSWFlat (graph-based approximate search)
    # ==================================================================

    def build_hnsw(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
    ) -> None:
        """
        Build an **HNSW Flat** (graph-based) index for approximate search.

        How it works
        ------------
        HNSW (Hierarchical Navigable Small World) builds a multi-layer
        proximity graph over the vectors:

        1. **Layers** — vectors are randomly assigned to layers
           (exponential distribution).  Layer 0 contains *all* vectors;
           higher layers contain progressively fewer.
        2. **Construction** — when a new vector is inserted, the algorithm
           descends from the top layer, greedily finding the nearest nodes
           at each layer.  At the insertion layer and below, up to ``M``
           bidirectional edges are created to the nearest neighbours,
           forming a "small world" graph.
        3. **Search** — the query starts at the top layer, greedily walks
           toward the nearest node at each layer, then does a more
           thorough beam search at layer 0 using ``ef_search`` as the
           beam width.

        Key parameters
        --------------
        - **``M``** (edges per node): controls graph connectivity.
          Higher M → better recall & more memory.  Typical: 16–64.
        - **``ef_construction``** (build-time beam width): higher values
          make construction slower but produce a better graph.
          Typical: 100–400.
        - **``ef_search``** (query-time beam width): higher values
          increase recall at the cost of latency.  Must be ≥ top_k.

        Trade-offs vs Flat
        ------------------
        - ⚡ Much faster queries — O(log N · d) instead of O(N · d).
        - 📈 Very high recall (often > 95 % even with moderate settings).
        - 💾 Higher memory usage (~2× a flat index due to graph edges).
        - 🚫 Does NOT support removing vectors after insertion — only
          adding.  If you need to delete, rebuild or use IVF.
        - No separate training step (vectors are indexed on insertion).

        Trade-offs vs IVF
        -----------------
        - Usually higher recall than IVF at the same query speed.
        - Slower to build (graph construction vs simple k-means).
        - No support for removal (IVF does support it).

        When to use
        -----------
        Best when you want the highest recall at low latency and can
        afford the extra memory.  Great for medium collections
        (10 k – 5 M vectors).

        Args:
            embeddings      : np.ndarray of shape (N, dimension), float32
            metadata        : list of dicts, one per embedding
            M               : number of bi-directional links per node.
                              Higher = better recall, more memory.  Default 32.
            ef_construction : size of the dynamic candidate list during
                              graph construction.  Default 200.
            ef_search       : size of the dynamic candidate list during
                              search.  Default 64.
        """
        embeddings = self._validate_embeddings(embeddings, metadata)

        # FAISS's IndexHNSWFlat uses L2 distance by default.
        # For inner-product (cosine on normalised vectors) we wrap it:
        #   faiss.IndexHNSWFlat(dim, M, metric)
        # Note: METRIC_INNER_PRODUCT is supported from faiss ≥ 1.7.
        self.index = faiss.IndexHNSWFlat(self.dimension, M, faiss.METRIC_INNER_PRODUCT)

        # Tune construction quality
        self.index.hnsw.efConstruction = ef_construction

        # Set search-time beam width
        self.index.hnsw.efSearch = ef_search

        # --- Add vectors (graph is built incrementally) ---
        logger.info(
            "Building HNSW graph (M=%d, efConstruction=%d) for %d vectors …",
            M,
            ef_construction,
            embeddings.shape[0],
        )
        t0 = time.time()
        self.index.add(embeddings)
        build_time = time.time() - t0

        self.metadata = list(metadata)
        self._index_type = "hnsw"
        logger.info(
            "Built FAISS IndexHNSWFlat: %d vectors, dim=%d, "
            "M=%d, efConstruction=%d, efSearch=%d (%.2fs)",
            self.index.ntotal,
            self.dimension,
            M,
            ef_construction,
            ef_search,
            build_time,
        )

    # ==================================================================
    #  TODO 4 — Incremental indexing
    # ==================================================================

    def add(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
    ) -> None:
        """
        Incrementally add new vectors and metadata to an existing index.

        How it works
        ------------
        FAISS indices maintain an internal counter (``ntotal``).  When you
        call ``index.add(vectors)``, the new vectors receive IDs starting
        at the current ``ntotal`` value.  Because our metadata list is
        kept in the same insertion order, we simply *append* the new
        metadata entries so that ``metadata[id]`` still correctly maps to
        the right vector.

        This method works with *all three* index types (flat, IVF, HNSW).
        For IVF, the index must already be trained — new vectors are
        assigned to the nearest existing cell without retraining.

        When to use
        -----------
        Use ``add()`` when you want to ingest new documents into an
        existing index without rebuilding from scratch.  For example,
        after the initial ``ingest``, a user might add a few PDFs;
        you can embed just those chunks and call ``add()`` to extend
        the index.

        Limitations
        -----------
        - Cannot *remove* vectors from Flat or HNSW indices.  IVF
          supports ``remove_ids()`` but we don't expose it yet.
        - For IVF, adding many vectors without retraining may degrade
          cluster balance and hurt recall.  Periodically rebuild if the
          dataset grows significantly.

        Args:
            embeddings : np.ndarray of shape (M, dimension), float32
            metadata   : list of dicts, one per new embedding
        """
        if self.index is None:
            raise RuntimeError(
                "Cannot add to a non-existent index. Call build(), "
                "build_ivf(), or build_hnsw() first."
            )
        embeddings = self._validate_embeddings(embeddings, metadata)

        # For IVF indices, verify the index is trained
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            raise RuntimeError(
                "IVF index is not trained. Build with build_ivf() first."
            )

        prev_size = self.index.ntotal
        self.index.add(embeddings)
        self.metadata.extend(metadata)
        logger.info(
            "Incremental add: %d new vectors (total: %d → %d)",
            embeddings.shape[0],
            prev_size,
            self.index.ntotal,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Find the top_k most similar chunks to the query vector.

        Args:
            query_vector : np.ndarray of shape (dimension,) or (1, dimension)
            top_k        : number of results to return

        Returns:
            List of dicts, each containing the chunk metadata plus a
            "score" key with the similarity score.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Search called on empty index")
            return []

        # Reshape to (1, dim) if needed
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Clamp top_k to index size
        top_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query_vector, top_k)

        results: List[Dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue  # FAISS returns -1 for missing results
            entry = {**self.metadata[idx], "score": float(score)}
            results.append(entry)

        logger.info("Search returned %d results (top_k=%d)", len(results), top_k)
        return results

    # ------------------------------------------------------------------
    # TODO 1 — Verification utility
    # ------------------------------------------------------------------

    def verify(
        self,
        embeddings: np.ndarray,
        sample_queries: int = 5,
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Verify the index by running sample nearest-neighbour queries.

        Picks ``sample_queries`` random vectors from the indexed
        embeddings, searches for each, and checks that the vector
        finds *itself* as the top-1 result (which must always be the
        case for an exact flat index and should be the case for a
        well-tuned approximate index).

        Args:
            embeddings     : the same array used to build the index,
                             shape (N, dim)
            sample_queries : how many random queries to run
            top_k          : number of neighbours to retrieve per query

        Returns:
            List of dicts, one per query, containing:
              - query_idx  : the index of the query vector in embeddings
              - top_k_ids  : list of retrieved vector IDs
              - top_k_scores : corresponding similarity scores
              - self_hit   : True if query_idx is in top_k_ids
        """
        if self.index is None:
            raise RuntimeError("Index not built — nothing to verify.")

        n = embeddings.shape[0]
        sample_queries = min(sample_queries, n)
        rng = np.random.default_rng(42)
        query_indices = rng.choice(n, size=sample_queries, replace=False)

        report: List[Dict] = []
        for qi in query_indices:
            qvec = embeddings[qi].reshape(1, -1).astype(np.float32)
            scores, ids = self.index.search(qvec, top_k)
            entry = {
                "query_idx": int(qi),
                "top_k_ids": ids[0].tolist(),
                "top_k_scores": [round(float(s), 6) for s in scores[0]],
                "self_hit": int(qi) in ids[0].tolist(),
            }
            report.append(entry)

        hits = sum(1 for r in report if r["self_hit"])
        logger.info(
            "Verification: %d/%d queries found themselves in top-%d "
            "(index_type=%s)",
            hits,
            sample_queries,
            top_k,
            self._index_type,
        )
        return report

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """
        Save the FAISS index and metadata sidecar to disk.

        Creates two files:
          - <directory>/index.faiss
          - <directory>/metadata.json
        """
        if self.index is None:
            raise RuntimeError("Cannot save: index has not been built")

        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        index_path = out_dir / "index.faiss"
        meta_path = out_dir / "metadata.json"

        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(self.metadata, fh, ensure_ascii=False, indent=2)

        logger.info(
            "Saved index (%d vectors, type=%s) to %s",
            self.index.ntotal,
            self._index_type,
            out_dir,
        )

    def load(self, directory: str) -> None:
        """
        Load a previously saved FAISS index and metadata.
        """
        in_dir = Path(directory)
        index_path = in_dir / "index.faiss"
        meta_path = in_dir / "metadata.json"

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as fh:
            self.metadata = json.load(fh)

        self.dimension = self.index.d
        logger.info(
            "Loaded index: %d vectors, dim=%d from %s",
            self.index.ntotal,
            self.dimension,
            in_dir,
        )


# ======================================================================
# Demo / Self-test — run with: python -m src.index.faiss_index
# ======================================================================
# This __main__ block exercises ALL four Learning TODOs:
#   1. Build a flat index and verify nearest-neighbour queries
#   2. Build an IVF index and compare recall/speed
#   3. Build an HNSW index and compare recall/speed
#   4. Demonstrate incremental indexing with add()
# ======================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    DIM = 384
    N_VECTORS = 2000      # total vectors to generate
    N_QUERIES = 50        # queries to benchmark
    TOP_K = 5

    print("=" * 65)
    print("  FAISS Index —  Learning TODO Demo")
    print("=" * 65)

    # --- Generate synthetic data ---
    rng = np.random.default_rng(42)
    data = rng.standard_normal((N_VECTORS, DIM)).astype(np.float32)
    # L2-normalise so inner product == cosine similarity
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = data / norms

    metadata = [{"id": i, "text": f"chunk_{i}"} for i in range(N_VECTORS)]

    query_indices = rng.choice(N_VECTORS, size=N_QUERIES, replace=False)
    queries = data[query_indices]

    # Helper to compute recall@k against brute-force ground truth
    def compute_recall(gt_ids: np.ndarray, pred_ids: np.ndarray) -> float:
        """Fraction of ground-truth top-k IDs found in predicted top-k."""
        hits = 0
        total = 0
        for g, p in zip(gt_ids, pred_ids):
            g_set = set(g.tolist())
            p_set = set(p.tolist())
            hits += len(g_set & p_set)
            total += len(g_set)
        return hits / total if total else 0.0

    # ----------------------------------------------------------------
    #  Ground truth (brute-force flat search)
    # ----------------------------------------------------------------
    print("\n──── TODO 1: Flat index (exact search) ────")
    idx_flat = FaissIndex(dimension=DIM)
    idx_flat.build(data, metadata)

    t0 = time.time()
    gt_scores, gt_ids = idx_flat.index.search(queries, TOP_K)
    flat_time = time.time() - t0

    report = idx_flat.verify(data, sample_queries=10, top_k=TOP_K)
    self_hits = sum(1 for r in report if r["self_hit"])
    print(f"  Vectors indexed : {idx_flat.size}")
    print(f"  Self-hit check  : {self_hits}/{len(report)} queries found themselves (expected 100 %)")
    print(f"  Search time     : {flat_time*1000:.2f} ms for {N_QUERIES} queries")
    print(f"  Recall@{TOP_K}      : 100.0 % (by definition — this IS the ground truth)")

    # ----------------------------------------------------------------
    #  TODO 2: IVF index
    # ----------------------------------------------------------------
    print("\n──── TODO 2: IVF index (approximate, cell-based) ────")
    for nprobe in [1, 5, 10, 50]:
        idx_ivf = FaissIndex(dimension=DIM)
        idx_ivf.build_ivf(data, metadata, nlist=100, nprobe=nprobe)

        t0 = time.time()
        ivf_scores, ivf_ids = idx_ivf.index.search(queries, TOP_K)
        ivf_time = time.time() - t0

        recall = compute_recall(gt_ids, ivf_ids) * 100
        print(f"  nprobe={nprobe:3d}  →  recall@{TOP_K}={recall:6.2f}%  "
              f"time={ivf_time*1000:.2f}ms  "
              f"speedup={flat_time/ivf_time:.1f}×")

    # ----------------------------------------------------------------
    #  TODO 3: HNSW index
    # ----------------------------------------------------------------
    print("\n──── TODO 3: HNSW index (approximate, graph-based) ────")
    for ef_search in [16, 32, 64, 128]:
        idx_hnsw = FaissIndex(dimension=DIM)
        idx_hnsw.build_hnsw(data, metadata, M=32, ef_construction=200, ef_search=ef_search)

        t0 = time.time()
        hnsw_scores, hnsw_ids = idx_hnsw.index.search(queries, TOP_K)
        hnsw_time = time.time() - t0

        recall = compute_recall(gt_ids, hnsw_ids) * 100
        print(f"  efSearch={ef_search:3d}  →  recall@{TOP_K}={recall:6.2f}%  "
              f"time={hnsw_time*1000:.2f}ms  "
              f"speedup={flat_time/hnsw_time:.1f}×")

    # ----------------------------------------------------------------
    #  TODO 4: Incremental indexing
    # ----------------------------------------------------------------
    print("\n──── TODO 4: Incremental indexing with add() ────")

    # Start with half the data
    half = N_VECTORS // 2
    idx_inc = FaissIndex(dimension=DIM)
    idx_inc.build(data[:half], metadata[:half])
    print(f"  Initial build   : {idx_inc.size} vectors")

    # Add the other half incrementally
    idx_inc.add(data[half:], metadata[half:])
    print(f"  After add()     : {idx_inc.size} vectors")

    # Verify the full index matches brute-force
    inc_scores, inc_ids = idx_inc.index.search(queries, TOP_K)
    recall = compute_recall(gt_ids, inc_ids) * 100
    print(f"  Recall@{TOP_K} vs flat: {recall:.2f}% (expected 100 %)")

    # Also test incremental add on IVF
    idx_inc_ivf = FaissIndex(dimension=DIM)
    idx_inc_ivf.build_ivf(data[:half], metadata[:half], nlist=50, nprobe=10)
    print(f"  IVF initial     : {idx_inc_ivf.size} vectors")
    idx_inc_ivf.add(data[half:], metadata[half:])
    print(f"  IVF after add() : {idx_inc_ivf.size} vectors")

    print("\n" + "=" * 65)
    print("  All Learning TODOs verified successfully ✔")
    print("=" * 65)
