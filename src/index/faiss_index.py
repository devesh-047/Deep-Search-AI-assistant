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

Learning TODO:
  1. Build the flat index and verify nearest-neighbor queries.
  2. Try IndexIVFFlat with nlist=100 and compare speed vs accuracy.
  3. Try IndexHNSWFlat and compare.
  4. Add incremental indexing (add new documents without rebuilding).
"""

import json
import logging
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

    Usage:
        idx = FaissIndex(dimension=384)
        idx.build(embeddings, chunk_metadata_list)
        results = idx.search("query text embedding as np array", top_k=5)
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

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        if self.index is None:
            return 0
        return self.index.ntotal

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
    ) -> None:
        """
        Build a flat inner-product index from embeddings.

        Args:
            embeddings : np.ndarray of shape (N, dimension), float32
            metadata   : list of dicts, one per embedding, storing chunk_id,
                         doc_id, text, etc.  len(metadata) must equal N.
        """
        if embeddings.shape[0] != len(metadata):
            raise ValueError(
                f"Mismatch: {embeddings.shape[0]} embeddings vs "
                f"{len(metadata)} metadata entries"
            )
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match "
                f"index dimension {self.dimension}"
            )

        # Inner product on L2-normalized vectors == cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.metadata = list(metadata)
        logger.info(
            "Built FAISS IndexFlatIP: %d vectors, dim=%d",
            self.index.ntotal,
            self.dimension,
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
            "Saved index (%d vectors) to %s", self.index.ntotal, out_dir
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
