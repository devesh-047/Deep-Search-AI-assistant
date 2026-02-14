"""
Embedding Encoder
==================
Converts text chunks into dense vector representations using
sentence-transformers/all-MiniLM-L6-v2.

Model choice rationale:
  - all-MiniLM-L6-v2 produces 384-dimensional embeddings
  - It is small (80 MB) and fast on CPU
  - It is well-supported by sentence-transformers and can be exported
    to ONNX / OpenVINO IR format
  - It provides good quality for English text retrieval tasks
  - Max input: 256 word-pieces (~200 words).  Our chunker targets this.

Design decisions:
  - Embeddings are computed in batches for efficiency
  - Results are returned as numpy arrays for direct FAISS ingestion
  - The encoder exposes a simple interface that the OpenVINO variant
    (src/openvino/ov_embeddings.py) mirrors, allowing a drop-in swap

Learning TODO:
  1. Run the encoder on sample chunks and inspect embedding shapes.
  2. Compute cosine similarity between related and unrelated chunks.
  3. Export the model to ONNX format.
  4. Convert the ONNX model to OpenVINO IR using mo (model optimizer).
  5. Benchmark sentence-transformers vs OpenVINO inference times.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Guard the sentence-transformers import
try:
    from sentence_transformers import SentenceTransformer

    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

# ---------------------------------------------------------------------------
# Default model identifier -- override via configs/settings.yaml
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # output dimension of all-MiniLM-L6-v2


class EmbeddingEncoder:
    """
    Wraps sentence-transformers to produce dense embeddings from text.

    Usage:
        encoder = EmbeddingEncoder()
        vectors = encoder.encode(["Hello world", "Another sentence"])
        # vectors.shape == (2, 384)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            model_name : HuggingFace model identifier
            device     : "cpu" or "cuda" (cpu recommended for stability)
            cache_dir  : where to cache downloaded model weights
        """
        if not ST_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required. "
                "Install: pip install sentence-transformers"
            )
        self.model_name = model_name
        self.device = device
        logger.info("Loading embedding model: %s on %s", model_name, device)
        self.model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=cache_dir,
        )
        # Verify expected dimension
        test_emb = self.model.encode(["test"], convert_to_numpy=True)
        self._dim = test_emb.shape[1]
        logger.info("Embedding dimension: %d", self._dim)

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimensionality."""
        return self._dim

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of text strings into dense vectors.

        Args:
            texts         : list of strings to encode
            batch_size    : number of texts per forward pass
            show_progress : display a progress bar (requires tqdm)
            normalize     : L2-normalize vectors (recommended for cosine sim)

        Returns:
            np.ndarray of shape (len(texts), dimension), dtype float32
        """
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        logger.info("Encoded %d texts -> shape %s", len(texts), embeddings.shape)
        return embeddings.astype(np.float32)

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Convenience method: encode one string, return 1-D vector."""
        return self.encode([text], normalize=normalize)[0]

    def save_embeddings(
        self, embeddings: np.ndarray, output_path: str
    ) -> Path:
        """
        Save embeddings to a .npy file for later reuse.

        Args:
            embeddings  : numpy array of shape (N, dim)
            output_path : file path ending in .npy
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out), embeddings)
        logger.info("Saved embeddings (%s) to %s", embeddings.shape, out)
        return out

    @staticmethod
    def load_embeddings(path: str) -> np.ndarray:
        """Load a previously saved embedding array."""
        arr = np.load(path)
        logger.info("Loaded embeddings from %s: shape %s", path, arr.shape)
        return arr


# ---------------------------------------------------------------------------
# Demo/Test: Run encoder and compute cosine similarity (TODO 1 & 2)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    import sys
    print("[Encoder Demo] Running sample embedding and similarity test...")
    try:
        encoder = EmbeddingEncoder()
    except ImportError as e:
        print("ERROR:", e)
        sys.exit(1)

    # Sample texts: related and unrelated
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast brown fox leaps over a sleepy dog.",  # semantically similar
        "Quantum mechanics describes the behavior of particles at atomic scales.",  # unrelated
    ]
    embeddings = encoder.encode(texts)
    print(f"Embeddings shape: {embeddings.shape}")
    for i, emb in enumerate(embeddings):
        print(f"  Text {i}: {emb.shape}, norm={np.linalg.norm(emb):.4f}")

    # Cosine similarity matrix
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print("\nCosine similarity matrix:")
    for i in range(len(texts)):
        row = []
        for j in range(len(texts)):
            sim = cosine_sim(embeddings[i], embeddings[j])
            row.append(f"{sim:.3f}")
        print(f"  {i}: {row}")

    print("\n[Done]")
