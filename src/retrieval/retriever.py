"""
Retriever
==========
Orchestrates the query flow: takes a natural-language query, encodes it,
searches the FAISS index, and returns ranked context chunks.

This is the "R" in RAG.  The retriever does NOT generate answers -- it
only finds relevant context.  Answer generation is handled by the LLM
module (src/llm/ollama_client.py).

Pipeline:
  1. User query (string)
  2. Encode query -> dense vector (via EmbeddingEncoder)
  3. Search FAISS index -> top-k chunk metadata + scores
  4. Return ranked list of context chunks

Design decisions:
  - Retrieval is embedding-based, NOT LLM-based.  The LLM is only used
    for final answer generation after context is retrieved.
  - The retriever is stateless -- it receives dependencies via constructor.

Learning TODO:
  1. Implement re-ranking using cross-encoder models.
  2. Add query preprocessing (spelling correction, expansion).
  3. Add hybrid retrieval (combine dense + sparse / BM25).
  4. Add metadata filtering (e.g., search only PDFs, or only a specific dataset).
"""

import logging
from typing import List, Dict, Optional

import numpy as np

from src.embeddings.encoder import EmbeddingEncoder
from src.index.faiss_index import FaissIndex

logger = logging.getLogger(__name__)


class RetrieverResult:
    """
    A single retrieval result with score and chunk information.
    """

    def __init__(self, chunk_id: str, doc_id: str, text: str, score: float, metadata: Dict):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.text = text
        self.score = score
        self.metadata = metadata

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ")
        return f"RetrieverResult(score={self.score:.4f}, chunk='{preview}...')"

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
        }


class Retriever:
    """
    Embedding-based retriever over chunkeed documents.

    Usage:
        encoder = EmbeddingEncoder()
        index = FaissIndex(dimension=384)
        index.load("data/processed/index")

        retriever = Retriever(encoder=encoder, index=index)
        results = retriever.query("What is the invoice total?", top_k=5)
    """

    def __init__(self, encoder: EmbeddingEncoder, index: FaissIndex):
        self.encoder = encoder
        self.index = index

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[RetrieverResult]:
        """
        Retrieve the most relevant chunks for a natural-language query.

        Args:
            query_text      : the user's question or search string
            top_k           : maximum number of results
            score_threshold : if set, discard results below this score

        Returns:
            List of RetrieverResult objects, sorted by descending score.
        """
        if not query_text.strip():
            logger.warning("Empty query, returning no results")
            return []

        # Step 1: encode the query
        query_vector = self.encoder.encode_single(query_text, normalize=True)

        # Step 2: search the index
        raw_results = self.index.search(query_vector, top_k=top_k)

        # Step 3: wrap results
        results: List[RetrieverResult] = []
        for entry in raw_results:
            if score_threshold is not None and entry["score"] < score_threshold:
                continue
            result = RetrieverResult(
                chunk_id=entry.get("chunk_id", "unknown"),
                doc_id=entry.get("doc_id", "unknown"),
                text=entry.get("text", ""),
                score=entry["score"],
                metadata={
                    k: v
                    for k, v in entry.items()
                    if k not in ("chunk_id", "doc_id", "text", "score")
                },
            )
            results.append(result)

        logger.info(
            "Query '%s' -> %d results (top_k=%d, threshold=%s)",
            query_text[:60],
            len(results),
            top_k,
            score_threshold,
        )
        return results

    def format_context(self, results: List[RetrieverResult], max_chars: int = 3000) -> str:
        """
        Format retrieval results into a context string suitable for LLM
        prompt construction.

        Each chunk is numbered and separated.  The total length is capped
        at max_chars to stay within LLM context limits.

        Args:
            results   : list of RetrieverResult from .query()
            max_chars : maximum total characters in the context block

        Returns:
            Formatted string ready to insert into an LLM prompt.
        """
        parts: List[str] = []
        total = 0
        for i, r in enumerate(results, 1):
            header = f"[Source {i} | score={r.score:.3f} | doc={r.doc_id}]"
            block = f"{header}\n{r.text}\n"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n".join(parts)
