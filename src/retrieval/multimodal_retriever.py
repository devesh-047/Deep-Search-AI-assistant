"""
Multimodal Retriever
=====================
Extends the existing retrieval pipeline with CLIP-based image retrieval
and score fusion — entirely additive, no changes to the core ``Retriever``.

Design:
    The multimodal retriever wraps the existing text-based ``Retriever``
    and adds a parallel CLIP embedding search path.  For each query:

        query
         ↓
        ┌─────────────────────┬────────────────────────┐
        │  MiniLM text embed  │  CLIP text embed       │
        │       ↓             │       ↓                │
        │  FAISS text index   │  FAISS CLIP index      │
        │       ↓             │       ↓                │
        │  text results       │  image results         │
        └─────────────────────┴────────────────────────┘
                     ↓                    ↓
                  ┌──────────────────────────┐
                  │   Score Fusion (weighted) │
                  └──────────────────────────┘
                              ↓
                      merged results

    The fusion is a simple weighted average of normalised scores.
    Text results use weight ``text_weight`` (default 0.6) and CLIP
    results use ``clip_weight`` (default 0.4).

Integration:
    This module does NOT modify the existing ``Retriever`` class.
    It is used alongside it in ``cli.py`` when ``--multimodal`` is passed.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MultimodalRetriever:
    """
    Combines text-based retrieval (MiniLM + FAISS) with CLIP-based
    image retrieval for multimodal search.

    Usage::

        from src.retrieval.multimodal_retriever import MultimodalRetriever

        mm = MultimodalRetriever(
            text_encoder=encoder,      # EmbeddingEncoder or OVEmbeddingEncoder
            text_index=index,          # existing FaissIndex
            clip_encoder=clip_enc,     # CLIPEncoder
            clip_index=clip_idx,       # FAISS index over CLIP embeddings
        )
        results = mm.query("find slides with charts", top_k=5)
    """

    def __init__(
        self,
        text_encoder,
        text_index,
        clip_encoder=None,
        clip_index=None,
        text_weight: float = 0.6,
        clip_weight: float = 0.4,
    ):
        """
        Args:
            text_encoder  : MiniLM embedding encoder (EmbeddingEncoder or OVEmbeddingEncoder).
            text_index    : FAISS index built from MiniLM embeddings.
            clip_encoder  : CLIPEncoder instance (optional — if None, CLIP
                            path is skipped and results are text-only).
            clip_index    : FAISS index built from CLIP image embeddings
                            (optional — if None, CLIP path is skipped).
            text_weight   : Weight for text retrieval scores in fusion.
            clip_weight   : Weight for CLIP retrieval scores in fusion.
        """
        self.text_encoder = text_encoder
        self.text_index = text_index
        self.clip_encoder = clip_encoder
        self.clip_index = clip_index
        self.text_weight = text_weight
        self.clip_weight = clip_weight

    @property
    def has_clip(self) -> bool:
        """Check if CLIP retrieval is available."""
        return (
            self.clip_encoder is not None
            and self.clip_index is not None
            and getattr(self.clip_encoder, "is_available", False)
        )

    def query(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Perform multimodal retrieval.

        1. Text path: encode query with MiniLM → search text FAISS index
        2. CLIP path: encode query with CLIP text encoder → search CLIP index
        3. Fuse: weighted score combination → deduplicate → sort → top_k

        Args:
            query_text : The user's natural-language query.
            top_k      : Maximum number of results to return.

        Returns:
            List of result dicts with keys:
                doc_id, chunk_id, text, score, metadata, retrieval_source
        """
        # --- Text retrieval path ---
        text_results = self._text_search(query_text, top_k=top_k * 2)

        # --- CLIP retrieval path ---
        clip_results = []
        if self.has_clip:
            clip_results = self._clip_search(query_text, top_k=top_k * 2)

        # --- Fusion ---
        if clip_results:
            merged = self._fuse_results(text_results, clip_results, top_k)
        else:
            merged = text_results[:top_k]

        logger.info(
            "Multimodal query '%s': %d text + %d clip → %d fused",
            query_text[:60],
            len(text_results),
            len(clip_results),
            len(merged),
        )
        return merged

    def _text_search(self, query_text: str, top_k: int) -> List[Dict]:
        """Run standard MiniLM text retrieval."""
        try:
            query_vec = self.text_encoder.encode_single(
                query_text, normalize=True
            )
            results = self.text_index.search(query_vec, top_k=top_k)
            for r in results:
                r["retrieval_source"] = "text"
            return results
        except Exception as exc:
            logger.error("Text retrieval failed: %s", exc)
            return []

    def _clip_search(self, query_text: str, top_k: int) -> List[Dict]:
        """Run CLIP text-to-image retrieval."""
        try:
            clip_vec = self.clip_encoder.encode_text(query_text)
            if clip_vec is None:
                return []
            results = self.clip_index.search(clip_vec, top_k=top_k)
            for r in results:
                r["retrieval_source"] = "clip"
            return results
        except Exception as exc:
            logger.error("CLIP retrieval failed: %s", exc)
            return []

    def _fuse_results(
        self,
        text_results: List[Dict],
        clip_results: List[Dict],
        top_k: int,
    ) -> List[Dict]:
        """
        Fuse text and CLIP results using weighted score combination.

        Scores are min-max normalised within each source before fusion
        to ensure they are on a comparable scale.
        """
        # Normalise scores within each source
        text_results = self._normalise_scores(text_results)
        clip_results = self._normalise_scores(clip_results)

        # Collect all results keyed by a unique identifier
        fused: Dict[str, Dict] = {}

        for r in text_results:
            key = r.get("chunk_id", r.get("doc_id", id(r)))
            if key not in fused:
                fused[key] = {**r, "fused_score": 0.0}
            fused[key]["fused_score"] += r["score"] * self.text_weight

        for r in clip_results:
            key = r.get("chunk_id", r.get("doc_id", id(r)))
            if key not in fused:
                fused[key] = {**r, "fused_score": 0.0}
            fused[key]["fused_score"] += r["score"] * self.clip_weight

        # Sort by fused score and take top_k
        sorted_results = sorted(
            fused.values(), key=lambda x: x["fused_score"], reverse=True
        )

        # Replace score with fused_score for downstream compatibility
        for r in sorted_results:
            r["score"] = r.pop("fused_score")

        return sorted_results[:top_k]

    @staticmethod
    def _normalise_scores(results: List[Dict]) -> List[Dict]:
        """Min-max normalise scores to [0, 1] range."""
        if not results:
            return results

        scores = [r["score"] for r in results]
        min_s = min(scores)
        max_s = max(scores)
        span = max_s - min_s

        if span < 1e-9:
            # All scores equal — set all to 1.0
            for r in results:
                r["score"] = 1.0
        else:
            for r in results:
                r["score"] = (r["score"] - min_s) / span

        return results
