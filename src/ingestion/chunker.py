"""
Text Chunker
==============
Splits normalized document text into overlapping chunks suitable for
embedding and retrieval.

Why chunk?
  Embedding models have a maximum input length (typically 256-512 tokens
  for all-MiniLM-L6-v2).  Large documents must be split into smaller
  pieces so that each piece can be independently embedded and retrieved.

  Overlap between chunks ensures that sentences at chunk boundaries are
  not lost -- a query might match text that spans two chunks, and the
  overlap provides continuity.

Chunking strategy:
  This module implements fixed-size character chunking with configurable
  overlap.  This is the simplest strategy and a good starting point.

Learning TODO:
  1. Implement sentence-aware chunking (split on sentence boundaries).
  2. Implement recursive chunking (split on paragraphs, then sentences,
     then characters as a fallback).
  3. Add token-based chunking using the tokenizer from the embedding model.
  4. Experiment with chunk sizes and measure retrieval quality.
"""

import logging
from typing import List, Dict
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default chunking parameters -- override via configs/settings.yaml
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 512      # characters per chunk
DEFAULT_CHUNK_OVERLAP = 64    # characters of overlap between consecutive chunks


@dataclass
class TextChunk:
    """
    A single chunk of text with provenance metadata.

    Attributes:
        chunk_id  : globally unique identifier (doc_id + chunk index)
        doc_id    : ID of the parent document
        text      : the chunk text
        index     : ordinal position within the parent document
        metadata  : inherited and chunk-specific metadata
    """
    chunk_id: str
    doc_id: str
    text: str
    index: int
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


class TextChunker:
    """
    Splits document text into fixed-size overlapping chunks.

    Usage:
        chunker = TextChunker(chunk_size=512, overlap=64)
        chunks = chunker.chunk_document(normalized_doc)
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        if overlap >= chunk_size:
            raise ValueError(
                f"Overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, doc_id: str, base_metadata: Dict = None) -> List[TextChunk]:
        """
        Split a single text string into chunks.

        Args:
            text           : the full document text
            doc_id         : parent document ID (used to build chunk_id)
            base_metadata  : metadata inherited from the parent document

        Returns:
            List of TextChunk objects.  Empty list if text is empty.
        """
        if not text or not text.strip():
            logger.debug("Empty text for doc_id=%s, skipping chunking", doc_id)
            return []

        base_metadata = base_metadata or {}
        chunks: List[TextChunk] = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Build chunk metadata
            chunk_meta = {
                **base_metadata,
                "char_start": start,
                "char_end": min(end, len(text)),
            }

            chunk = TextChunk(
                chunk_id=f"{doc_id}_chunk_{idx:04d}",
                doc_id=doc_id,
                text=chunk_text,
                index=idx,
                metadata=chunk_meta,
            )
            chunks.append(chunk)

            # Advance the window
            start += self.chunk_size - self.overlap
            idx += 1

        logger.info(
            "Chunked doc_id=%s into %d chunks (size=%d, overlap=%d)",
            doc_id, len(chunks), self.chunk_size, self.overlap,
        )
        return chunks

    def chunk_documents(self, documents) -> List[TextChunk]:
        """
        Chunk a list of NormalizedDocument objects.

        Args:
            documents: iterable of objects with .doc_id, .text, .metadata

        Returns:
            Flat list of all TextChunk objects across all documents.
        """
        all_chunks: List[TextChunk] = []
        for doc in documents:
            doc_chunks = self.chunk_text(
                text=doc.text,
                doc_id=doc.doc_id,
                base_metadata=doc.metadata,
            )
            all_chunks.extend(doc_chunks)
        logger.info("Total chunks across %d documents: %d", len(documents), len(all_chunks))
        return all_chunks
