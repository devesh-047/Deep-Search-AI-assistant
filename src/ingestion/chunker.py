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

import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default chunking parameters -- override via configs/settings.yaml
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 512      # characters per chunk
DEFAULT_CHUNK_OVERLAP = 64    # characters of overlap between consecutive chunks
DEFAULT_MIN_CHUNK_LENGTH = 30 # discard chunks shorter than this

# Regex for splitting text into sentences.
# Handles: periods, question marks, exclamation marks followed by whitespace.
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

# Separators for recursive chunking, tried in order from coarsest to finest.
_RECURSIVE_SEPARATORS = [
    "\n\n",  # paragraph break
    "\n",    # line break
    ". ",    # sentence end
    "? ",    # question end
    "! ",    # exclamation end
    "; ",    # semicolon
    ", ",    # comma
    " ",     # word boundary
    "",      # character-level (last resort)
]


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
    Splits document text into overlapping chunks.

    Supports multiple strategies:
        - ``"fixed"``     : fixed-size character window (default)
        - ``"sentence"``  : sentence-boundary-aware chunking
        - ``"recursive"`` : paragraph → sentence → character fallback
        - ``"token"``     : token-count-based using the embedding tokenizer

    Usage::

        chunker = TextChunker(chunk_size=512, overlap=64)
        chunks = chunker.chunk_document(normalized_doc)

        # Sentence-aware
        chunker = TextChunker(strategy="sentence")

        # Recursive
        chunker = TextChunker(strategy="recursive")

        # Token-based (requires tokenizer)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        chunker = TextChunker(strategy="token", tokenizer=tok, chunk_size=128)
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
        min_chunk_length: int = DEFAULT_MIN_CHUNK_LENGTH,
        strategy: str = "fixed",
        tokenizer=None,
    ):
        if overlap >= chunk_size:
            raise ValueError(
                f"Overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_length = min_chunk_length
        self.strategy = strategy
        self.tokenizer = tokenizer

        # Validate strategy
        valid = {"fixed", "sentence", "recursive", "token"}
        if strategy not in valid:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from {valid}")
        if strategy == "token" and tokenizer is None:
            raise ValueError(
                "Token-based chunking requires a tokenizer.  Pass tokenizer= "
                "(e.g. AutoTokenizer.from_pretrained('...'))"
            )

    def chunk_text(self, text: str, doc_id: str, base_metadata: Dict = None) -> List[TextChunk]:
        """
        Split a single text string into chunks using the configured strategy.

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

        # Dispatch to the appropriate strategy.
        if self.strategy == "sentence":
            chunks = self._chunk_sentence(text, doc_id, base_metadata)
        elif self.strategy == "recursive":
            chunks = self._chunk_recursive(text, doc_id, base_metadata)
        elif self.strategy == "token":
            chunks = self._chunk_token(text, doc_id, base_metadata)
        else:
            chunks = self._chunk_fixed(text, doc_id, base_metadata)

        # Filter out very short chunks.
        chunks = [c for c in chunks if len(c.text.strip()) >= self.min_chunk_length]

        logger.info(
            "Chunked doc_id=%s into %d chunks (size=%d, overlap=%d)",
            doc_id, len(chunks), self.chunk_size, self.overlap,
        )
        return chunks

    # ------------------------------------------------------------------ #
    # Strategy: fixed-size character window (original)                    #
    # ------------------------------------------------------------------ #

    def _chunk_fixed(self, text: str, doc_id: str, base_metadata: Dict) -> List[TextChunk]:
        """Sliding character window with overlap."""
        chunks: List[TextChunk] = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            chunk_meta = {
                **base_metadata,
                "char_start": start,
                "char_end": min(end, len(text)),
                "strategy": "fixed",
            }

            chunk = TextChunk(
                chunk_id=f"{doc_id}_chunk_{idx:04d}",
                doc_id=doc_id,
                text=chunk_text,
                index=idx,
                metadata=chunk_meta,
            )
            chunks.append(chunk)

            start += self.chunk_size - self.overlap
            idx += 1

        return chunks

    # ------------------------------------------------------------------ #
    # Strategy: sentence-aware chunking                                   #
    # ------------------------------------------------------------------ #

    def _chunk_sentence(self, text: str, doc_id: str, base_metadata: Dict) -> List[TextChunk]:
        """
        Sentence-boundary-aware chunking.

        Splits text into sentences first, then greedily packs sentences
        into chunks up to ``chunk_size`` characters.  Overlap is achieved
        by carrying the last few sentences of chunk N into chunk N+1.

        This avoids cutting sentences in half, producing more coherent
        embeddings.
        """
        sentences = _SENTENCE_SPLIT_RE.split(text)
        # Filter out empty strings from the split.
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        chunks: List[TextChunk] = []
        idx = 0
        sent_idx = 0

        while sent_idx < len(sentences):
            # Greedily pack sentences into a chunk.
            current_chunk_sents: List[str] = []
            current_len = 0

            while sent_idx < len(sentences):
                sent = sentences[sent_idx]
                added_len = len(sent) + (1 if current_chunk_sents else 0)  # +1 for space

                if current_len + added_len > self.chunk_size and current_chunk_sents:
                    # This sentence would exceed the limit; stop here.
                    break

                current_chunk_sents.append(sent)
                current_len += added_len
                sent_idx += 1

            chunk_text = " ".join(current_chunk_sents)

            chunk_meta = {
                **base_metadata,
                "sentence_count": len(current_chunk_sents),
                "strategy": "sentence",
            }

            chunks.append(TextChunk(
                chunk_id=f"{doc_id}_chunk_{idx:04d}",
                doc_id=doc_id,
                text=chunk_text,
                index=idx,
                metadata=chunk_meta,
            ))
            idx += 1

            # Overlap: rewind by counting sentences that fit within
            # the overlap character budget.
            if sent_idx < len(sentences) and self.overlap > 0:
                overlap_len = 0
                rewind = 0
                for i in range(len(current_chunk_sents) - 1, -1, -1):
                    s_len = len(current_chunk_sents[i]) + 1
                    if overlap_len + s_len > self.overlap:
                        break
                    overlap_len += s_len
                    rewind += 1
                if rewind > 0:
                    sent_idx -= rewind

        return chunks

    # ------------------------------------------------------------------ #
    # Strategy: recursive chunking                                        #
    # ------------------------------------------------------------------ #

    def _chunk_recursive(
        self, text: str, doc_id: str, base_metadata: Dict,
        _separators: Optional[List[str]] = None,
    ) -> List[TextChunk]:
        """
        Recursive chunking.

        Tries to split on the coarsest separator first (paragraphs),
        falling back to finer separators (sentences, words, characters)
        when pieces are still too large.

        This preserves document structure as much as possible while
        ensuring every chunk fits within ``chunk_size``.
        """
        separators = _separators if _separators is not None else list(_RECURSIVE_SEPARATORS)

        if not separators:
            # No separators left — force character-level split.
            return self._chunk_fixed(text, doc_id, base_metadata)

        sep = separators[0]
        remaining_seps = separators[1:]

        # Split on the current separator.
        if sep == "":
            pieces = list(text)  # character-level
        else:
            pieces = text.split(sep)

        # Merge small pieces greedily and recurse on large ones.
        merged_chunks: List[TextChunk] = []
        current_pieces: List[str] = []
        current_len = 0
        idx = len(merged_chunks)

        for piece in pieces:
            piece_len = len(piece) + (len(sep) if current_pieces else 0)

            if current_len + piece_len <= self.chunk_size:
                current_pieces.append(piece)
                current_len += piece_len
            else:
                # Flush the accumulated pieces as a chunk.
                if current_pieces:
                    chunk_text = sep.join(current_pieces)
                    merged_chunks.append(TextChunk(
                        chunk_id=f"{doc_id}_chunk_{len(merged_chunks):04d}",
                        doc_id=doc_id,
                        text=chunk_text,
                        index=len(merged_chunks),
                        metadata={**base_metadata, "strategy": "recursive"},
                    ))
                    current_pieces = []
                    current_len = 0

                # If this single piece is still too large, recurse with
                # a finer separator.
                if len(piece) > self.chunk_size:
                    sub_chunks = self._chunk_recursive(
                        piece, doc_id, base_metadata, remaining_seps,
                    )
                    # Re-number chunk IDs.
                    for sc in sub_chunks:
                        sc.chunk_id = f"{doc_id}_chunk_{len(merged_chunks):04d}"
                        sc.index = len(merged_chunks)
                        merged_chunks.append(sc)
                else:
                    current_pieces = [piece]
                    current_len = len(piece)

        # Flush remaining.
        if current_pieces:
            chunk_text = sep.join(current_pieces)
            merged_chunks.append(TextChunk(
                chunk_id=f"{doc_id}_chunk_{len(merged_chunks):04d}",
                doc_id=doc_id,
                text=chunk_text,
                index=len(merged_chunks),
                metadata={**base_metadata, "strategy": "recursive"},
            ))

        return merged_chunks

    # ------------------------------------------------------------------ #
    # Strategy: token-based chunking                                      #
    # ------------------------------------------------------------------ #

    def _chunk_token(self, text: str, doc_id: str, base_metadata: Dict) -> List[TextChunk]:
        """
        Token-count-based chunking using the embedding model's tokenizer.

        Instead of counting characters, this counts actual tokens so
        chunks align exactly with the model's input limits.  ``chunk_size``
        is interpreted as **max tokens** (not characters) in this mode.

        Requires ``self.tokenizer`` to be set (e.g. a HuggingFace
        AutoTokenizer).
        """
        # Tokenize the full text.
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        total_tokens = len(tokens)

        if total_tokens == 0:
            return []

        chunks: List[TextChunk] = []
        start = 0
        idx = 0
        step = self.chunk_size - self.overlap

        while start < total_tokens:
            end = min(start + self.chunk_size, total_tokens)
            chunk_token_ids = tokens[start:end]

            # Decode tokens back to text.
            chunk_text = self.tokenizer.decode(chunk_token_ids, skip_special_tokens=True)

            chunk_meta = {
                **base_metadata,
                "token_start": start,
                "token_end": end,
                "token_count": end - start,
                "strategy": "token",
            }

            chunks.append(TextChunk(
                chunk_id=f"{doc_id}_chunk_{idx:04d}",
                doc_id=doc_id,
                text=chunk_text,
                index=idx,
                metadata=chunk_meta,
            ))

            start += step
            idx += 1

        return chunks

    def chunk_documents(self, documents) -> List[TextChunk]:
        """
        Chunk a list of NormalizedDocument objects.

        Args:
            documents: iterable of objects with .doc_id, .text, .metadata

        Returns:
            Flat list of all TextChunk objects across all documents.
        """
        try:
            from tqdm import tqdm
            iterator = tqdm(documents, desc="Chunking", unit="doc")
        except ImportError:
            iterator = documents
            
        all_chunks: List[TextChunk] = []
        for doc in iterator:
            doc_chunks = self.chunk_text(
                text=doc.text,
                doc_id=doc.doc_id,
                base_metadata=doc.metadata,
            )
            all_chunks.extend(doc_chunks)
        logger.info("Total chunks across %d documents: %d", len(documents), len(all_chunks))
        return all_chunks

    # ------------------------------------------------------------------ #
    # Experimentation utility                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def experiment_chunk_sizes(
        text: str,
        doc_id: str = "experiment",
        sizes: Optional[List[int]] = None,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
        strategies: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Compare different chunk sizes and strategies on the same text.

        Prints a formatted table and returns the raw results.  Useful for
        tuning the chunking parameters before committing to a config.

        Args:
            text       : sample text to chunk.
            doc_id     : dummy doc_id for chunk IDs.
            sizes      : list of chunk_size values to test.
            overlap    : overlap to use for all tests.
            strategies : list of strategy names to test.

        Returns:
            List of dicts with keys: strategy, chunk_size, overlap,
            num_chunks, avg_chunk_len, min_chunk_len, max_chunk_len.

        Usage::

            from src.ingestion.chunker import TextChunker
            text = open("some_doc.txt").read()
            results = TextChunker.experiment_chunk_sizes(text)
        """
        if sizes is None:
            sizes = [256, 512, 1024]
        if strategies is None:
            strategies = ["fixed", "sentence", "recursive"]

        results: List[Dict] = []

        for strategy in strategies:
            for size in sizes:
                adj_overlap = min(overlap, size - 1)
                chunker = TextChunker(
                    chunk_size=size,
                    overlap=adj_overlap,
                    strategy=strategy,
                    min_chunk_length=0,  # don't filter during experiments
                )
                chunks = chunker.chunk_text(text, doc_id)
                lengths = [len(c.text) for c in chunks]

                row = {
                    "strategy": strategy,
                    "chunk_size": size,
                    "overlap": adj_overlap,
                    "num_chunks": len(chunks),
                    "avg_chunk_len": sum(lengths) / len(lengths) if lengths else 0,
                    "min_chunk_len": min(lengths) if lengths else 0,
                    "max_chunk_len": max(lengths) if lengths else 0,
                }
                results.append(row)

        # Print a formatted table.
        header = f"{'Strategy':<12} {'Size':>6} {'Overlap':>7} {'Chunks':>7} {'Avg Len':>8} {'Min':>5} {'Max':>5}"
        logger.info("Chunk size experiment on %d chars:", len(text))
        print(f"\n  {header}")
        print(f"  {'─' * len(header)}")
        for r in results:
            print(
                f"  {r['strategy']:<12} {r['chunk_size']:>6} {r['overlap']:>7} "
                f"{r['num_chunks']:>7} {r['avg_chunk_len']:>8.0f} "
                f"{r['min_chunk_len']:>5} {r['max_chunk_len']:>5}"
            )
        print()

        return results
