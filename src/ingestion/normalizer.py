"""
Document Normalizer
====================
Converts heterogeneous ``RawDocument`` objects into a unified
``NormalizedDocument`` schema suitable for chunking and embedding.

Unified schema::

    {
        "doc_id":     str,   # deterministic hash of doc_key
        "source":     str,   # dataset name or file path
        "doc_type":   str,   # "form", "document_image", "classified_image", ...
        "text":       str,   # extracted plain text
        "image_path": str,   # path to image on disk (or null)
        "metadata":   dict,  # arbitrary per-document metadata
    }

Why normalise?
    Different datasets produce different record structures.  The normaliser
    flattens everything into a single representation so that downstream
    stages (chunker, embedder, indexer) need zero dataset-specific logic.

Learning TODO
-------------
1. Add deduplication (skip documents whose doc_id already exists in
   the processed directory).
2. Add language detection (langdetect or fasttext) and store in metadata.
3. Strip boilerplate headers/footers common in corporate documents.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List

from src.ingestion.loader import RawDocument

logger = logging.getLogger(__name__)


def _make_doc_id(key: str) -> str:
    """
    Create a deterministic document ID from the document key.
    Uses SHA-256 truncated to 12 hex characters for brevity.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


class NormalizedDocument:
    """
    Uniform representation of any document in the pipeline.

    Every downstream component (chunker, embedder, indexer) works with
    this class, regardless of the original dataset or file format.
    """

    def __init__(
        self,
        doc_id: str,
        source: str,
        doc_type: str,
        text: str,
        image_path: str,
        metadata: Dict,
    ):
        self.doc_id = doc_id
        self.source = source
        self.doc_type = doc_type
        self.text = text
        self.image_path = image_path
        self.metadata = metadata

    def __repr__(self) -> str:
        return (
            f"NormalizedDocument(id='{self.doc_id}', "
            f"type='{self.doc_type}', "
            f"text_len={len(self.text)})"
        )

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "doc_type": self.doc_type,
            "text": self.text,
            "image_path": self.image_path,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "NormalizedDocument":
        return cls(
            doc_id=data["doc_id"],
            source=data["source"],
            doc_type=data["doc_type"],
            text=data["text"],
            image_path=data.get("image_path", ""),
            metadata=data.get("metadata", {}),
        )


class DocumentNormalizer:
    """
    Transforms a list of ``RawDocument`` objects into ``NormalizedDocument``
    objects and persists them as JSON-lines.

    Usage::

        normalizer = DocumentNormalizer(output_dir="data/processed/normalised")
        norm_docs = normalizer.normalize(raw_docs)
        normalizer.save(norm_docs)
    """

    def __init__(self, output_dir: str = "data/processed/normalised"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def normalize(self, raw_docs: List[RawDocument]) -> List[NormalizedDocument]:
        """
        Convert raw documents to the unified schema.

        Documents with empty text are still included -- they will receive
        text later via the OCR pipeline.
        """
        normalized: List[NormalizedDocument] = []
        for raw in raw_docs:
            doc_id = _make_doc_id(raw.doc_key)
            doc_type = self._classify_type(raw)
            norm = NormalizedDocument(
                doc_id=doc_id,
                source=raw.source,
                doc_type=doc_type,
                text=raw.text.strip(),
                image_path=raw.image_path or "",
                metadata=raw.metadata,
            )
            normalized.append(norm)
        logger.info("Normalised %d documents", len(normalized))
        return normalized

    def save(self, docs: List[NormalizedDocument], filename: str = "documents.jsonl") -> Path:
        """
        Persist normalised documents as a JSON-lines file.
        Each line is one JSON object -- easy to stream and append.
        """
        out_path = self.output_dir / filename
        with open(out_path, "w", encoding="utf-8") as fh:
            for doc in docs:
                fh.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
        logger.info("Saved %d normalised documents to %s", len(docs), out_path)
        return out_path

    def load(self, filename: str = "documents.jsonl") -> List[NormalizedDocument]:
        """Load previously saved normalised documents."""
        out_path = self.output_dir / filename
        if not out_path.exists():
            logger.warning("No normalised documents found at %s", out_path)
            return []
        docs: List[NormalizedDocument] = []
        with open(out_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    docs.append(NormalizedDocument.from_dict(json.loads(line)))
        logger.info("Loaded %d normalised documents from %s", len(docs), out_path)
        return docs

    @staticmethod
    def _classify_type(raw: RawDocument) -> str:
        """Assign a canonical document type based on dataset and metadata."""
        dataset = raw.metadata.get("dataset", "")
        if dataset == "funsd":
            return "form"
        elif dataset == "docvqa":
            return "document_image"
        elif dataset == "rvl_cdip":
            return "classified_image"
        # Fallback for file-based loading paths.
        if raw.image_path:
            return "image"
        if raw.text:
            return "text"
        return "unknown"
