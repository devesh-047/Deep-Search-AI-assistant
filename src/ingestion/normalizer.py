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

Features
--------
* **Deduplication** -- documents whose ``doc_id`` already exists in a
  previously saved JSONL file are skipped during normalisation.
* **Language detection** -- when ``langdetect`` is installed, the detected
  language code (e.g. ``en``, ``de``) is stored in ``metadata["language"]``.
* **Boilerplate stripping** -- common corporate headers/footers
  (page numbers, confidentiality notices, "Page X of Y", etc.) are
  removed before the text is stored.
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from src.ingestion.loader import RawDocument

logger = logging.getLogger(__name__)


def _make_doc_id(key: str) -> str:
    """
    Create a deterministic document ID from the document key.
    Uses SHA-256 truncated to 12 hex characters for brevity.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Language detection helper
# ---------------------------------------------------------------------------

def _detect_language(text: str) -> Optional[str]:
    """
    Detect the language of *text* using ``langdetect``.

    Returns the ISO 639-1 code (e.g. ``"en"``, ``"de"``) or ``None`` if
    detection fails or the library is not installed.  Short texts
    (< 20 characters) are skipped because they produce unreliable results.
    """
    if not text or len(text.strip()) < 20:
        return None
    try:
        from langdetect import detect, LangDetectException
    except ImportError:
        logger.debug(
            "langdetect is not installed -- skipping language detection.  "
            "Install it with:  pip install langdetect"
        )
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None


# ---------------------------------------------------------------------------
# Boilerplate stripping helpers
# ---------------------------------------------------------------------------

# Compiled once at import time for performance.
_BOILERPLATE_PATTERNS: List[re.Pattern] = [
    # "Page X of Y" or "Page X"
    re.compile(r"(?i)^\s*page\s+\d+(?:\s+of\s+\d+)?\s*$", re.MULTILINE),
    # Bare page numbers on their own line
    re.compile(r"^\s*-?\s*\d{1,4}\s*-?\s*$", re.MULTILINE),
    # Confidentiality / disclaimer notices
    re.compile(
        r"(?i)^\s*(?:confidential|privileged|do not distribute|internal use only"
        r"|proprietary|strictly private|not for public release).*$",
        re.MULTILINE,
    ),
    # "DRAFT" watermarks
    re.compile(r"(?i)^\s*-{0,3}\s*draft\s*-{0,3}\s*$", re.MULTILINE),
    # Common email/fax header lines
    re.compile(
        r"(?i)^\s*(?:from:|to:|cc:|bcc:|sent:|date:|subject:|fax:)\s*$",
        re.MULTILINE,
    ),
]


def _strip_boilerplate(text: str) -> str:
    """
    Remove common corporate boilerplate lines from *text*.

    Each pattern in ``_BOILERPLATE_PATTERNS`` is applied in order.  Matched
    lines are replaced with empty strings, then runs of 3+ blank lines are
    collapsed to 2.
    """
    for pattern in _BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)
    # Collapse excessive blank lines left behind.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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

    def __init__(self, output_dir: str = "data/processed/normalised",
                 deduplicate: bool = True,
                 detect_language: bool = True,
                 strip_boilerplate: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.deduplicate = deduplicate
        self.detect_language = detect_language
        self.strip_boilerplate = strip_boilerplate

    # ------------------------------------------------------------------ #
    # Deduplication helpers                                                #
    # ------------------------------------------------------------------ #

    def _load_existing_ids(self, filename: str = "documents.jsonl") -> Set[str]:
        """
        Scan the previously-saved JSONL file and return all ``doc_id``
        values found there.  Returns an empty set if the file does not
        exist yet.
        """
        out_path = self.output_dir / filename
        ids: Set[str] = set()
        if not out_path.exists():
            return ids
        with open(out_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        ids.add(json.loads(line)["doc_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        logger.info("Dedup: %d existing document IDs loaded from %s",
                     len(ids), out_path)
        return ids

    # ------------------------------------------------------------------ #
    # Core normalisation                                                  #
    # ------------------------------------------------------------------ #

    def normalize(self, raw_docs: List[RawDocument],
                  jsonl_filename: str = "documents.jsonl"
                  ) -> List[NormalizedDocument]:
        """
        Convert raw documents to the unified schema.

        * If ``deduplicate`` is enabled, documents whose ``doc_id`` already
          appears in the saved JSONL file are silently skipped.
        * If ``strip_boilerplate`` is enabled, common corporate
          headers/footers are removed from the text.
        * If ``detect_language`` is enabled and ``langdetect`` is installed,
          the detected language code is stored in
          ``metadata["language"]``.
        * Documents with empty text are still included -- they will receive
          text later via the OCR pipeline.
        """
        # --- Deduplication: collect already-processed IDs ----------------
        existing_ids: Set[str] = set()
        if self.deduplicate:
            existing_ids = self._load_existing_ids(jsonl_filename)

        try:
            from tqdm import tqdm
            iterator = tqdm(raw_docs, desc="Normalizing", unit="doc")
        except ImportError:
            iterator = raw_docs

        normalized: List[NormalizedDocument] = []
        skipped = 0
        for raw in iterator:
            doc_id = _make_doc_id(raw.doc_key)

            # Skip duplicates.
            if doc_id in existing_ids:
                skipped += 1
                continue
            # Also guard against duplicates *within* this batch.
            existing_ids.add(doc_id)

            # --- Text cleaning -------------------------------------------
            text = raw.text.strip()
            if text and self.strip_boilerplate:
                text = _strip_boilerplate(text)

            # --- Language detection --------------------------------------
            metadata = dict(raw.metadata)  # shallow copy to avoid mutation
            if self.detect_language:
                lang = _detect_language(text)
                # Fall back to "en" if detection is unavailable or fails.
                metadata["language"] = lang or "en"

            doc_type = self._classify_type(raw)
            norm = NormalizedDocument(
                doc_id=doc_id,
                source=raw.source,
                doc_type=doc_type,
                text=text,
                image_path=raw.image_path or "",
                metadata=metadata,
            )
            normalized.append(norm)

        if skipped:
            logger.info("Dedup: skipped %d duplicate documents", skipped)
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
