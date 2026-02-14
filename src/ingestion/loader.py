"""
Dataset Loader
===============
Loads raw datasets from disk into a uniform ``RawDocument`` list that the
rest of the pipeline can consume.

Two loading paths are supported:

1. **HuggingFace Arrow datasets** (primary) -- the datasets stored under
   ``/mnt/d/Openvino-project/data/raw`` were saved with
   ``datasets.Dataset.save_to_disk()`` and live in Arrow format.  This
   loader uses ``datasets.load_from_disk()`` to reload them.

2. **Loose files** (secondary) -- plain text, PDF, DOCX, PPTX, and images
   found by walking a directory tree.

Design notes
------------
* The loader **never writes** to the raw data directory.
* Images are extracted from HuggingFace ``Image`` columns and saved to the
  OCR cache so that Tesseract can process them.
* Text columns (e.g. FUNSD ``words``) are joined into plain text here so
  that downstream stages see a single string per record.
* PDF extraction uses **pdfplumber** for reliable text and layout handling.
* DOCX extraction uses **python-docx** to iterate over paragraphs.
* PPTX extraction uses **python-pptx** to iterate over slide text frames.
* ``load_dataset`` accepts an optional ``splits`` parameter so callers can
  restrict loading to specific splits (e.g. only ``train``).
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unified document representation returned by every loader path
# ---------------------------------------------------------------------------
class RawDocument:
    """
    Container for a single loaded document.

    Attributes
    ----------
    doc_key : str
        Unique key across the dataset (e.g. ``funsd_train_0042``).
    source : str
        Where the record came from (dataset name or file path).
    text : str
        Extracted plain text.  Empty for image-only records (OCR fills it
        in later).
    image_path : str or None
        Path to a saved image if the record contained one.
    metadata : dict
        Arbitrary per-record information (bounding boxes, labels, etc.).
    """

    def __init__(
        self,
        doc_key: str,
        source: str,
        text: str = "",
        image_path: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        self.doc_key = doc_key
        self.source = source
        self.text = text
        self.image_path = image_path
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return (
            f"RawDocument(key='{self.doc_key}', source='{self.source}', "
            f"text_len={len(self.text)}, has_image={self.image_path is not None})"
        )

    def to_dict(self) -> Dict:
        return {
            "doc_key": self.doc_key,
            "source": self.source,
            "text_length": len(self.text),
            "image_path": self.image_path,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Arrow-format dataset loading helpers
# ---------------------------------------------------------------------------

def _load_arrow_dataset(dataset_path: str):
    """
    Load a HuggingFace dataset saved with ``save_to_disk()``.

    Returns either a ``datasets.Dataset`` or ``datasets.DatasetDict``
    depending on whether the saved artifact has splits.

    Raises ImportError if the ``datasets`` library is not installed.
    """
    try:
        from datasets import load_from_disk
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required to load Arrow datasets.  "
            "Install it with:  pip install datasets"
        )
    return load_from_disk(dataset_path)


def _extract_image(image_obj, save_path: str) -> str:
    """
    Save a HuggingFace ``Image`` feature to disk as PNG.

    Parameters
    ----------
    image_obj : PIL.Image.Image
        The image object from the Arrow dataset.
    save_path : str
        Destination file path.

    Returns
    -------
    str
        The absolute path where the image was saved.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image_obj.save(save_path, format="PNG")
    return os.path.abspath(save_path)


# ---------------------------------------------------------------------------
# Loose-file text extraction helpers
# ---------------------------------------------------------------------------

def _extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file using **pdfplumber**.

    Iterates over every page in the PDF and concatenates the extracted text
    with double newlines between pages.  Falls back to an empty string for
    pages that yield no text (e.g. scanned images -- those should be routed
    through the OCR module instead).

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the ``.pdf`` file.

    Returns
    -------
    str
        The concatenated plain-text content of all pages.

    Raises
    ------
    ImportError
        If ``pdfplumber`` is not installed.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "The 'pdfplumber' library is required to extract text from PDFs.  "
            "Install it with:  pip install pdfplumber"
        )

    pages_text: List[str] = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
    return "\n\n".join(pages_text)


def _extract_text_from_docx(file_path: str) -> str:
    """
    Extract text from a DOCX file using **python-docx**.

    Iterates over every paragraph in the document and joins them with
    newlines.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the ``.docx`` file.

    Returns
    -------
    str
        The concatenated plain-text content of all paragraphs.

    Raises
    ------
    ImportError
        If ``python-docx`` is not installed.
    """
    try:
        import docx
    except ImportError:
        raise ImportError(
            "The 'python-docx' library is required to extract text from DOCX files.  "
            "Install it with:  pip install python-docx"
        )

    doc = docx.Document(file_path)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text)


def _extract_text_from_pptx(file_path: str) -> str:
    """
    Extract text from a PPTX file using **python-pptx**.

    Iterates over every slide, then every shape with a text frame, and
    concatenates all paragraph text.  Slides are separated by double
    newlines.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the ``.pptx`` file.

    Returns
    -------
    str
        The concatenated plain-text content of all slides.

    Raises
    ------
    ImportError
        If ``python-pptx`` is not installed.
    """
    try:
        from pptx import Presentation
    except ImportError:
        raise ImportError(
            "The 'python-pptx' library is required to extract text from PPTX files.  "
            "Install it with:  pip install python-pptx"
        )

    prs = Presentation(file_path)
    slides_text: List[str] = []
    for slide in prs.slides:
        parts: List[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    if paragraph.text.strip():
                        parts.append(paragraph.text)
        if parts:
            slides_text.append("\n".join(parts))
    return "\n\n".join(slides_text)


# Map file extensions to their extraction functions.
_FILE_EXTRACTORS = {
    ".txt": lambda path: Path(path).read_text(encoding="utf-8", errors="replace"),
    ".pdf": _extract_text_from_pdf,
    ".docx": _extract_text_from_docx,
    ".pptx": _extract_text_from_pptx,
}

# Image extensions that should be routed to the OCR module.
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}


# ---------------------------------------------------------------------------
# Per-dataset record handlers
# ---------------------------------------------------------------------------
# Each handler knows how to turn one record from a specific Arrow dataset
# into a RawDocument.  Adding a new dataset means writing a new handler.

def _handle_funsd_record(record: Dict, idx: int, split: str,
                         image_dir: str) -> RawDocument:
    """
    FUNSD record layout:
        id      : str            -- document identifier
        words   : List[str]      -- individual words
        bboxes  : List[List[int]]-- bounding boxes per word
        ner_tags: List[int]      -- NER labels per word
        image   : PIL.Image      -- scanned form image

    Strategy: join the ``words`` list into a single text string separated
    by spaces.  Also save the image for optional OCR verification.
    """
    words = record.get("words", [])
    text = " ".join(words) if words else ""

    # Save the image so the OCR module can re-process it if needed.
    image_path = None
    if record.get("image") is not None:
        fname = f"funsd_{split}_{idx:05d}.png"
        image_path = _extract_image(
            record["image"],
            os.path.join(image_dir, "funsd", fname),
        )

    return RawDocument(
        doc_key=f"funsd_{split}_{idx:05d}",
        source="funsd",
        text=text,
        image_path=image_path,
        metadata={
            "dataset": "funsd",
            "split": split,
            "record_index": idx,
            "word_count": len(words),
            "has_bboxes": bool(record.get("bboxes")),
        },
    )


def _handle_docvqa_record(record: Dict, idx: int,
                          image_dir: str) -> RawDocument:
    """
    DocVQA record layout:
        questionId           : str
        question             : str
        question_types       : List[str]
        image                : PIL.Image
        docId                : int
        ucsf_document_id     : str
        ucsf_document_page_no: str
        answers              : List[str]
        data_split           : str

    Strategy: the ``question`` and ``answers`` are stored as text metadata.
    The document image is saved for OCR -- the actual document text must be
    extracted from the image, not from the Q/A pair.
    """
    image_path = None
    if record.get("image") is not None:
        fname = f"docvqa_{idx:05d}.png"
        image_path = _extract_image(
            record["image"],
            os.path.join(image_dir, "docvqa", fname),
        )

    return RawDocument(
        doc_key=f"docvqa_{idx:05d}",
        source="docvqa",
        text="",  # Must be filled by OCR on the document image.
        image_path=image_path,
        metadata={
            "dataset": "docvqa",
            "record_index": idx,
            "question": record.get("question", ""),
            "answers": record.get("answers", []),
            "question_id": record.get("questionId", ""),
            "doc_id": record.get("docId"),
        },
    )


def _handle_rvl_cdip_record(record: Dict, idx: int,
                            image_dir: str) -> RawDocument:
    """
    RVL-CDIP record layout:
        image : PIL.Image
        label : int  (0-15, representing 16 document categories)

    Strategy: image-only dataset.  Save the image for OCR.
    """
    image_path = None
    if record.get("image") is not None:
        fname = f"rvl_cdip_{idx:05d}.png"
        image_path = _extract_image(
            record["image"],
            os.path.join(image_dir, "rvl_cdip", fname),
        )

    # Human-readable label map for the 16 RVL-CDIP classes.
    label_names = [
        "letter", "form", "email", "handwritten", "advertisement",
        "scientific_report", "scientific_publication", "specification",
        "file_folder", "news_article", "budget", "invoice", "presentation",
        "questionnaire", "resume", "memo",
    ]
    label_int = record.get("label", -1)
    label_str = label_names[label_int] if 0 <= label_int < 16 else "unknown"

    return RawDocument(
        doc_key=f"rvl_cdip_{idx:05d}",
        source="rvl_cdip",
        text="",  # Must be filled by OCR.
        image_path=image_path,
        metadata={
            "dataset": "rvl_cdip",
            "record_index": idx,
            "label": label_int,
            "label_name": label_str,
        },
    )


# ---------------------------------------------------------------------------
# Main loader class
# ---------------------------------------------------------------------------

# Map dataset names to their handler functions.
_RECORD_HANDLERS = {
    "funsd": _handle_funsd_record,
    "docvqa": _handle_docvqa_record,
    "rvl_cdip": _handle_rvl_cdip_record,
}


class DatasetLoader:
    """
    Load one or more HuggingFace Arrow datasets (or loose files) and return
    a list of ``RawDocument`` objects.

    Parameters
    ----------
    raw_data_root : str
        Path to the top-level raw data directory (e.g.
        ``/mnt/d/Openvino-project/data/raw``).
    image_cache_dir : str
        Where to write extracted images so the OCR module can find them.

    Usage
    -----
    ::

        loader = DatasetLoader(
            raw_data_root="/mnt/d/Openvino-project/data/raw",
            image_cache_dir="data/processed/ocr_cache",
        )
        docs = loader.load_dataset("funsd")
    """

    def __init__(self, raw_data_root: str, image_cache_dir: str):
        self.raw_data_root = Path(raw_data_root)
        self.image_cache_dir = Path(image_cache_dir)

        if not self.raw_data_root.exists():
            raise FileNotFoundError(
                f"Raw data root does not exist: {self.raw_data_root}"
            )
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def load_dataset(self, name: str, max_records: int = 0,
                     splits: Optional[List[str]] = None) -> List[RawDocument]:
        """
        Load a named dataset and return RawDocument objects.

        Parameters
        ----------
        name : str
            One of the registered dataset names (``funsd``, ``docvqa``,
            ``rvl_cdip``).
        max_records : int
            If > 0, load at most this many records (useful for quick tests).
        splits : list of str or None
            If provided, only load the specified splits (e.g.
            ``["train"]``).  Ignored for datasets without splits.

        Returns
        -------
        list of RawDocument
        """
        dataset_path = self.raw_data_root / name
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {dataset_path}"
            )

        logger.info("Loading dataset '%s' from %s", name, dataset_path)
        ds = _load_arrow_dataset(str(dataset_path))

        handler = _RECORD_HANDLERS.get(name)
        if handler is None:
            logger.warning(
                "No handler registered for dataset '%s'.  "
                "Returning empty list. Register a handler in loader.py.",
                name,
            )
            return []

        return self._iterate_dataset(ds, name, handler, max_records, splits)

    def load_directory(self, dir_path: str,
                       max_records: int = 0) -> List[RawDocument]:
        """
        Walk a directory tree and load loose files (txt, pdf, docx, pptx,
        images) into ``RawDocument`` objects.

        Text is extracted inline for supported document types.  Image files
        are recorded with an ``image_path`` so the OCR module can process
        them later.

        Parameters
        ----------
        dir_path : str
            Path to the directory to walk.
        max_records : int
            If > 0, load at most this many files.

        Returns
        -------
        list of RawDocument
        """
        root = Path(dir_path)
        if not root.exists():
            raise FileNotFoundError(f"Directory not found: {root}")

        documents: List[RawDocument] = []
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            if max_records > 0 and len(documents) >= max_records:
                break

            ext = file_path.suffix.lower()
            rel = str(file_path.relative_to(root))

            # Supported text-bearing document formats.
            if ext in _FILE_EXTRACTORS:
                try:
                    text = _FILE_EXTRACTORS[ext](str(file_path))
                except Exception as exc:
                    logger.warning("Failed to extract text from %s: %s",
                                   file_path, exc)
                    text = ""
                documents.append(RawDocument(
                    doc_key=rel,
                    source=str(file_path),
                    text=text,
                    metadata={"file_type": ext},
                ))

            # Image files -- defer to OCR.
            elif ext in _IMAGE_EXTENSIONS:
                documents.append(RawDocument(
                    doc_key=rel,
                    source=str(file_path),
                    text="",
                    image_path=str(file_path.resolve()),
                    metadata={"file_type": ext},
                ))
            else:
                logger.debug("Skipping unsupported file: %s", file_path)

        logger.info("Directory '%s': %d documents loaded", dir_path,
                     len(documents))
        return documents

    def list_available_datasets(self) -> List[str]:
        """Return names of subdirectories under raw_data_root."""
        if not self.raw_data_root.exists():
            return []
        return sorted(
            d.name
            for d in self.raw_data_root.iterdir()
            if d.is_dir()
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _iterate_dataset(self, ds, name: str, handler, max_records: int,
                         splits: Optional[List[str]] = None
                         ) -> List[RawDocument]:
        """
        Walk the loaded dataset (or DatasetDict) and apply the handler
        to each record.

        Parameters
        ----------
        splits : list of str or None
            If provided and the dataset is a ``DatasetDict``, only iterate
            over the named splits.  Unknown split names are logged as
            warnings and skipped.
        """
        from datasets import DatasetDict

        documents: List[RawDocument] = []
        image_dir = str(self.image_cache_dir)

        if isinstance(ds, DatasetDict):
            # Determine which splits to process.
            available_splits = list(ds.keys())
            if splits is not None:
                selected = []
                for s in splits:
                    if s in ds:
                        selected.append(s)
                    else:
                        logger.warning(
                            "Requested split '%s' not found in dataset '%s'. "
                            "Available splits: %s", s, name, available_splits,
                        )
                target_splits = selected
            else:
                target_splits = available_splits

            # E.g. FUNSD has train / test splits.
            for split_name in target_splits:
                split_ds = ds[split_name]
                count = 0
                for idx, record in enumerate(split_ds):
                    if max_records > 0 and len(documents) >= max_records:
                        break
                    doc = handler(record, idx, split_name, image_dir)
                    documents.append(doc)
                    count += 1
                logger.info(
                    "  split '%s': loaded %d records", split_name, count
                )
        else:
            # Single dataset (no splits).
            for idx, record in enumerate(ds):
                if max_records > 0 and len(documents) >= max_records:
                    break
                # Handlers for non-split datasets don't take a split arg.
                doc = handler(record, idx, image_dir)
                documents.append(doc)

        logger.info(
            "Dataset '%s': %d total documents loaded", name, len(documents)
        )
        return documents
