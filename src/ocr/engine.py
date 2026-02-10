"""
OCR Engine
===========
Extracts text from images and scanned document pages using Tesseract OCR.

Architecture decision:
  Tesseract is the baseline OCR engine because it is:
    - Free and open source
    - Easy to install on Windows + WSL
    - Sufficient for printed text in English
    - A well-understood baseline for measuring improvements

  PaddleOCR is the planned upgrade path because it:
    - Supports more languages and scripts
    - Has higher accuracy on complex layouts
    - Can run on OpenVINO backend (important for this project)

  This module wraps Tesseract behind a clean interface so that swapping
  in PaddleOCR later requires changing only this file.

Prerequisites:
  - Tesseract must be installed on the system:
      WSL:     sudo apt install tesseract-ocr
      Windows: download from https://github.com/UB-Mannheim/tesseract/wiki
  - pytesseract Python package: pip install pytesseract
  - Pillow for image loading: pip install Pillow

Learning TODO:
  1. Test Tesseract on sample images from FUNSD and SROIE datasets.
  2. Experiment with preprocessing (grayscale, thresholding, deskew).
  3. Implement the PaddleOCR backend behind the same interface.
  4. Add OpenVINO backend for PaddleOCR inference.
"""

import logging
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Guard imports so the module can be loaded even if deps are missing.
# This lets the CLI show helpful errors instead of crashing on import.
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class OCREngine:
    """
    Unified OCR interface.  Currently backed by Tesseract.

    Usage:
        engine = OCREngine(backend="tesseract")
        text = engine.extract_text("path/to/image.png")
    """

    SUPPORTED_BACKENDS = {"tesseract", "paddleocr"}

    def __init__(self, backend: str = "tesseract", lang: str = "eng"):
        """
        Args:
            backend : "tesseract" (implemented) or "paddleocr" (placeholder)
            lang    : Tesseract language code, e.g. "eng", "deu", "fra"
        """
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown OCR backend '{backend}'. "
                f"Supported: {self.SUPPORTED_BACKENDS}"
            )
        self.backend = backend
        self.lang = lang
        self._validate_backend()

    def _validate_backend(self) -> None:
        """Check that the selected backend's dependencies are available."""
        if self.backend == "tesseract":
            if not PIL_AVAILABLE:
                raise ImportError("Pillow is required for OCR. Install: pip install Pillow")
            if not TESSERACT_AVAILABLE:
                raise ImportError(
                    "pytesseract is required. Install: pip install pytesseract\n"
                    "Also install Tesseract binary: sudo apt install tesseract-ocr"
                )
        elif self.backend == "paddleocr":
            logger.warning(
                "PaddleOCR backend is a placeholder. "
                "Implement in Phase 2 after Tesseract baseline works."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_text(self, image_path: str) -> str:
        """
        Extract text from a single image file.

        Args:
            image_path: path to an image file (PNG, JPG, TIFF, BMP)

        Returns:
            Extracted text as a string.  Empty string on failure.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error("Image not found: %s", image_path)
            return ""

        if self.backend == "tesseract":
            return self._tesseract_extract(image_path)
        elif self.backend == "paddleocr":
            return self._paddleocr_extract(image_path)
        return ""

    def extract_with_metadata(self, image_path: str) -> Dict:
        """
        Extract text and return it alongside metadata.

        Returns:
            Dict with keys: "text", "backend", "language", "source"
        """
        text = self.extract_text(image_path)
        return {
            "text": text,
            "backend": self.backend,
            "language": self.lang,
            "source": str(image_path),
            "char_count": len(text),
        }

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _tesseract_extract(self, image_path: Path) -> str:
        """Run Tesseract OCR on a single image."""
        try:
            img = Image.open(image_path)
            # Convert to RGB if necessary (Tesseract handles RGB well)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            text = pytesseract.image_to_string(img, lang=self.lang)
            logger.info(
                "Tesseract extracted %d chars from %s", len(text), image_path
            )
            return text.strip()
        except Exception as exc:
            logger.error("Tesseract failed on %s: %s", image_path, exc)
            return ""

    def _paddleocr_extract(self, image_path: Path) -> str:
        """
        Placeholder for PaddleOCR backend.

        Learning TODO:
          1. pip install paddleocr paddlepaddle
          2. Initialize PaddleOCR(use_angle_cls=True, lang='en')
          3. Run ocr on the image and concatenate detected text lines
          4. Later: convert PaddleOCR models to OpenVINO IR format
             and use openvino.runtime for inference.
        """
        logger.warning(
            "[PLACEHOLDER] PaddleOCR backend not yet implemented. "
            "Returning empty string for %s",
            image_path,
        )
        return ""


def preprocess_image(image_path: str) -> Optional[str]:
    """
    Optional preprocessing step to improve OCR accuracy.

    Applies:
      - Grayscale conversion
      - Adaptive thresholding (via OpenCV)

    Learning TODO:
      - pip install opencv-python
      - Experiment with different thresholding methods
      - Measure impact on FUNSD / SROIE accuracy

    Returns:
        Path to the preprocessed image, or None on failure.
    """
    try:
        import cv2
        import numpy as np

        img = cv2.imread(str(image_path))
        if img is None:
            logger.error("OpenCV could not read: %s", image_path)
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Adaptive threshold to handle varying illumination
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        out_path = str(Path(image_path).with_suffix(".preprocessed.png"))
        cv2.imwrite(out_path, thresh)
        logger.info("Preprocessed image saved to %s", out_path)
        return out_path

    except ImportError:
        logger.warning("OpenCV not installed. Skipping preprocessing.")
        return None
    except Exception as exc:
        logger.error("Preprocessing failed for %s: %s", image_path, exc)
        return None
