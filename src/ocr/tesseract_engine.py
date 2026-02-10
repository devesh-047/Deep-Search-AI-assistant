"""
Tesseract OCR Engine
=====================
Extracts text from document images using Tesseract via the pytesseract
Python binding.

Prerequisites:
    1. System package:   sudo apt install tesseract-ocr
    2. Python binding:   pip install pytesseract pillow

Design notes:
    - Images can optionally be preprocessed (grayscale + adaptive threshold)
      to improve OCR accuracy on noisy scans.
    - Each image produces a single text string; bounding-box-level results
      are available via ``extract_with_boxes`` for layout-aware pipelines.
    - The engine operates on images already extracted to disk (by the
      DatasetLoader) and referenced via ``NormalizedDocument.image_path``.

Learning TODO:
    1. Run OCR on a FUNSD sample and compare output to the ground-truth
       ``words`` column.
    2. Tune the confidence threshold and measure the effect on text quality.
    3. Try different Tesseract page-segmentation modes (--psm flag).
    4. Test preprocessing on/off and measure word-error-rate differences.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Guard imports -- pytesseract and PIL may not be installed yet.
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class TesseractEngine:
    """
    Tesseract-based OCR for document images.

    Usage::

        engine = TesseractEngine(lang="eng", preprocess=True)
        text = engine.extract_text("data/processed/ocr_cache/funsd/funsd_train_00000.png")
    """

    def __init__(
        self,
        lang: str = "eng",
        preprocess: bool = True,
        confidence_threshold: int = 40,
    ):
        """
        Args:
            lang                 : Tesseract language code (e.g. "eng").
            preprocess           : Apply adaptive-threshold preprocessing.
            confidence_threshold : Discard words with confidence below this
                                   value (0-100).
        """
        if not TESSERACT_AVAILABLE:
            raise ImportError(
                "pytesseract is required.  Install: pip install pytesseract  "
                "Also install the system binary: sudo apt install tesseract-ocr"
            )
        if not PIL_AVAILABLE:
            raise ImportError("Pillow is required.  Install: pip install pillow")

        self.lang = lang
        self.preprocess = preprocess
        self.confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def extract_text(self, image_path: str) -> str:
        """
        Extract plain text from a single image.

        Returns an empty string if the image cannot be read or OCR yields
        nothing above the confidence threshold.
        """
        img = self._load_image(image_path)
        if img is None:
            return ""

        if self.preprocess and CV2_AVAILABLE:
            img = self._preprocess(img)

        try:
            text = pytesseract.image_to_string(img, lang=self.lang)
            return text.strip()
        except Exception as exc:
            logger.error("OCR failed for %s: %s", image_path, exc)
            return ""

    def extract_with_boxes(self, image_path: str) -> List[Dict]:
        """
        Extract text with bounding-box and confidence information.

        Returns a list of dicts with keys: ``text``, ``conf``, ``left``,
        ``top``, ``width``, ``height``.  Words below the confidence
        threshold are excluded.
        """
        img = self._load_image(image_path)
        if img is None:
            return []

        if self.preprocess and CV2_AVAILABLE:
            img = self._preprocess(img)

        try:
            data = pytesseract.image_to_data(
                img, lang=self.lang, output_type=pytesseract.Output.DICT
            )
        except Exception as exc:
            logger.error("OCR (boxes) failed for %s: %s", image_path, exc)
            return []

        results: List[Dict] = []
        n_words = len(data["text"])
        for i in range(n_words):
            conf = int(data["conf"][i])
            word = data["text"][i].strip()
            if conf < self.confidence_threshold or not word:
                continue
            results.append({
                "text": word,
                "conf": conf,
                "left": data["left"][i],
                "top": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i],
            })
        logger.info(
            "OCR boxes for %s: %d words (threshold=%d)",
            image_path, len(results), self.confidence_threshold,
        )
        return results

    def batch_extract(self, image_paths: List[str]) -> List[Tuple[str, str]]:
        """
        Run OCR on multiple images.

        Returns a list of (image_path, extracted_text) tuples.
        """
        results: List[Tuple[str, str]] = []
        for path in image_paths:
            text = self.extract_text(path)
            results.append((path, text))
        logger.info("Batch OCR: processed %d images", len(results))
        return results

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_image(image_path: str) -> Optional["Image.Image"]:
        """Load an image from disk, returning None on failure."""
        p = Path(image_path)
        if not p.exists():
            logger.warning("Image not found: %s", image_path)
            return None
        try:
            return Image.open(p)
        except Exception as exc:
            logger.error("Failed to open image %s: %s", image_path, exc)
            return None

    @staticmethod
    def _preprocess(img: "Image.Image") -> "Image.Image":
        """
        Adaptive-threshold preprocessing for noisy scans.

        Steps:
            1. Convert to grayscale
            2. Apply Gaussian blur (reduces noise)
            3. Apply adaptive threshold (enhances text / background contrast)
            4. Convert back to PIL Image for pytesseract

        Requires opencv-python (cv2) and numpy.
        """
        if not CV2_AVAILABLE:
            return img

        # PIL -> numpy (grayscale)
        gray = np.array(img.convert("L"))

        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )

        # numpy -> PIL
        return Image.fromarray(binary)
