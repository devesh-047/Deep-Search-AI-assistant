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
    - Supports Tesseract page-segmentation modes (PSM) for different
      document layouts.
    - Includes utilities for comparing OCR output with ground truth and
      tuning confidence/preprocessing settings.
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
        psm: int = 3,
    ):
        """
        Args:
            lang                 : Tesseract language code (e.g. "eng").
            preprocess           : Apply adaptive-threshold preprocessing.
            confidence_threshold : Discard words with confidence below this
                                   value (0-100).
            psm                  : Page segmentation mode (0-13). Common:
                                   3 = Fully automatic (default)
                                   6 = Uniform block of text
                                   11 = Sparse text, find as much as possible
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
        self.psm = psm

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
            config = f"--psm {self.psm}"
            text = pytesseract.image_to_string(img, lang=self.lang, config=config)
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
            config = f"--psm {self.psm}"
            data = pytesseract.image_to_data(
                img, lang=self.lang, config=config, output_type=pytesseract.Output.DICT
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
        try:
            from tqdm import tqdm
            iterator = tqdm(image_paths, desc="OCR", unit="img")
        except ImportError:
            iterator = image_paths
            
        results: List[Tuple[str, str]] = []
        for path in iterator:
            text = self.extract_text(path)
            results.append((path, text))
        logger.info("Batch OCR: processed %d images", len(results))
        return results

    def compare_with_ground_truth(
        self, image_path: str, ground_truth_words: List[str]
    ) -> Dict:
        """
        Compare OCR output against ground-truth words.

        Useful for evaluating OCR accuracy on datasets like FUNSD that
        include ground-truth word annotations.

        Args:
            image_path         : Path to the image.
            ground_truth_words : List of expected words (e.g. from FUNSD).

        Returns:
            dict with keys:
                - ocr_text       : The extracted text string.
                - ocr_words      : List of OCR words.
                - ground_truth   : List of ground-truth words.
                - word_accuracy  : % of ground-truth words found in OCR.
                - precision      : % of OCR words that match ground truth.
                - recall         : % of ground-truth words found in OCR.
        """
        ocr_text = self.extract_text(image_path)
        ocr_words = [w.lower() for w in ocr_text.split() if w.strip()]
        gt_words = [w.lower() for w in ground_truth_words if w.strip()]

        # Calculate word-level metrics
        gt_set = set(gt_words)
        ocr_set = set(ocr_words)

        matches = gt_set & ocr_set
        precision = len(matches) / len(ocr_set) if ocr_set else 0.0
        recall = len(matches) / len(gt_set) if gt_set else 0.0

        return {
            "ocr_text": ocr_text,
            "ocr_words": ocr_words,
            "ground_truth": gt_words,
            "word_accuracy": recall * 100,  # % of GT words found
            "precision": precision * 100,
            "recall": recall * 100,
            "ocr_word_count": len(ocr_words),
            "gt_word_count": len(gt_words),
        }

    def tune_settings(
        self,
        image_path: str,
        ground_truth_words: List[str],
        test_configs: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        Test multiple OCR configurations and rank by accuracy.

        Useful for finding optimal settings for a specific document type.

        Args:
            image_path         : Path to a representative image.
            ground_truth_words : Expected words for accuracy measurement.
            test_configs       : List of config dicts with keys:
                                 'preprocess', 'confidence_threshold', 'psm'.
                                 If None, uses a sensible default grid.

        Returns:
            List of result dicts sorted by recall (best first), each with:
                - config         : The tested configuration.
                - word_accuracy  : % of ground-truth words found.
                - ocr_word_count : Number of words extracted.
        """
        if test_configs is None:
            # Default test grid
            test_configs = [
                {"preprocess": False, "confidence_threshold": 40, "psm": 3},
                {"preprocess": True, "confidence_threshold": 40, "psm": 3},
                {"preprocess": True, "confidence_threshold": 30, "psm": 3},
                {"preprocess": True, "confidence_threshold": 50, "psm": 3},
                {"preprocess": True, "confidence_threshold": 40, "psm": 6},
                {"preprocess": True, "confidence_threshold": 40, "psm": 11},
            ]

        results: List[Dict] = []
        for cfg in test_configs:
            # Create a temporary engine with this config
            engine = TesseractEngine(
                lang=self.lang,
                preprocess=cfg.get("preprocess", True),
                confidence_threshold=cfg.get("confidence_threshold", 40),
                psm=cfg.get("psm", 3),
            )
            comparison = engine.compare_with_ground_truth(
                image_path, ground_truth_words
            )
            results.append({
                "config": cfg,
                "word_accuracy": comparison["word_accuracy"],
                "precision": comparison["precision"],
                "recall": comparison["recall"],
                "ocr_word_count": comparison["ocr_word_count"],
            })

        # Sort by recall (descending)
        results.sort(key=lambda r: r["recall"], reverse=True)
        logger.info(
            "Tuned %d configs for %s. Best recall: %.1f%%",
            len(results), image_path, results[0]["recall"],
        )
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
