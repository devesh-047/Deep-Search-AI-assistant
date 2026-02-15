"""
Frame OCR
==========
Runs Tesseract OCR on sampled video frames and associates the extracted
text with timestamps.

Low-content frames (those with very little or no text) are automatically
skipped to reduce noise in the final document.

Dependencies:
    pip install pytesseract  (already in requirements.txt)
    System: sudo apt install tesseract-ocr
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class FrameOCRResult:
    """OCR output for a single video frame.

    Attributes
    ----------
    frame_path : str
        Path to the frame image.
    timestamp : float
        Video timestamp of this frame (seconds).
    text : str
        Extracted text from the frame.
    word_count : int
        Number of words extracted.
    """
    frame_path: str
    timestamp: float
    text: str
    word_count: int


class FrameOCR:
    """Run OCR on sampled frames.

    Usage::

        ocr = FrameOCR(min_word_count=3)
        results = ocr.extract_batch(sampled_frames)
        for r in results:
            print(f"[{r.timestamp:.1f}s] ({r.word_count} words): {r.text[:80]}...")

    Parameters
    ----------
    lang : str
        Tesseract language code (default: ``"eng"``).
    min_word_count : int
        Skip frames with fewer words than this (default: 3).
        Frames with very little text (e.g. plain backgrounds) are noise.
    preprocess : bool
        Apply adaptive-threshold preprocessing before OCR (default: True).
    confidence_threshold : int
        Discard words below this confidence level (0-100, default: 40).
    """

    def __init__(
        self,
        lang: str = "eng",
        min_word_count: int = 3,
        preprocess: bool = True,
        confidence_threshold: int = 40,
    ):
        self.lang = lang
        self.min_word_count = min_word_count
        self.preprocess = preprocess
        self.confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_batch(self, frames) -> List[FrameOCRResult]:
        """Run OCR on a list of ``SampledFrame`` objects.

        Parameters
        ----------
        frames : list
            List of ``SampledFrame`` objects (from ``FrameSampler.sample()``).

        Returns
        -------
        List[FrameOCRResult]
            OCR results for frames that contain meaningful text.
            Frames with fewer than ``min_word_count`` words are excluded.
        """
        if not frames:
            return []

        try:
            import pytesseract
        except ImportError:
            logger.error(
                "pytesseract is not installed.  "
                "Install with: pip install pytesseract"
            )
            return []

        results: List[FrameOCRResult] = []

        pbar = tqdm(
            frames,
            desc="Running OCR on frames",
            unit="frame",
            leave=False,
        )

        for frame in pbar:
            text = self._extract_text(frame.frame_path, pytesseract)
            words = text.split()
            word_count = len(words)

            if word_count >= self.min_word_count:
                results.append(FrameOCRResult(
                    frame_path=frame.frame_path,
                    timestamp=frame.timestamp,
                    text=text,
                    word_count=word_count,
                ))

        logger.info(
            "OCR completed: %d/%d frames had meaningful text (>=%d words)",
            len(results), len(frames), self.min_word_count,
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_text(self, image_path: str, pytesseract_module) -> str:
        """Run Tesseract on a single image with optional preprocessing."""
        import cv2

        img = cv2.imread(image_path)
        if img is None:
            logger.debug("Could not read image: %s", image_path)
            return ""

        if self.preprocess:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            img = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2,
            )

        try:
            # Use TSV output to filter by confidence.
            data = pytesseract_module.image_to_data(
                img, lang=self.lang, output_type=pytesseract_module.Output.DICT
            )
            words = []
            for i, conf in enumerate(data["conf"]):
                try:
                    if int(conf) >= self.confidence_threshold:
                        word = data["text"][i].strip()
                        if word:
                            words.append(word)
                except (ValueError, TypeError):
                    continue
            return " ".join(words)
        except Exception as exc:
            logger.debug("OCR failed on %s: %s", image_path, exc)
            return ""
