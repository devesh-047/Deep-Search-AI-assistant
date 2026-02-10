"""
PaddleOCR Engine (Placeholder)
===============================
This module is a **learning hook** for upgrading the OCR pipeline from
Tesseract to PaddleOCR with an OpenVINO backend.

PaddleOCR advantages over Tesseract:
    - Built-in layout analysis (text detection + recognition + direction)
    - Higher accuracy on complex document layouts
    - Models can be converted to OpenVINO IR for hardware acceleration
      on Intel CPU, iGPU, or NPU

Current status:
    NOT IMPLEMENTED.  The class API mirrors TesseractEngine so it can
    serve as a drop-in replacement once implemented.

Learning TODO:
    1. Install PaddleOCR:  pip install paddleocr paddlepaddle
    2. Download the lightweight English detection + recognition models.
    3. Run PaddleOCR on the same samples used for Tesseract and compare.
    4. Export Paddle models to ONNX.
    5. Convert ONNX to OpenVINO IR with ``mo`` (Model Optimizer).
    6. Use openvino.runtime to run the IR models and replace Paddle inference.
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class PaddleOCREngine:
    """
    Placeholder for PaddleOCR-based text extraction.

    The interface matches ``TesseractEngine`` so that switching engines
    requires only changing the import and constructor call.

    All methods currently raise ``NotImplementedError`` with instructions
    for the student.
    """

    def __init__(
        self,
        lang: str = "en",
        use_openvino: bool = False,
        device: str = "CPU",
    ):
        """
        Args:
            lang          : PaddleOCR language code.
            use_openvino  : If True, run inference via OpenVINO IR models.
            device        : OpenVINO device string ("CPU", "GPU", "NPU").
        """
        self.lang = lang
        self.use_openvino = use_openvino
        self.device = device

        logger.warning(
            "PaddleOCREngine is a PLACEHOLDER.  See the docstring for "
            "implementation steps."
        )

    def extract_text(self, image_path: str) -> str:
        """
        Extract plain text from a single image.

        Implementation steps:
            1. Load image with cv2.imread(image_path)
            2. Create PaddleOCR instance:
                   from paddleocr import PaddleOCR
                   ocr = PaddleOCR(use_angle_cls=True, lang=self.lang)
            3. Run:   result = ocr.ocr(image_path, cls=True)
            4. Flatten result into a single string.
        """
        raise NotImplementedError(
            "PaddleOCR not yet implemented.  See docstring for steps."
        )

    def extract_with_boxes(self, image_path: str) -> List[Dict]:
        """
        Extract text with bounding boxes (mirrors TesseractEngine API).

        PaddleOCR returns bounding polygons natively -- convert them to
        the same dict format as TesseractEngine.extract_with_boxes.
        """
        raise NotImplementedError(
            "PaddleOCR box extraction not yet implemented."
        )

    def batch_extract(self, image_paths: List[str]) -> List[Tuple[str, str]]:
        """Run OCR on multiple images."""
        raise NotImplementedError(
            "PaddleOCR batch extraction not yet implemented."
        )
