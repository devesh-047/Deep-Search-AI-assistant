"""
OCR subpackage -- text extraction from document images.

Two engines:
    TesseractEngine  -- baseline, always available
    PaddleOCREngine  -- upgrade with OpenVINO backend acceleration
"""

from src.ocr.tesseract_engine import TesseractEngine

# PaddleOCR is optional — export only when available.
try:
    from src.ocr.paddle_engine import PaddleOCREngine
except ImportError:
    pass
