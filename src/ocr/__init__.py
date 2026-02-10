"""
OCR subpackage -- text extraction from document images.

Two engines:
    TesseractEngine  -- baseline, always available
    PaddleOCREngine  -- future upgrade with OpenVINO backend (placeholder)
"""

from src.ocr.tesseract_engine import TesseractEngine
