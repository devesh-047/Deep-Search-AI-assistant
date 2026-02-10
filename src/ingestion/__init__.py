"""
Ingestion subpackage -- loading, normalising, and chunking documents.

Pipeline flow:
    DatasetLoader  -->  DocumentNormalizer  -->  TextChunker
    (Arrow/files)       (unified schema)         (embedding-ready chunks)
"""

from src.ingestion.loader import DatasetLoader, RawDocument
from src.ingestion.normalizer import DocumentNormalizer, NormalizedDocument
from src.ingestion.chunker import TextChunker, TextChunk
