"""
Deep Search AI Assistant -- root package.

This package contains the complete RAG pipeline:
    ingestion  -> load and normalise raw documents
    ocr        -> extract text from images via Tesseract / PaddleOCR
    embeddings -> encode text chunks into dense vectors
    index      -> build and query a FAISS vector store
    retrieval  -> orchestrate query-time search
    llm        -> generate answers via Ollama (Mistral 7B) or OpenVINO GenAI
    openvino   -> hardware-aware acceleration helpers
"""

__version__ = "0.1.0"
