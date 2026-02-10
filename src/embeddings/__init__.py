"""
Embeddings subpackage -- text-to-vector encoding.

Two backends:
    EmbeddingEncoder      -- sentence-transformers (PyTorch, current)
    OVEmbeddingEncoder    -- OpenVINO IR (future acceleration target)
"""

from src.embeddings.encoder import EmbeddingEncoder
