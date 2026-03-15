"""
Embeddings subpackage -- text-to-vector encoding.

Three backends:
    EmbeddingEncoder      -- sentence-transformers (PyTorch)
    OVEmbeddingEncoder    -- OpenVINO IR acceleration
    CLIPEncoder           -- CLIP multimodal embeddings (images + text)
"""

from src.embeddings.encoder import EmbeddingEncoder

# Optional backends — export only when available.
try:
    from src.embeddings.openvino_encoder import OVEmbeddingEncoder
except ImportError:
    pass

try:
    from src.embeddings.clip_encoder import CLIPEncoder
except ImportError:
    pass
