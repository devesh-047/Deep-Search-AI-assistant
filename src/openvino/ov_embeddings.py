"""
OpenVINO Embeddings (Learning Placeholder)
============================================
This module defines the interface for running the embedding model
(all-MiniLM-L6-v2) using OpenVINO Runtime instead of PyTorch.

STATUS: PLACEHOLDER -- not yet functional.
This file exists as a learning hook for Phase 9 of the roadmap.

Why OpenVINO for embeddings?
  - The sentence-transformers model runs on PyTorch by default.
  - Converting to OpenVINO IR format allows:
      * Faster inference on Intel CPUs (vectorized kernels, INT8 quantization)
      * Execution on Intel iGPU or NPU via the same API
      * Reduced memory footprint
      * No PyTorch dependency at inference time

Conversion workflow (what the student will implement):
  1. Export sentence-transformers model to ONNX:
       from sentence_transformers import SentenceTransformer
       model = SentenceTransformer("all-MiniLM-L6-v2")
       # Use optimum-intel or manual torch.onnx.export

  2. Convert ONNX to OpenVINO IR:
       from openvino.tools import mo
       mo.convert_model("model.onnx", ...)

  3. Load IR in OpenVINO Runtime:
       from openvino.runtime import Core
       core = Core()
       model = core.read_model("model.xml")
       compiled = core.compile_model(model, "CPU")  # or "GPU", "NPU"

  4. Run inference:
       result = compiled([input_ids, attention_mask])
       embeddings = result[0]  # post-process: mean pooling + normalize

Required packages:
  pip install openvino openvino-dev optimum[openvino]

Learning TODO:
  1. Follow the conversion workflow above.
  2. Implement the OVEmbeddingEncoder class below.
  3. Verify output matches the PyTorch encoder (cosine sim > 0.99).
  4. Benchmark: compare latency of PyTorch vs OpenVINO.
  5. Try different devices: CPU, GPU, NPU (if available).
  6. Try INT8 quantization with NNCF and measure accuracy/speed tradeoff.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Guard OpenVINO import
try:
    from openvino.runtime import Core

    OV_AVAILABLE = True
except ImportError:
    OV_AVAILABLE = False


class OVEmbeddingEncoder:
    """
    OpenVINO-accelerated embedding encoder.

    Drop-in replacement for src/embeddings/encoder.EmbeddingEncoder.
    Mirrors the same public API: encode(), encode_single(), dimension.

    STATUS: skeleton only.  The student must complete the implementation.
    """

    def __init__(
        self,
        model_dir: str = "models/ov_embeddings",
        device: str = "CPU",
    ):
        """
        Args:
            model_dir : directory containing model.xml and model.bin
            device    : OpenVINO device string -- "CPU", "GPU", "NPU"
        """
        self.model_dir = Path(model_dir)
        self.device = device
        self._compiled_model = None
        self._dim = 384  # all-MiniLM-L6-v2 output dimension

        if not OV_AVAILABLE:
            logger.warning(
                "OpenVINO runtime not installed. "
                "Install: pip install openvino"
            )
            return

        # ----------------------------------------------------------
        # TODO (Student): Load the model
        # ----------------------------------------------------------
        # model_xml = self.model_dir / "model.xml"
        # if model_xml.exists():
        #     core = Core()
        #     model = core.read_model(str(model_xml))
        #     self._compiled_model = core.compile_model(model, device)
        #     logger.info("Loaded OV embedding model on %s", device)
        # else:
        #     logger.warning("OV model not found at %s", model_xml)
        logger.info(
            "[PLACEHOLDER] OVEmbeddingEncoder created. "
            "Model loading not yet implemented."
        )

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode texts using OpenVINO inference.

        TODO (Student):
          1. Tokenize texts using the saved tokenizer
          2. Run inference through self._compiled_model
          3. Apply mean pooling over token embeddings
          4. Optionally L2-normalize
          5. Return np.ndarray of shape (len(texts), self._dim)
        """
        logger.warning(
            "[PLACEHOLDER] OV encode() not implemented. "
            "Returning zero vectors for %d texts.",
            len(texts),
        )
        return np.zeros((len(texts), self._dim), dtype=np.float32)

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode a single text string."""
        return self.encode([text], normalize=normalize)[0]


def list_available_devices() -> List[str]:
    """
    List OpenVINO devices available on this system.

    Useful for hardware-aware execution:
      - "CPU"  : always available
      - "GPU"  : Intel iGPU (if present and drivers installed)
      - "NPU"  : Intel NPU (AI PC, Meteor Lake+)

    Returns:
        List of device strings, e.g. ["CPU", "GPU"]
    """
    if not OV_AVAILABLE:
        logger.warning("OpenVINO not installed. Cannot list devices.")
        return []

    core = Core()
    devices = core.available_devices
    logger.info("Available OpenVINO devices: %s", devices)
    return devices
