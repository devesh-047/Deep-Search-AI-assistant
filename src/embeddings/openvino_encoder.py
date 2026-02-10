"""
OpenVINO Embedding Encoder (Placeholder)
==========================================
Drop-in replacement for ``EmbeddingEncoder`` that runs the all-MiniLM-L6-v2
model via OpenVINO Runtime instead of PyTorch / sentence-transformers.

Why OpenVINO for embeddings?
    - Reduced latency on Intel CPUs via graph optimizations and threading
    - Access to iGPU and NPU for concurrent inference alongside CPU workloads
    - Smaller runtime footprint (no PyTorch dependency for inference)

Conversion workflow (the student must implement this):
    1. Export sentence-transformers model to ONNX:
           from sentence_transformers import SentenceTransformer
           model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
           model.save("models/st-minilm")
       Then use optimum-intel or the ONNX export:
           optimum-cli export onnx --model sentence-transformers/all-MiniLM-L6-v2 models/onnx/

    2. Convert ONNX to OpenVINO IR:
           mo --input_model models/onnx/model.onnx --output_dir models/ov/all-MiniLM-L6-v2

    3. Load the IR in this module and run inference.

Current status:
    NOT IMPLEMENTED.  ``encode()`` returns zero vectors of the correct shape
    so that downstream code does not crash while the student works on earlier
    pipeline stages.

Learning TODO:
    1. Install openvino + openvino-dev.
    2. Complete the ONNX export and IR conversion above.
    3. Implement the tokeniser (use transformers.AutoTokenizer).
    4. Run inference with openvino.runtime.Core and compare to PyTorch output.
    5. Benchmark CPU vs iGPU latency.
"""

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384


class OVEmbeddingEncoder:
    """
    OpenVINO-accelerated embedding encoder.

    Interface mirrors ``EmbeddingEncoder`` so it can be swapped in without
    changing retrieval or indexing code.
    """

    def __init__(
        self,
        model_xml: str = "",
        device: str = "CPU",
        dimension: int = EMBEDDING_DIM,
    ):
        """
        Args:
            model_xml : Path to the OpenVINO IR ``.xml`` file.
            device    : OpenVINO device string ("CPU", "GPU", "NPU").
            dimension : Expected embedding dimension.
        """
        self.model_xml = model_xml
        self.device = device
        self._dim = dimension
        self._compiled_model = None

        logger.warning(
            "OVEmbeddingEncoder is a PLACEHOLDER.  encode() returns zero vectors.  "
            "See docstring for implementation steps."
        )

        # Attempt to load the model if a path was given.
        if model_xml:
            self._try_load(model_xml, device)

    def _try_load(self, model_xml: str, device: str) -> None:
        """
        Try to compile the IR model.  Fails gracefully if openvino is
        not installed or the model file is missing.
        """
        try:
            from openvino.runtime import Core
            core = Core()
            model = core.read_model(model=model_xml)
            self._compiled_model = core.compile_model(model=model, device_name=device)
            logger.info("Loaded OpenVINO embedding model on %s", device)
        except ImportError:
            logger.warning("openvino not installed -- using placeholder mode")
        except Exception as exc:
            logger.warning("Failed to load OV model: %s -- using placeholder mode", exc)

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
        Encode texts into embedding vectors.

        PLACEHOLDER: returns zero vectors until the student implements
        the tokeniser + OpenVINO inference path described in the docstring.
        """
        if self._compiled_model is not None:
            # TODO: tokenise texts, run compiled_model, collect outputs.
            #   input_ids, attention_mask = tokeniser(texts)
            #   result = self._compiled_model({"input_ids": ..., "attention_mask": ...})
            #   embeddings = result[output_key]
            #   (apply mean pooling + normalisation)
            pass

        # Fallback: zero vectors of the expected shape.
        logger.warning(
            "OVEmbeddingEncoder.encode() returning ZERO vectors for %d texts",
            len(texts),
        )
        return np.zeros((len(texts), self._dim), dtype=np.float32)

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        return self.encode([text], normalize=normalize)[0]


def list_available_devices() -> List[str]:
    """
    Return the OpenVINO devices available on this machine.
    Useful for the ``cli.py devices`` command.
    """
    try:
        from openvino.runtime import Core
        return Core().available_devices
    except ImportError:
        logger.warning("openvino not installed -- cannot list devices")
        return []
    except Exception as exc:
        logger.error("Failed to query OpenVINO devices: %s", exc)
        return []
