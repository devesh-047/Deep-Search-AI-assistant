"""
OpenVINO GenAI LLM Client (Placeholder)
=========================================
Drop-in replacement for ``OllamaClient`` that runs LLM inference via
OpenVINO GenAI instead of Ollama.

Why OpenVINO GenAI for LLMs?
    - Eliminates the dependency on a separate Ollama server process
    - Provides INT4/INT8 weight compression for lower memory usage
    - Can target Intel CPU, iGPU, or NPU through a unified API
    - Pairs with optimum-intel for easy model conversion

Conversion workflow (for the student):
    1. Install optimum-intel:
           pip install optimum-intel[openvino]

    2. Convert Mistral 7B to OpenVINO IR:
           optimum-cli export openvino \\
               --model mistralai/Mistral-7B-Instruct-v0.2 \\
               --weight-format int4 \\
               models/ov/mistral-7b-instruct/

    3. Load the model in this module and run inference using
       ``openvino_genai.LLMPipeline`` (or ``optimum.intel.OVModelForCausalLM``).

Current status:
    NOT IMPLEMENTED.  ``generate()`` returns a placeholder message directing
    the user to Ollama.

Learning TODO:
    1. Install ``openvino-genai`` package.
    2. Run the conversion command above (requires ~16 GB RAM for INT4).
    3. Implement ``generate()`` using ``openvino_genai.LLMPipeline``.
    4. Compare generation quality and latency with Ollama.
    5. Experiment with INT4 vs INT8 quantisation.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class OVLLMClient:
    """
    Placeholder for OpenVINO GenAI LLM inference.

    The interface mirrors ``OllamaClient`` so that switching backends
    requires only changing the import and constructor.
    """

    def __init__(
        self,
        model_dir: str = "",
        device: str = "CPU",
    ):
        """
        Args:
            model_dir : Path to the OpenVINO IR model directory.
            device    : OpenVINO device string ("CPU", "GPU", "NPU").
        """
        self.model_dir = model_dir
        self.device = device
        self._pipeline = None

        logger.warning(
            "OVLLMClient is a PLACEHOLDER.  generate() returns a stub message.  "
            "See docstring for implementation steps."
        )

        if model_dir:
            self._try_load(model_dir, device)

    def _try_load(self, model_dir: str, device: str) -> None:
        """Attempt to load the model.  Fails gracefully."""
        try:
            import openvino_genai
            self._pipeline = openvino_genai.LLMPipeline(model_dir, device)
            logger.info("Loaded OpenVINO LLM from %s on %s", model_dir, device)
        except ImportError:
            logger.warning("openvino-genai not installed -- using placeholder mode")
        except Exception as exc:
            logger.warning("Failed to load OV LLM: %s -- using placeholder mode", exc)

    def is_available(self) -> bool:
        """Check if the model is loaded."""
        return self._pipeline is not None

    def generate(
        self,
        question: str,
        context: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate an answer.

        PLACEHOLDER: returns a stub message until the student implements
        the OpenVINO GenAI inference pipeline.
        """
        if self._pipeline is not None:
            # TODO: construct prompt from context + question, then:
            #   result = self._pipeline.generate(prompt, max_new_tokens=max_tokens, ...)
            #   return result
            pass

        return (
            "[PLACEHOLDER] OVLLMClient is not yet implemented.  "
            "Use OllamaClient for answer generation.  "
            "See src/llm/openvino_llm.py docstring for implementation steps."
        )
