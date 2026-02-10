"""
OpenVINO LLM Inference (Learning Placeholder)
===============================================
This module defines the interface for running a local LLM using
OpenVINO GenAI instead of Ollama.

STATUS: PLACEHOLDER -- not yet functional.
This file exists as a learning hook for Phase 10 of the roadmap.

Why OpenVINO GenAI for LLM inference?
  - Ollama uses llama.cpp under the hood, which is already efficient,
    but it does not leverage Intel-specific hardware optimizations.
  - OpenVINO GenAI provides:
      * Optimized transformer inference on Intel CPUs (AMX, AVX-512)
      * INT4/INT8 weight compression for reduced memory
      * Execution on Intel iGPU or NPU
      * Unified API across all Intel hardware
      * No dependency on NVIDIA CUDA

Workflow (what the student will implement):
  1. Convert Mistral-7B-Instruct to OpenVINO IR:
       optimum-cli export openvino \\
         --model mistralai/Mistral-7B-Instruct-v0.2 \\
         --weight-format int4 \\
         models/ov_llm/

  2. Use OpenVINO GenAI for text generation:
       import openvino_genai as ov_genai
       pipe = ov_genai.LLMPipeline("models/ov_llm/", "CPU")
       result = pipe.generate("prompt here", max_new_tokens=256)

Required packages:
  pip install openvino openvino-genai optimum[openvino]

Learning TODO:
  1. Export Mistral-7B to OpenVINO IR with INT4 compression.
  2. Implement the OVLLMClient class below.
  3. Compare output quality: Ollama vs OpenVINO GenAI.
  4. Benchmark latency and throughput on CPU.
  5. Try GPU device for comparison.
  6. Implement streaming token generation.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Guard OpenVINO GenAI import
try:
    import openvino_genai as ov_genai

    OV_GENAI_AVAILABLE = True
except ImportError:
    OV_GENAI_AVAILABLE = False

# Reuse the same RAG prompt templates as the Ollama client
from src.llm.ollama_client import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE


class OVLLMClient:
    """
    OpenVINO GenAI-based LLM client.

    Drop-in replacement for src/llm/ollama_client.OllamaClient.
    Mirrors the same public API: generate(), is_available().

    STATUS: skeleton only.  The student must complete the implementation.
    """

    def __init__(
        self,
        model_dir: str = "models/ov_llm",
        device: str = "CPU",
    ):
        """
        Args:
            model_dir : directory containing the OpenVINO IR model files
            device    : OpenVINO device -- "CPU", "GPU", "NPU"
        """
        self.model_dir = Path(model_dir)
        self.device = device
        self._pipeline = None

        if not OV_GENAI_AVAILABLE:
            logger.warning(
                "openvino-genai not installed. "
                "Install: pip install openvino-genai"
            )
            return

        # ----------------------------------------------------------
        # TODO (Student): Load the LLM pipeline
        # ----------------------------------------------------------
        # if (self.model_dir / "openvino_model.xml").exists():
        #     self._pipeline = ov_genai.LLMPipeline(
        #         str(self.model_dir), self.device
        #     )
        #     logger.info("Loaded OV LLM pipeline on %s", device)
        # else:
        #     logger.warning("OV LLM model not found at %s", self.model_dir)
        logger.info(
            "[PLACEHOLDER] OVLLMClient created. "
            "Pipeline loading not yet implemented."
        )

    def is_available(self) -> bool:
        """Check if the OpenVINO LLM pipeline is loaded."""
        return self._pipeline is not None

    def generate(
        self,
        question: str,
        context: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate an answer using the OpenVINO LLM pipeline.

        TODO (Student):
          1. Build the prompt from RAG_SYSTEM_PROMPT + RAG_USER_TEMPLATE
          2. Call self._pipeline.generate(prompt, max_new_tokens=max_tokens)
          3. Return the generated text

        For now, returns a placeholder message.
        """
        if not self.is_available():
            return (
                "[PLACEHOLDER] OpenVINO LLM not loaded. "
                "Use OllamaClient as the active backend."
            )

        # ----------------------------------------------------------
        # TODO (Student): implement generation
        # ----------------------------------------------------------
        # system_msg = RAG_SYSTEM_PROMPT.format(context=context)
        # user_msg = RAG_USER_TEMPLATE.format(question=question)
        # full_prompt = f"{system_msg}\n{user_msg}"
        # config = ov_genai.GenerationConfig()
        # config.max_new_tokens = max_tokens
        # config.temperature = temperature
        # result = self._pipeline.generate(full_prompt, config)
        # return result.strip()

        return "[PLACEHOLDER] OpenVINO LLM generation not yet implemented."
