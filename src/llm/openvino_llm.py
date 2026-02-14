"""
OpenVINO GenAI LLM Client
===========================
Drop-in replacement for ``OllamaClient`` that runs LLM inference via
OpenVINO GenAI instead of Ollama.

Why OpenVINO GenAI for LLMs?
    - Eliminates the dependency on a separate Ollama server process
    - Provides INT4/INT8 weight compression for lower memory usage
    - Can target Intel CPU, iGPU, or NPU through a unified API
    - Pairs with optimum-intel for easy model conversion

Conversion workflow:
    1. Install optimum-intel:
           pip install optimum-intel[openvino]

    2. Convert Mistral 7B to OpenVINO IR:
           optimum-cli export openvino \\
               --model mistralai/Mistral-7B-Instruct-v0.2 \\
               --weight-format int4 \\
               models/ov/mistral-7b-instruct/

    3. This module loads the converted model and runs inference using
       ``openvino_genai.LLMPipeline`` (primary) or
       ``optimum.intel.OVModelForCausalLM`` (fallback).

Learning TODO (all implemented):
    1. ✅ Install ``openvino-genai`` package.
    2. ✅ Run the conversion command above (requires ~16 GB RAM for INT4).
    3. ✅ Implement ``generate()`` using ``openvino_genai.LLMPipeline``.
    4. ✅ Compare generation quality and latency with Ollama.
    5. ✅ Experiment with INT4 vs INT8 quantisation.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RAG prompt template (matches OllamaClient's default template)
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context.
Use ONLY the information from the context below to answer the question.
If the context does not contain enough information to answer, say so honestly.
Do not make up information.

CONTEXT:
{context}"""

RAG_USER_TEMPLATE = """Based on the context provided, please answer the following question:

{question}"""


class OVLLMClient:
    """
    OpenVINO GenAI LLM inference client.

    The interface mirrors ``OllamaClient`` so that switching backends
    requires only changing the import and constructor.

    Two backends are supported:
        1. ``openvino_genai.LLMPipeline`` — lightweight, purpose-built
           for text generation on OpenVINO devices.  Preferred.
        2. ``optimum.intel.OVModelForCausalLM`` — HuggingFace-compatible
           wrapper.  Used as fallback if openvino-genai is not installed.

    Usage::

        client = OVLLMClient(
            model_dir="models/ov/mistral-7b-instruct",
            device="CPU",
        )
        if client.is_available():
            answer = client.generate(
                question="What is the total amount?",
                context="The invoice shows a total of $150.00",
            )
    """

    def __init__(
        self,
        model_dir: str = "",
        device: str = "CPU",
    ):
        """
        Args:
            model_dir : Path to the OpenVINO IR model directory.
                        The directory should contain the converted model
                        files (.xml, .bin, tokenizer config, etc.).
            device    : OpenVINO device string ("CPU", "GPU", "NPU").
        """
        self.model_dir = model_dir
        self.device = device
        self._pipeline = None       # openvino_genai.LLMPipeline
        self._ov_model = None       # optimum OVModelForCausalLM
        self._tokenizer = None      # transformers AutoTokenizer
        self._backend = None        # "genai" or "optimum" or None

        if model_dir:
            self._try_load(model_dir, device)
        else:
            logger.info(
                "OVLLMClient created without model_dir — call is_available() "
                "before generate()."
            )

    def _try_load(self, model_dir: str, device: str) -> None:
        """
        Attempt to load the model using available backends.

        Tries openvino_genai first (lightweight, fast), then falls back
        to optimum.intel (HuggingFace-compatible).

        Why two backends?
        -----------------
        - ``openvino_genai`` is purpose-built for text generation and
          handles tokenization + generation in one call.  However, it
          requires the ``openvino-genai`` pip package which may not be
          installed.
        - ``optimum.intel`` uses HuggingFace's AutoTokenizer +
          OVModelForCausalLM, which is more widely available but
          heavier.
        """
        model_path = Path(model_dir)
        if not model_path.exists():
            logger.warning("Model directory does not exist: %s", model_dir)
            return

        # --- Backend 1: openvino_genai.LLMPipeline ---
        try:
            import openvino_genai

            self._pipeline = openvino_genai.LLMPipeline(
                str(model_path), device
            )
            self._backend = "genai"
            logger.info(
                "Loaded OpenVINO LLM (genai backend) from %s on %s",
                model_dir, device,
            )
            return
        except ImportError:
            logger.debug("openvino-genai not installed, trying optimum fallback")
        except Exception as exc:
            logger.warning("openvino_genai failed: %s — trying optimum", exc)

        # --- Backend 2: optimum.intel.OVModelForCausalLM ---
        try:
            from optimum.intel import OVModelForCausalLM
            from transformers import AutoTokenizer

            self._ov_model = OVModelForCausalLM.from_pretrained(
                str(model_path), device=device
            )
            self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self._backend = "optimum"
            logger.info(
                "Loaded OpenVINO LLM (optimum backend) from %s on %s",
                model_dir, device,
            )
            return
        except ImportError:
            logger.warning(
                "Neither openvino-genai nor optimum-intel installed. "
                "Install one of: pip install openvino-genai | "
                "pip install optimum-intel[openvino]"
            )
        except Exception as exc:
            logger.warning("optimum.intel failed: %s", exc)

        logger.warning(
            "OVLLMClient could not load model from %s — "
            "generate() will return a fallback message.",
            model_dir,
        )

    def is_available(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self._backend is not None

    @staticmethod
    def build_rag_prompt(
        question: str,
        context: str,
        template: str = "default",
    ) -> Dict[str, str]:
        """
        Build RAG prompt messages.

        Matches OllamaClient.build_rag_prompt() interface so the
        calling code in cli.py doesn't need to change.

        Args:
            question : the user's question
            context  : formatted context from Retriever
            template : prompt template name (only "default" supported)

        Returns:
            Dict with "system" and "user" keys.
        """
        system_msg = RAG_SYSTEM_PROMPT.format(context=context) if context else ""
        user_msg = RAG_USER_TEMPLATE.format(question=question)
        return {"system": system_msg, "user": user_msg}

    def generate(
        self,
        question: str,
        context: str = "",
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        template: str = "default",
        preset: Optional[str] = None,
        debug: bool = False,
    ) -> str:
        """
        Generate an answer using RAG prompt construction.

        The flow:
          1. Build RAG prompt from template + context + question
          2. Send to the OpenVINO LLM pipeline
          3. Return the generated text

        When the model is not loaded, returns a helpful fallback message.

        Args:
            question    : the user's natural-language question
            context     : retrieved context chunks
            temperature : controls randomness (lower = more deterministic)
            top_p       : nucleus sampling threshold
            max_tokens  : maximum tokens in the generated response
            template    : prompt template name
            preset      : generation preset (unused, kept for API compat)
            debug       : if True, log the full prompt

        Returns:
            The generated answer as a string.
        """
        # Build prompt
        prompt_parts = self.build_rag_prompt(question, context, template)
        system_msg = prompt_parts["system"]
        user_msg = prompt_parts["user"]

        if debug:
            print("\n" + "=" * 60)
            print("DEBUG: OpenVINO LLM Prompt")
            print("=" * 60)
            print(f"Backend: {self._backend}")
            print(f"Temperature: {temperature}, Top-p: {top_p}")
            print(f"--- SYSTEM ---\n{system_msg}")
            print(f"--- USER ---\n{user_msg}")
            print("=" * 60 + "\n")

        # Combine into a single prompt for the model
        # Mistral Instruct format: [INST] {system}\n{user} [/INST]
        if system_msg:
            full_prompt = f"[INST] {system_msg}\n\n{user_msg} [/INST]"
        else:
            full_prompt = f"[INST] {user_msg} [/INST]"

        # --- Generate with openvino_genai backend ---
        if self._backend == "genai":
            return self._generate_genai(
                full_prompt, temperature, top_p, max_tokens
            )

        # --- Generate with optimum backend ---
        if self._backend == "optimum":
            return self._generate_optimum(
                full_prompt, temperature, top_p, max_tokens
            )

        # --- No backend available ---
        return (
            "[OVLLMClient] Model not loaded. To use OpenVINO for LLM inference:\n"
            "  1. pip install openvino-genai\n"
            "  2. Convert model: optimum-cli export openvino "
            "--model mistralai/Mistral-7B-Instruct-v0.2 "
            "--weight-format int4 models/ov/mistral-7b-instruct/\n"
            "  3. Pass model_dir='models/ov/mistral-7b-instruct' to OVLLMClient\n"
            "\nFalling back: use 'python cli.py ask' with Ollama instead."
        )

    def _generate_genai(
        self, prompt: str, temperature: float, top_p: float, max_tokens: int
    ) -> str:
        """
        Generate using openvino_genai.LLMPipeline.

        The LLMPipeline handles tokenization internally, so we just
        pass the string prompt and generation config.

        How LLMPipeline works:
        ----------------------
        1. Tokenizes the prompt using the bundled tokenizer
        2. Runs the OpenVINO model in an autoregressive loop:
           - Feed token IDs through the model
           - Sample the next token from the output logits
           - Append the new token and repeat
        3. Decodes the generated token IDs back to text
        4. Stops when max_tokens reached or EOS token generated
        """
        try:
            import openvino_genai

            config = openvino_genai.GenerationConfig()
            config.max_new_tokens = max_tokens
            config.temperature = temperature
            config.top_p = top_p
            # do_sample must be True for temperature/top_p to take effect
            config.do_sample = temperature > 0

            start = time.perf_counter()
            result = self._pipeline.generate(prompt, config)
            elapsed = time.perf_counter() - start

            # result is a string (the generated text)
            answer = str(result).strip()
            logger.info(
                "OpenVINO LLM (genai): %d chars in %.1fs for: '%s'",
                len(answer), elapsed, prompt[:60],
            )
            return answer

        except Exception as exc:
            logger.error("OpenVINO genai generation failed: %s", exc)
            return f"[ERROR] OpenVINO generation failed: {exc}"

    def _generate_optimum(
        self, prompt: str, temperature: float, top_p: float, max_tokens: int
    ) -> str:
        """
        Generate using optimum.intel OVModelForCausalLM.

        This backend uses the standard HuggingFace generate() API:
        1. Tokenize the prompt with AutoTokenizer
        2. Call model.generate() with the token IDs
        3. Decode the output tokens back to text

        This pathway is heavier than genai but more flexible
        (supports all HuggingFace generation parameters).
        """
        try:
            start = time.perf_counter()

            inputs = self._tokenizer(prompt, return_tensors="pt")
            input_len = inputs["input_ids"].shape[1]

            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p

            output_ids = self._ov_model.generate(
                **inputs, **gen_kwargs
            )
            # Slice off the input tokens to get only the generated part
            new_tokens = output_ids[0][input_len:]
            answer = self._tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()

            elapsed = time.perf_counter() - start
            logger.info(
                "OpenVINO LLM (optimum): %d chars in %.1fs for: '%s'",
                len(answer), elapsed, prompt[:60],
            )
            return answer

        except Exception as exc:
            logger.error("OpenVINO optimum generation failed: %s", exc)
            return f"[ERROR] OpenVINO generation failed: {exc}"

    def benchmark(
        self,
        prompt: str = "What is machine learning?",
        max_tokens: int = 100,
        n_runs: int = 3,
    ) -> Dict[str, float]:
        """
        Benchmark LLM generation speed.

        Runs generation multiple times and reports:
        - Total time per run
        - Tokens per second (approximate)
        - Characters generated per second

        Useful for comparing CPU vs GPU performance, or INT4 vs INT8.

        Args:
            prompt     : Test prompt to generate from
            max_tokens : Max tokens to generate per run
            n_runs     : Number of timed runs

        Returns:
            Dict with timing statistics.
        """
        if not self.is_available():
            return {"error": "Model not loaded"}

        times = []
        char_counts = []

        for i in range(n_runs):
            start = time.perf_counter()
            result = self.generate(
                question=prompt,
                max_tokens=max_tokens,
                temperature=0.0,  # deterministic for benchmarking
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            char_counts.append(len(result))
            logger.info("Benchmark run %d: %.2fs, %d chars", i + 1, elapsed, len(result))

        import numpy as np
        times_arr = np.array(times)
        chars_arr = np.array(char_counts)

        return {
            "backend": self._backend,
            "device": self.device,
            "mean_time_s": float(times_arr.mean()),
            "min_time_s": float(times_arr.min()),
            "max_time_s": float(times_arr.max()),
            "mean_chars": float(chars_arr.mean()),
            "chars_per_sec": float(chars_arr.mean() / times_arr.mean()),
        }
