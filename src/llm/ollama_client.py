"""
Ollama LLM Client
===================
Handles answer generation using a locally-running Mistral 7B Instruct
model served by Ollama.

Architecture decisions:
  - Ollama provides a simple HTTP API for local LLM inference.
  - The LLM is used ONLY for answer generation, NOT for retrieval.
  - Retrieval is handled by embedding-based search (src/retrieval/).
  - The RAG prompt is constructed here: context chunks from the retriever
    are injected into the system prompt before the user's question.

Why Mistral 7B Instruct?
  - 7B parameters: runs on consumer hardware with ~6 GB VRAM or CPU
  - Instruct-tuned: follows instructions well for Q&A tasks
  - Open license: suitable for offline, private use
  - Good balance of quality vs resource requirements

Prerequisites:
  1. Install Ollama: https://ollama.ai/download
  2. Pull the model: ollama pull mistral
  3. Ollama must be running: ollama serve (background process)

Learning TODO (all implemented):
  1. ✅ Test basic generation with Ollama API.
  2. ✅ Implement RAG prompt template and measure answer quality.
  3. ✅ Experiment with temperature and top_p for different use cases.
  4. ✅ Add streaming response support.
  5. ✅ Replace with OpenVINO GenAI inference (stub ready).
"""

import json
import logging
import time
import sys
from typing import Optional, Dict, List, Generator

logger = logging.getLogger(__name__)

try:
    import urllib.request
    import urllib.error

    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral"

# ---------------------------------------------------------------------------
#  TODO 2 — RAG Prompt Templates
# ---------------------------------------------------------------------------
# We provide multiple prompt templates so you can experiment with
# different prompting strategies. Each template defines a system
# prompt and a user message format.
#
# Why templates matter:
#   The LLM has no memory of your documents. It receives the context
#   and question once, and must produce an answer from that single
#   input. The *way* you frame the context (system prompt) directly
#   affects:
#     - Whether the LLM stays grounded (no hallucination)
#     - Whether it cites sources
#     - How concise or verbose the answer is
#     - Whether it admits uncertainty when context is insufficient

RAG_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context.
Use ONLY the information from the context below to answer the question.
If the context does not contain enough information to answer, say so honestly.
Do not make up information.

CONTEXT:
{context}
"""

RAG_USER_TEMPLATE = """Based on the context provided, please answer the following question:

{question}"""


# ------------------------------------------------------------------
# Additional prompt templates for experimentation (TODO 2)
# ------------------------------------------------------------------

# Template 2: Strict citation — forces the LLM to reference
# specific source numbers from format_context() output.
RAG_SYSTEM_PROMPT_CITED = """You are a precise document analysis assistant.
Answer questions using ONLY the numbered sources below.
For every claim in your answer, cite the source in [Source N] format.
If no source supports an answer, say "The provided documents do not contain this information."

SOURCES:
{context}
"""

RAG_USER_TEMPLATE_CITED = """Question: {question}

Provide a clear, sourced answer:"""


# Template 3: Concise — for short, factual answers (like dates,
# names, amounts). Uses lower temperature for determinism.
RAG_SYSTEM_PROMPT_CONCISE = """Extract the answer from the context below.
Reply with ONLY the answer — no explanation, no preamble.
If the answer is not in the context, reply "Not found."

Context:
{context}
"""

RAG_USER_TEMPLATE_CONCISE = """{question}"""


# Template registry for easy switching
PROMPT_TEMPLATES: Dict[str, Dict[str, str]] = {
    "default": {
        "system": RAG_SYSTEM_PROMPT,
        "user": RAG_USER_TEMPLATE,
        "description": "Balanced: grounded answers, admits uncertainty",
    },
    "cited": {
        "system": RAG_SYSTEM_PROMPT_CITED,
        "user": RAG_USER_TEMPLATE_CITED,
        "description": "Strict citations: every claim references [Source N]",
    },
    "concise": {
        "system": RAG_SYSTEM_PROMPT_CONCISE,
        "user": RAG_USER_TEMPLATE_CONCISE,
        "description": "Short factual answers: dates, names, amounts",
    },
}


# ---------------------------------------------------------------------------
#  TODO 3 — Temperature / top_p Presets
# ---------------------------------------------------------------------------
# Temperature and top_p control how the LLM samples its next token.
#
# Temperature:
#   - Controls the "randomness" of the output.
#   - At temperature=0: the model always picks the highest-probability
#     token (greedy decoding). Deterministic but may be repetitive.
#   - At temperature=1: sampling follows the learned distribution.
#     More varied but can hallucinate.
#   - At temperature>1: distribution is flattened — more random,
#     creative, but less reliable.
#
# top_p (nucleus sampling):
#   - Instead of considering ALL possible next tokens, only consider
#     the smallest set whose cumulative probability >= top_p.
#   - top_p=0.9: consider the top ~90% probability mass.
#   - top_p=0.1: consider only the very highest-probability tokens.
#   - Lower top_p = more deterministic, higher top_p = more diverse.
#
# For RAG (question answering over documents):
#   - LOW temperature (0.1–0.3) is best: we want factual, grounded
#     answers, not creative writing.
#   - HIGH temperature (0.7–1.0) is for brainstorming, creative tasks.
#
# Presets below capture common use cases:

GENERATION_PRESETS: Dict[str, Dict] = {
    "precise": {
        "temperature": 0.1,
        "top_p": 0.5,
        "description": "Most deterministic. Best for factual Q&A, dates, amounts.",
    },
    "balanced": {
        "temperature": 0.3,
        "top_p": 0.9,
        "description": "Default RAG setting. Grounded but natural language.",
    },
    "creative": {
        "temperature": 0.7,
        "top_p": 0.95,
        "description": "More varied responses. Good for summaries, explanations.",
    },
    "exploratory": {
        "temperature": 1.0,
        "top_p": 1.0,
        "description": "Maximum diversity. Not recommended for RAG — may hallucinate.",
    },
}


class OllamaClient:
    """
    Client for the Ollama local LLM server.

    Supports:
      - Basic generation (TODO 1)
      - RAG prompt construction with multiple templates (TODO 2)
      - Temperature/top_p presets (TODO 3)
      - Streaming responses (TODO 4)

    Usage:
        client = OllamaClient()
        if client.is_available():
            answer = client.generate("What is this document about?", context="...")

    With presets:
        answer = client.generate(question, context=context, preset="precise")

    With streaming:
        for chunk in client.generate_stream(question, context=context):
            print(chunk, end="", flush=True)

    With templates:
        answer = client.generate(question, context=context, template="cited")
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MODEL,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if Ollama server is running and the model is pulled."""
        try:
            url = f"{self.base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m["name"] for m in data.get("models", [])]
                # Ollama model names may include tags like ":latest"
                available = any(self.model in m for m in models)
                if not available:
                    logger.warning(
                        "Model '%s' not found. Available: %s. "
                        "Run: ollama pull %s",
                        self.model,
                        models,
                        self.model,
                    )
                return available
        except Exception as exc:
            logger.error("Ollama not reachable at %s: %s", self.base_url, exc)
            return False

    def list_models(self) -> List[str]:
        """List all models available on the Ollama server."""
        try:
            url = f"{self.base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    # ------------------------------------------------------------------
    #  TODO 2 — RAG Prompt Building
    # ------------------------------------------------------------------

    @staticmethod
    def build_rag_prompt(
        question: str,
        context: str,
        template: str = "default",
    ) -> Dict[str, str]:
        """
        Build the system and user messages for a RAG query.

        Why this is a separate method:
            Making prompt construction explicit lets you inspect,
            log, and debug the exact text sent to the LLM. This is
            critical for understanding RAG answer quality.

        Args:
            question : the user's question
            context  : formatted context from Retriever.format_context()
            template : one of "default", "cited", "concise"

        Returns:
            Dict with "system" and "user" keys containing the
            formatted prompt strings.
        """
        tmpl = PROMPT_TEMPLATES.get(template, PROMPT_TEMPLATES["default"])
        system_msg = tmpl["system"].format(context=context) if context else ""
        user_msg = tmpl["user"].format(question=question)
        return {"system": system_msg, "user": user_msg}

    # ------------------------------------------------------------------
    #  Generation (enhanced with TODOs 2 + 3)
    # ------------------------------------------------------------------

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

        The full flow:
          1. Select temperature/top_p from preset (if given)
          2. Build RAG prompt from template + context + question
          3. Send to Ollama API
          4. Return the generated text

        Args:
            question    : the user's natural-language question
            context     : retrieved context chunks (from Retriever.format_context)
            temperature : controls randomness (lower = more deterministic)
            top_p       : nucleus sampling threshold (lower = less diverse)
            max_tokens  : maximum tokens in the generated response
            template    : prompt template name ("default", "cited", "concise")
            preset      : generation preset name ("precise", "balanced",
                          "creative", "exploratory") — overrides temperature/top_p
            debug       : if True, log the full prompt to the console

        Returns:
            The generated answer as a string.
        """
        # TODO 3: Apply preset if specified
        if preset and preset in GENERATION_PRESETS:
            p = GENERATION_PRESETS[preset]
            temperature = p["temperature"]
            top_p = p["top_p"]
            logger.info("Using preset '%s': temp=%.1f, top_p=%.2f", preset, temperature, top_p)

        # TODO 2: Build prompt from template
        prompt_parts = self.build_rag_prompt(question, context, template)
        system_msg = prompt_parts["system"]
        user_msg = prompt_parts["user"]

        # Debug output — lets you see exactly what the LLM receives
        if debug:
            print("\n" + "=" * 60)
            print("DEBUG: RAG Prompt Sent to LLM")
            print("=" * 60)
            print(f"Template: {template}")
            print(f"Temperature: {temperature}, Top-p: {top_p}")
            print(f"--- SYSTEM ---\n{system_msg}")
            print(f"--- USER ---\n{user_msg}")
            print("=" * 60 + "\n")

        payload = {
            "model": self.model,
            "prompt": user_msg,
            "system": system_msg,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }

        try:
            url = f"{self.base_url}/api/generate"
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            start_time = time.time()
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            elapsed = time.time() - start_time

            answer = result.get("response", "").strip()

            # Extract Ollama response metadata
            eval_count = result.get("eval_count", 0)
            eval_duration = result.get("eval_duration", 0)
            tokens_per_sec = (
                eval_count / (eval_duration / 1e9) if eval_duration else 0
            )

            logger.info(
                "Generated answer (%d chars, %d tokens, %.1f tok/s, "
                "%.1fs) for question: '%s'",
                len(answer),
                eval_count,
                tokens_per_sec,
                elapsed,
                question[:60],
            )
            return answer

        except urllib.error.URLError as exc:
            logger.error("Ollama request failed: %s", exc)
            return f"[ERROR] Could not reach Ollama at {self.base_url}. Is it running?"
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            return f"[ERROR] Generation failed: {exc}"

    # ------------------------------------------------------------------
    #  TODO 4 — Streaming Response Support
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        question: str,
        context: str = "",
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        template: str = "default",
        preset: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Generate an answer with streaming (token-by-token) output.

        Why streaming?
        --------------
        Ollama (and most LLMs) generate text one token at a time.
        Without streaming, you wait for the ENTIRE response (which
        could take 10-30 seconds for a 7B model on CPU) before seeing
        anything.

        With streaming, each token is sent to the client as soon as
        it's generated. The user sees the answer being "typed out"
        in real time, which:
          - Feels much faster (time-to-first-token is ~1-2 seconds)
          - Gives immediate feedback that the system is working
          - Allows the user to stop early if the answer is wrong

        How Ollama streaming works
        --------------------------
        When ``stream=True``, Ollama responds with NDJSON (newline-
        delimited JSON). Each line is a JSON object with a
        ``"response"`` field containing one or a few tokens::

            {"response": "The"}
            {"response": " invoice"}
            {"response": " total"}
            {"response": " is"}
            {"response": " $500"}
            {"response": ".", "done": true, "eval_count": 12, ...}

        We yield each ``response`` field as it arrives, so the caller
        can print it immediately.

        Usage::

            for chunk in client.generate_stream(question, context=ctx):
                print(chunk, end="", flush=True)
            print()  # newline after stream ends

        Args:
            question    : the user's question
            context     : formatted context from retriever
            temperature : randomness control
            top_p       : nucleus sampling threshold
            max_tokens  : max tokens to generate
            template    : prompt template name
            preset      : generation preset name

        Yields:
            String chunks (typically 1 token each) as they arrive.
        """
        # Apply preset
        if preset and preset in GENERATION_PRESETS:
            p = GENERATION_PRESETS[preset]
            temperature = p["temperature"]
            top_p = p["top_p"]

        prompt_parts = self.build_rag_prompt(question, context, template)

        payload = {
            "model": self.model,
            "prompt": prompt_parts["user"],
            "system": prompt_parts["system"],
            "stream": True,  # <-- streaming enabled
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        }

        try:
            url = f"{self.base_url}/api/generate"
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=300) as resp:
                # Read NDJSON line by line
                for line in resp:
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done", False):
                            # Log final stats
                            eval_count = chunk.get("eval_count", 0)
                            eval_duration = chunk.get("eval_duration", 0)
                            tps = (
                                eval_count / (eval_duration / 1e9)
                                if eval_duration else 0
                            )
                            logger.info(
                                "Stream complete: %d tokens, %.1f tok/s",
                                eval_count,
                                tps,
                            )
                            return
                    except json.JSONDecodeError:
                        continue

        except urllib.error.URLError as exc:
            yield f"\n[ERROR] Could not reach Ollama at {self.base_url}: {exc}"
        except Exception as exc:
            yield f"\n[ERROR] Streaming failed: {exc}"

    # ------------------------------------------------------------------
    #  Context-free generation (unchanged)
    # ------------------------------------------------------------------

    def generate_without_context(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """
        Generate a response without RAG context.
        Useful for testing or non-retrieval tasks.

        Note: for the search assistant, always prefer generate() with context.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            url = f"{self.base_url}/api/generate"
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "").strip()
        except Exception as exc:
            logger.error("LLM generation (no context) failed: %s", exc)
            return f"[ERROR] Generation failed: {exc}"

    # ------------------------------------------------------------------
    #  TODO 2 — Answer Quality Measurement
    # ------------------------------------------------------------------

    @staticmethod
    def measure_answer_quality(
        answer: str,
        context: str,
        question: str,
    ) -> Dict[str, float]:
        """
        Simple heuristic metrics for RAG answer quality.

        Why measure quality?
        --------------------
        In RAG, the biggest risk is **hallucination** — the LLM
        generating information not present in the context. These
        simple metrics give you a rough signal:

        Metrics:
          - **groundedness** — what fraction of answer words also
            appear in the context? Higher = more grounded. A score
            of 0.5+ usually indicates the answer draws from context.
          - **relevance** — what fraction of question words appear
            in the answer? Higher = the answer addresses the question.
          - **length_ratio** — answer length / context length.
            Very high ratios may indicate fabrication; very low may
            indicate the LLM is being too terse.
          - **uncertainty_flag** — 1.0 if the answer contains phrases
            like "I don't know" or "not enough information"
            (appropriate when context is insufficient).

        These are NOT rigorous evaluation metrics. For production,
        use dedicated tools like RAGAS, DeepEval, or manual evaluation.

        Args:
            answer   : the LLM's answer
            context  : the context that was provided
            question : the original question

        Returns:
            Dict of metric_name -> score (0.0 to 1.0).
        """
        # Tokenise crudely
        import re
        def tokenise(text: str) -> set:
            return set(re.findall(r"\w+", text.lower()))

        answer_words = tokenise(answer)
        context_words = tokenise(context)
        question_words = tokenise(question)

        # Remove common stopwords for more meaningful overlap
        stop = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "shall", "can", "to",
            "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "above",
            "below", "and", "but", "or", "nor", "not", "so", "yet",
            "it", "its", "this", "that", "these", "those", "i", "you",
            "he", "she", "we", "they", "me", "him", "her", "us", "them",
        }
        answer_content = answer_words - stop
        context_content = context_words - stop
        question_content = question_words - stop

        # Groundedness: answer words found in context
        if answer_content:
            groundedness = len(answer_content & context_content) / len(answer_content)
        else:
            groundedness = 0.0

        # Relevance: question words found in answer
        if question_content:
            relevance = len(question_content & answer_content) / len(question_content)
        else:
            relevance = 0.0

        # Length ratio
        length_ratio = len(answer) / max(len(context), 1)

        # Uncertainty detection
        uncertainty_phrases = [
            "i don't know", "not enough information", "cannot determine",
            "not found", "no information", "does not contain",
            "cannot answer", "insufficient", "unclear",
        ]
        uncertainty_flag = 1.0 if any(
            phrase in answer.lower() for phrase in uncertainty_phrases
        ) else 0.0

        return {
            "groundedness": round(groundedness, 3),
            "relevance": round(relevance, 3),
            "length_ratio": round(length_ratio, 3),
            "uncertainty_flag": uncertainty_flag,
        }


# ======================================================================
#  TODO 5 — OpenVINO GenAI Stub
# ======================================================================

class OVLLMClient:
    """
    Placeholder for OpenVINO GenAI-based LLM inference.

    Why replace Ollama?
    -------------------
    Ollama is convenient (download and run), but it:
      - Runs its own server process (extra dependency)
      - Uses its own quantization (llama.cpp GGUF format)
      - Does not leverage Intel-specific hardware optimizations

    OpenVINO GenAI provides:
      - Direct inference without a server process
      - INT4/INT8 weight compression via NNCF
      - Optimized execution on Intel CPU, iGPU, and NPU
      - Smaller deployment footprint (no Ollama binary)

    How it would work
    -----------------
    1. Convert Mistral 7B to OpenVINO IR using optimum-intel:
       ``optimum-cli export openvino --model mistralai/Mistral-7B-Instruct-v0.1
         --weight-format int4 models/openvino/mistral-7b/``

    2. Load and run with ``openvino_genai.LLMPipeline``:
       ``pipe = ov_genai.LLMPipeline(model_path, device="CPU")``
       ``answer = pipe.generate(prompt, max_new_tokens=512)``

    3. This class would implement the same interface as OllamaClient
       (generate, generate_stream) but use OpenVINO instead of HTTP.

    Status: Placeholder — implement when ready for Phase 9.
    """

    def __init__(self, model_path: str, device: str = "CPU"):
        self.model_path = model_path
        self.device = device
        self._pipeline = None
        logger.warning(
            "OVLLMClient is a placeholder. "
            "Install openvino-genai and convert a model first."
        )

    def is_available(self) -> bool:
        try:
            import openvino_genai  # noqa: F401
            return True
        except ImportError:
            return False

    def generate(self, question: str, context: str = "", **kwargs) -> str:
        return (
            "[OVLLMClient] Not implemented. "
            "See src/llm/openvino_llm.py for the full placeholder."
        )


# ======================================================================
#  TODO 1 — Basic Generation Test
# ======================================================================

if __name__ == "__main__":
    """
    Test script for the Ollama client.

    Run:  python -m src.llm.ollama_client

    Tests:
      1. Server connectivity
      2. Basic generation (no context)
      3. RAG generation (with mock context)
      4. Prompt template comparison
      5. Temperature experiment
      6. Streaming output
      7. Answer quality metrics
    """
    import textwrap

    def section(title: str) -> None:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

    client = OllamaClient()

    # ------------------------------------------------------------------
    #  Test 1: Connectivity
    # ------------------------------------------------------------------
    section("Test 1: Ollama Server Connectivity")
    available = client.is_available()
    print(f"  Server: {client.base_url}")
    print(f"  Model:  {client.model}")
    print(f"  Status: {'✓ Available' if available else '✗ Not available'}")

    if available:
        models = client.list_models()
        print(f"  Models: {models}")

    if not available:
        print("\n  ⚠ Ollama is not running. Start it with: ollama serve")
        print("  ⚠ Then pull the model: ollama pull mistral")
        print("  Remaining tests will be skipped.")
        sys.exit(0)

    # ------------------------------------------------------------------
    #  Test 2: Basic generation (no RAG context)
    # ------------------------------------------------------------------
    section("Test 2: Basic Generation (no context)")
    answer = client.generate_without_context(
        "What is Retrieval Augmented Generation (RAG) in one sentence?",
        temperature=0.3,
    )
    print(f"\n  Q: What is RAG?\n  A: {textwrap.fill(answer, width=70, initial_indent='     ', subsequent_indent='     ')}")

    # ------------------------------------------------------------------
    #  Test 3: RAG generation (with mock context)
    # ------------------------------------------------------------------
    section("Test 3: RAG Generation (with context)")
    mock_context = """[Source 1 | score=0.85 | doc=invoice_001]
Invoice Number: INV-2024-0042
Date: March 15, 2024
Company: Acme Corporation
Total Amount: $1,250.00
Payment Terms: Net 30

[Source 2 | score=0.72 | doc=invoice_001]
Bill To: John Smith, 456 Oak Avenue, Springfield, IL 62704
Ship To: Same as billing address
"""

    question = "What is the total amount on the invoice?"
    answer = client.generate(question=question, context=mock_context)
    print(f"\n  Q: {question}")
    print(f"  A: {textwrap.fill(answer, width=70, initial_indent='     ', subsequent_indent='     ')}")

    # ------------------------------------------------------------------
    #  Test 4: Template comparison
    # ------------------------------------------------------------------
    section("Test 4: Prompt Template Comparison")
    for tmpl_name, tmpl in PROMPT_TEMPLATES.items():
        print(f"\n  --- Template: {tmpl_name} ---")
        print(f"  Description: {tmpl['description']}")
        ans = client.generate(
            question="Who is the invoice billed to?",
            context=mock_context,
            template=tmpl_name,
        )
        print(f"  Answer: {textwrap.fill(ans, width=65, initial_indent='', subsequent_indent='         ')}")

    # ------------------------------------------------------------------
    #  Test 5: Temperature experiment
    # ------------------------------------------------------------------
    section("Test 5: Temperature / Preset Experiment")
    for preset_name in ["precise", "balanced", "creative"]:
        p = GENERATION_PRESETS[preset_name]
        ans = client.generate(
            question="What company issued this invoice?",
            context=mock_context,
            preset=preset_name,
        )
        print(f"\n  Preset: {preset_name} (temp={p['temperature']}, top_p={p['top_p']})")
        print(f"  Answer: {ans[:200]}")

    # ------------------------------------------------------------------
    #  Test 6: Streaming
    # ------------------------------------------------------------------
    section("Test 6: Streaming Response")
    print("\n  Streaming answer: ", end="", flush=True)
    full_answer = ""
    for chunk in client.generate_stream(
        question="What is the payment term?",
        context=mock_context,
    ):
        print(chunk, end="", flush=True)
        full_answer += chunk
    print("\n")

    # ------------------------------------------------------------------
    #  Test 7: Answer quality metrics
    # ------------------------------------------------------------------
    section("Test 7: Answer Quality Metrics")
    metrics = OllamaClient.measure_answer_quality(
        answer=full_answer,
        context=mock_context,
        question="What is the payment term?",
    )
    for metric, value in metrics.items():
        bar = "█" * int(value * 20)
        print(f"  {metric:<20}: {value:.3f}  {bar}")

    # Test with a hallucinated answer
    fake = "The invoice was signed by Albert Einstein on the Moon."
    fake_metrics = OllamaClient.measure_answer_quality(
        answer=fake, context=mock_context, question="Who signed the invoice?"
    )
    print(f"\n  Hallucinated answer metrics:")
    for metric, value in fake_metrics.items():
        bar = "█" * int(value * 20)
        print(f"  {metric:<20}: {value:.3f}  {bar}")

    section("All Tests Complete!")
