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

Learning TODO:
  1. Test basic generation with Ollama API.
  2. Implement RAG prompt template and measure answer quality.
  3. Experiment with temperature and top_p for different use cases.
  4. Add streaming response support.
  5. Replace with OpenVINO GenAI inference (src/openvino/ov_llm.py).
"""

import json
import logging
from typing import Optional, Dict, List

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
# RAG prompt template
# ---------------------------------------------------------------------------
RAG_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context.
Use ONLY the information from the context below to answer the question.
If the context does not contain enough information to answer, say so honestly.
Do not make up information.

CONTEXT:
{context}
"""

RAG_USER_TEMPLATE = """Based on the context provided, please answer the following question:

{question}"""


class OllamaClient:
    """
    Client for the Ollama local LLM server.

    Usage:
        client = OllamaClient()
        if client.is_available():
            answer = client.generate("What is this document about?", context="...")
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

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        question: str,
        context: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate an answer using RAG prompt construction.

        Args:
            question    : the user's natural-language question
            context     : retrieved context chunks (from Retriever.format_context)
            temperature : controls randomness (lower = more deterministic)
            max_tokens  : maximum tokens in the generated response

        Returns:
            The generated answer as a string.
        """
        # Build the RAG prompt
        system_msg = RAG_SYSTEM_PROMPT.format(context=context) if context else ""
        user_msg = RAG_USER_TEMPLATE.format(question=question)

        payload = {
            "model": self.model,
            "prompt": user_msg,
            "system": system_msg,
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
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                answer = result.get("response", "").strip()
                logger.info(
                    "Generated answer (%d chars) for question: '%s'",
                    len(answer),
                    question[:60],
                )
                return answer

        except urllib.error.URLError as exc:
            logger.error("Ollama request failed: %s", exc)
            return f"[ERROR] Could not reach Ollama at {self.base_url}. Is it running?"
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            return f"[ERROR] Generation failed: {exc}"

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
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "").strip()
        except Exception as exc:
            logger.error("LLM generation (no context) failed: %s", exc)
            return f"[ERROR] Generation failed: {exc}"
