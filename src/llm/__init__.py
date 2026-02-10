"""
LLM subpackage -- answer generation using local language models.

Two backends:
    OllamaClient    -- Mistral 7B via Ollama HTTP API (current)
    OVLLMClient     -- OpenVINO GenAI inference (future placeholder)
"""

from src.llm.ollama_client import OllamaClient
