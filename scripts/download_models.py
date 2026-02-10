"""
Download Models Script
=======================
One-time setup: download and cache the models required by the pipeline.

This script is idempotent -- running it again skips already-downloaded
models.

Models downloaded:
    1. sentence-transformers/all-MiniLM-L6-v2  (embedding model, ~80 MB)
    2. Mistral 7B Instruct via Ollama           (LLM, ~4 GB)

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --skip-ollama   (embedding only)
"""

import argparse
import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def download_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    """
    Download the embedding model by importing sentence-transformers.

    The library caches the model under ~/.cache/huggingface/ on first load,
    so subsequent calls are fast.
    """
    logger.info("Downloading embedding model: %s", model_name)
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        dim = model.get_sentence_embedding_dimension()
        logger.info(
            "Embedding model ready: %s (dim=%d)", model_name, dim
        )
    except ImportError:
        logger.error(
            "sentence-transformers not installed.  "
            "Run: pip install sentence-transformers"
        )
        sys.exit(1)
    except Exception as exc:
        logger.error("Failed to download embedding model: %s", exc)
        sys.exit(1)


def pull_ollama_model(model: str = "mistral") -> None:
    """
    Pull the LLM model via the Ollama CLI.

    Requires Ollama to be installed and the ``ollama`` command on PATH.
    """
    logger.info("Pulling Ollama model: %s", model)
    try:
        result = subprocess.run(
            ["ollama", "pull", model],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            logger.info("Ollama model '%s' pulled successfully", model)
        else:
            logger.error("ollama pull failed:\n%s", result.stderr)
    except FileNotFoundError:
        logger.error(
            "Ollama CLI not found.  Install from https://ollama.ai/download  "
            "Then run:  ollama pull %s",
            model,
        )
    except subprocess.TimeoutExpired:
        logger.error("ollama pull timed out (model may be very large)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download models required by Deep Search AI Assistant"
    )
    parser.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Skip pulling the Ollama LLM model",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model name for the embedding encoder",
    )
    parser.add_argument(
        "--llm-model",
        default="mistral",
        help="Ollama model name for the LLM",
    )
    args = parser.parse_args()

    download_embedding_model(args.embedding_model)

    if not args.skip_ollama:
        pull_ollama_model(args.llm_model)
    else:
        logger.info("Skipping Ollama model pull (--skip-ollama)")

    logger.info("Model download complete.")


if __name__ == "__main__":
    main()
