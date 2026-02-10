"""
Deep Search AI Assistant -- Command Line Interface
=====================================================
Entry point for all user-facing operations.

Commands:
  ingest   -- Load, normalise, OCR, chunk, embed, and index a dataset
  search   -- Semantic search over indexed documents (no LLM)
  ask      -- RAG question answering (retrieve + LLM generation)
  stats    -- Show index and pipeline statistics
  devices  -- List available OpenVINO hardware devices

Usage examples:
  python cli.py ingest --dataset funsd
  python cli.py ingest --dataset docvqa --max-records 20
  python cli.py search "Find the invoice number"
  python cli.py ask "What is the total amount on the invoice?"
  python cli.py stats
  python cli.py devices

Design notes:
  - Uses argparse from the standard library (no extra dependencies).
  - Each command maps to a handler function that orchestrates the
    relevant pipeline modules.
  - Errors are caught and displayed with helpful messages.
  - Logging is configured at startup based on --verbose flag.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (resolve regardless of where the script is invoked from)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Configuration defaults -- these match configs/settings.yaml
# ---------------------------------------------------------------------------
RAW_DATA_ROOT = Path("/mnt/d/Openvino-project/data/raw")
# Fallback for native Windows (not WSL) execution.
if not RAW_DATA_ROOT.exists():
    RAW_DATA_ROOT = Path(r"D:\Openvino-project\data\raw")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
NORMALISED_DIR = PROCESSED_DIR / "normalised"
OCR_CACHE_DIR = PROCESSED_DIR / "ocr_cache"
CHUNKS_DIR = PROCESSED_DIR / "chunks"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"
INDEX_DIR = PROCESSED_DIR / "faiss"


def setup_logging(verbose: bool = False) -> None:
    """Configure root logger."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ===================================================================
# Command handlers
# ===================================================================

def cmd_ingest(args: argparse.Namespace) -> None:
    """
    Ingest pipeline: load -> OCR -> normalise -> chunk -> embed -> index.

    This is the full data preparation pipeline.  Each stage writes its
    output to data/processed/ so that subsequent stages can be re-run
    independently.
    """
    from src.ingestion.loader import DatasetLoader
    from src.ingestion.normalizer import DocumentNormalizer
    from src.ingestion.chunker import TextChunker
    from src.ocr.tesseract_engine import TesseractEngine
    from src.embeddings.encoder import EmbeddingEncoder
    from src.index.faiss_index import FaissIndex

    dataset = args.dataset
    max_records = getattr(args, "max_records", 0)
    logging.info("=== INGEST PIPELINE START (dataset=%s) ===", dataset)

    # --- Step 1: Load dataset from Arrow format ---
    logging.info("Step 1/6: Loading dataset '%s' from %s", dataset, RAW_DATA_ROOT)
    loader = DatasetLoader(
        raw_data_root=str(RAW_DATA_ROOT),
        image_cache_dir=str(OCR_CACHE_DIR),
    )

    available = loader.list_available_datasets()
    if dataset not in available:
        print(f"ERROR: Dataset '{dataset}' not found.\n"
              f"Available datasets: {', '.join(available)}")
        sys.exit(1)

    raw_docs = loader.load_dataset(dataset, max_records=max_records)
    if not raw_docs:
        logging.warning("No documents loaded.")
        return
    logging.info("  Loaded %d documents", len(raw_docs))

    # --- Step 2: OCR for image documents ---
    logging.info("Step 2/6: Running OCR on documents that need text extraction")
    ocr_count = 0
    try:
        ocr_engine = TesseractEngine(preprocess=True)
        for doc in raw_docs:
            if doc.image_path and not doc.text:
                extracted = ocr_engine.extract_text(doc.image_path)
                if extracted:
                    doc.text = extracted
                    ocr_count += 1
        logging.info("  OCR extracted text from %d images", ocr_count)
    except ImportError:
        logging.warning(
            "  OCR skipped (pytesseract not installed).  "
            "Documents without text will have empty text fields."
        )

    # --- Step 3: Normalise ---
    logging.info("Step 3/6: Normalising documents")
    normalizer = DocumentNormalizer(output_dir=str(NORMALISED_DIR))
    norm_docs = normalizer.normalize(raw_docs)
    normalizer.save(norm_docs)
    logging.info("  Normalised %d documents", len(norm_docs))

    # --- Step 4: Chunk ---
    logging.info("Step 4/6: Chunking documents")
    chunker = TextChunker(chunk_size=512, overlap=64)
    docs_with_text = [d for d in norm_docs if d.text.strip()]
    chunks = chunker.chunk_documents(docs_with_text)
    if not chunks:
        logging.warning(
            "No text chunks produced.  Documents may lack extractable text.  "
            "Ensure OCR is working for image-based datasets."
        )
        return
    logging.info(
        "  Produced %d chunks from %d documents (of %d total)",
        len(chunks), len(docs_with_text), len(norm_docs),
    )

    # --- Step 5: Embed ---
    logging.info("Step 5/6: Generating embeddings")
    encoder = EmbeddingEncoder(device="cpu")
    chunk_texts = [c.text for c in chunks]
    embeddings = encoder.encode(chunk_texts, batch_size=64, show_progress=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    encoder.save_embeddings(embeddings, str(EMBEDDINGS_DIR / "chunk_embeddings.npy"))
    logging.info("  Generated embeddings: shape %s", embeddings.shape)

    # --- Step 6: Build FAISS index ---
    logging.info("Step 6/6: Building FAISS index")
    chunk_metadata = [c.to_dict() for c in chunks]
    index = FaissIndex(dimension=encoder.dimension)
    index.build(embeddings, chunk_metadata)
    index.save(str(INDEX_DIR))
    logging.info("  Index built: %d vectors", index.size)

    logging.info("=== INGEST PIPELINE COMPLETE ===")
    print(
        f"\nIngestion complete:\n"
        f"  Documents loaded:  {len(raw_docs)}\n"
        f"  OCR extractions:   {ocr_count}\n"
        f"  With text:         {len(docs_with_text)}\n"
        f"  Chunks created:    {len(chunks)}\n"
        f"  Vectors indexed:   {index.size}"
    )


def cmd_search(args: argparse.Namespace) -> None:
    """
    Semantic search: encode query -> search FAISS index -> display results.
    No LLM involved -- pure embedding-based retrieval.
    """
    from src.embeddings.encoder import EmbeddingEncoder
    from src.index.faiss_index import FaissIndex
    from src.retrieval.retriever import Retriever

    query = args.query
    top_k = args.top_k

    logging.info("=== SEARCH: '%s' (top_k=%d) ===", query, top_k)

    encoder = EmbeddingEncoder(device="cpu")
    index = FaissIndex()
    try:
        index.load(str(INDEX_DIR))
    except FileNotFoundError:
        print("ERROR: Index not found. Run 'python cli.py ingest --dataset <name>' first.")
        sys.exit(1)

    retriever = Retriever(encoder=encoder, index=index)
    results = retriever.query(query, top_k=top_k)

    if not results:
        print("No results found.")
        return

    print(f"\n{'='*60}")
    print(f"Search results for: \"{query}\"")
    print(f"{'='*60}")
    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {r.score:.4f}) ---")
        print(f"Document: {r.doc_id}")
        print(f"Chunk:    {r.chunk_id}")
        preview = r.text[:300].replace("\n", " ")
        print(f"Text:     {preview}")
    print(f"\n{'='*60}")


def cmd_ask(args: argparse.Namespace) -> None:
    """
    RAG question answering: retrieve context + generate answer with LLM.
    """
    from src.embeddings.encoder import EmbeddingEncoder
    from src.index.faiss_index import FaissIndex
    from src.retrieval.retriever import Retriever
    from src.llm.ollama_client import OllamaClient

    question = args.question
    top_k = args.top_k

    logging.info("=== ASK: '%s' ===", question)

    encoder = EmbeddingEncoder(device="cpu")
    index = FaissIndex()
    try:
        index.load(str(INDEX_DIR))
    except FileNotFoundError:
        print("ERROR: Index not found. Run 'python cli.py ingest --dataset <name>' first.")
        sys.exit(1)

    # Retrieve context
    retriever = Retriever(encoder=encoder, index=index)
    results = retriever.query(question, top_k=top_k)
    context = retriever.format_context(results, max_chars=3000)

    if not results:
        print("No relevant context found in the knowledge base.")
        return

    # Generate answer
    llm = OllamaClient()
    if not llm.is_available():
        print(
            "WARNING: Ollama is not available.  Showing retrieved context only.\n"
            "Start Ollama with:  ollama serve\n"
            "Pull model with:    ollama pull mistral\n"
        )
        print(context)
        return

    print(f"\nRetrieved {len(results)} context chunks.  Generating answer...\n")
    answer = llm.generate(question=question, context=context)
    print(f"{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")
    print(f"\n{answer}\n")
    print(f"{'='*60}")
    print(f"Based on {len(results)} retrieved chunks.")


def cmd_stats(args: argparse.Namespace) -> None:
    """Display statistics about the current state of the pipeline."""
    print(f"\n{'='*60}")
    print("Deep Search AI Assistant -- Pipeline Statistics")
    print(f"{'='*60}")

    # Raw data
    print(f"\nRaw data root: {RAW_DATA_ROOT}")
    if RAW_DATA_ROOT.exists():
        datasets = sorted(d.name for d in RAW_DATA_ROOT.iterdir() if d.is_dir())
        print(f"  Datasets: {', '.join(datasets) if datasets else 'none'}")
    else:
        print("  [NOT FOUND]")

    # Normalised documents
    norm_file = NORMALISED_DIR / "documents.jsonl"
    print(f"\nNormalised documents: {norm_file}")
    if norm_file.exists():
        with open(norm_file) as f:
            doc_count = sum(1 for line in f if line.strip())
        print(f"  Document count: {doc_count}")
    else:
        print("  [NOT YET CREATED -- run ingest]")

    # Embeddings
    emb_file = EMBEDDINGS_DIR / "chunk_embeddings.npy"
    print(f"\nEmbeddings: {emb_file}")
    if emb_file.exists():
        import numpy as np
        emb = np.load(str(emb_file))
        print(f"  Shape: {emb.shape}")
        print(f"  Dtype: {emb.dtype}")
        print(f"  Size:  {emb_file.stat().st_size / 1024:.1f} KB")
    else:
        print("  [NOT YET CREATED -- run ingest]")

    # FAISS index
    index_file = INDEX_DIR / "index.faiss"
    meta_file = INDEX_DIR / "metadata.json"
    print(f"\nFAISS index: {index_file}")
    if index_file.exists():
        print(f"  Index size: {index_file.stat().st_size / 1024:.1f} KB")
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            print(f"  Indexed chunks: {len(meta)}")
    else:
        print("  [NOT YET CREATED -- run ingest]")

    # Ollama
    print(f"\nLLM backend:")
    try:
        from src.llm.ollama_client import OllamaClient
        client = OllamaClient()
        if client.is_available():
            print(f"  Ollama: AVAILABLE (model: {client.model})")
        else:
            print(f"  Ollama: model '{client.model}' not found or server not running")
    except Exception as e:
        print(f"  Ollama: NOT REACHABLE ({e})")

    print(f"\n{'='*60}")


def cmd_devices(args: argparse.Namespace) -> None:
    """List available OpenVINO devices."""
    print(f"\n{'='*60}")
    print("OpenVINO Device Discovery")
    print(f"{'='*60}\n")
    try:
        from src.openvino.device_manager import DeviceManager
        dm = DeviceManager()
        devices = dm.list_devices()
        if devices:
            for d in devices:
                props = dm.device_properties(d)
                name = props.get("FULL_DEVICE_NAME", "")
                print(f"  {d:8s}  {name}")
        else:
            print("  No devices found.  Is openvino installed?")
            print("  Install: pip install openvino")
    except ImportError as e:
        print(f"  OpenVINO not available: {e}")
    print(f"\n{'='*60}")


# ===================================================================
# Argument parser
# ===================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deep-search",
        description=(
            "Multimodal RAG-based search assistant for personal knowledge bases.  "
            "Powered by OpenVINO for efficient on-device inference."
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- ingest --
    p_ingest = subparsers.add_parser(
        "ingest",
        help="Ingest and index documents from a raw dataset",
    )
    p_ingest.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name: funsd, docvqa, or rvl_cdip",
    )
    p_ingest.add_argument(
        "--max-records",
        type=int,
        default=0,
        dest="max_records",
        help="Limit the number of records loaded (0 = all, default: 0)",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    # -- search --
    p_search = subparsers.add_parser(
        "search",
        help="Semantic search over indexed documents (no LLM)",
    )
    p_search.add_argument(
        "query",
        type=str,
        help="Natural-language search query",
    )
    p_search.add_argument(
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="Number of results to return (default: 5)",
    )
    p_search.set_defaults(func=cmd_search)

    # -- ask --
    p_ask = subparsers.add_parser(
        "ask",
        help="Ask a question (RAG: retrieval + LLM generation)",
    )
    p_ask.add_argument(
        "question",
        type=str,
        help="Natural-language question",
    )
    p_ask.add_argument(
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="Number of context chunks to retrieve (default: 5)",
    )
    p_ask.set_defaults(func=cmd_ask)

    # -- stats --
    p_stats = subparsers.add_parser(
        "stats",
        help="Show pipeline and index statistics",
    )
    p_stats.set_defaults(func=cmd_stats)

    # -- devices --
    p_devices = subparsers.add_parser(
        "devices",
        help="List available OpenVINO hardware devices",
    )
    p_devices.set_defaults(func=cmd_devices)

    return parser


# ===================================================================
# Main entry point
# ===================================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(verbose=getattr(args, "verbose", False))

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as exc:
        logging.error("Command failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
