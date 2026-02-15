"""
Deep Search AI Assistant -- Command Line Interface
=====================================================
Entry point for all user-facing operations.

Commands:
  ingest         -- Load, normalise, OCR, chunk, embed, and index a dataset
  ingest-videos  -- Ingest video dataset (MSR-VTT captions + OCR frames) into JSON
  search         -- Semantic search over indexed documents (no LLM)
  ask            -- RAG question answering (retrieve + LLM generation)
  stats          -- Show index and pipeline statistics
  devices        -- List available OpenVINO hardware devices
  ocr-tune       -- Test and tune OCR settings on sample images

Usage examples:
  python cli.py ingest --dataset funsd
  python cli.py ingest --dataset docvqa --max-records 20
  python cli.py ingest-videos --dataset msrvtt
  python cli.py ingest-videos --max-videos 5
  python cli.py search "Find the invoice number"
  python cli.py ask "What is the total amount on the invoice?"
  python cli.py stats
  python cli.py devices
  python cli.py ocr-tune --dataset funsd --sample 0

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
import os
import sys
import time
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
try:
    if not RAW_DATA_ROOT.exists():
        RAW_DATA_ROOT = Path(r"D:\Openvino-project\data\raw")
except OSError:
    # WSL mount may not be available (e.g. drive not mounted).
    RAW_DATA_ROOT = Path(r"D:\Openvino-project\data\raw")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
NORMALISED_DIR = PROCESSED_DIR / "normalised"
OCR_CACHE_DIR = PROCESSED_DIR / "ocr_cache"
CHUNKS_DIR = PROCESSED_DIR / "chunks"
EMBEDDINGS_DIR = PROCESSED_DIR / "embeddings"
INDEX_DIR = PROCESSED_DIR / "faiss"

# Load settings from YAML
SETTINGS = {}
try:
    import yaml
    settings_path = PROJECT_ROOT / "configs" / "settings.yaml"
    if settings_path.exists():
        with open(settings_path, "r", encoding="utf-8") as f:
            SETTINGS = yaml.safe_load(f) or {}
except Exception as e:
    print(f"Warning: Could not load settings.yaml: {e}", file=sys.stderr)
    SETTINGS = {}



# ---------------------------------------------------------------------------
# ANSI colour helpers (disabled when piped or on dumb terminals)
# ---------------------------------------------------------------------------
_USE_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty() and os.environ.get("TERM") != "dumb"


class _C:
    """ANSI escape shortcuts."""
    RESET   = "\033[0m"   if _USE_COLOR else ""
    BOLD    = "\033[1m"   if _USE_COLOR else ""
    DIM     = "\033[2m"   if _USE_COLOR else ""
    GREEN   = "\033[32m"  if _USE_COLOR else ""
    YELLOW  = "\033[33m"  if _USE_COLOR else ""
    BLUE    = "\033[34m"  if _USE_COLOR else ""
    MAGENTA = "\033[35m"  if _USE_COLOR else ""
    CYAN    = "\033[36m"  if _USE_COLOR else ""
    RED     = "\033[31m"  if _USE_COLOR else ""
    WHITE   = "\033[97m"  if _USE_COLOR else ""


# Unicode symbols for progress feedback.
SYM_CHECK  = "✔" if _USE_COLOR else "[OK]"
SYM_CROSS  = "✘" if _USE_COLOR else "[FAIL]"
SYM_ARROW  = "→" if _USE_COLOR else "->"
SYM_DOT    = "●" if _USE_COLOR else "*"
SYM_WARN   = "⚠" if _USE_COLOR else "[!]"


def _header(title: str, width: int = 60) -> None:
    """Print a prominent section header."""
    print()
    print(f"{_C.BOLD}{_C.CYAN}{'─' * width}{_C.RESET}")
    print(f"{_C.BOLD}{_C.CYAN}  {title}{_C.RESET}")
    print(f"{_C.BOLD}{_C.CYAN}{'─' * width}{_C.RESET}")


def _step(num: int, total: int, desc: str) -> None:
    """Print a step indicator at the start of a pipeline stage."""
    print(f"\n  {_C.BOLD}{_C.BLUE}[{num}/{total}]{_C.RESET} {_C.WHITE}{desc}{_C.RESET}")


def _done(msg: str) -> None:
    """Print a green check-marked completion line."""
    print(f"      {_C.GREEN}{SYM_CHECK} {msg}{_C.RESET}")


def _warn(msg: str) -> None:
    """Print a yellow warning line."""
    print(f"      {_C.YELLOW}{SYM_WARN} {msg}{_C.RESET}")


def _info(msg: str) -> None:
    """Print a dim informational line."""
    print(f"      {_C.DIM}{msg}{_C.RESET}")


def _error(msg: str) -> None:
    """Print a red error line."""
    print(f"      {_C.RED}{SYM_CROSS} {msg}{_C.RESET}")


def _summary_table(rows: list[tuple[str, str]], width: int = 60) -> None:
    """Print a key-value summary box."""
    print()
    print(f"  {_C.BOLD}{'─' * (width - 4)}{_C.RESET}")
    for label, value in rows:
        print(f"  {_C.DIM}{label:<22}{_C.RESET}{_C.BOLD}{value}{_C.RESET}")
    print(f"  {_C.BOLD}{'─' * (width - 4)}{_C.RESET}")


def _elapsed(start: float) -> str:
    """Return a human-friendly elapsed time string."""
    secs = time.time() - start
    if secs < 60:
        return f"{secs:.1f}s"
    return f"{int(secs // 60)}m {secs % 60:.1f}s"


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

# Libraries that produce noisy output at INFO level.
_NOISY_LOGGERS = [
    "httpx", "httpcore", "urllib3", "requests",
    "huggingface_hub", "sentence_transformers",
    "faiss", "faiss.loader",
    "datasets", "PIL", "filelock",
    "transformers", "tokenizers",
    "src.ingestion.loader", "src.ingestion.normalizer",
    "src.ingestion.chunker", "src.ocr",
    "src.embeddings.encoder", "src.index.faiss_index",
    "src.retrieval.retriever", "src.llm",
]


def setup_logging(verbose: bool = False) -> None:
    """
    Configure root logger.

    In normal mode, third-party library loggers are silenced to WARNING
    so only the pipeline's own messages are visible.  Use ``--verbose``
    to see everything.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if not verbose:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)


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
    t0 = time.time()

    _header(f"Ingest Pipeline  {SYM_ARROW}  dataset={_C.MAGENTA}{dataset}{_C.RESET}{_C.BOLD}{_C.CYAN}")
    if max_records:
        _info(f"Max records: {max_records}")

    # --- Step 1: Load dataset from Arrow format ---
    _step(1, 6, "Loading dataset")
    loader = DatasetLoader(
        raw_data_root=str(RAW_DATA_ROOT),
        image_cache_dir=str(OCR_CACHE_DIR),
    )

    available = loader.list_available_datasets()
    if dataset not in available:
        _error(f"Dataset '{dataset}' not found.")
        _info(f"Available: {', '.join(available)}")
        sys.exit(1)

    raw_docs = loader.load_dataset(dataset, max_records=max_records)
    if not raw_docs:
        _warn("No documents loaded.")
        return
    _done(f"{len(raw_docs)} documents loaded")

    # --- Step 2: OCR for image documents ---
    _step(2, 6, "Running OCR on image documents")
    ocr_count = 0
    
    # Read OCR engine from settings
    ocr_settings = SETTINGS.get("ocr", {})
    engine_name = ocr_settings.get("engine", "tesseract").lower()
    
    ocr_engine = None
    try:
        if engine_name == "paddleocr":
            # Try PaddleOCR first
            try:
                from src.ocr.paddle_engine import PaddleOCREngine
                paddle_lang = ocr_settings.get("paddleocr_lang", "en")
                use_ov = ocr_settings.get("paddleocr_use_openvino", False)
                device = "CPU"  # Could be made configurable
                ocr_engine = PaddleOCREngine(
                    lang=paddle_lang,
                    use_openvino=use_ov,
                    device=device,
                    confidence_threshold=ocr_settings.get("confidence_threshold", 40)
                )
                _info(f"Using PaddleOCR (lang={paddle_lang}, openvino={use_ov})")
            except ImportError:
                _warn("PaddleOCR not installed, falling back to Tesseract")
                _info("Install with: pip install paddleocr paddlepaddle")
        
        # Fallback to Tesseract or if explicitly requested
        if ocr_engine is None:
            from src.ocr.tesseract_engine import TesseractEngine
            preprocess = ocr_settings.get("preprocess", True)
            ocr_engine = TesseractEngine(
                lang=ocr_settings.get("tesseract_lang", "eng"),
                preprocess=preprocess,
                confidence_threshold=ocr_settings.get("confidence_threshold", 40)
            )
            _info(f"Using Tesseract (preprocess={preprocess})")
        
        # Collect all images that need OCR
        images_to_ocr = []
        image_doc_map = {}  # map image_path -> doc for updating later
        for doc in raw_docs:
            if doc.image_path and not doc.text:
                images_to_ocr.append(doc.image_path)
                image_doc_map[doc.image_path] = doc
        
        if images_to_ocr:
            # Use batch_extract with progress bar
            ocr_results = ocr_engine.batch_extract(images_to_ocr)
            for image_path, extracted_text in ocr_results:
                if extracted_text:
                    image_doc_map[image_path].text = extracted_text
                    ocr_count += 1
            _done(f"Extracted text from {ocr_count} images")
        else:
            _info("No images needed OCR (all documents already have text)")
    except ImportError as e:
        _warn(f"OCR skipped — {e}")
        _info("Install Tesseract: pip install pytesseract + sudo apt install tesseract-ocr")
        _info("Or install PaddleOCR: pip install paddleocr paddlepaddle")

    # --- Step 3: Normalise ---
    _step(3, 6, "Normalising documents")
    normalizer = DocumentNormalizer(output_dir=str(NORMALISED_DIR))
    norm_docs = normalizer.normalize(raw_docs)
    normalizer.save(norm_docs)
    _done(f"{len(norm_docs)} documents normalised")

    # --- Step 4: Chunk ---
    _step(4, 6, "Chunking text into passages")
    chunker = TextChunker(chunk_size=512, overlap=64)
    docs_with_text = [d for d in norm_docs if d.text.strip()]
    chunks = chunker.chunk_documents(docs_with_text)
    if not chunks:
        _warn("No text chunks produced — documents may lack extractable text.")
        _info("Ensure OCR is working for image-based datasets.")
        return
    _done(f"{len(chunks)} chunks from {len(docs_with_text)} documents")

    # --- Step 5: Embed ---
    # Phase 10: Check settings.yaml to decide between PyTorch and OpenVINO
    _step(5, 6, "Generating embeddings")
    encoder = None
    device_override = getattr(args, "device", None)
    try:
        from src.openvino.device_manager import DeviceManager
        dm = DeviceManager()
        if dm.is_openvino_enabled():
            model_ir = dm.get_embedding_model_path()
            # Validate device availability through DeviceManager
            if device_override:
                ov_device = dm.select(device_override)
            else:
                ov_device = dm.select_from_settings()
            if model_ir and Path(model_ir).exists():
                from src.embeddings.openvino_encoder import OVEmbeddingEncoder
                encoder = OVEmbeddingEncoder(
                    model_xml=model_ir, device=ov_device
                )
                _info(f"Using OpenVINO encoder on {ov_device}")
            else:
                _warn(f"OV model not found at '{model_ir}', falling back to PyTorch")
    except Exception as exc:
        _info(f"OpenVINO not available ({exc}), using PyTorch encoder")

    if encoder is None:
        encoder = EmbeddingEncoder(device="cpu")
        _info("Using PyTorch/sentence-transformers encoder")

    chunk_texts = [c.text for c in chunks]
    embeddings = encoder.encode(chunk_texts, batch_size=64, show_progress=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    # save_embeddings only exists on the PyTorch encoder
    import numpy as _np
    _np.save(str(EMBEDDINGS_DIR / "chunk_embeddings.npy"), embeddings)
    _done(f"Shape {embeddings.shape}  ({embeddings.dtype})")

    # --- Step 6: Build FAISS index ---
    _step(6, 6, "Building FAISS index")
    chunk_metadata = [c.to_dict() for c in chunks]
    index = FaissIndex(dimension=encoder.dimension)
    index.build(embeddings, chunk_metadata)
    index.save(str(INDEX_DIR))
    _done(f"{index.size} vectors indexed")

    # --- Summary ---
    _summary_table([
        ("Documents loaded",  str(len(raw_docs))),
        ("OCR extractions",   str(ocr_count)),
        ("With text",         str(len(docs_with_text))),
        ("Chunks created",    str(len(chunks))),
        ("Vectors indexed",   str(index.size)),
        ("Time elapsed",      _elapsed(t0)),
    ])
    print(f"  {_C.GREEN}{_C.BOLD}{SYM_CHECK} Ingestion complete{_C.RESET}\n")


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

    _header(f"Search  {SYM_ARROW}  \"{_C.WHITE}{query}{_C.RESET}{_C.BOLD}{_C.CYAN}\"")

    # Phase 10: device-aware encoder selection
    encoder = None
    device_override = getattr(args, "device", None)
    try:
        from src.openvino.device_manager import DeviceManager
        dm = DeviceManager()
        if dm.is_openvino_enabled():
            model_ir = dm.get_embedding_model_path()
            # Validate device availability through DeviceManager
            if device_override:
                ov_device = dm.select(device_override)
            else:
                ov_device = dm.select_from_settings()
            if model_ir and Path(model_ir).exists():
                from src.embeddings.openvino_encoder import OVEmbeddingEncoder
                encoder = OVEmbeddingEncoder(
                    model_xml=model_ir, device=ov_device
                )
    except Exception:
        pass
    if encoder is None:
        encoder = EmbeddingEncoder(device="cpu")

    index = FaissIndex()
    try:
        index.load(str(INDEX_DIR))
    except FileNotFoundError:
        _error("Index not found.")
        _info("Run:  python cli.py ingest --dataset <name>")
        sys.exit(1)

    retriever = Retriever(encoder=encoder, index=index)
    results = retriever.query(query, top_k=top_k)

    if not results:
        _warn("No results found.")
        return

    print()
    for i, r in enumerate(results, 1):
        score_color = _C.GREEN if r.score >= 0.5 else (_C.YELLOW if r.score >= 0.3 else _C.RED)
        print(f"  {_C.BOLD}{_C.BLUE}#{i}{_C.RESET}  "
              f"{score_color}score {r.score:.4f}{_C.RESET}  "
              f"{_C.DIM}doc={r.doc_id}  chunk={r.chunk_id}{_C.RESET}")
        preview = r.text[:300].replace("\n", " ").strip()
        print(f"     {preview}")
        print()

    _info(f"Showing {len(results)} result(s) for top_k={top_k}")
    print()


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

    _header(f"Ask  {SYM_ARROW}  \"{_C.WHITE}{question}{_C.RESET}{_C.BOLD}{_C.CYAN}\"")

    # Phase 10: device-aware encoder selection
    encoder = None
    device_override = getattr(args, "device", None)
    try:
        from src.openvino.device_manager import DeviceManager
        dm = DeviceManager()
        if dm.is_openvino_enabled():
            model_ir = dm.get_embedding_model_path()
            # Validate device availability through DeviceManager
            if device_override:
                ov_device = dm.select(device_override)
            else:
                ov_device = dm.select_from_settings()
            if model_ir and Path(model_ir).exists():
                from src.embeddings.openvino_encoder import OVEmbeddingEncoder
                encoder = OVEmbeddingEncoder(
                    model_xml=model_ir, device=ov_device
                )
    except Exception:
        pass
    if encoder is None:
        encoder = EmbeddingEncoder(device="cpu")

    index = FaissIndex()
    try:
        index.load(str(INDEX_DIR))
    except FileNotFoundError:
        _error("Index not found.")
        _info("Run:  python cli.py ingest --dataset <name>")
        sys.exit(1)

    # Retrieve context
    _step(1, 2, "Retrieving relevant context")
    retriever = Retriever(encoder=encoder, index=index)
    results = retriever.query(question, top_k=top_k)
    context = retriever.format_context(results, max_chars=3000)

    if not results:
        _warn("No relevant context found in the knowledge base.")
        return
    _done(f"{len(results)} chunks retrieved")

    # Generate answer
    _step(2, 2, "Generating answer with LLM")
    
    # Read LLM provider from settings
    llm_settings = SETTINGS.get("llm", {})
    provider = llm_settings.get("provider", "ollama").lower()
    
    llm = None
    if provider == "openvino":
        # Try OpenVINO LLM
        try:
            from src.llm.openvino_llm import OVLLMClient
            from src.openvino.device_manager import DeviceManager
            
            dm = DeviceManager()
            ov_settings = SETTINGS.get("openvino", {})
            llm_model_dir = ov_settings.get("llm_model_dir", "")
            device = ov_settings.get("device", "CPU")
            
            if llm_model_dir and Path(llm_model_dir).exists():
                llm = OVLLMClient(model_dir=llm_model_dir, device=device)
                if llm.is_available():
                    _info(f"Using OpenVINO LLM from {llm_model_dir} on {device}")
                else:
                    _warn("OpenVINO LLM model failed to load, falling back to Ollama")
                    llm = None
            else:
                _warn(f"OpenVINO LLM model not found at '{llm_model_dir}', falling back to Ollama")
                _info("Convert a model first: optimum-cli export openvino --model mistralai/Mistral-7B-Instruct-v0.2 --weight-format int4 models/ov/mistral-7b-instruct")
        except ImportError as e:
            _warn(f"OpenVINO LLM not available ({e}), falling back to Ollama")
    
    # Fallback to Ollama or if explicitly requested
    if llm is None:
        from src.llm.ollama_client import OllamaClient
        llm = OllamaClient()
        if llm.is_available():
            _info("Using Ollama LLM")
        
    if not llm.is_available():
        _warn("No LLM available — showing retrieved context only.")
        _info("For Ollama: ollama serve && ollama pull mistral")
        _info("For OpenVINO: Set llm.provider='openvino' in settings.yaml and convert a model")
        print(f"\n{context}")
        return

    _info("Thinking...")
    answer = llm.generate(question=question, context=context)
    _done("Answer generated")

    print(f"\n  {_C.BOLD}{_C.WHITE}Q: {question}{_C.RESET}")
    print(f"\n  {_C.GREEN}{answer}{_C.RESET}")
    print(f"\n  {_C.DIM}Based on {len(results)} retrieved chunks.{_C.RESET}\n")


def cmd_stats(args: argparse.Namespace) -> None:
    """Display statistics about the current state of the pipeline."""
    _header("Pipeline Statistics")

    def _stat_line(label: str, value: str, ok: bool = True) -> None:
        icon = f"{_C.GREEN}{SYM_CHECK}{_C.RESET}" if ok else f"{_C.YELLOW}{SYM_WARN}{_C.RESET}"
        print(f"  {icon} {_C.DIM}{label:<26}{_C.RESET}{value}")

    # Raw data
    print(f"\n  {_C.BOLD}Raw Data{_C.RESET}")
    if RAW_DATA_ROOT.exists():
        datasets = sorted(d.name for d in RAW_DATA_ROOT.iterdir() if d.is_dir())
        _stat_line("Root", str(RAW_DATA_ROOT))
        _stat_line("Datasets", ", ".join(datasets) if datasets else "none")
    else:
        _stat_line("Root", f"{RAW_DATA_ROOT}  (not found)", ok=False)

    # Normalised documents
    print(f"\n  {_C.BOLD}Normalised Documents{_C.RESET}")
    norm_file = NORMALISED_DIR / "documents.jsonl"
    if norm_file.exists():
        with open(norm_file) as f:
            doc_count = sum(1 for line in f if line.strip())
        _stat_line("File", str(norm_file.relative_to(PROJECT_ROOT)))
        _stat_line("Document count", str(doc_count))
    else:
        _stat_line("Status", "Not created yet — run ingest", ok=False)

    # Embeddings
    print(f"\n  {_C.BOLD}Embeddings{_C.RESET}")
    emb_file = EMBEDDINGS_DIR / "chunk_embeddings.npy"
    if emb_file.exists():
        import numpy as np
        emb = np.load(str(emb_file))
        _stat_line("Shape", str(emb.shape))
        _stat_line("Dtype", str(emb.dtype))
        _stat_line("Size", f"{emb_file.stat().st_size / 1024:.1f} KB")
    else:
        _stat_line("Status", "Not created yet — run ingest", ok=False)

    # FAISS index
    print(f"\n  {_C.BOLD}FAISS Index{_C.RESET}")
    index_file = INDEX_DIR / "index.faiss"
    meta_file = INDEX_DIR / "metadata.json"
    if index_file.exists():
        _stat_line("Index size", f"{index_file.stat().st_size / 1024:.1f} KB")
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            _stat_line("Indexed chunks", str(len(meta)))
    else:
        _stat_line("Status", "Not created yet — run ingest", ok=False)

    # LLM
    print(f"\n  {_C.BOLD}LLM Backend{_C.RESET}")
    try:
        from src.llm.ollama_client import OllamaClient
        client = OllamaClient()
        if client.is_available():
            _stat_line("Ollama", f"available  (model: {client.model})")
        else:
            _stat_line("Ollama", f"model '{client.model}' not found or server not running", ok=False)
    except Exception as e:
        _stat_line("Ollama", f"not reachable ({e})", ok=False)

    print()


def cmd_devices(args: argparse.Namespace) -> None:
    """List available OpenVINO devices with detailed properties."""
    _header("OpenVINO Device Discovery")
    print()
    try:
        from src.openvino.device_manager import DeviceManager
        dm = DeviceManager()
        devices = dm.list_devices()
        if devices:
            for d in devices:
                props = dm.device_properties(d)
                name = props.get("FULL_DEVICE_NAME", "")
                arch = props.get("DEVICE_ARCHITECTURE", "")
                opt_req = props.get("OPTIMAL_NUMBER_OF_INFER_REQUESTS", "")
                print(f"  {_C.GREEN}{SYM_DOT}{_C.RESET} {_C.BOLD}{d:8s}{_C.RESET}  {name}")
                if arch:
                    print(f"             {_C.DIM}Architecture: {arch}{_C.RESET}")
                if opt_req:
                    print(f"             {_C.DIM}Optimal parallel requests: {opt_req}{_C.RESET}")

            # Show settings.yaml device preference
            print()
            selected = dm.select_from_settings()
            ov_enabled = dm.is_openvino_enabled()
            print(f"  {_C.BOLD}Settings:{_C.RESET}")
            print(f"    OpenVINO enabled: {_C.GREEN if ov_enabled else _C.YELLOW}{ov_enabled}{_C.RESET}")
            print(f"    Selected device:  {_C.GREEN}{selected}{_C.RESET}")
            model_ir = dm.get_embedding_model_path()
            ir_exists = Path(model_ir).exists() if model_ir else False
            ir_color = _C.GREEN if ir_exists else _C.RED
            print(f"    Embedding IR:     {ir_color}{model_ir} {'(found)' if ir_exists else '(NOT FOUND)'}{_C.RESET}")

            # Optional benchmark
            if getattr(args, "benchmark", False) and model_ir and ir_exists:
                print(f"\n  {_C.BOLD}Benchmarking...{_C.RESET}")
                results = dm.benchmark_devices(model_xml=model_ir)
                for dev, stats in results.items():
                    if "error" in stats:
                        print(f"    {dev}: {_C.RED}error — {stats['error']}{_C.RESET}")
                    else:
                        print(
                            f"    {_C.GREEN}{SYM_DOT}{_C.RESET} {dev:8s}  "
                            f"mean={stats['mean_ms']:.2f}ms  "
                            f"min={stats['min_ms']:.2f}ms  "
                            f"max={stats['max_ms']:.2f}ms"
                        )
        else:
            _warn("No devices found.")
            _info("Install OpenVINO:  pip install openvino")
    except ImportError as e:
        _warn(f"OpenVINO not available: {e}")
    print()


def cmd_ocr_tune(args: argparse.Namespace) -> None:
    """
    Test and tune OCR settings on a sample image from a dataset.
    
    Loads a single document, extracts ground-truth words, and tests
    multiple OCR configurations to find the optimal settings.
    """
    from src.ingestion.loader import DatasetLoader
    from src.ocr.tesseract_engine import TesseractEngine

    dataset = args.dataset
    sample_idx = args.sample
    show_comparison = args.compare

    _header(f"OCR Tuning  {SYM_ARROW}  dataset={_C.MAGENTA}{dataset}{_C.RESET}{_C.BOLD}{_C.CYAN}  sample={sample_idx}")

    # --- Load dataset ---
    _step(1, 3, "Loading sample document")
    loader = DatasetLoader(
        raw_data_root=str(RAW_DATA_ROOT),
        image_cache_dir=str(OCR_CACHE_DIR),
    )

    available = loader.list_available_datasets()
    if dataset not in available:
        _error(f"Dataset '{dataset}' not found.")
        _info(f"Available: {', '.join(available)}")
        sys.exit(1)

    raw_docs = loader.load_dataset(dataset, max_records=sample_idx + 1)
    if len(raw_docs) <= sample_idx:
        _error(f"Sample index {sample_idx} out of range (loaded {len(raw_docs)} docs)")
        sys.exit(1)

    doc = raw_docs[sample_idx]
    if not doc.image_path:
        _error(f"Sample {sample_idx} has no image.")
        sys.exit(1)
    if not doc.text:
        _warn(f"Sample {sample_idx} has no ground-truth text — comparison skipped.")
        ground_truth_words = []
    else:
        ground_truth_words = doc.text.split()

    _done(f"Loaded: {doc.doc_key}")
    _info(f"Image: {doc.image_path}")
    _info(f"Ground-truth words: {len(ground_truth_words)}")

    # --- Single OCR comparison (if requested) ---
    if show_comparison and ground_truth_words:
        _step(2, 3, "Running OCR comparison")
        try:
            engine = TesseractEngine(preprocess=True, confidence_threshold=40, psm=3)
            result = engine.compare_with_ground_truth(doc.image_path, ground_truth_words)
            
            _done("OCR completed")
            print(f"\n  {_C.BOLD}Results{_C.RESET}")
            print(f"    {_C.DIM}OCR words:{_C.RESET}          {result['ocr_word_count']}")
            print(f"    {_C.DIM}Ground-truth words:{_C.RESET} {result['gt_word_count']}")
            print(f"    {_C.GREEN}Precision:{_C.RESET}         {result['precision']:.1f}%")
            print(f"    {_C.GREEN}Recall:{_C.RESET}            {result['recall']:.1f}%")
            print(f"    {_C.GREEN}Word accuracy:{_C.RESET}     {result['word_accuracy']:.1f}%")
            
            if result['word_accuracy'] < 70:
                print(f"\n    {_C.YELLOW}{SYM_WARN} Low accuracy — try tuning settings{_C.RESET}")
            
            print(f"\n  {_C.BOLD}OCR Preview{_C.RESET}")
            preview = result['ocr_text'][:500]
            print(f"    {_C.DIM}{preview}...{_C.RESET}")
        except ImportError as e:
            _error(f"Missing dependency: {e}")
            sys.exit(1)

    # --- Settings tuning ---
    if not show_comparison:
        _step(2, 3, "Testing multiple OCR configurations")
        if not ground_truth_words:
            _warn("No ground truth available — cannot measure accuracy.")
            _info("Tuning requires datasets with text annotations (e.g. FUNSD).")
            return

        try:
            engine = TesseractEngine()
            results = engine.tune_settings(doc.image_path, ground_truth_words)
            
            _done(f"Tested {len(results)} configurations")
            
            print(f"\n  {_C.BOLD}Top 5 Configurations{_C.RESET}")
            print(f"  {_C.DIM}{'Rank':<6}{'Recall':<10}{'Precision':<12}{'Words':<8}{'Config'}{_C.RESET}")
            print(f"  {_C.DIM}{'─'*70}{_C.RESET}")
            
            for i, r in enumerate(results[:5], 1):
                cfg = r['config']
                preproc = "✓" if cfg.get('preprocess') else "✗"
                conf = cfg.get('confidence_threshold', 40)
                psm = cfg.get('psm', 3)
                
                recall_color = _C.GREEN if r['recall'] >= 80 else (_C.YELLOW if r['recall'] >= 60 else _C.RED)
                
                print(f"  {_C.BOLD}#{i:<5}{_C.RESET}"
                      f"{recall_color}{r['recall']:>5.1f}%{_C.RESET}   "
                      f"{r['precision']:>6.1f}%     "
                      f"{r['ocr_word_count']:>4}    "
                      f"{_C.DIM}preprocess={preproc} conf={conf} psm={psm}{_C.RESET}")
            
            print()
            best = results[0]
            print(f"  {_C.GREEN}{_C.BOLD}{SYM_CHECK} Best config:{_C.RESET} "
                  f"{_C.BOLD}preprocess={best['config'].get('preprocess')}, "
                  f"confidence_threshold={best['config'].get('confidence_threshold')}, "
                  f"psm={best['config'].get('psm')}{_C.RESET}")
            
        except ImportError as e:
            _error(f"Missing dependency: {e}")
            sys.exit(1)

    _step(3, 3, "Done")
    print()


# ===================================================================
# Video ingestion handler
# ===================================================================

def cmd_ingest_videos(args: argparse.Namespace) -> None:
    """
    Video ingestion pipeline.

    Two modes depending on dataset:
    - **Caption mode** (MSR-VTT): use pre-written captions from annotation JSON.
      Skips audio extraction and Whisper transcription entirely.
    - **Transcript mode** (custom datasets): extract audio → Whisper → OCR → merge.

    Processed output is written to data/processed/videos/.
    """
    from src.video.video_loader import VideoLoader
    from src.video.frame_sampler import FrameSampler
    from src.video.frame_ocr import FrameOCR
    from src.video.video_document_builder import VideoDocumentBuilder

    t0 = time.time()

    # ---- Read settings ------------------------------------------------
    video_settings = SETTINGS.get("video", {})

    # Video root: CLI override > settings.yaml > default
    video_root = getattr(args, "video_root", None)
    if not video_root:
        video_root = video_settings.get(
            "video_data_root",
            "/mnt/d/Openvino-project/data/raw/archive/data/MSRVTT/MSRVTT",
        )
    # Handle WSL / Windows path fallback
    video_root_path = Path(video_root)
    if not video_root_path.exists():
        alt = Path(r"D:\Openvino-project\data\raw\archive\data\MSRVTT\MSRVTT")
        try:
            if alt.exists():
                video_root_path = alt
                video_root = str(alt)
        except OSError:
            pass

    frame_interval = float(video_settings.get("frame_interval", 5))
    enable_whisper = video_settings.get("enable_whisper", False)
    whisper_model  = video_settings.get("whisper_model_size", "small")
    whisper_device = video_settings.get("whisper_device", "cpu")
    enable_ocr     = video_settings.get("enable_frame_ocr", True)
    ocr_min_words  = int(video_settings.get("ocr_min_words", 3))
    chunk_interval = float(video_settings.get("chunk_interval", 30))
    output_dir     = video_settings.get("output_dir", "data/processed/videos")
    output_dir     = str(PROJECT_ROOT / output_dir)

    max_videos = getattr(args, "max_videos", 0)
    source_name = getattr(args, "dataset", "msrvtt")

    _header(f"Video Ingest Pipeline  {SYM_ARROW}  source={_C.MAGENTA}{source_name}{_C.RESET}{_C.BOLD}{_C.CYAN}")
    _info(f"Video root: {video_root}")
    _info(f"Frame interval: {frame_interval}s | Whisper: {'ON' if enable_whisper else 'OFF (caption mode)'} | OCR: {enable_ocr}")
    if max_videos:
        _info(f"Max videos: {max_videos}")

    # ---- Step 1: Discover videos --------------------------------------
    _step(1, 5, "Discovering video files")
    loader = VideoLoader(root=video_root, max_files=max_videos)
    videos = loader.discover()

    if not videos:
        _warn(f"No video files found under {video_root}")
        _info("Check the path and ensure .mp4/.avi/.mkv files exist.")
        return

    # Report caption info
    videos_with_captions = sum(1 for v in videos if v.captions)
    if videos_with_captions:
        _done(f"Found {len(videos)} video(s) — {videos_with_captions} with pre-loaded captions")
    else:
        _done(f"Found {len(videos)} video file(s)")

    # ---- Step 2: Initialise pipeline components -----------------------
    _step(2, 5, "Initialising pipeline components")
    frames_dir = str(Path(output_dir) / "frames")

    frame_sampler = FrameSampler(
        output_dir=frames_dir,
        interval_seconds=frame_interval,
    )
    frame_ocr = FrameOCR(
        min_word_count=ocr_min_words,
    ) if enable_ocr else None
    builder = VideoDocumentBuilder(
        output_dir=output_dir,
        chunk_interval=chunk_interval,
    )

    # Whisper components (only loaded if needed)
    audio_extractor = None
    transcriber = None
    if enable_whisper:
        from src.video.audio_extractor import AudioExtractor
        from src.video.transcription import WhisperTranscriber
        audio_dir = str(Path(output_dir) / "audio")
        audio_extractor = AudioExtractor(output_dir=audio_dir)
        transcriber = WhisperTranscriber(
            model_size=whisper_model,
            device=whisper_device,
        )
    _done("Components ready")

    # ---- Steps 3-4: Process each video --------------------------------
    all_docs = []
    from tqdm import tqdm as _tqdm

    pbar = _tqdm(
        videos,
        desc="Processing videos",
        unit="video",
    )

    for video in pbar:
        pbar.set_postfix_str(Path(video.video_path).name)
        has_captions = bool(video.captions)

        # ---- Transcript path (Whisper) ----------------------------
        transcript_segments = []
        raw_segments = []
        if enable_whisper and not has_captions:
            _step(3, 5, f"Transcribing: {Path(video.video_path).name}")
            wav_path = audio_extractor.extract(video.video_path)
            if wav_path:
                _info("Transcribing audio...")
                raw_segments = transcriber.transcribe(wav_path)
                transcript_segments = transcriber.chunk_segments(
                    raw_segments, interval=chunk_interval
                )
                _done(f"{len(transcript_segments)} transcript chunks")
            else:
                _warn("No audio — skipping transcription")

        # ---- Frame sampling + OCR ---------------------------------
        _step(3 if has_captions else 4, 5,
              f"Extracting frames: {Path(video.video_path).name}")
        sampled_frames = frame_sampler.sample(
            video.video_path, video_id=video.video_id
        )
        _done(f"{len(sampled_frames)} frames sampled")

        ocr_results = []
        if enable_ocr and sampled_frames:
            _step(4 if has_captions else 5, 5,
                  f"Running OCR on frames: {Path(video.video_path).name}")
            ocr_results = frame_ocr.extract_batch(sampled_frames)
            _done(f"{len(ocr_results)} frames with text")
        elif not enable_ocr:
            _info("Frame OCR disabled in settings")

        # ---- Build document ---------------------------------------
        duration = frame_sampler.get_video_duration(video.video_path)

        if has_captions:
            # Caption mode (MSR-VTT) — use pre-loaded captions
            _step(5, 5, f"Building document (caption mode): {video.video_id}")
            doc = builder.build_from_captions(
                video_id=video.video_id,
                source=source_name,
                captions=video.captions,
                video_path=video.video_path,
                duration=duration,
                ocr_results=ocr_results,
                extra_metadata=video.metadata,
            )
        else:
            # Transcript mode (Whisper)
            _step(5, 5, f"Building document (transcript mode): {video.video_id}")
            doc = builder.build(
                video_id=video.video_id,
                source=source_name,
                transcript_segments=(
                    raw_segments if raw_segments else []
                ),
                ocr_results=ocr_results,
                video_path=video.video_path,
                duration=duration,
                extra_metadata=video.metadata,
            )

        all_docs.append(doc)
        _done(f"{len(doc.chunks)} chunks, {len(doc.text)} chars")

    pbar.close()

    # ---- Save all documents -------------------------------------------
    if all_docs:
        _info("Saving all video documents...")
        builder.save_batch(all_docs)
        _done(f"Saved {len(all_docs)} video documents to {output_dir}")

    # ---- Summary ------------------------------------------------------
    total_chunks = sum(len(d.chunks) for d in all_docs)
    total_chars = sum(len(d.text) for d in all_docs)
    _summary_table([
        ("Videos processed",   str(len(all_docs))),
        ("Total chunks",       str(total_chunks)),
        ("Total characters",   str(total_chars)),
        ("Output directory",   output_dir),
        ("Time elapsed",       _elapsed(t0)),
    ])
    print(f"  {_C.GREEN}{_C.BOLD}{SYM_CHECK} Video ingestion complete{_C.RESET}\n")


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
    p_ingest.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override OpenVINO device (CPU, GPU, NPU, AUTO). Default: from settings.yaml",
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
    p_search.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override OpenVINO device (CPU, GPU, NPU, AUTO). Default: from settings.yaml",
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
    p_ask.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override OpenVINO device (CPU, GPU, NPU, AUTO). Default: from settings.yaml",
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
    p_devices.add_argument(
        "--benchmark",
        action="store_true",
        help="Run inference benchmark on all available devices",
    )
    p_devices.set_defaults(func=cmd_devices)

    # -- ocr-tune --
    p_ocr = subparsers.add_parser(
        "ocr-tune",
        help="Test and tune OCR settings on sample images",
    )
    p_ocr.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (must have ground-truth text, e.g. funsd)",
    )
    p_ocr.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Sample index to test (default: 0)",
    )
    p_ocr.add_argument(
        "--compare",
        action="store_true",
        help="Show single OCR comparison instead of tuning multiple configs",
    )
    p_ocr.set_defaults(func=cmd_ocr_tune)

    # -- ingest-videos --
    p_video = subparsers.add_parser(
        "ingest-videos",
        help="Ingest video files: extract audio, transcribe, OCR frames, build JSON",
    )
    p_video.add_argument(
        "--dataset",
        type=str,
        default="msrvtt",
        help="Video dataset source name (default: msrvtt)",
    )
    p_video.add_argument(
        "--video-root",
        type=str,
        default=None,
        dest="video_root",
        help="Override path to video dataset directory (default: from settings.yaml)",
    )
    p_video.add_argument(
        "--max-videos",
        type=int,
        default=0,
        dest="max_videos",
        help="Limit the number of videos to process (0 = all, default: 0)",
    )
    p_video.set_defaults(func=cmd_ingest_videos)

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
        print(f"\n  {_C.YELLOW}Interrupted by user.{_C.RESET}")
        sys.exit(130)
    except ImportError as exc:
        print(f"\n  {_C.RED}{SYM_CROSS} Missing dependency: {exc}{_C.RESET}")
        sys.exit(1)
    except FileNotFoundError as exc:
        print(f"\n  {_C.RED}{SYM_CROSS} File not found: {exc}{_C.RESET}")
        sys.exit(1)
    except Exception as exc:
        if getattr(args, "verbose", False):
            logging.error("Command failed: %s", exc, exc_info=True)
        else:
            print(f"\n  {_C.RED}{SYM_CROSS} Error: {exc}{_C.RESET}")
            print(f"  {_C.DIM}Run with --verbose for full traceback.{_C.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
