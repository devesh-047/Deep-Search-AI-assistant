# Deep Search AI Assistant (GSoC 2026 OpenVINO)

An advanced multimodal RAG system designed for AI PCs, integrating local LLMs with OpenVINO optimization. This project ingests documents and videos, creating a unified knowledge base queryable via natural language.

## Project Overview

This application demonstrates a complete offline RAG pipeline optimized for Intel hardware. It moves beyond simple text search by incorporating:

1.  **Multimodal Ingestion**: Processes PDFs (OCR) and Videos (MSR-VTT dataset).
2.  **Semantic Search**: Uses vector embeddings to find relevant content regardless of exact keyword matches.
3.  **Local LLM Integration**: Uses OpenVINO-optimized models (Mistral/Llama3) for generation.
4.  **Hardware Acceleration**: Leverages Intel CPU/iGPU/NPU via OpenVINO Runtime.

## Architecture

```text
       [ User Query ]
             |
             v
   +---------------------+
   |   Embedding Model   |
   |     (CPU / iGPU)    |
   +----------+----------+
              |
              v
   +---------------------+
   |   Vector Database   |
   |  (Context Retrieval)|
   +----------+----------+
              |
              v
   +---------------------+
   |    LLM Inference    |
   |     (OpenVINO™)     |
   |    (NPU / iGPU)     |
   +----------+----------+
              |
              v
      [ Final Response ]
```

The pipeline consists of modular stages:

1.  **Ingestion Layer**:
    *   **Documents**: PDF/Image ingestion using OCR (Tesseract/PaddleOCR).
    *   **Videos**: MSR-VTT dataset processing (Caption-based or Transcript-based).
2.  **Normalization Layer**: Converts all inputs into a unified `NormalizedDocument` schema.
3.  **Embedding Layer**: Generates vector representations using `all-MiniLM-L6-v2` (OpenVINO optimized) and optionally CLIP (`clip-vit-base-patch32`) for multimodal image-text embeddings.
4.  **Indexing Layer**: Stores vectors in FAISS indices for efficient similarity search (separate text and CLIP indices).
5.  **Retrieval & Generation**:
    *   Retrieves relevant chunks based on query similarity.
    *   Optionally fuses text and CLIP results for multimodal queries (e.g. "find slides with charts").
    *   Augments LLM prompt with context.
    *   Generates answers using local OpenVINO LLM.

## Project Structure

```
├── cli.py                     # Main entry point (CLI)
├── configs/
│   ├── models.yaml            # Model definitions (LLM, Embeddings)
│   └── settings.yaml          # Main configuration file
├── data/
│   ├── raw/                   # Input datasets (MSR-VTT, Docs)
│   └── processed/             # Normalized/Intermediate JSONs
├── scripts/
│   ├── download_models.py     # Setup utility to cache models
│   └── verify_setup.py        # Environmental sanity check
└── src/
    ├── benchmark/             # Performance benchmarking module
    │   ├── embedding_benchmark.py  # PyTorch vs OpenVINO embedding comparison
    │   ├── llm_benchmark.py        # LLM generation speed measurement
    │   └── system_metrics.py       # CPU and memory instrumentation (psutil)
    ├── embeddings/            # Vector generation (OpenVINO)
    ├── index/                 # FAISS vector store management
    ├── ingestion/             # Document loader & normalizer
    ├── llm/                   # LLM interfaces (OpenVINO/Ollama)
    ├── ocr/                   # Optical Character Recognition engines
    ├── openvino/              # Device management & optimization
    ├── retrieval/             # Semantic search logic
    └── video/                 # MSR-VTT video processing pipeline
```

## Dataset Details

### MSR-VTT (Video Retrieval)
*   **Location**: `data/raw/archive/data/MSRVTT`
*   **Description**: Large-scale video benchmark for video-to-text retrieval.
*   **Content**: 10,000 diverse video clips (music, news, sports, cartoons) with 200,000 natural language captions.
*   **Role**: Used to test video ingestion, captioning, and semantic search over temporal media.

### DocVQA (Visual Question Answering)
*   **Location**: `data/raw/docvqa`
*   **Description**: A dataset designed for Visual Question Answering on document images.
*   **Content**: Diverse document types (invoices, reports, forms) with question-answer pairs.
*   **Role**: Evaluates the RAG pipeline's ability to answer specific queries grounded in complex visual documents.

### FUNSD (Form Understanding)
*   **Location**: `data/raw/funsd`
*   **Description**: Form Understanding in Noisy Scanned Documents.
*   **Content**: Scanned forms with noisy text and spatial layout annotations.
*   **Role**: Tests the robustness of the OCR engine and spatial reasoning capabilities.

### RVL-CDIP (Document Classification)
*   **Location**: `data/raw/rvl_cdip`
*   **Description**: Ryerson Vision Lab Complex Document Information Processing.
*   **Content**: 400,000 grayscale images spanning 16 document categories (letters, memos, emails, etc.).
*   **Role**: Provides a large-scale corpus for stress-testing document ingestion and indexing performance.

### Supported Data Types

The system is designed to handle diverse multimodal inputs:

| Data Type | Supported Formats | Processing Method |
| :--- | :--- | :--- |
| **Documents** | `.pdf`, `.docx`, `.pptx`, `.txt` | Text extraction (pdfplumber / python-docx / python-pptx) |
| **Images** | `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp` | OCR (Tesseract / PaddleOCR with OpenVINO) + CLIP embeddings |
| **Videos** | `.mp4`, `.avi`, `.mov`, `.mkv` | Frame Sampling + Audio Transcription (Whisper) + Caption Integration |
| **Metadata** | `.json`, `.txt` | Structural parsing (e.g., MSR-VTT annotations) |

## Setup Guide

### Prerequisites
*   OS: Windows 11 (via WSL2 Ubuntu 20.04 recommended) or Linux.
*   Python: 3.8 - 3.11.
*   System Dependencies: `ffmpeg` (for video), `tesseract-ocr` (for docs).

### Installation

1.  **Clone the repository**:
    ```bash
    git clone <repo-url>
    cd Deep-Search-AI-assistant
    ```

2.  **Create Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install System Tools (Ubuntu/WSL)**:
    ```bash
    sudo apt update
    sudo apt install ffmpeg tesseract-ocr
    ```

## LLM Setup

The system supports two LLM backends. You only need **one** of them.

---

### Option A — Ollama (recommended for getting started)

Ollama runs Mistral locally over a REST API. No GPU required.

**1. Install Ollama**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
On Windows/WSL, download the installer from https://ollama.com/download.

**2. Start the Ollama server**
```bash
ollama serve
```
Keep this running in a separate terminal (or run it as a background service).

**3. Pull the Mistral model**
```bash
ollama pull mistral
```
This downloads ~4 GB. Run it once — the model is cached afterwards.

**4. Verify it works**
```bash
ollama run mistral "Hello, what are you?"
```

**5. Configure the project to use Ollama**

In `configs/settings.yaml`, set:
```yaml
llm:
  provider: ollama
  model: mistral
  endpoint: http://localhost:11434
```

That's it. `python cli.py ask "..."` will now use Ollama automatically.

---

### Option B — OpenVINO GenAI (on-device, no server needed)

OpenVINO runs the LLM directly on Intel CPU / iGPU / NPU without Ollama.
This is faster on Intel hardware and requires no background server.

**1. Convert a model to OpenVINO IR format**

Install the conversion tool if not already present:
```bash
pip install optimum-intel[openvino]
```

Convert Mistral 7B Instruct with INT4 weight compression (requires ~4 GB disk):
```bash
optimum-cli export openvino \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --weight-format int4 \
    models/ov/mistral-7b-instruct
```

Or convert a smaller model for lower-memory systems (Phi-3 Mini, ~2 GB):
```bash
optimum-cli export openvino \
    --model microsoft/Phi-3-mini-4k-instruct \
    --weight-format int4 \
    models/ov/phi-3-mini
```

**2. Configure the project to use OpenVINO**

In `configs/settings.yaml`, set:
```yaml
llm:
  provider: openvino

openvino:
  enabled: true
  device: CPU          # or GPU, NPU, AUTO
  llm_model_dir: models/ov/mistral-7b-instruct
```

**3. Select the best device**
```bash
# List available Intel devices (CPU / iGPU / NPU)
python3 cli.py devices
```

| Device | Use when |
| :--- | :--- |
| `CPU` | Always available, good baseline |
| `GPU` | Intel iGPU (Iris Xe) — faster than CPU for large models |
| `NPU` | Intel AI Boost NPU (Core Ultra) — lowest power, good for inference |
| `AUTO` | Let OpenVINO pick the best available device automatically |

Set your preferred device in `configs/settings.yaml` under `openvino.device`.

**4. Verify**
```bash
python3 cli.py ask "What is in the document?"
```
The pipeline will log `Using OpenVINO LLM from models/ov/...` when the backend is active.

---

### Switching between backends

Edit `configs/settings.yaml` and change `llm.provider`:

```yaml
llm:
  provider: ollama      # ← change to "openvino" to switch
```

No code changes required.

## Configuration

Settings are managed in `configs/settings.yaml`. Key sections:

*   `llm`: Select model provider (OpenVINO vs Ollama) and model name.
*   `openvino`: OpenVINO device, embedding IR path, and LLM model directory.
*   `ocr`: OCR engine selection (`tesseract` or `paddleocr`) and preprocessing options.
*   `faiss`: FAISS index type settings (`IndexFlatIP` is default).
*   `video`: MSR-VTT dataset path and processing mode.
    *   `enable_whisper`: Set to `false` for caption mode (faster), `true` for transcripts.
*   `clip`: CLIP multimodal retrieval settings (enable/disable, model name, OpenVINO flag).

## How to Run

### 1. Ingest Documents
Runs the full pipeline for a document dataset: load → OCR → normalise → chunk → embed → build FAISS index.
```bash
# Ingest from DocVQA dataset (limit to 5 records for quick demo)
python3 cli.py ingest --dataset docvqa --max-records 5

# Ingest from FUNSD (form understanding dataset)
python3 cli.py ingest --dataset funsd --max-records 5

# Ingest full dataset (no limit)
python3 cli.py ingest --dataset docvqa
```
> **Note:** The `ingest` command builds the FAISS index automatically as its final step — no separate build step is needed.

### 2. Ingest Videos
Processes MSR-VTT dataset into normalised JSON documents.
```bash
# Default (MSR-VTT caption mode, all videos)
python3 cli.py ingest-videos

# Limit to 5 videos for a quick demo
python3 cli.py ingest-videos --max-videos 5
```

### 3. Search
Perform semantic search (retrieval only, no LLM).
```bash
python3 cli.py search "cartoon character playing guitar"
```

### 4. Ask (RAG)
Generate answers using the LLM.
```bash
python3 cli.py ask "What is the date mentioned in the approval letter?"
```

### 5. Inspect Pipeline State
```bash
# Show index size, document count, LLM availability
python3 cli.py stats

# List available OpenVINO devices (CPU / iGPU / NPU)
python3 cli.py devices
```

## Performance Evaluation

### Why Benchmarking Matters

Real-world AI workloads require quantifiable evidence of efficiency improvements.
This project includes a dedicated benchmarking module (`src/benchmark/`) that
measures the concrete impact of OpenVINO optimization — not through claims, but
through measured numbers on the same hardware under identical conditions.

### Measured Metrics

| Metric | Description |
| :--- | :--- |
| **Avg Latency (ms)** | Mean wall-clock time per encode or generate call across timed iterations |
| **Throughput (samples/sec)** | Texts embedded per second; higher is better |
| **Tokens/sec** | Output tokens generated per second; higher is better |
| **Mean CPU (%)** | Average CPU utilization during timed runs (via psutil) |
| **Peak RSS (MB)** | Maximum resident memory consumed during inference |
| **Model load time (s)** | One-time cost to load/compile the model; excluded from latency |

Warmup iterations are always excluded from reported numbers to avoid
measuring JIT compilation and cache-warming artifacts. All iterations use
`time.perf_counter()` for sub-millisecond precision.

### How to Run Benchmarks

```bash
# Run all benchmarks (embedding + LLM)
python3 cli.py benchmark --all

# Embedding only (PyTorch vs OpenVINO comparison)
python3 cli.py benchmark --embeddings

# LLM only
python3 cli.py benchmark --llm

# Custom batch size and iteration count
python3 cli.py benchmark --embeddings --batch-size 32 --iterations 50 --warmup 5
```

### Benchmark Results (Measured on Project Hardware)

**Embedding Benchmark** — 64 texts, batch size 16, 20 timed iterations (3 warmup):

```
Embedding Benchmark Results
--------------------------------------------------
Corpus size  : 64 texts
Batch size   : 16
Iterations   : 20  (warmup: 3)

PyTorch CPU:
  Avg Latency  : 527.24 ms
  Min Latency  : 445.19 ms
  Max Latency  : 977.01 ms
  Throughput   : 121.4 samples/sec
  Mean CPU     : 606.8%
  Peak RSS     : 873.1 MB
  Model load   : 5.025 s

OpenVINO CPU:
  Avg Latency  : 466.03 ms
  Min Latency  : 419.88 ms
  Max Latency  : 591.76 ms
  Throughput   : 137.3 samples/sec
  Mean CPU     : 548.1%
  Peak RSS     : 1051.7 MB
  Model load   : 2.611 s

Speedup      : 1.13x  (OpenVINO vs PyTorch)
--------------------------------------------------
```

**Key observations from the embedding benchmark:**

- OpenVINO delivers a **1.13x latency reduction** and a **more consistent** latency
  distribution (Min–Max spread of 172 ms vs 532 ms for PyTorch), demonstrating
  that graph-level optimizations reduce variance even where raw speedup is modest.
- **OpenVINO model load time is 48% faster** (2.6 s vs 5.0 s), which matters when
  the encoder is cold-started per CLI invocation.
- CPU utilization is **9% lower under OpenVINO** (548% vs 607%), indicating more
  efficient use of available cores for the same workload.
- The speedup margin is hardware-dependent. On Intel iGPU or NPU (`--device GPU`
  / `--device NPU` in `settings.yaml`) significantly larger gains are expected.

**LLM Benchmark** — hardware requirement note:

Running a 7B-parameter model (Mistral or Llama3) in INT4 format requires
approximately **4–6 GB of available RAM** after other processes. On constrained
systems the OS will terminate the process before generation begins. To run the
LLM benchmark successfully:

```bash
# Verify available memory first
free -h

# Run LLM benchmark only (embedding model already unloaded)
python3 cli.py benchmark --llm

# Or use a smaller model configured in settings.yaml
```

> All numbers above are real measurements captured with `time.perf_counter()`
> on project hardware (WSL2 / Ubuntu 20.04). Results will vary by CPU model,
> available RAM, and thread contention from background processes.

### How OpenVINO Improves Inference

OpenVINO applies a set of hardware-targeted transformations between model
export and execution:

1. **Graph Fusion** — Adjacent operations (e.g., MatMul + Bias + Activation)
   are merged into single hardware kernels, reducing memory bandwidth.
2. **INT8 Quantization** — Weight values are compressed from FP32 to INT8,
   halving memory traffic and enabling vectorized integer math on AVX-512.
3. **Thread Parallelism** — The runtime automatically tiles workloads across
   physical cores using its own thread pool, eliminating PyTorch overhead.
4. **Device-Specific Dispatch** — The same model can run on CPU, iGPU, or
   NPU by changing a single device string, with device-optimal code paths
   selected automatically.

These optimizations are transparent to application code: the encoder and
LLM client interfaces are identical regardless of backend.

## OpenVINO Integration

This project explicitly leverages OpenVINO for:

1.  **Embeddings**: `sentence-transformers` models are exported to OpenVINO IR format for faster CPU inference.
2.  **LLM Inference**: Supports `optimum-intel` to run LLMs directly on Intel CPU/GPU/NPU without external servers.
3.  **OCR**: PaddleOCR inference accelerated via OpenVINO ONNX backend (default engine; Tesseract fallback).
4.  **CLIP Vision**: CLIP image encoder optionally compiled to OpenVINO IR for accelerated multimodal embeddings.

Device selection (`CPU`, `GPU`, `NPU`) is configurable in `settings.yaml`.

## Metadata-Aware Staged Retrieval (Enhancement)

An optional metadata pre-filtering stage can be enabled to narrow the vector search space before ANN search. This improves relevance and reduces latency for queries that reference specific years, file types, or content modalities.

```bash
# Enable with --metadata-filtering flag
python3 cli.py search "Find 2024 invoices" --metadata-filtering
python3 cli.py ask "What is in the presentation slides?" --metadata-filtering

# Run comparison benchmark
python3 scripts/compare_retrieval_modes.py
```

See [README_metadata_retrieval.md](README_metadata_retrieval.md) for full documentation on architecture, usage, and interpretation.

## Multimodal Document Support

The ingestion pipeline supports the following document types out of the box:

| Format | Library | Extracted Content |
| :--- | :--- | :--- |
| **PDF** | `pdfplumber` | Full page text |
| **DOCX** | `python-docx` | Headings, paragraphs, and tables (pipe-separated) |
| **PPTX** | `python-pptx` | Slide-numbered output with titles, content text, and tables |
| **Images** | Tesseract / PaddleOCR (OpenVINO) | OCR-extracted text + CLIP multimodal embeddings |
| **Videos** | Whisper + OpenCV | Captions, audio transcription, and frame OCR |
| **TXT** | Built-in | Raw text |

All formats are normalised into the unified `NormalizedDocument` schema and indexed identically.

### DOCX Output Format

DOCX files are extracted with structural markers:

```
Heading: Project Plan

Paragraph:
This document describes the project timeline.

Table:
Task | Owner | Deadline
Design | Alice | 2024-01-15
Build | Bob | 2024-03-01
```

### PPTX Output Format

PPTX files are extracted with slide numbering and titles:

```
Slide 1
Title: Introduction
Content: Welcome to the quarterly review.

Slide 2
Title: Results
Content: Revenue increased by 15%.
Table:
Quarter | Revenue
Q1 | $1.2M
Q2 | $1.4M
```

## Local File Query Mode

You can ingest a local file or directory and ask a question in a single command.
The system will automatically ingest, index, retrieve, and answer.

Supports **documents** (PDF, DOCX, PPTX, TXT), **images** (PNG, JPG, etc.), and **videos** (MP4, AVI, MKV, MOV, WebM, FLV, WMV) — all in one command.

```bash
# Query a directory of documents
python cli.py --path ./documents --ask "What is the payment amount?"

# Query a single file
python cli.py --path ./report.pdf --ask "Summarize this document"

# Query a video file — automatically extracts frames, runs OCR, transcribes audio
python cli.py --path video.mp4 --ask "What is the main topic of this video?"

# Query a mixed folder (docs + images + videos)
python cli.py --path ./mixed_folder --ask "Find mentions of invoices"

# With metadata-aware filtering
python cli.py --path ./docs --ask "Find invoices from 2019" --metadata-filtering

# Control the number of retrieved chunks
python cli.py --path ./docs --ask "Key findings" --top-k 10
```

For video files, the pipeline automatically:
- Extracts frames at 5-second intervals using OpenCV
- Runs Tesseract OCR on frames to capture on-screen text
- Generates BLIP captions for every Nth frame (natural-language scene descriptions)
- Extracts audio and transcribes it with Whisper
- Merges OCR + captions + transcript into a unified text representation
- CLIP visual search augments retrieval for non-text frames
- Falls back gracefully: if one component fails, the others continue

The pipeline executes:

1. **Load** — Detect file types and extract text (PDF, DOCX, PPTX, TXT, images, videos)
2. **Video Processing** — For video files: frame sampling → OCR → BLIP captioning → Whisper transcription
3. **OCR** — Run Tesseract/PaddleOCR on image files
4. **Normalise** — Clean text, detect language, deduplicate
5. **Chunk** — Split text into 512-character passages with overlap
6. **Embed** — Generate dense embeddings (PyTorch or OpenVINO)
7. **CLIP** — Generate CLIP image embeddings for visual retrieval (when enabled)
8. **Index** — Build FAISS vector index
9. **Retrieve + Answer** — Semantic search (+ multimodal CLIP fusion with `--multimodal`) + LLM-generated answer

## AI Usage Disclosure

**Use of AI Assistance**:
AI tools were used during the development of this project for the following specific tasks:

1.  **Code Structure**: Generating boilerplate command-line interface (CLI) logic and forming the repository structure.
2.  **Documentation**: Assisting in structuring and formatting this README file.
3.  **Debugging**: Analyzing stack traces to identify missing dependencies and for other debuggung purposes.
4.  **Dataset Selection**: Suggesting appropriate public datasets (Eg: MSR-VTT) for multimodal testing.

The core architecture, logical implementation of the RAG pipeline, OpenVINO integration strategies, and final code verification were performed manually by the developer. All code has been reviewed and validated.
