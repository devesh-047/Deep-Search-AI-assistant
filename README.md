# Deep Search AI Assistant

## Multimodal RAG-Based Search on Personal Knowledge Base for AI PC

An offline, privacy-preserving personal AI assistant that builds a multimodal
knowledge base from document images (forms, invoices, memos, scanned PDFs) and
answers natural-language questions using Retrieval Augmented Generation (RAG).
Designed for deployment on Intel AI PCs with OpenVINO acceleration.

This project aligns with the
[OpenVINO GSoC project description](https://github.com/openvinotoolkit/openvino/wiki/GSoC)
for "Deep Search AI Assistant on Multimodal Personal Database for AIPC."

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Why RAG](#3-why-rag)
4. [Why OpenVINO](#4-why-openvino)
5. [Why Python](#5-why-python-over-c)
6. [Model Stack](#6-model-stack)
7. [Directory Structure](#7-directory-structure)
8. [Setup Guide](#8-setup-guide)
9. [CLI Reference](#9-cli-reference)
10. [Step-by-Step Learning Roadmap](#10-step-by-step-learning-roadmap)
11. [Data Flow Rules](#11-data-flow-rules)
12. [OpenVINO Integration Roadmap](#12-openvino-integration-roadmap)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Project Overview

### What this project does

Deep Search is a core function of a personal AI assistant.  It enables
information extraction and fuzzy semantic queries across document images,
scanned forms, invoices, memos, and other personal documents.

The system:

- Creates a **multimodal personal knowledge base** from document images
- Uses **Retrieval Augmented Generation (RAG)** to enhance a local LLM with
  retrieved context
- Runs **entirely offline** -- no cloud services, no data leaves the machine
- Protects **user privacy** by design
- Deploys efficiently on **Intel AI PCs** using OpenVINO Runtime
- Supports **hardware-aware execution** across CPU, iGPU, and NPU

### What users can do

- Ask natural-language questions about their personal documents
- Perform fuzzy / semantic search across a heterogeneous document collection
- Retrieve and summarize information from private multimodal data
- Get answers grounded in actual document content (not hallucinated)

### What this project is NOT

This is a **learning-oriented project skeleton**.  It is not a finished product.
The code is correct, modular, and production-style in structure, but many
components are surface-level implementations or placeholders.  The intent is
that a motivated student builds understanding by progressively implementing
each pipeline stage, guided by this README.

No training or fine-tuning is performed.  No benchmarking suites are included.
No cloud services are used.

---

## 2. System Architecture

```
+------------------------------------------------------------------+
|                        CLI  (cli.py)                              |
|   ingest | search | ask | stats | devices                        |
+-----+--------+--------+--------+--------+-----------------------+
      |        |        |        |        |
      v        |        |        |        v
+----------+   |    +--------+   |   +-----------+
| INGEST   |   |    | SEARCH |   |   | OPENVINO  |
| PIPELINE |   |    | (pure  |   |   | device    |
|          |   |    | embed) |   |   | manager   |
+----+-----+   |    +---+----+   |   +-----------+
     |         |        |        |
     v         |        v        v
+---------+    |   +---------+  +---------+
| Step 1  |    |   |Embedding|  |Retriever|
| Load    |    |   |Encoder  |  |  +LLM   |
| Dataset |    |   +----+----+  +----+----+
+----+----+    |        |            |
     |         |        v            v
     v         |   +---------+  +---------+
| Step 2  |    |   | FAISS   |  | Ollama  |
| OCR     |    |   | Index   |  | Mistral |
| Extract |    |   +---------+  +---------+
+----+----+    |
     |         |
     v         |
| Step 3  |    |       DATA FLOW
| Normal- |    |       =========
| ize     |    |
+----+----+    |       Raw Data (read-only)
     |         |         D:\Openvino-project\data\raw\   (/mnt/d/... in WSL)
     v         |              |
| Step 4  |    |              v  (load, extract images)
| Chunk   |    |       Processed Data (write)
+----+----+    |         Deep-Search-AI-assistant/data/processed/
     |         |              |
     v         |              +-- ocr_cache/     (extracted PNGs)
| Step 5  |    |              +-- normalised/    (documents.jsonl)
| Embed   |    |              +-- chunks/        (future)
+----+----+    |              +-- embeddings/    (chunk_embeddings.npy)
     |         |              +-- faiss/         (index.faiss + metadata.json)
     v         |
| Step 6  |    |
| Index   |    |
+---------+    |

QUERY FLOW:
  User question
       |
       v
  Encode query --> search FAISS --> top-k chunks --> format context
       |                                                  |
       v                                                  v
  [search command stops here]              [ask command continues]
                                                          |
                                                          v
                                                 LLM generates answer
                                                 grounded in context
```

### Pipeline stages in detail

| Stage | Module | Input | Output |
|-------|--------|-------|--------|
| 1. Load | `src/ingestion/loader.py` | Arrow datasets from `data/raw/` | `List[RawDocument]` |
| 2. OCR | `src/ocr/tesseract_engine.py` | Document images (PNG) | Extracted text strings |
| 3. Normalize | `src/ingestion/normalizer.py` | `List[RawDocument]` | `List[NormalizedDocument]` (unified schema) |
| 4. Chunk | `src/ingestion/chunker.py` | `NormalizedDocument.text` | `List[TextChunk]` (512-char windows) |
| 5. Embed | `src/embeddings/encoder.py` | `List[str]` (chunk texts) | `np.ndarray` shape `(N, 384)` |
| 6. Index | `src/index/faiss_index.py` | Embeddings + metadata | FAISS `IndexFlatIP` |
| 7. Retrieve | `src/retrieval/retriever.py` | Query string | `List[RetrieverResult]` |
| 8. Generate | `src/llm/ollama_client.py` | Question + context | Natural-language answer |

---

## 3. Why RAG

### The problem with standalone LLMs

A local LLM (Mistral 7B) has been trained on public internet data.  It knows
nothing about personal documents -- invoices, internal memos, handwritten
forms.  Asking it "What is the invoice total?" without context produces
hallucinations.

### What RAG solves

Retrieval Augmented Generation separates the pipeline into two stages:

1. **Retrieval**: Find the most relevant document chunks for the question using
   embedding-based semantic search.  This is fast, deterministic, and
   grounded in actual data.

2. **Generation**: Feed the retrieved context into the LLM prompt.  The LLM
   synthesizes an answer using only the provided context.

### Why this architecture matters

- **Grounded answers**: The LLM can only reference text that actually exists
  in your documents.
- **No fine-tuning needed**: The LLM is used as-is.  Updating the knowledge
  base means re-indexing, not re-training.
- **Privacy**: Documents never leave the machine.  Embeddings and the FAISS
  index are local artifacts.
- **Transparency**: Every answer cites which document chunks were used.
- **Separation of concerns**: Retrieval quality and generation quality can be
  improved independently.

### What RAG does NOT do

- It does not make the LLM "understand" your documents.  It gives the LLM
  relevant context at query time.
- It does not eliminate hallucination entirely.  If the retrieved context is
  poor, the LLM may still produce incorrect answers.
- It does not replace a database.  RAG is optimized for fuzzy semantic queries,
  not exact lookups.

---

## 4. Why OpenVINO

### The deployment target

This project targets Intel AI PCs -- consumer laptops and desktops with:

- Intel CPUs (Alder Lake, Raptor Lake, Meteor Lake, Arrow Lake)
- Integrated GPUs (Intel Xe, Arc)
- Neural Processing Units (NPU on Meteor Lake and later)

OpenVINO is Intel's inference optimization toolkit.  It converts trained models
into an optimized Intermediate Representation (IR) and executes them
efficiently across all three device types through a single API.

### Specific benefits for this project

| Benefit | Explanation |
|---------|-------------|
| CPU optimization | Graph-level optimizations, operator fusion, threading |
| iGPU access | Offload embedding or OCR inference to the integrated GPU |
| NPU access | Best performance-per-watt for sustained inference on Meteor Lake+ |
| No PyTorch at inference | Smaller runtime footprint, faster cold start |
| Unified API | Same code targets CPU, GPU, and NPU via device string |
| INT8/FP16 compression | Reduce model size and increase throughput |

### What gets accelerated

1. **Embedding model** (all-MiniLM-L6-v2): Convert to OpenVINO IR, run on CPU
   or iGPU for faster chunk encoding during ingestion and query.
2. **OCR models** (PaddleOCR, future): PaddlePaddle detection/recognition
   models convert to IR for hardware-accelerated text extraction.
3. **LLM** (Mistral 7B, future): Use `openvino-genai` or `optimum-intel` to
   run the LLM on CPU with INT4 weight compression, eliminating the
   dependency on Ollama.

### Current status in this project

OpenVINO integration is structured as a **learning progression**:

- Phase 1 (current): CPU-first with PyTorch/sentence-transformers and Ollama
- Phase 2: Convert embedding model to OpenVINO IR (`src/openvino/model_converter.py`)
- Phase 3: Replace PyTorch encoder with `OVEmbeddingEncoder` (`src/embeddings/openvino_encoder.py`)
- Phase 4: Add PaddleOCR with OpenVINO backend (`src/ocr/paddle_engine.py`)
- Phase 5: Replace Ollama with `OVLLMClient` (`src/llm/openvino_llm.py`)

---

## 5. Why Python Over C++

OpenVINO supports both Python and C++ APIs.  This project uses Python.

### Reasons

1. **Learning curve**: Python is accessible to most GSoC applicants.  The
   focus of this project is RAG system design and OpenVINO integration, not
   language mastery.

2. **Rapid iteration**: RAG pipelines involve many moving parts (OCR,
   embeddings, vector search, LLM prompting).  Python enables fast
   experimentation.

3. **Ecosystem**: sentence-transformers, FAISS, pytesseract, PaddleOCR,
   HuggingFace datasets, and Ollama all have first-class Python APIs.
   Using C++ for any of these would require reimplementation or FFI wrapping.

4. **OpenVINO GenAI compatibility**: `openvino-genai` provides the
   `LLMPipeline` class for turnkey LLM inference in Python.  The C++ GenAI
   API exists but is less documented.

5. **Prototyping vs production**: This is a learning project.  Correctness and
   clarity outweigh raw performance.  Once the pipeline is validated in
   Python, performance-critical paths can be profiled and selectively
   rewritten.

### When C++ would make sense

- Embedding a search engine into a native desktop application (Qt, WinUI)
- Deploying on resource-constrained edge devices
- Eliminating Python interpreter overhead in a latency-critical service

---

## 6. Model Stack

### Embedding Model: all-MiniLM-L6-v2

| Property | Value |
|----------|-------|
| Source | `sentence-transformers/all-MiniLM-L6-v2` |
| Dimension | 384 |
| Max tokens | 256 word-pieces |
| Size | ~80 MB |
| License | Apache-2.0 |
| Runtime | sentence-transformers (current), OpenVINO IR (future) |

**Why this model**: Small, fast, stable on CPU, well-supported by
sentence-transformers, and converts cleanly to ONNX and OpenVINO IR.
Good quality for English document retrieval.

### OCR: Tesseract (baseline)

| Property | Value |
|----------|-------|
| Source | System package (`apt install tesseract-ocr`) + `pytesseract` |
| Purpose | Extract text from scanned document images |
| GPU required | No |
| Future upgrade | PaddleOCR with OpenVINO backend |

**Why Tesseract**: Universal availability, zero GPU dependency, sufficient for
baseline text extraction.  PaddleOCR (`src/ocr/paddle_engine.py`) is
scaffolded as a future upgrade with higher accuracy and OpenVINO
acceleration.

### Vector Database: FAISS (CPU)

| Property | Value |
|----------|-------|
| Package | `faiss-cpu` |
| Index type | `IndexFlatIP` (inner product on L2-normalized vectors = cosine similarity) |
| Scalability | Millions of vectors on CPU |

**Why FAISS**: Industry standard for dense vector retrieval.  CPU-only build
avoids CUDA version conflicts.  Flat index provides exact search (no
approximation errors) at this dataset scale.

### LLM: Mistral 7B Instruct via Ollama

| Property | Value |
|----------|-------|
| Source | `ollama pull mistral` |
| Parameters | 7B |
| Quantization | Q4_0 (Ollama default) |
| VRAM | ~4.2 GB (GPU) or runs on CPU with ~8 GB RAM |
| License | Apache-2.0 |
| Runtime | Ollama HTTP API (current), OpenVINO GenAI (future) |

**Why Mistral 7B**: Open license, instruct-tuned for Q&A, runs on consumer
hardware.  Ollama handles model management and quantization.

**Critical design rule**: The LLM is used ONLY for answer generation.
Retrieval is entirely embedding-based.  The LLM never participates in the
search process.

### Multimodal (Future): CLIP ViT-B/32

| Property | Value |
|----------|-------|
| Source | `openai/clip-vit-base-patch32` |
| Purpose | Text-image alignment for cross-modal retrieval |
| Status | Placeholder only (`configs/models.yaml`) |

---

## 7. Directory Structure

```
Deep-Search-AI-assistant/
|
+-- cli.py                          # Main entry point (argparse CLI)
+-- requirements.txt                # Python dependencies
+-- README.md                       # This file
+-- .gitignore                      # Git ignore rules
|
+-- configs/
|   +-- settings.yaml               # Pipeline settings (paths, params)
|   +-- models.yaml                 # Model registry and metadata
|
+-- data/
|   +-- processed/                  # All pipeline outputs (gitignored)
|       +-- .gitkeep
|       +-- ocr_cache/              # Extracted PNGs from Arrow images
|       +-- normalised/             # documents.jsonl (unified schema)
|       +-- chunks/                 # (future) serialized chunks
|       +-- embeddings/             # chunk_embeddings.npy
|       +-- faiss/                  # index.faiss + metadata.json
|
+-- src/
|   +-- __init__.py                 # Package root
|   |
|   +-- ingestion/
|   |   +-- __init__.py
|   |   +-- loader.py               # Load Arrow datasets -> RawDocument
|   |   +-- normalizer.py           # RawDocument -> NormalizedDocument
|   |   +-- chunker.py              # NormalizedDocument -> TextChunk
|   |
|   +-- ocr/
|   |   +-- __init__.py
|   |   +-- tesseract_engine.py     # Tesseract OCR (baseline, working)
|   |   +-- paddle_engine.py        # PaddleOCR (placeholder, future)
|   |
|   +-- embeddings/
|   |   +-- __init__.py
|   |   +-- encoder.py              # sentence-transformers encoder (working)
|   |   +-- openvino_encoder.py     # OpenVINO IR encoder (placeholder)
|   |
|   +-- index/
|   |   +-- __init__.py
|   |   +-- faiss_index.py          # FAISS build/search/save/load (working)
|   |
|   +-- retrieval/
|   |   +-- __init__.py
|   |   +-- retriever.py            # Query encode -> search -> results (working)
|   |
|   +-- llm/
|   |   +-- __init__.py
|   |   +-- ollama_client.py        # Ollama Mistral client (working)
|   |   +-- openvino_llm.py         # OpenVINO GenAI LLM (placeholder)
|   |
|   +-- openvino/
|       +-- __init__.py
|       +-- device_manager.py       # OpenVINO device detection (working)
|       +-- model_converter.py      # ONNX -> IR conversion (partial)
|
+-- scripts/
|   +-- download_models.py          # Download embedding model + pull Ollama
|   +-- verify_setup.py             # Verify all dependencies are installed
|
+-- logs/
    +-- .gitkeep
```

### File-by-file purpose

| File | Lines | Status | Purpose |
|------|-------|--------|---------|
| `cli.py` | ~478 | Working | All CLI commands, orchestrates pipeline |
| `src/ingestion/loader.py` | ~404 | Working | Arrow dataset loading, image extraction |
| `src/ingestion/normalizer.py` | ~185 | Working | Unified document schema |
| `src/ingestion/chunker.py` | ~155 | Working | Fixed-size character chunking |
| `src/ocr/tesseract_engine.py` | ~221 | Working | Tesseract text extraction |
| `src/ocr/paddle_engine.py` | ~96 | Placeholder | PaddleOCR stub |
| `src/embeddings/encoder.py` | ~155 | Working | sentence-transformers encoding |
| `src/embeddings/openvino_encoder.py` | ~148 | Placeholder | OpenVINO embedding stub |
| `src/index/faiss_index.py` | ~212 | Working | FAISS index management |
| `src/retrieval/retriever.py` | ~162 | Working | Query -> retrieve -> format context |
| `src/llm/ollama_client.py` | ~216 | Working | Ollama RAG generation |
| `src/llm/openvino_llm.py` | ~113 | Placeholder | OpenVINO GenAI stub |
| `src/openvino/device_manager.py` | ~117 | Working | Device detection and selection |
| `src/openvino/model_converter.py` | ~145 | Partial | ONNX to IR conversion |
| `scripts/verify_setup.py` | ~155 | Working | Dependency verification |
| `scripts/download_models.py` | ~116 | Working | Model download helper |

---

## 8. Setup Guide

### 8.1 Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| OS | Windows 10/11 + WSL2 | Ubuntu 22.04 recommended inside WSL |
| Python | 3.9 - 3.11 | 3.12+ may have library compatibility issues |
| CUDA | 11.5 | Only if using NVIDIA GPU (RTX 2050); not needed for CPU-first path |
| RAM | 16 GB minimum | Ollama with Mistral 7B uses ~8 GB |
| Disk | 10 GB free | Models + datasets + processed artifacts |

### 8.2 WSL Setup Assumptions

This project lives on the native WSL filesystem at `~/projects/Deep-Search-AI-assistant`
for performance and symlink compatibility.  The raw data remains on the Windows
filesystem at `/mnt/d/Openvino-project/data/raw` (accessible as
`D:\Openvino-project\data\raw` from Windows).

The code tries the WSL path first, then falls back to the Windows path
(see `cli.py` lines 44-47).

### 8.3 Python Virtual Environment

```bash
# Inside WSL
cd ~/projects/Deep-Search-AI-assistant

# Create a virtual environment (do NOT commit this directory)
python3 -m venv .venv

# Activate
source .venv/bin/activate    # Linux/WSL
# .venv\Scripts\activate     # Windows PowerShell

# Verify Python version
python --version   # Should print 3.9.x, 3.10.x, or 3.11.x
```

The `.venv/` directory is in `.gitignore` and must never be committed.

### 8.4 Installing Dependencies

```bash
# With the virtual environment activated:
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- `numpy`, `Pillow`, `opencv-python`, `tqdm`, `PyYAML` -- core utilities
- `datasets` -- HuggingFace Arrow format loading
- `pytesseract` -- Python binding for Tesseract OCR
- `sentence-transformers` -- embedding model (pulls PyTorch as a dependency)
- `faiss-cpu` -- vector similarity search

### 8.5 Tesseract OCR Binary

`pytesseract` is a Python wrapper.  It requires the Tesseract binary:

```bash
# Ubuntu/WSL
sudo apt update
sudo apt install tesseract-ocr

# Verify
tesseract --version
```

Without the binary, `pytesseract` will raise an error at runtime.

### 8.6 CUDA 11.5 Caveats

**This project defaults to CPU execution.**  CUDA is not required for the
baseline pipeline.  However, if you plan to use the NVIDIA RTX 2050:

- CUDA 11.5 is an older toolkit version.  Many libraries ship wheels built
  against CUDA 11.8 or 12.x.
- PyTorch installed via `pip install sentence-transformers` may pull a CUDA
  12.x build that is incompatible with your driver.
- `requirements.txt` intentionally does NOT pin CUDA-specific versions.
  If you want GPU acceleration for sentence-transformers:

```bash
# Check your driver's maximum supported CUDA version:
nvidia-smi

# Install the matching PyTorch build BEFORE requirements.txt:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# (cu118 is the closest available to 11.5 and usually works)

# Then install the rest:
pip install -r requirements.txt
```

- `faiss-cpu` is used deliberately.  `faiss-gpu` requires CUDA headers at
  install time and provides no benefit for the index sizes in this project.

**Recommendation**: Start with CPU.  Add GPU only after the full pipeline works.

### 8.7 NVIDIA vs OpenVINO Runtime

| Scenario | Runtime | Notes |
|----------|---------|-------|
| CPU-first (baseline) | PyTorch CPU + Ollama | Default, no GPU needed |
| NVIDIA GPU embedding | PyTorch CUDA | Set `device: "cuda"` in `settings.yaml` |
| Intel iGPU embedding | OpenVINO IR on GPU | Requires IR conversion + `openvino` |
| Intel NPU | OpenVINO IR on NPU | Requires Meteor Lake+ hardware + drivers |
| LLM on CPU (no Ollama) | OpenVINO GenAI | Future: eliminates Ollama dependency |

The NVIDIA RTX 2050 on this machine can be used for PyTorch-based embedding
but is NOT used for OpenVINO (OpenVINO targets Intel hardware).  The two
runtimes serve different purposes and can coexist.

### 8.8 Ollama Installation

Ollama serves the Mistral 7B LLM locally.  It is needed only for the `ask`
command (RAG answer generation).

```bash
# Install Ollama (Linux/WSL)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the model (~4 GB download)
ollama pull mistral

# Start the server (runs in background)
ollama serve
```

Verify:

```bash
curl http://localhost:11434/api/tags
# Should list "mistral:latest" in the models array
```

If not using WSL, download the Windows installer from https://ollama.ai/download.

### 8.9 Verifying the Setup

Run the verification script:

```bash
python scripts/verify_setup.py
```

This checks:
- Python version (>= 3.9)
- All required Python packages
- Tesseract binary on PATH
- Ollama server reachability
- Raw data directory existence

### 8.10 Download Models

Pre-download models so the first `ingest` run does not stall:

```bash
python scripts/download_models.py
# Skipping Ollama (if already pulled):
python scripts/download_models.py --skip-ollama
```

### 8.11 Common Setup Failures and Fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: pytesseract` | Package not installed | `pip install pytesseract` |
| `TesseractNotFoundError` | Binary not installed | `sudo apt install tesseract-ocr` |
| `CUDA error: no kernel image` | PyTorch CUDA version mismatch | Install matching PyTorch version (see 8.6) |
| `ValueError: cannot reshape array` | Corrupted embeddings file | Delete `data/processed/embeddings/` and re-run ingest |
| `FileNotFoundError: data/raw/funsd` | Wrong working directory | Run from `Deep-Search-AI-assistant/` root |
| `Connection refused: localhost:11434` | Ollama not running | `ollama serve` in a separate terminal |
| `Model 'mistral' not found` | Model not pulled | `ollama pull mistral` |
| `MemoryError during embedding` | Too many texts at once | Use `--max-records 20` to limit dataset size |
| `ImportError: openvino` | OpenVINO not installed (expected) | This is optional -- install when ready for Phase 9 |
| `numpy version conflict` | sentence-transformers pulls incompatible numpy | `pip install numpy>=1.24,<2.0` |

---

## 9. CLI Reference

All commands are run from the project root directory.

### ingest -- Build the knowledge base

```bash
python cli.py ingest --dataset funsd
python cli.py ingest --dataset docvqa --max-records 20
python cli.py ingest --dataset rvl_cdip
```

Runs the full pipeline: load -> OCR -> normalize -> chunk -> embed -> index.
Each stage writes output to `data/processed/`.

- `--dataset` (required): One of `funsd`, `docvqa`, `rvl_cdip`
- `--max-records` (optional): Limit number of records (default: 0 = all)
- `-v` / `--verbose`: Enable debug logging

### search -- Semantic search (no LLM)

```bash
python cli.py search "Find the invoice number"
python cli.py search "employee name" --top-k 3
```

Encodes the query, searches FAISS, displays results.  No LLM is involved.

- Positional arg: the search query
- `--top-k` (optional): Number of results (default: 5)

### ask -- RAG question answering

```bash
python cli.py ask "What is the total amount on the invoice?"
python cli.py ask "Who signed this form?" --top-k 3
```

Retrieves context via semantic search, then generates an answer using the LLM.
Requires Ollama to be running with Mistral pulled.

### stats -- Pipeline statistics

```bash
python cli.py stats
```

Shows: raw datasets available, normalized document count, embedding shape,
FAISS index size, Ollama availability.

### devices -- OpenVINO device listing

```bash
python cli.py devices
```

Lists OpenVINO-compatible hardware on the machine.  Requires `openvino`
package to be installed.

---

## 10. Step-by-Step Learning Roadmap

Each phase below is a self-contained learning step.  Complete them in order.
Each phase builds on the previous one.

---

### Phase 1: Dataset Normalization

**Goal**: Understand how heterogeneous raw datasets are loaded and unified
into a common document representation.

**Files to read and modify**:
- `src/ingestion/loader.py` -- Loading logic and per-dataset handlers
- `src/ingestion/normalizer.py` -- Unified `NormalizedDocument` schema
- `configs/settings.yaml` -- Dataset registry

**What to implement**:
- Read through `loader.py` and understand how `DatasetLoader.load_dataset()`
  dispatches to per-dataset handlers (`_handle_funsd_record`, etc.)
- Trace the data flow from Arrow file to `RawDocument`
- Run the loader on each dataset and examine the output:

```bash
python cli.py ingest --dataset funsd --max-records 5
```

**Expected output**:
- Console shows "Loaded 5 documents"
- Images appear in `data/processed/ocr_cache/funsd/`
- `data/processed/normalised/documents.jsonl` contains 5 lines

**How to verify correctness**:

```python
# In a Python REPL:
import json
with open("data/processed/normalised/documents.jsonl") as f:
    docs = [json.loads(line) for line in f]
print(len(docs))              # Should be 5
print(docs[0].keys())         # Should have: doc_id, source, doc_type, text, image_path, metadata
print(docs[0]["doc_type"])    # Should be "form" for FUNSD
print(len(docs[0]["text"]))   # Should be > 0 (FUNSD has word annotations)
```

**Checkpoint**:
- [x] `python cli.py ingest --dataset funsd --max-records 5` runs without error
- [x] `data/processed/normalised/documents.jsonl` has 5 non-empty lines
- [x] Each line has `doc_id`, `source`, `doc_type`, `text`, `metadata`

**Common mistakes**:
- Running from the wrong directory (the raw data path will not resolve)
- Forgetting to install the `datasets` package
- Not understanding that FUNSD has text from the `words` column while DocVQA
  and RVL-CDIP have empty text (image-only, needing OCR)

---

### Phase 2: OCR Extraction

**Goal**: Extract text from document images using Tesseract OCR.

**Files to read and modify**:
- `src/ocr/tesseract_engine.py` -- Main OCR engine
- `cli.py` lines 110-125 -- Where OCR is invoked in the ingest pipeline

**What to implement**:
- Read through `TesseractEngine.__init__()` and `extract_text()`
- Understand the preprocessing pipeline: grayscale -> Gaussian blur -> adaptive threshold
- Run OCR on DocVQA (which has no pre-extracted text):

```bash
python cli.py ingest --dataset docvqa --max-records 5
```

**Expected output**:
- Console shows "OCR extracted text from N images" where N > 0
- The normalized documents.jsonl now has non-empty text fields for image-heavy records

**How to verify correctness**:

```python
import json
with open("data/processed/normalised/documents.jsonl") as f:
    docs = [json.loads(line) for line in f]
non_empty = [d for d in docs if len(d["text"]) > 0]
print(f"Documents with text: {len(non_empty)} / {len(docs)}")
```

For DocVQA, all 5 documents should have text after OCR.

**Checkpoint**:
- [x] DocVQA ingest extracts text from document images
- [x] OCR output is visible in the normalized documents
- [x] `extract_with_boxes()` returns word-level bounding boxes

**Common mistakes**:
- Tesseract binary not installed (`sudo apt install tesseract-ocr`)
- Image path does not exist (check that `ocr_cache/` was populated by the loader)
- Low confidence threshold causing too much noise; high threshold causing empty output
- Preprocessing making things worse on already-clean images (try `preprocess=False`)

---

### Phase 3: Unified Document Schema

**Goal**: Understand why normalization exists and how it decouples
downstream code from dataset-specific logic.

**Files to read and modify**:
- `src/ingestion/normalizer.py` -- The normalizer and `NormalizedDocument`

**What to implement**:
- Read `_classify_type()` and understand how it maps datasets to document types
- Add a deduplication check: if a `doc_id` already exists in the output file,
  skip that document instead of overwriting
- Add a `language` field to metadata (use `langdetect` or hardcode "en" for now)

**Expected output**:
- `documents.jsonl` has one entry per unique document
- Each entry has a `doc_type` field matching the dataset

**How to verify correctness**:

```python
import json
with open("data/processed/normalised/documents.jsonl") as f:
    docs = [json.loads(line) for line in f]
doc_ids = [d["doc_id"] for d in docs]
print(f"Unique IDs: {len(set(doc_ids))} / {len(doc_ids)}")  # Should be equal
print(set(d["doc_type"] for d in docs))  # Shows the distinct types
```

**Checkpoint**:
- [x] Every document has a non-empty `doc_id` (SHA-256 derived)
- [x] `doc_type` is correctly assigned for each dataset
- [x] No duplicate `doc_id` values in the output

**Common mistakes**:
- Modifying `RawDocument` instead of `NormalizedDocument` (wrong layer)
- Creating non-deterministic IDs (use the doc_key, not random UUIDs)

---

### Phase 4: Chunking Strategy

**Goal**: Split document text into overlapping windows suitable for embedding.

**Files to read and modify**:
- `src/ingestion/chunker.py` -- `TextChunker` and `TextChunk`
- `configs/settings.yaml` -- `chunk_size` and `chunk_overlap` parameters

**What to implement**:
- Read through `chunk_text()` and trace the sliding window logic
- Verify that chunks overlap correctly (chunk N's tail = chunk N+1's head)
- Experiment with different chunk sizes (256, 512, 1024) and observe the
  effect on chunk count

```bash
python cli.py ingest --dataset funsd --max-records 10
# Observe "Produced N chunks from M documents"
```

**Expected output**:
- Each document with text > 512 chars produces multiple chunks
- Short documents produce exactly 1 chunk
- Documents with empty text produce 0 chunks

**How to verify correctness**:

```python
from src.ingestion.chunker import TextChunker

chunker = TextChunker(chunk_size=512, overlap=64)
test_text = "A" * 1000  # 1000 characters
chunks = chunker.chunk_text(test_text, doc_id="test")
print(f"Chunks: {len(chunks)}")  # Should be 3
# chunk 0: chars 0-512
# chunk 1: chars 448-960
# chunk 2: chars 896-1000
for c in chunks:
    print(f"  {c.chunk_id}: len={len(c.text)}, start={c.metadata['char_start']}")
```

**Checkpoint**:
- [x] Overlap verification: last 64 chars of chunk N == first 64 chars of chunk N+1
- [x] Short text (< 512 chars) produces exactly 1 chunk
- [x] Empty text produces 0 chunks
- [x] `chunk_id` format is `{doc_id}_chunk_{index:04d}`

**Common mistakes**:
- Off-by-one in the sliding window (verify overlap boundary manually)
- Setting overlap >= chunk_size (the constructor raises ValueError)
- Not filtering out very short chunks (min_chunk_length in settings.yaml)

---

### Phase 5: Embedding Generation

**Goal**: Convert text chunks into dense 384-dimensional vectors.

**Files to read and modify**:
- `src/embeddings/encoder.py` -- `EmbeddingEncoder` wrapping sentence-transformers

**What to implement**:
- Read through `EmbeddingEncoder.__init__()` -- note the test embedding on init
- Run encoding on sample texts and inspect shapes:

```python
from src.embeddings.encoder import EmbeddingEncoder
import numpy as np

encoder = EmbeddingEncoder(device="cpu")
vectors = encoder.encode(["Hello world", "Invoice total: $500"])
print(vectors.shape)     # (2, 384)
print(vectors.dtype)     # float32

# Verify L2 normalization
norms = np.linalg.norm(vectors, axis=1)
print(norms)             # Should be [1.0, 1.0] (approximately)

# Compute cosine similarity (= dot product for normalized vectors)
sim = np.dot(vectors[0], vectors[1])
print(f"Similarity: {sim:.4f}")
```

**Expected output**:
- `chunk_embeddings.npy` appears in `data/processed/embeddings/`
- Shape is `(N_chunks, 384)` with dtype `float32`
- All vectors are L2-normalized (norm ~1.0)

**How to verify correctness**:

```bash
python cli.py ingest --dataset funsd --max-records 10
python cli.py stats
# Stats should show embedding shape and file size
```

```python
import numpy as np
emb = np.load("data/processed/embeddings/chunk_embeddings.npy")
norms = np.linalg.norm(emb, axis=1)
print(f"Shape: {emb.shape}")
print(f"Norm range: [{norms.min():.4f}, {norms.max():.4f}]")  # Should be ~1.0
```

**Checkpoint**:
- [x] Embeddings file exists and has correct shape
- [x] All vectors are approximately unit-normalized
- [x] Similar texts have higher cosine similarity than unrelated texts

**Common mistakes**:
- Forgetting to normalize (FAISS inner product assumes normalized vectors)
- Using `device="cuda"` without a compatible PyTorch CUDA build
- Running out of memory on large datasets (reduce `--max-records` or `batch_size`)

---

### Phase 6: FAISS Indexing

**Goal**: Build a FAISS vector index for similarity search.

**Files to read and modify**:
- `src/index/faiss_index.py` -- `FaissIndex` class

**What to implement**:
- Read through `build()` -- understand `IndexFlatIP` and why it works with
  L2-normalized vectors
- Read through `search()` -- understand the score/index arrays returned by FAISS
- Understand the metadata sidecar pattern: FAISS stores only vectors, metadata
  is stored in `metadata.json` and looked up by integer position

**Expected output after ingest**:
- `data/processed/faiss/index.faiss` -- the FAISS binary index
- `data/processed/faiss/metadata.json` -- parallel metadata array

**How to verify correctness**:

```python
import faiss
import json

index = faiss.read_index("data/processed/faiss/index.faiss")
print(f"Vectors: {index.ntotal}")
print(f"Dimension: {index.d}")  # Should be 384

with open("data/processed/faiss/metadata.json") as f:
    meta = json.load(f)
print(f"Metadata entries: {len(meta)}")  # Should equal index.ntotal
print(meta[0].keys())  # Should include chunk_id, doc_id, text, etc.
```

**Checkpoint**:
- [x] Index vector count matches metadata entry count
- [x] Dimension is 384
- [x] Load-save roundtrip preserves the index

**Common mistakes**:
- Dimension mismatch between embeddings and index (check `embeddings.shape[1]`)
- Forgetting that `IndexFlatIP` expects float32 arrays
- Not understanding that FAISS `search()` returns -1 for missing results

---

### Phase 7: Query Flow

**Goal**: Execute semantic search queries end-to-end.

**Files to read and modify**:
- `src/retrieval/retriever.py` -- `Retriever` class

**What to implement**:
- Read through `query()` -- trace the encode -> search -> wrap flow
- Read through `format_context()` -- understand how results become an LLM prompt

```bash
# First, build the index:
python cli.py ingest --dataset funsd

# Then search:
python cli.py search "invoice number"
python cli.py search "employee name" --top-k 3
```

**Expected output**:
- Ranked list of results with scores and text previews
- Higher scores indicate more semantically similar chunks
- Related queries return overlapping but re-ranked results

**How to verify correctness**:
- Run the same query twice -- results should be identical (deterministic)
- Run a very specific query and check that the top result is relevant
- Run a nonsensical query ("asdfghjkl") -- scores should be low

**Checkpoint**:
- [x] `python cli.py search "test"` returns results with scores
- [x] Results include `chunk_id`, `doc_id`, score, and text preview
- [x] Identical queries produce identical results

**Common mistakes**:
- Searching before running ingest (index does not exist yet)
- Not understanding that search is embedding-based (no keyword matching)
- Expecting BM25-style exact keyword match behavior from dense retrieval

---

### Phase 8: RAG Prompt Construction

**Goal**: Connect retrieval to LLM generation for question answering.

**Files to read and modify**:
- `src/llm/ollama_client.py` -- `OllamaClient` and RAG prompt templates
- `cli.py` lines 219-268 -- `cmd_ask()` handler

**What to implement**:
- Read the `RAG_SYSTEM_PROMPT` and `RAG_USER_TEMPLATE` templates
- Understand how `format_context()` output is injected into the system prompt
- Run the full RAG pipeline:

```bash
# Ensure Ollama is running and Mistral is pulled:
ollama serve &
ollama pull mistral

# Ask a question:
python cli.py ask "What types of forms are in the collection?"
```

**Expected output**:
- Retrieved context chunks are displayed (or logged)
- LLM generates an answer grounded in the retrieved context
- If Ollama is not running, a helpful error message is shown

**How to verify correctness**:
- The answer should reference information from the actual documents
- The answer should NOT contain information not in the retrieved context
- Try asking about something not in the knowledge base -- the LLM should
  say it does not have enough information

**Checkpoint**:
- [x] `python cli.py ask "test question"` runs without error (if Ollama is up)
- [x] Answer references retrieved document content
- [x] System prompt includes the RAG context block

**Common mistakes**:
- Ollama not running (`ollama serve` must be active)
- Model not pulled (`ollama pull mistral`)
- Context too long for the LLM's context window (max_chars in format_context)
- Expecting the LLM to "know" things not in the retrieved context

---

### Phase 9: OpenVINO Acceleration

**Goal**: Replace PyTorch-based embedding inference with OpenVINO for
optimized execution on Intel hardware.

**Files to read and modify**:
- `src/openvino/model_converter.py` -- ONNX to IR conversion
- `src/embeddings/openvino_encoder.py` -- `OVEmbeddingEncoder`
- `src/openvino/device_manager.py` -- Device selection

**What to implement**:

Step 1: Install OpenVINO

```bash
pip install openvino openvino-dev
python cli.py devices   # Should list at least "CPU"
```

Step 2: Export embedding model to ONNX

```bash
pip install optimum-intel[openvino]
optimum-cli export onnx \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    models/onnx/all-MiniLM-L6-v2/
```

Step 3: Convert ONNX to OpenVINO IR

```python
from src.openvino.model_converter import convert_onnx_to_ir
convert_onnx_to_ir(
    onnx_path="models/onnx/all-MiniLM-L6-v2/model.onnx",
    output_dir="models/ov/all-MiniLM-L6-v2",
)
# Produces: models/ov/all-MiniLM-L6-v2/model.xml + model.bin
```

Step 4: Implement `OVEmbeddingEncoder.encode()`

The student must:
1. Load the tokenizer (`transformers.AutoTokenizer`)
2. Tokenize input texts
3. Run inference through `self._compiled_model`
4. Apply mean pooling over token embeddings
5. L2-normalize the output

Step 5: Verify output matches PyTorch

```python
from src.embeddings.encoder import EmbeddingEncoder
from src.embeddings.openvino_encoder import OVEmbeddingEncoder
import numpy as np

pt_enc = EmbeddingEncoder(device="cpu")
ov_enc = OVEmbeddingEncoder(model_xml="models/ov/all-MiniLM-L6-v2/model.xml")

texts = ["Hello world", "Invoice total"]
pt_emb = pt_enc.encode(texts)
ov_emb = ov_enc.encode(texts)

diff = np.abs(pt_emb - ov_emb).max()
print(f"Max absolute difference: {diff}")  # Should be < 0.001
```

**Checkpoint**:
- [x] `python cli.py devices` lists at least CPU
- [x] ONNX export produces a valid model file
- [x] IR conversion produces .xml + .bin files
- [x] OVEmbeddingEncoder produces vectors within 0.001 of PyTorch output

**Common mistakes**:
- Not installing `optimum-intel` for the ONNX export
- Wrong input/output tensor names in the IR (inspect with `mo --help`)
- Forgetting mean pooling (OpenVINO returns raw token embeddings, not sentence embeddings)
- Forgetting L2 normalization after pooling

---

### Phase 10: Hardware-Aware Execution

**Goal**: Use the Device Manager to dynamically select the best available
device for inference.

**Files to read and modify**:
- `src/openvino/device_manager.py` -- `DeviceManager`
- `configs/settings.yaml` -- `openvino.device` setting

**What to implement**:
- Use `DeviceManager.select(preferred="GPU")` to try iGPU, falling back to CPU
- Modify the pipeline to read the device preference from `settings.yaml`
- Log which device is selected for each inference call
- Experiment with `AUTO` and `MULTI` device plugins (if your machine has
  multiple devices)

**Expected output**:
- `python cli.py devices` shows all detected hardware
- Embedding inference runs on the selected device without code changes
- Falling back to CPU when preferred device is unavailable

**How to verify correctness**:
- Run with `device: "GPU"` and check that it either uses the GPU or
  falls back gracefully
- Compare inference times between CPU and GPU for 100 embeddings
- Verify that outputs are numerically equivalent regardless of device

**Checkpoint**:
- [x] Device selection respects settings.yaml preference
- [x] Graceful fallback when preferred device is unavailable
- [x] Results are device-independent (same embeddings on CPU vs GPU)

**Common mistakes**:
- GPU may not be available without Intel GPU drivers (check `vainfo` on Linux)
- NPU requires specific driver versions on Windows/Linux
- `AUTO` plugin may not be available in older OpenVINO versions

---

## 11. Data Flow Rules

These rules are strict and must be observed throughout development.

### Rule 1: Raw data is read-only

```
Raw data path: /mnt/d/Openvino-project/data/raw
```

The pipeline reads from this path but **never writes to it**.  Raw data is
immutable.

### Rule 2: All outputs go to processed/

```
Processed data path: Deep-Search-AI-assistant/data/processed/
```

Every pipeline artifact -- extracted images, normalized documents, embeddings,
FAISS index -- lives under `data/processed/`.

### Rule 3: Processed data is regenerable

Everything in `data/processed/` can be deleted and recreated by running
`python cli.py ingest`.  This directory is gitignored (except for `.gitkeep`).

### Rule 4: Retrieval never touches raw data

The retrieval pipeline (search/ask commands) reads only from:
- `data/processed/faiss/` (the index)
- `data/processed/embeddings/` (if needed for reloading)

It never accesses `data/raw/` directly.

### Rule 5: One-way data flow

```
Raw data -> Loader -> OCR -> Normalizer -> Chunker -> Encoder -> Index
                                                                  |
                                                                  v
                                                            Query / RAG
```

Data flows forward only.  No stage writes back to a previous stage's output.

---

## 12. OpenVINO Integration Roadmap

This section maps the placeholder files to their intended implementation and
marks the current status.

| Component | File | Status | What to Do |
|-----------|------|--------|------------|
| Device detection | `src/openvino/device_manager.py` | Working | Functional as-is |
| Model conversion | `src/openvino/model_converter.py` | Partial | Test end-to-end with all-MiniLM-L6-v2 |
| OV embedding encoder | `src/embeddings/openvino_encoder.py` | Placeholder | Implement tokenizer + inference + pooling |
| OV LLM client | `src/llm/openvino_llm.py` | Placeholder | Implement via `openvino-genai.LLMPipeline` |
| PaddleOCR + OV | `src/ocr/paddle_engine.py` | Placeholder | Install PaddleOCR, convert models to IR |

### Conversion commands reference

```bash
# Embedding model: ONNX export
optimum-cli export onnx \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    models/onnx/all-MiniLM-L6-v2/

# Embedding model: ONNX -> OpenVINO IR
mo --input_model models/onnx/all-MiniLM-L6-v2/model.onnx \
   --output_dir models/ov/all-MiniLM-L6-v2/

# LLM: Mistral 7B -> OpenVINO IR (INT4)
optimum-cli export openvino \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --weight-format int4 \
    models/ov/mistral-7b-instruct/
```

---

## 13. Troubleshooting

### The CLI crashes immediately

Verify you are running from the project root:

```bash
cd ~/projects/Deep-Search-AI-assistant
python cli.py --help
```

### Ingestion produces zero chunks

This means no documents had extractable text.  For image-only datasets
(DocVQA, RVL-CDIP), OCR must succeed.  Check:

1. Is Tesseract installed? (`tesseract --version`)
2. Are images being extracted? (check `data/processed/ocr_cache/`)
3. Is the confidence threshold too high? (lower `confidence_threshold` in
   `settings.yaml`)

### Search returns no results

1. Have you run `ingest` first?
2. Check `python cli.py stats` -- are there indexed vectors?
3. Is the FAISS index file present? (`data/processed/faiss/index.faiss`)

### Ollama is not available for `ask` command

1. Is Ollama installed? (`which ollama`)
2. Is the server running? (`ollama serve` in another terminal)
3. Is the model pulled? (`ollama list` should show `mistral`)
4. Is the port correct? (default: 11434, check `settings.yaml`)

### OpenVINO not found

This is expected in Phase 1-8.  Install when ready:

```bash
pip install openvino openvino-dev
```

### WSL path issues

The code handles both Windows (`D:\...`) and WSL (`/mnt/d/...`) paths.  If
you encounter path errors, check:

1. Which environment you are running in (PowerShell vs WSL bash)
2. That the raw data directory exists at the expected path
3. That `cli.py` lines 44-47 correctly resolve the path for your environment

---

## License

This project is for educational and research purposes, aligned with the
OpenVINO GSoC program.  Individual model licenses are documented in
`configs/models.yaml`.
