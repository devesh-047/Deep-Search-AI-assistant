# Deep Search AI Assistant (GSoC 2024 OpenVINO)

An advanced multimodal RAG system designed for AI PCs, integrating local LLMs with OpenVINO optimization. This project ingests documents and videos, creating a unified knowledge base queryable via natural language.

## Project Overview

This application demonstrates a complete offline RAG pipeline optimized for Intel hardware. It moves beyond simple text search by incorporating:

1.  **Multimodal Ingestion**: Processes PDFs (OCR) and Videos (MSR-VTT dataset).
2.  **Semantic Search**: Uses vector embeddings to find relevant content regardless of exact keyword matches.
3.  **Local LLM Integration**: Uses OpenVINO-optimized models (Mistral/Llama3) for generation.
4.  **Hardware Acceleration**: Leverages Intel CPU/iGPU/NPU via OpenVINO Runtime.

## Architecture

The pipeline consists of modular stages:

1.  **Ingestion Layer**:
    *   **Documents**: PDF/Image ingestion using OCR (Tesseract/PaddleOCR).
    *   **Videos**: MSR-VTT dataset processing (Caption-based or Transcript-based).
2.  **Normalization Layer**: Converts all inputs into a unified `NormalizedDocument` schema.
3.  **Embedding Layer**: Generates vector representations using `all-MiniLM-L6-v2` (OpenVINO optimized).
4.  **Indexing Layer**: Stores vectors in a FAISS index for efficient similarity search.
5.  **Retrieval & Generation**:
    *   Retrieves relevant chunks based on query similarity.
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
| **Documents** | `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tiff` | OCR (Tesseract) + Text Normalization |
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

## Configuration

Settings are managed in `configs/settings.yaml`. Key sections:

*   `llm`: Select model provider (OpenVINO vs Ollama).
*   `vector_db`: FAISS index settings.
*   `video`: MSR-VTT dataset path and processing mode.
    *   `enable_whisper`: Set to `false` for caption mode (faster), `true` for transcripts.

## How to Run

### 1. Ingest Documents
Processes PDFs and images from `data/raw/docs`.
```bash
python3 cli.py ingest-documents
```

### 2. Ingest Videos
Processes MSR-VTT dataset.
```bash
# Default (MSR-VTT caption mode)
python3 cli.py ingest-videos
```

### 3. Build Index
Creates FAISS vector index from processed data.
```bash
python3 cli.py build-index
```

### 4. Search
Perform semantic search (retrieval only).
```bash
python3 cli.py search "cartoon character playing guitar"
```

### 5. Ask (RAG)
Generate answers using the LLM.
```bash
python3 cli.py ask "What is the total amount on this invoice?"
```

## OpenVINO Integration

This project explicitly leverages OpenVINO for:

1.  **Embeddings**: `sentence-transformers` models are exported to OpenVINO IR format for faster CPU inference.
2.  **LLM Inference**: Supports `optimum-intel` to run LLMs directly on Intel CPU/GPU/NPU without external servers.
3.  **OCR (Planned)**: PaddleOCR inference optimized via OpenVINO.

Device selection (`CPU`, `GPU`, `NPU`) is configurable in `settings.yaml`.

## AI Usage Disclosure

**Use of AI Assistance**:
AI tools were used during the development of this project for the following specific tasks:

1.  **Code Structure**: Generating boilerplate command-line interface (CLI) logic.
2.  **Documentation**: Assisting in structuring and formatting this README file.
3.  **Debugging**: Analyzing stack traces to identify missing dependencies.
4.  **Dataset Selection**: Suggesting appropriate public datasets (MSR-VTT) for multimodal testing.

The core architecture, logical implementation of the RAG pipeline, OpenVINO integration strategies, and final code verification were performed manually by the developer. All code has been reviewed and validated.
