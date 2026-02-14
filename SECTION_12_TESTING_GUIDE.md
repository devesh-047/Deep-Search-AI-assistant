# Section 12: OpenVINO Integration — Complete Testing Guide

## ✅ What's Implemented

All 5 components from the OpenVINO Integration Roadmap are now **fully implemented**:

| Component | Status | File |
|-----------|--------|------|
| Device detection | ✅ Complete | `src/openvino/device_manager.py` |
| Model conversion | ✅ Complete | `src/openvino/model_converter.py` |
| OV embedding encoder | ✅ Complete | `src/embeddings/openvino_encoder.py` |
| **OV LLM client** | ✅ **Complete** | `src/llm/openvino_llm.py` |
| **PaddleOCR + OV** | ✅ **Complete** | `src/ocr/paddle_engine.py` |

---

## 🚀 Quick Test (No Installation Required)

These work **right now** with your existing setup:

```bash
# 1. Test device detection
python cli.py devices

# 2. Test OpenVINO embedding encoder (already enabled in settings.yaml)
python cli.py ingest --dataset docvqa --max-records 10
python cli.py search "invoice total"

# 3. Test progress bars (added to all slow operations)
# You'll see: OCR, Normalizing, Chunking, Encoding progress bars
```

---

## 📦 Part 4: OpenVINO LLM Client

### What It Does
Replaces Ollama with OpenVINO GenAI for local LLM inference. Benefits:
- ✅ No separate server process needed (unlike Ollama)
- ✅ INT4/INT8 quantization → lower memory usage
- ✅ Runs on Intel CPU, iGPU, or NPU
- ✅ Dual backend: `openvino-genai` (fast) or `optimum-intel` (fallback)

### Installation

```bash
# Option 1: openvino-genai (recommended, lightweight)
pip install openvino-genai

# Option 2: optimum-intel (heavier, more compatible)
pip install optimum-intel[openvino]
```

### Model Conversion

Convert Mistral 7B to OpenVINO IR format (INT4 quantization):

```bash
# This requires ~16 GB RAM and takes 5-10 minutes
optimum-cli export openvino \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --weight-format int4 \
    models/ov/mistral-7b-instruct/
```

**What this creates:**
```
models/ov/mistral-7b-instruct/
├── openvino_model.xml          # Model architecture
├── openvino_model.bin          # INT4 weights (~4 GB)
├── tokenizer_config.json       # Tokenizer settings
├── tokenizer.json              # Vocabulary
└── ...
```

### Configuration

Edit `configs/settings.yaml`:

```yaml
llm:
  provider: "openvino"  # Change from "ollama" to "openvino"

openvino:
  enabled: true
  device: "CPU"  # or "GPU" if you have Intel iGPU
  llm_model_dir: "models/ov/mistral-7b-instruct"  # Path to converted model
```

### Testing

```bash
# Test the ask command (RAG with LLM)
python cli.py ask "What is the total amount on the invoice?"
```

**Expected output:**
```
────────────────────────────────────────────────────────────
  Ask  →  "What is the total amount on the invoice?"
────────────────────────────────────────────────────────────

  [1/2] Retrieving relevant context
      ℹ Using OpenVINO encoder on CPU
      ✔ 5 chunks retrieved

  [2/2] Generating answer with LLM
      ℹ Using OpenVINO LLM from models/ov/mistral-7b-instruct on CPU
      ℹ Thinking...
      ✔ Answer generated

  Q: What is the total amount on the invoice?

  Based on the document, the total amount is $150.00.

  Based on 5 retrieved chunks.
```

### Benchmarking

Compare Ollama vs OpenVINO performance:

```bash
# Create a test script
cat > test_llm_benchmark.py << 'EOF'
from src.llm.ollama_client import OllamaClient
from src.llm.openvino_llm import OVLLMClient

# Test Ollama
ollama = OllamaClient()
if ollama.is_available():
    print("Ollama benchmark:")
    result = ollama.generate(
        question="What is machine learning?",
        max_tokens=100
    )
    print(f"  Generated {len(result)} chars")

# Test OpenVINO
ov_llm = OVLLMClient(model_dir="models/ov/mistral-7b-instruct", device="CPU")
if ov_llm.is_available():
    print("\nOpenVINO LLM benchmark:")
    stats = ov_llm.benchmark(
        prompt="What is machine learning?",
        max_tokens=100,
        n_runs=3
    )
    print(f"  Backend: {stats['backend']}")
    print(f"  Device: {stats['device']}")
    print(f"  Mean time: {stats['mean_time_s']:.2f}s")
    print(f"  Chars/sec: {stats['chars_per_sec']:.1f}")
EOF

python test_llm_benchmark.py
```

---

## 📦 Part 5: PaddleOCR Engine

### What It Does
Replaces Tesseract with PaddleOCR for better accuracy on complex layouts (forms, invoices). Benefits:
- ✅ Deep learning-based (vs Tesseract's traditional CV)
- ✅ Built-in layout analysis
- ✅ Better accuracy on Asian languages
- ✅ Optional OpenVINO backend for acceleration

### Installation

```bash
# Install PaddleOCR
pip install paddleocr paddlepaddle

# For OpenVINO backend (optional)
pip install paddle2onnx
```

### Configuration

Edit `configs/settings.yaml`:

```yaml
ocr:
  engine: "paddleocr"  # Change from "tesseract" to "paddleocr"
  paddleocr_lang: "en"
  paddleocr_use_openvino: false  # Set to true for OpenVINO acceleration
  confidence_threshold: 40
```

### Testing

```bash
# Ingest with PaddleOCR
python cli.py ingest --dataset docvqa --max-records 10
```

**Expected output:**
```
  [2/6] Running OCR on image documents
      ℹ Using PaddleOCR (lang=en, openvino=False)
PaddleOCR: 100%|████████████| 10/10 [00:25<00:00,  2.5s/img]
      ✔ Extracted text from 10 images
```

### Comparing Tesseract vs PaddleOCR

```bash
# Test both engines on the same dataset
cat > test_ocr_comparison.py << 'EOF'
from src.ocr.tesseract_engine import TesseractEngine
from src.ocr.paddle_engine import PaddleOCREngine, PADDLE_AVAILABLE
from pathlib import Path

# Get a test image
test_image = "data/processed/ocr_cache/docvqa/docvqa_00000.png"

if not Path(test_image).exists():
    print(f"Test image not found: {test_image}")
    print("Run: python cli.py ingest --dataset docvqa --max-records 1")
    exit(1)

# Tesseract
print("Tesseract OCR:")
tess = TesseractEngine(preprocess=True)
tess_text = tess.extract_text(test_image)
print(f"  Extracted {len(tess_text)} chars")
print(f"  Preview: {tess_text[:100]}...")

# PaddleOCR
if PADDLE_AVAILABLE:
    print("\nPaddleOCR:")
    paddle = PaddleOCREngine(lang="en")
    paddle_text = paddle.extract_text(test_image)
    print(f"  Extracted {len(paddle_text)} chars")
    print(f"  Preview: {paddle_text[:100]}...")
else:
    print("\nPaddleOCR not installed")
    print("Install with: pip install paddleocr paddlepaddle")
EOF

python test_ocr_comparison.py
```

---

## 🎯 Complete End-to-End Test

Test the entire pipeline with all OpenVINO components:

```bash
# 1. Configure everything
cat > configs/settings.yaml << 'EOF'
# ... (keep existing paths/datasets sections) ...

ocr:
  engine: "paddleocr"
  paddleocr_lang: "en"
  paddleocr_use_openvino: false

llm:
  provider: "openvino"

openvino:
  enabled: true
  device: "CPU"
  embedding_model_ir: "models/ov/all-MiniLM-L6-v2/model.xml"
  llm_model_dir: "models/ov/mistral-7b-instruct"
EOF

# 2. Ingest with PaddleOCR + OpenVINO embeddings
python cli.py ingest --dataset docvqa --max-records 10

# 3. Search with OpenVINO embeddings
python cli.py search "invoice total amount"

# 4. Ask with OpenVINO LLM
python cli.py ask "What is the total amount on the invoice?"

# 5. Check device info
python cli.py devices --benchmark
```

---

## 📊 Progress Bars Added

All slow operations now show `tqdm` progress bars:

| Operation | Progress Bar | File |
|-----------|-------------|------|
| OCR (Tesseract) | `OCR: 5/10 img [00:15<00:15, 3.0s/img]` | `src/ocr/tesseract_engine.py` |
| OCR (PaddleOCR) | `PaddleOCR: 5/10 img [00:12<00:12, 2.4s/img]` | `src/ocr/paddle_engine.py` |
| Normalizing | `Normalizing: 10/10 doc [00:00<00:00, 50doc/s]` | `src/ingestion/normalizer.py` |
| Chunking | `Chunking: 10/10 doc [00:00<00:00, 100doc/s]` | `src/ingestion/chunker.py` |
| Encoding | `Encoding: 5/15 batch [00:02<00:06, 1.5batch/s]` | `src/embeddings/openvino_encoder.py` |

---

## 🔧 Troubleshooting

### OpenVINO LLM not loading

**Problem:** `_warn("OpenVINO LLM model failed to load")`

**Solutions:**
1. Check model path exists: `ls models/ov/mistral-7b-instruct/`
2. Verify conversion completed: `ls -lh models/ov/mistral-7b-instruct/*.bin`
3. Check RAM: INT4 model needs ~6 GB RAM
4. Try fallback backend: `pip install optimum-intel[openvino]`

### PaddleOCR not working

**Problem:** `PaddleOCR not installed, falling back to Tesseract`

**Solutions:**
1. Install: `pip install paddleocr paddlepaddle`
2. Download models (automatic on first run, needs internet)
3. Check disk space: models are ~100 MB

### Progress bars not showing

**Problem:** No progress bars during OCR/encoding

**Solutions:**
1. Install tqdm: `pip install tqdm` (should already be in requirements.txt)
2. Check if running in non-interactive terminal
3. Verify you're using the updated code (run `git status`)

---

## 📝 Summary

**What to install for full OpenVINO integration:**

```bash
# Core (already installed)
pip install openvino openvino-dev optimum-intel

# LLM support (choose one)
pip install openvino-genai  # Recommended
# OR
pip install optimum-intel[openvino]  # Fallback

# OCR support (optional)
pip install paddleocr paddlepaddle

# Progress bars (already in requirements.txt)
pip install tqdm
```

**What to convert:**

```bash
# LLM model (required for OpenVINO LLM)
optimum-cli export openvino \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --weight-format int4 \
    models/ov/mistral-7b-instruct/
```

**What to test:**

```bash
# 1. Current setup (works now)
python cli.py devices
python cli.py ingest --dataset docvqa --max-records 10
python cli.py search "invoice"

# 2. After installing openvino-genai + converting model
python cli.py ask "What is the total amount?"

# 3. After installing paddleocr
# (Change settings.yaml: ocr.engine = "paddleocr")
python cli.py ingest --dataset docvqa --max-records 10
```
