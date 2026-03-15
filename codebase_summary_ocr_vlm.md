# Codebase Summary — OCR & VLM Additions

This document covers only the new OpenVINO PaddleOCR and CLIP multimodal
retrieval features added to the Deep Search AI Assistant.

## OCR Additions

### PaddleOCR OpenVINO Integration

`PaddleOCREngine` in `src/ocr/paddle_engine.py` now uses `DeviceManager`
to validate the OpenVINO device string and fall back gracefully:

```
use_openvino=True → import openvino → DeviceManager.select(device) → use_onnx=True
```

If OpenVINO is not installed, PaddleOCR runs with its default PaddlePaddle backend.

### Default OCR Engine Change

`configs/settings.yaml` sets the default OCR engine to PaddleOCR:

```yaml
ocr:
  engine: "paddleocr"
  paddleocr_use_openvino: true
```

### Fallback Logic

The CLI (`cli.py` lines 256-283) already had PaddleOCR → Tesseract fallback.
If PaddleOCR import fails, `TesseractEngine` is used automatically.
`src/ocr/engine.py`'s `_paddleocr_extract` now delegates to `PaddleOCREngine`
instead of returning empty string.

---

## VLM Additions

### CLIP Encoder Module

`src/embeddings/clip_encoder.py` provides:

- `encode_image(path) → np.ndarray (512,)` — CLIP image embedding
- `encode_text(query) → np.ndarray (512,)` — CLIP text embedding
- Optional OpenVINO acceleration for the vision encoder (on-the-fly ONNX → IR conversion)
- All embeddings are L2-normalised for FAISS inner-product compatibility

Model: `openai/clip-vit-base-patch32` (512-dimensional)

### Image Embedding During Ingestion

During `python cli.py ingest`, an additive Step 7 runs after text indexing:

1. Collect documents with valid `image_path`
2. Encode each image with CLIP
3. Build a separate FAISS index at `data/processed/faiss/clip/`
4. Metadata includes `modality="image"` and `embedding_type="clip"`

The existing text FAISS index is unchanged.

### Query Embedding Fusion

`src/retrieval/multimodal_retriever.py` wraps the existing `Retriever`:

1. Text path: MiniLM embedding → text FAISS index
2. CLIP path: CLIP text embedding → CLIP FAISS index
3. Fusion: weighted score combination (text=0.6, clip=0.4)

Activated via `--multimodal` flag: `python cli.py search "charts" --multimodal`

### Metadata Compatibility

CLIP-indexed documents use the same `NormalizedDocument` schema. Additional
metadata fields (`modality`, `embedding_type`) are additive — they do not
alter existing fields.

---

## Video Ingestion in `--path` Mode

### Overview

The `--path` CLI mode now supports video files (`.mp4`, `.avi`, `.mkv`, `.mov`,
`.webm`, `.flv`, `.wmv`) alongside documents and images. Video processing happens
automatically — no separate commands needed.

```bash
python cli.py --path video.mp4 --ask "What is the main topic?"
python cli.py --path ./mixed_folder --ask "Find references to AI"
```

### Pipeline

For each detected video file, `_process_video_file()` in `src/ingestion/loader.py`
runs the existing video pipeline modules:

1. **Frame Sampling** — `FrameSampler` extracts frames at configured intervals (default: 5s)
2. **Frame OCR** — `FrameOCR` runs Tesseract on extracted frames
3. **Frame Captioning** — `FrameCaptioner` generates BLIP captions for every Nth frame
4. **Audio Extraction** — `AudioExtractor` extracts audio track to WAV
5. **Whisper Transcription** — `WhisperTranscriber` transcribes audio to text segments
6. **Document Building** — `VideoDocumentBuilder` merges captions + OCR + transcript into a `VideoDocument`
7. **Conversion** — `VideoDocument.text` is wrapped into a `RawDocument` for the existing pipeline

### Fallback Behavior

- If Whisper fails → captions + OCR text used
- If OCR finds nothing → captions + transcript used
- If captioning fails → OCR + transcript used
- If all three fail → CLIP visual search on raw frames
- Corrupted videos → logged and skipped (no crash)

### Metadata Fields

Video documents include metadata for filtering compatibility:

```python
{
    "file_type": "video",
    "modality": "video",
    "duration": 120.5,
    "frame_count": 24,
    "ocr_frames_with_text": 8,
    "transcript_segments": 15,
    "caption_count": 5,
    "caption_interval": 5,
}
```

---

## Frame Captioning (VLM Integration)

### Overview

`src/video/frame_captioner.py` uses `Salesforce/blip-image-captioning-base` to
generate natural-language descriptions for sampled video frames.  This dramatically
improves video RAG quality because:

- **OCR** only captures text visible on screen
- **Whisper** only captures spoken audio
- **BLIP captions** describe the *visual content* (objects, scenes, actions)

### Example

For a frame showing a dark UI with art generation controls:
```
Frame at 15.0s: a computer screen showing a user interface with a drawing
```

### Configuration

```yaml
video:
  enable_frame_captioning: true
  caption_interval: 5          # caption every 5th frame
  caption_model: "Salesforce/blip-image-captioning-base"
  caption_use_openvino: false   # set true for OpenVINO acceleration
```

### How Captions Improve RAG

Without captions, a query like "What is shown in the video?" fails because:
- OCR found no text on screen
- Whisper may not describe visual content

With BLIP captions, the indexed text now includes descriptions like:
```
Frame at 5.0s: a person standing in front of a whiteboard
Frame at 15.0s: a computer screen with code being displayed
Frame at 25.0s: a graph showing performance metrics
```

The RAG retriever matches these against the query, producing meaningful answers.

---

## Files Modified

| File | Change |
| :--- | :--- |
| `src/ingestion/loader.py` | Added `_VIDEO_EXTENSIONS`, `_process_video_file()` with BLIP captioning step, video dispatch in `load_directory()` and `load_path()` |
| `src/ocr/paddle_engine.py` | Added DeviceManager integration for OpenVINO device validation |
| `src/ocr/engine.py` | Replaced placeholder `_paddleocr_extract` with PaddleOCREngine delegation |
| `src/ocr/__init__.py` | Added conditional PaddleOCREngine export |
| `src/video/__init__.py` | Added conditional FrameCaptioner export |
| `src/embeddings/__init__.py` | Added conditional CLIPEncoder and OVEmbeddingEncoder exports |
| `configs/settings.yaml` | Changed default to PaddleOCR+OpenVINO; added CLIP config, BLIP captioning config |
| `cli.py` | Added CLIP ingestion step, `--multimodal` flags, video modality, CLIP frame search |
| `README.md` | Updated architecture, features, OpenVINO, video pipeline, and Local File Query sections |

## Files Created

| File | Purpose |
| :--- | :--- |
| `src/video/frame_captioner.py` | BLIP frame captioning module (VLM scene descriptions) |
| `src/embeddings/clip_encoder.py` | CLIP multimodal encoder (image + text) with OpenVINO acceleration |
| `src/retrieval/multimodal_retriever.py` | Score fusion retriever combining text and CLIP retrieval |
| `codebase_summary_ocr_vlm.md` | This document |

## Files Reused (not modified)

| File | Role |
| :--- | :--- |
| `src/video/frame_sampler.py` | Frame extraction at configured intervals |
| `src/video/frame_ocr.py` | Tesseract OCR on video frames |
| `src/video/audio_extractor.py` | Audio track extraction to WAV |
| `src/video/transcription.py` | Whisper transcription to text segments |
| `src/video/video_document_builder.py` | Merging transcript + OCR into VideoDocument |
| `src/video/video_loader.py` | Video file discovery (used by `ingest-videos`, not by `--path`) |

