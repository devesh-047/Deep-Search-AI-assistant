# OpenVINO Compiled Model Caching — Codebase Summary

## What Was Added

Compiled-model caching for `OVEmbeddingEncoder` using OpenVINO's
`Core.export_model()` / `Core.import_model()` API.

### New code in `src/embeddings/openvino_encoder.py`

| Element | Type | Purpose |
| :--- | :--- | :--- |
| `import hashlib` | import | MD5 hash for cache key |
| `_CACHE_DIR` | `Path` constant | `~/.deepsearch/cache/` |
| `_get_cache_key(model_path, device)` | module-level function | Returns deterministic hex digest |
| Cache load block inside `_try_load()` | 20 lines | `import_model()` on hit; falls through on miss/corruption |
| Cache save block inside `_try_load()` | 12 lines | `export_model()` after successful compilation |

---

## Where the Change Is

**File**: `src/embeddings/openvino_encoder.py`  
**Class**: `OVEmbeddingEncoder`  
**Method**: `_try_load(model_xml, device)`

The change is confined to the section between `core = Core()` and the
input/output tensor discovery loop.  No other method was modified.

---

## Why It Is Safe

1. **Entire `_try_load()` is already in a try/except** — any exception anywhere in the method falls through to `logger.warning(..., "using placeholder mode")`.  The new cache code inherits this safety net.

2. **Cache I/O is double-wrapped** — the load attempt and the save attempt each have their own inner `try/except`.  A corrupted blob triggers a `logger.warning` and sets `cache_loaded = False`, causing a normal recompile.

3. **The compiled model object is used identically** — whether it comes from `compile_model()` or `import_model()`, it is the same `CompiledModel` type and is assigned to `self._compiled_model`.  All downstream code (`self._input_names`, `self._output_name`, `self._compiled_model(infer_inputs)`) is unchanged.

4. **Cache key includes device** — `md5(f"{model_path}_{device}")` means CPU and GPU blobs never collide.

5. **`OVLLMClient` was intentionally left unmodified** — the LLM backend uses `openvino_genai.LLMPipeline`, which manages its own internal compiled model and does not expose a `CompiledModel` handle.  Caching there would require intercepting an opaque internal object, which is unsafe.

6. **CLI behavior unchanged** — no flags, arguments, or outputs were modified. The only user-visible change is two console prints on first/subsequent runs.

---

## Performance Impact

| Scenario | Expected time |
| :--- | :--- |
| Cold start (no blob) | ~2.6 s (matches existing benchmark) |
| Warm start (blob exists) | ~0.4 – 0.8 s (2–4× faster) |

The actual saving depends on:
- CPU model and core count
- OpenVINO version (later versions compile faster)
- Model size (all-MiniLM-L6-v2 is small; gains scale with model size)

---

## Cache Location

```
~/.deepsearch/cache/
└── <md5hex>.blob    # one file per (model_path, device) pair
```

To clear: `rm -rf ~/.deepsearch/cache/`

The directory is created automatically with `Path.mkdir(parents=True, exist_ok=True)`.

---

## Files Changed

| File | Change type |
| :--- | :--- |
| `src/embeddings/openvino_encoder.py` | Modified (caching logic added) |
| `README.md` | Updated (new "Compiled Model Caching" subsection) |
| `codebase_summary_ov_cache.md` | Created (this file) |

---

## Files NOT Changed

All other files are untouched:

- `cli.py` — unchanged
- `src/llm/openvino_llm.py` — unchanged (LLM caching skipped: unsafe)
- `src/openvino/device_manager.py` — unchanged
- `configs/settings.yaml` — unchanged
- `src/openvino/ov_embeddings.py` — deprecated placeholder, unchanged
- `src/openvino/ov_llm.py` — deprecated placeholder, unchanged

---

## Testing Checklist

### Functional regression
- [ ] `python cli.py ingest --dataset funsd` — no crash, correct output
- [ ] `python cli.py ask "What is this document about?"` — answer unchanged

### Cache behaviour
- [ ] **First run**: prints `Compiling model...` then `Compiled model cached at ~/.deepsearch/cache/*.blob`
- [ ] **Second run**: prints `Loading compiled model from cache...` (no compilation)
- [ ] **Corrupted blob**: `echo "bad" > ~/.deepsearch/cache/*.blob && python cli.py ask "..."` → warning logged, recompile succeeds, no crash

### Benchmark comparison
Run with the existing benchmark module:
```bash
python cli.py benchmark --embeddings
```
Compare `Model load time` between first run and second run.
