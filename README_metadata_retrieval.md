# Metadata-Aware Staged Retrieval

## Overview

This enhancement adds **optional metadata pre-filtering** to the existing RAG pipeline, enabling a staged retrieval approach that reduces the vector search space before approximate nearest-neighbour (ANN) search.

The original retrieval path remains completely untouched. The new functionality is toggled via a CLI flag (`--metadata-filtering`) and lives in a separate module.

## Why Metadata Filtering Improves Production RAG

In production knowledge bases, documents span multiple years, file types, departments, and modalities. A pure vector search treats every chunk equally, leading to:

- **Wasted computation** — searching all 100k chunks when the user explicitly asks about "2024 invoices" is inefficient.
- **Noisy results** — semantically similar chunks from irrelevant file types dilute the top-k.
- **Higher latency** — larger search spaces mean slower retrieval, especially with brute-force FAISS indices.

Metadata pre-filtering addresses these by narrowing the candidate set *before* vector comparison, resulting in:

- Faster retrieval when metadata constraints are present
- More relevant results for constrained queries
- Zero degradation for unconstrained queries (graceful fallback)

## Architecture

```
                          ┌─────────────────────────┐
                          │      User Query          │
                          └────────────┬────────────┘
                                       │
                                       ▼
                          ┌─────────────────────────┐
                          │  Query Metadata Parser   │
                          │  (rule-based, no ML)     │
                          │  Detects: year, type,    │
                          │  directory, modality     │
                          └─────┬──────────┬────────┘
                                │          │
                      constraints       no constraints
                      detected          detected
                          │                │
                          ▼                │
                ┌──────────────────┐       │
                │  Metadata Store  │       │
                │  (in-memory idx) │       │
                │  Filter chunk    │       │
                │  IDs by metadata │       │
                └────────┬─────── ┘       │
                         │                 │
                    filtered IDs           │
                    (may be empty)         │
                         │                 │
               ┌─────────┴──────┐          │
               │  empty?        │          │
               │  → fallback ───┼──────────┤
               │  non-empty     │          │
               └────────┬───────┘          │
                        │                  │
                        ▼                  ▼
              ┌──────────────────────────────────┐
              │        FAISS ANN Search           │
              │  (restricted to filtered IDs      │
              │   OR full index if no filter)     │
              └──────────────┬───────────────────┘
                             │
                             ▼
              ┌──────────────────────────────────┐
              │         Top-k Results             │
              │  (with scores + metadata)         │
              └──────────────┬───────────────────┘
                             │
                             ▼
              ┌──────────────────────────────────┐
              │     LLM Context Construction      │
              │     & Answer Generation           │
              └──────────────────────────────────┘
```

## Components

### 1. Metadata Enrichment (`enrich_chunk_metadata()`)

During ingestion, chunk metadata is enriched with:

| Field              | Type       | Description                              |
|:------------------ |:---------- |:---------------------------------------- |
| `file_name`        | `str`      | Name of the source file                  |
| `file_type`        | `str`      | Normalised type: pdf, docx, pptx, image, video, text |
| `created_year`     | `int/None` | Year the document was created (if available) |
| `source_directory` | `str`      | Directory path of the source             |
| `page_number`      | `int/None` | Page number within a PDF                 |
| `slide_number`     | `int/None` | Slide number within a PPTX               |
| `modality`         | `str`      | Content type: text, image, or video      |

Enrichment is **backward-compatible**: it only adds new keys, never overwrites existing ones. Existing stored embeddings and indices remain functional.

### 2. Query Metadata Parser (`QueryMetadataParser`)

A lightweight, deterministic, rule-based parser that extracts metadata constraints from queries:

- **Year**: Any 4-digit number matching 1900–2099 (e.g., "invoices from 2024")
- **File type**: Keywords like "pdf", "slides", "presentation", "video", "report"
- **Directory**: Tokens prefixed with "folder:", "dir:", "from:" 
- **Modality**: Explicit terms like "text", "image", "video"

No ML models or LLM calls are used. Processing is O(n) over query tokens.

### 3. Staged Retrieval (`StagedRetriever`)

Located in `src/retrieval/metadata_filter.py`. Execution flow:

1. Parse query for metadata hints
2. If constraints found → filter chunk IDs via `MetadataStore`
3. If filtered set is non-empty → FAISS search restricted to those IDs
4. If filtered set is empty → fallback to full vector search
5. Return results with timing statistics

### 4. Metadata Store (`MetadataStore`)

An in-memory index built from the existing `metadata.json` sidecar file. Each entry is enriched on load, mapping FAISS vector IDs to structured metadata.

## How to Enable Metadata-Aware Mode

### Search

```bash
# Original mode (unchanged)
python cli.py search "Find invoice totals"

# Metadata-aware mode
python cli.py search "Find invoice totals from 2024" --metadata-filtering
```

### Ask (RAG)

```bash
# Original mode (unchanged)
python cli.py ask "What is the quarterly revenue?"

# Metadata-aware mode
python cli.py ask "What is the quarterly revenue from the 2024 report?" --metadata-filtering
```

### Re-ingest to get enriched metadata

To benefit from metadata enrichment on new ingestions:

```bash
python cli.py ingest --dataset funsd
```

Existing indices continue to work — the `MetadataStore` will infer missing fields from available data.

## How to Run the Comparison Script

```bash
python scripts/compare_retrieval_modes.py
python scripts/compare_retrieval_modes.py --top-k 5 --verbose
```

This runs both retrieval modes on 7 representative queries and outputs:

- Per-query detailed metrics
- Aggregated summary table
- Latency and memory comparison
- Observations and recommendations

Raw results are also saved as JSON to `data/processed/retrieval_comparison.json`.

## DocVQA Accuracy & Latency Benchmark

A second benchmark tests both retrieval modes against **real DocVQA question–answer pairs** (scanned business documents) sourced from the public `nielsr/docvqa_1200_examples` dataset on Hugging Face. It uses pre-extracted OCR words — no Tesseract run needed.

### What it measures

| Metric                | Description |
|:--------------------- |:----------- |
| **Recall@k**          | Fraction of questions where the correct source document appears in the top-k retrieved chunks (k = 1, 3, 5). |
| **Answer-in-context** | Fraction of questions where any ground-truth answer string is found verbatim in the concatenated retrieved context. |
| **Latency (ms)**      | End-to-end wall-clock time per query, including encoding. Percentiles (p50–p99) also reported. |
| **Vector comparisons**| FAISS vectors compared per query; reduction shows pre-filter effectiveness. |
| **Result-set overlap**| Fraction of staged top-k results that also appear in the original top-k (correctness parity check). |

### How to run

```bash
# Default: 60 examples, top-5
python scripts/docvqa_rag_benchmark.py

# Larger run with per-query output
python scripts/docvqa_rag_benchmark.py --n-docs 120 --top-k 5 --verbose
```

Results are saved to `data/processed/docvqa_bench_results.json`.

### Sample results (60 questions, 23 unique docs, top-5)

```
Metric                             Original       Staged
────────────────────────────────────────────────────────────
Avg latency (ms)                       23.0         21.8   (+5% faster)
Recall@1                              43.3%        43.3%   =
Recall@3                              60.0%        60.0%   =
Recall@5                              68.3%        68.3%   =
Answer-in-context                     55.0%        55.0%
Avg vector comparisons                   23           23   0.0% reduction
Avg result-set overlap               100.0%

Latency percentiles (ms)     Original   Staged
p50                              21.3     20.3
p75                              24.7     24.2
p90                              31.2     30.0
p95                              35.2     31.5
max                              41.6     33.6
```

### Why vector reduction is 0 % on DocVQA

DocVQA is a **homogeneous image corpus** — every document is a scanned form or business letter. When the pre-filter fires on a constraint like `file_type='image'`, it returns all documents, so FAISS must compare all vectors regardless.

Real savings appear on **mixed-modality** corpora (PDFs + videos + images). For example, a 10 000-vector index with 20 % images would reduce the search space by ~80 % for image-targeted queries.

### Why Recall@1 is ~43 % instead of ~100 %

`all-MiniLM-L6-v2` is a general-purpose sentence embedding model. DocVQA questions are often short and ambiguous (e.g. *"What is the name of the company?"*), which means multiple documents score similarly. Using a document-understanding model (LayoutLMv3, Donut) or a cross-encoder re-ranker would push Recall@1 significantly higher. Recall@5 at 68 % shows the right document is *reachable* — the bottleneck is ranking, not retrieval coverage.

## Interpretation of Results

### Key Metrics

| Metric                    | What it means                                              |
|:------------------------- |:---------------------------------------------------------- |
| Retrieval latency (ms)    | Total time from query to results (includes encoding)       |
| Vector comparisons        | Number of vectors FAISS compared (fewer = faster)          |
| Precision overlap         | Fraction of staged results also in original top-k          |
| Context tokens            | Approximate tokens in the LLM context (cost indicator)     |
| Peak memory (MB)          | Process RSS during retrieval                               |

### What to expect

- **Queries with metadata hints**: Staged retrieval may show reduced vector comparisons and comparable or lower latency.
- **Queries without hints**: Both modes produce identical results (staged mode falls back gracefully).
- **Small indices** (<1k vectors): Pre-filter overhead may slightly exceed brute-force savings. Benefits scale with index size.

## When to Use Original vs Metadata-Aware Mode

| Scenario                                        | Recommended Mode    |
|:----------------------------------------------- |:------------------- |
| Generic queries, no domain constraints           | Original            |
| User references specific year/type/modality      | Metadata-aware      |
| Small index (<1k vectors)                        | Original            |
| Large index (>10k vectors) + structured corpus   | Metadata-aware      |
| Latency-critical, need lowest possible overhead  | Original            |
| Enterprise KB with mixed file types and years    | Metadata-aware      |

## File Layout

```
src/retrieval/
├── __init__.py              # Unchanged
├── retriever.py             # Unchanged (original retrieval)
└── metadata_filter.py       # NEW: MetadataStore, QueryMetadataParser, StagedRetriever

scripts/
├── compare_retrieval_modes.py   # NEW: Synthetic query benchmark (7 queries)
└── docvqa_rag_benchmark.py      # NEW: DocVQA accuracy & latency benchmark

data/processed/
├── retrieval_comparison.json    # Output of compare_retrieval_modes.py
└── docvqa_bench_results.json    # Output of docvqa_rag_benchmark.py

README_metadata_retrieval.md    # NEW: This file
```

## Dependencies

No new dependencies are required. The module uses only:
- `re`, `time`, `os`, `pathlib` (stdlib)
- `numpy` (already required)
- `faiss` (already required)
- `psutil` (optional, for memory measurement in comparison script)
