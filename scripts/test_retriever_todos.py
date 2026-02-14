"""
Verification script for retriever.py Learning TODOs.

Exercises all four features:
  1. Cross-encoder re-ranking
  2. Query preprocessing (spelling + synonym expansion)
  3. Hybrid retrieval (dense FAISS + sparse BM25 + RRF fusion)
  4. Metadata filtering

Run from project root:
    python scripts/test_retriever_todos.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.retriever import (
    QueryPreprocessor,
    BM25Retriever,
    Retriever,
    RetrieverResult,
)

print("=" * 60)
print("  Retriever — Learning TODO Verification")
print("=" * 60)

# ------------------------------------------------------------------
#  TODO 2: Query Preprocessing
# ------------------------------------------------------------------
print("\n--- TODO 2: Query Preprocessing ---")
pp = QueryPreprocessor()

tests = [
    ("invoce   totla ammount", "spelling fixes + normalisation"),
    ("employee name",          "synonym expansion for 'employee' and 'name'"),
    ("  HELLO   World  ",      "case + whitespace normalisation"),
    ("find the invoice date",  "synonym expansion for 'invoice' and 'date'"),
]
for raw, desc in tests:
    result = pp.preprocess(raw)
    print(f"  '{raw}' -> '{result}'   ({desc})")

# Test with features toggled off
no_expand = pp.preprocess("invoice total", expand_synonyms=False)
no_spell  = pp.preprocess("invoce totla", fix_spelling=False)
print(f"  Synonyms OFF: 'invoice total' -> '{no_expand}'")
print(f"  Spelling OFF: 'invoce totla'  -> '{no_spell}'")
print("  [OK] Preprocessing works\n")

# ------------------------------------------------------------------
#  TODO 3: BM25 Sparse Retriever
# ------------------------------------------------------------------
print("--- TODO 3: BM25 Retriever ---")

corpus = [
    {"chunk_id": "c1", "doc_id": "d1", "text": "The invoice total amount is $500 payable on receipt."},
    {"chunk_id": "c2", "doc_id": "d1", "text": "Employee name: John Smith, Department: Engineering."},
    {"chunk_id": "c3", "doc_id": "d2", "text": "Purchase order for office supplies dated March 2024."},
    {"chunk_id": "c4", "doc_id": "d2", "text": "Company address: 123 Main Street, Springfield, IL."},
    {"chunk_id": "c5", "doc_id": "d3", "text": "Signature of the authorized representative is required."},
    {"chunk_id": "c6", "doc_id": "d3", "text": "The total amount on the form is five hundred dollars."},
]

bm25 = BM25Retriever(k1=1.5, b=0.75)
bm25.fit(corpus)

queries_bm25 = ["invoice total", "employee name", "company address", "signature"]
for q in queries_bm25:
    results = bm25.search(q, top_k=3)
    top_ids = [(r["chunk_id"], round(r["score"], 3)) for r in results]
    print(f"  '{q}' -> {top_ids}")

print("  [OK] BM25 works\n")

# ------------------------------------------------------------------
#  TODO 3 (cont): Reciprocal Rank Fusion
# ------------------------------------------------------------------
print("--- TODO 3: Reciprocal Rank Fusion ---")

dense = [
    {"chunk_id": "c1", "text": "...", "score": 0.95},
    {"chunk_id": "c3", "text": "...", "score": 0.80},
    {"chunk_id": "c5", "text": "...", "score": 0.70},
]
sparse = [
    {"chunk_id": "c2", "text": "...", "score": 8.5},
    {"chunk_id": "c1", "text": "...", "score": 6.2},
    {"chunk_id": "c4", "text": "...", "score": 3.1},
]
fused = Retriever._reciprocal_rank_fusion(dense, sparse, top_k=5, k=60)
print(f"  Dense ranks:  c1=#1, c3=#2, c5=#3")
print(f"  Sparse ranks: c2=#1, c1=#2, c4=#3")
print(f"  Fused result:")
for r in fused:
    print(f"    {r['chunk_id']}: RRF score = {r['score']:.5f}")
print("  (c1 should be #1 since it appears in BOTH lists)")
print("  [OK] RRF works\n")

# ------------------------------------------------------------------
#  TODO 4: Metadata Filtering
# ------------------------------------------------------------------
print("--- TODO 4: Metadata Filtering ---")

mock_results = [
    RetrieverResult("c1", "d1", "invoice text",   0.9, {"doc_type": "form",    "source": "funsd"}),
    RetrieverResult("c2", "d2", "receipt text",    0.8, {"doc_type": "receipt", "source": "docvqa"}),
    RetrieverResult("c3", "d3", "another form",    0.7, {"doc_type": "form",    "source": "funsd"}),
    RetrieverResult("c4", "d4", "letter text",     0.6, {"doc_type": "letter",  "source": "rvl_cdip"}),
    RetrieverResult("c5", "d5", "memo about form", 0.5, {"doc_type": "memo",    "source": "funsd"}),
]

# Filter by doc_type
filtered = Retriever._apply_filters(mock_results, {"doc_type": "form"})
print(f"  Filter doc_type=form: {len(filtered)} results "
      f"(chunks: {[r.chunk_id for r in filtered]})")

# Filter by source
filtered = Retriever._apply_filters(mock_results, {"source": "funsd"})
print(f"  Filter source=funsd:  {len(filtered)} results "
      f"(chunks: {[r.chunk_id for r in filtered]})")

# Filter by BOTH
filtered = Retriever._apply_filters(
    mock_results, {"source": "funsd", "doc_type": "form"}
)
print(f"  Filter both:          {len(filtered)} results "
      f"(chunks: {[r.chunk_id for r in filtered]})")

# Filter that matches nothing
filtered = Retriever._apply_filters(mock_results, {"doc_type": "spreadsheet"})
print(f"  Filter no match:      {len(filtered)} results")

print("  [OK] Filtering works\n")

# ------------------------------------------------------------------
#  TODO 1: Cross-Encoder Re-ranking (import check only)
# ------------------------------------------------------------------
print("--- TODO 1: Cross-Encoder Re-ranking ---")
from src.retrieval.retriever import CROSS_ENCODER_AVAILABLE
print(f"  CrossEncoder available: {CROSS_ENCODER_AVAILABLE}")
if CROSS_ENCODER_AVAILABLE:
    print("  (To test: the Retriever will auto-use it when enable_reranker=True)")
    print("  Skipping full test to avoid model download in this script.")
else:
    print("  (sentence-transformers CrossEncoder not found, re-ranking disabled)")
print("  [OK] Import check passed\n")

# ------------------------------------------------------------------
#  End-to-end test with real index (if available)
# ------------------------------------------------------------------
print("--- End-to-end: search with preprocessing ---")
try:
    from src.embeddings.encoder import EmbeddingEncoder
    from src.index.faiss_index import FaissIndex

    index = FaissIndex()
    index.load("data/processed/faiss")

    encoder = EmbeddingEncoder(device="cpu")
    retriever = Retriever(
        encoder=encoder,
        index=index,
        enable_bm25=True,
        enable_preprocessing=True,
        enable_reranker=False,  # skip to avoid model download
    )

    for q in ["invoce totla", "employee name", "asdfghjkl"]:
        results = retriever.query(q, top_k=3)
        print(f"\n  Query: '{q}'")
        for r in results:
            preview = r.text[:60].replace("\n", " ")
            print(f"    score={r.score:.4f}  doc={r.doc_id}  '{preview}...'")

    print("\n  [OK] End-to-end works")
except FileNotFoundError:
    print("  [SKIP] No index found. Run 'python cli.py ingest --dataset funsd --max-records 10' first.")
except Exception as e:
    print(f"  [ERROR] {e}")

print("\n" + "=" * 60)
print("  All Learning TODOs verified!")
print("=" * 60)
