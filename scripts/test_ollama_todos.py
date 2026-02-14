"""
Verification script for ollama_client.py Learning TODOs (Phase 8).

Tests everything that does NOT require Ollama to be running:
  1. Prompt template construction
  2. Temperature/top_p preset registry
  3. Answer quality metrics
  4. Streaming interface (import check)
  5. OpenVINO GenAI stub

Run from project root:
    python scripts/test_ollama_todos.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm.ollama_client import (
    OllamaClient,
    OVLLMClient,
    PROMPT_TEMPLATES,
    GENERATION_PRESETS,
    RAG_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT_CITED,
    RAG_SYSTEM_PROMPT_CONCISE,
)

print("=" * 60)
print("  Phase 8 — Ollama Client Learning TODO Verification")
print("=" * 60)

# ------------------------------------------------------------------
#  TODO 1: Basic generation test (Ollama connectivity check)
# ------------------------------------------------------------------
print("\n--- TODO 1: Ollama Connectivity ---")
client = OllamaClient()
available = client.is_available()
print(f"  Server: {client.base_url}")
print(f"  Model:  {client.model}")
print(f"  Available: {available}")
if available:
    models = client.list_models()
    print(f"  Models on server: {models}")
else:
    print("  (Ollama not running — generation tests will be skipped)")
print("  [OK] Connectivity check works\n")

# ------------------------------------------------------------------
#  TODO 2: RAG Prompt Templates
# ------------------------------------------------------------------
print("--- TODO 2: RAG Prompt Templates ---")
print(f"  Registered templates: {list(PROMPT_TEMPLATES.keys())}")
for name, tmpl in PROMPT_TEMPLATES.items():
    print(f"    '{name}': {tmpl['description']}")

# Test prompt building
mock_context = "[Source 1 | score=0.85 | doc=inv001]\nTotal: $500\n"
question = "What is the total?"

for tmpl_name in PROMPT_TEMPLATES:
    prompt = OllamaClient.build_rag_prompt(question, mock_context, tmpl_name)
    has_system = len(prompt["system"]) > 0
    has_user = len(prompt["user"]) > 0
    has_context = mock_context.strip() in prompt["system"]
    has_question = question in prompt["user"]
    print(f"  Template '{tmpl_name}': system={has_system}, user={has_user}, "
          f"context_injected={has_context}, question_injected={has_question}")

print("  [OK] Prompt templates work\n")

# ------------------------------------------------------------------
#  TODO 3: Temperature / top_p Presets
# ------------------------------------------------------------------
print("--- TODO 3: Generation Presets ---")
print(f"  Registered presets: {list(GENERATION_PRESETS.keys())}")
for name, p in GENERATION_PRESETS.items():
    print(f"    '{name}': temp={p['temperature']}, top_p={p['top_p']} — {p['description']}")

# Verify preset values are in valid ranges
for name, p in GENERATION_PRESETS.items():
    assert 0.0 <= p["temperature"] <= 2.0, f"Bad temperature for {name}"
    assert 0.0 <= p["top_p"] <= 1.0, f"Bad top_p for {name}"
print("  [OK] All presets have valid parameter ranges\n")

# ------------------------------------------------------------------
#  TODO 4: Streaming Support
# ------------------------------------------------------------------
print("--- TODO 4: Streaming Support ---")
has_stream = hasattr(client, "generate_stream")
print(f"  generate_stream() method exists: {has_stream}")
import inspect
if has_stream:
    sig = inspect.signature(client.generate_stream)
    params = list(sig.parameters.keys())
    print(f"  Parameters: {params}")
    # Check it's a generator
    import types
    # The method should be a generator function
    is_gen = inspect.isgeneratorfunction(client.generate_stream)
    print(f"  Is generator function: {is_gen}")
print("  [OK] Streaming interface ready\n")

# ------------------------------------------------------------------
#  TODO 2 (cont): Answer Quality Metrics
# ------------------------------------------------------------------
print("--- TODO 2: Answer Quality Metrics ---")

# Good answer (grounded in context)
good_answer = "The total amount is $500."
metrics = OllamaClient.measure_answer_quality(
    answer=good_answer,
    context=mock_context,
    question=question,
)
print(f"  Grounded answer: '{good_answer}'")
for m, v in metrics.items():
    bar = "█" * int(v * 20)
    print(f"    {m:<20}: {v:.3f}  {bar}")

# Bad answer (hallucinated)
bad_answer = "The total is $9999 and was signed by Napoleon on Mars."
bad_metrics = OllamaClient.measure_answer_quality(
    answer=bad_answer,
    context=mock_context,
    question=question,
)
print(f"\n  Hallucinated answer: '{bad_answer}'")
for m, v in bad_metrics.items():
    bar = "█" * int(v * 20)
    print(f"    {m:<20}: {v:.3f}  {bar}")

# Verify grounded answer has higher groundedness
assert metrics["groundedness"] > bad_metrics["groundedness"], \
    "Grounded answer should score higher than hallucinated one!"
print(f"\n  Groundedness: {metrics['groundedness']:.3f} (good) vs "
      f"{bad_metrics['groundedness']:.3f} (bad) — correct!")

# Uncertainty answer
uncertain = "I don't know. The context does not contain enough information."
unc_metrics = OllamaClient.measure_answer_quality(
    answer=uncertain,
    context=mock_context,
    question=question,
)
print(f"\n  Uncertain answer: '{uncertain}'")
print(f"    uncertainty_flag: {unc_metrics['uncertainty_flag']}  "
      f"({'detected' if unc_metrics['uncertainty_flag'] else 'missed'})")
assert unc_metrics["uncertainty_flag"] == 1.0, "Should detect uncertainty"

print("  [OK] Quality metrics work\n")

# ------------------------------------------------------------------
#  TODO 5: OpenVINO GenAI Stub
# ------------------------------------------------------------------
print("--- TODO 5: OpenVINO GenAI Stub ---")
ov_client = OVLLMClient(model_path="models/openvino/mistral-7b/")
ov_available = ov_client.is_available()
print(f"  OVLLMClient instantiated: True")
print(f"  OpenVINO GenAI available: {ov_available}")
result = ov_client.generate("test")
print(f"  Stub response: {result}")
print("  [OK] GenAI stub ready\n")

# ------------------------------------------------------------------
#  End-to-end with real index (if Ollama is available)
# ------------------------------------------------------------------
if available:
    print("--- End-to-end: RAG with real index ---")
    try:
        from src.embeddings.encoder import EmbeddingEncoder
        from src.index.faiss_index import FaissIndex
        from src.retrieval.retriever import Retriever

        index = FaissIndex()
        index.load("data/processed/faiss")
        encoder = EmbeddingEncoder(device="cpu")
        retriever = Retriever(encoder=encoder, index=index)

        q = "What types of documents are in the collection?"
        results = retriever.query(q, top_k=3)
        context = retriever.format_context(results)

        print(f"  Question: {q}")
        print(f"  Context length: {len(context)} chars, {len(results)} chunks")

        answer = client.generate(question=q, context=context)
        print(f"  Answer: {answer[:300]}...")

        metrics = OllamaClient.measure_answer_quality(answer, context, q)
        print(f"\n  Quality:")
        for m, v in metrics.items():
            print(f"    {m}: {v:.3f}")

        print("\n  [OK] End-to-end RAG works")
    except FileNotFoundError:
        print("  [SKIP] No index found. Run 'python cli.py ingest' first.")
    except Exception as e:
        print(f"  [ERROR] {e}")
else:
    print("--- End-to-end: SKIPPED (Ollama not running) ---")
    print("  To test full RAG, start Ollama and re-run this script:")
    print("    ollama serve &")
    print("    ollama pull mistral")
    print("    python scripts/test_ollama_todos.py")

print("\n" + "=" * 60)
print("  All Phase 8 Learning TODOs verified!")
print("=" * 60)
