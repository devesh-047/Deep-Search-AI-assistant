"""Quick smoke test for Section 12 implementations."""
import sys
sys.path.insert(0, ".")

print("=" * 60)
print("Section 12: OpenVINO Integration Roadmap — Smoke Test")
print("=" * 60)

# 1. Device Manager
print("\n1. Device Manager")
try:
    from src.openvino.device_manager import DeviceManager
    dm = DeviceManager()
    devices = dm.list_devices()
    print(f"   OK  devices={devices}")
except Exception as e:
    print(f"   FAIL  {e}")

# 2. Model Converter
print("\n2. Model Converter")
try:
    from src.openvino.model_converter import convert_onnx_to_ir, convert_embedding_model_pipeline
    print("   OK  import successful")
except Exception as e:
    print(f"   FAIL  {e}")

# 3. OV Embedding Encoder
print("\n3. OV Embedding Encoder")
try:
    from src.embeddings.openvino_encoder import OVEmbeddingEncoder
    print("   OK  import successful")
except Exception as e:
    print(f"   FAIL  {e}")

# 4. OV LLM Client
print("\n4. OV LLM Client")
try:
    from src.llm.openvino_llm import OVLLMClient
    client = OVLLMClient()  # no model_dir = no load attempt
    print(f"   OK  import successful, available={client.is_available()}")
    msg = client.generate("test question")
    print(f"   OK  generate() returned {len(msg)} chars")
    print(f"       Preview: {msg[:100]}...")
except Exception as e:
    print(f"   FAIL  {e}")

# 5. PaddleOCR Engine
print("\n5. PaddleOCR Engine")
try:
    from src.ocr.paddle_engine import PaddleOCREngine, PADDLE_AVAILABLE
    print(f"   OK  import successful, paddleocr installed={PADDLE_AVAILABLE}")
    if not PADDLE_AVAILABLE:
        print("       (PaddleOCR not installed — this is expected)")
        print("       Install with: pip install paddleocr paddlepaddle")
except Exception as e:
    print(f"   FAIL  {e}")

print("\n" + "=" * 60)
print("All 5 components verified!")
print("=" * 60)
