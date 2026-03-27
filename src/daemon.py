"""
Deep Search Daemon
==================
A lightweight, zero-dependency background server that keeps the heavily
imported frameworks (FAISS, PyTorch, OpenVINO, SentenceTransformers)
resident in memory.

By running this daemon, CLI commands offload their execution via
fast HTTP requests, dropping startup latency from ~20s to ~0.1s.
"""

import http.server
import json
import logging
import socketserver
import urllib.parse
import sys
from pathlib import Path

# Add project root to path securely
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Heavy imports are loaded ONCE when the daemon boots
from src.embeddings.encoder import EmbeddingEncoder
from src.index.faiss_index import FaissIndex
from src.retrieval.retriever import Retriever
import yaml

SETTINGS = {}
try:
    with open(PROJECT_ROOT / "configs" / "settings.yaml", "r", encoding="utf-8") as f:
        SETTINGS = yaml.safe_load(f) or {}
except Exception as e:
    logger.warning(f"Daemon could not load settings.yaml: {e}")

logger = logging.getLogger(__name__)

PORT = SETTINGS.get("daemon", {}).get("port", 8500)
HOST = "127.0.0.1"

# Global state
encoder = None
index = None
retriever = None

def init_components(data_dir: Path):
    """Preload all AI components into the Daemon's memory."""
    global encoder, index, retriever
    logger.info("Initializing resident OpenVINO/PyTorch models...")

    ov_settings = SETTINGS.get("openvino", {})
    ov_enabled = ov_settings.get("enabled", False)
    model_xml = ov_settings.get("embedding_model_ir", "")
    ov_device = ov_settings.get("device", "CPU")
    
    # Resolve relative model paths against project root
    if model_xml and not Path(model_xml).is_absolute():
        model_xml = str(PROJECT_ROOT / model_xml)

    try:
        from src.openvino.device_manager import DeviceManager
        dm = DeviceManager()
        ov_device = dm.select(ov_device)
        logger.info(f"Verified OpenVINO device: {ov_device}")
    except Exception as e:
        logger.warning(f"Device verification failed: {e}")

    # Load Encoder
    if ov_enabled and model_xml and Path(model_xml).exists():
        from src.embeddings.openvino_encoder import OVEmbeddingEncoder
        encoder = OVEmbeddingEncoder(model_xml=model_xml, device=ov_device)
        logger.info("Daemon spawned OVEmbeddingEncoder")
    else:
        encoder = EmbeddingEncoder()
        logger.info("Daemon spawned standard EmbeddingEncoder")

    # Load FAISS Index
    index_path = data_dir / "faiss_index.bin"
    index = FaissIndex(dimension=encoder.dimension)
    if index_path.exists():
        index.load(str(index_path))
        logger.info(f"Daemon loaded FAISS index ({index.size} vectors)")
    else:
        logger.warning(f"Index not found at {index_path}. Search will return empty.")

    # Load Retriever
    data_json = data_dir / "dataset.json"
    doc_data = {}
    if data_json.exists():
        with open(data_json, "r", encoding="utf-8") as f:
            doc_data = json.load(f)
            logger.info(f"Daemon loaded {len(doc_data)} documents from dataset")

    # Note: we disable multimodal retrieve in simple daemon for now
    retriever = Retriever(encoder=encoder, index=index)
    logger.info("Daemon initialization complete. Heavy components resident in RAM.")

class DaemonHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress standard logging for speed, unless error
        pass

    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "vectors": index.size if index else 0}).encode())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/search':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                req = json.loads(post_data.decode('utf-8'))
                query = req.get("query", "")
                top_k = req.get("top_k", 5)
                
                if not query:
                    raise ValueError("Query cannot be empty")
                if not retriever:
                    raise ValueError("Retriever not initialized on Daemon")

                # Perform the resident search
                results = retriever.query(query, top_k=top_k)
                res_dict = [
                    {"doc_id": r.doc_id, "score": float(r.score), "content": r.content, "metadata": r.metadata}
                    for r in results
                ]
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"results": res_dict}).encode('utf-8'))
                
            except Exception as e:
                logger.error(f"Daemon /search error: {e}")
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        else:
            self.send_error(404)

class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

def run_daemon(data_dir: Path):
    init_components(data_dir)
    with ReusableTCPServer((HOST, PORT), DaemonHandler) as httpd:
        logger.info(f"Deep Search Daemon serving on http://{HOST}:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.server_close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the Deep Search AI Daemon")
    parser.add_argument("--path", type=str, required=True, help="Path to the processed data directory")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_daemon(Path(args.path))
