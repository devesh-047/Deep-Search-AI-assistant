import faiss
import json
import os

# Check if files exist first to avoid crashes
index_path = "data/processed/faiss/index.faiss"
meta_path = "data/processed/faiss/metadata.json"

if os.path.exists(index_path) and os.path.exists(meta_path):
    index = faiss.read_index(index_path)
    print(f"Vectors: {index.ntotal}")
    print(f"Dimension: {index.d}")

    with open(meta_path) as f:
        meta = json.load(f)
    print(f"Metadata entries: {len(meta)}")
    if len(meta) > 0:
        print(f"Keys: {meta[0].keys()}")
else:
    print("Error: Ensure your data files are in data/processed/faiss/")
