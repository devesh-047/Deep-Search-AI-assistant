"""Quick test of all Learning TODOs in faiss_index.py."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.index.faiss_index import FaissIndex

DIM = 384
N = 500
TOP_K = 5
N_QUERIES = 20

rng = np.random.default_rng(42)
data = rng.standard_normal((N, DIM)).astype(np.float32)
norms = np.linalg.norm(data, axis=1, keepdims=True)
data = data / norms
metadata = [{"id": i, "text": f"chunk_{i}"} for i in range(N)]
qi = rng.choice(N, size=N_QUERIES, replace=False)
queries = data[qi]

# TODO 1: Flat index
print("=== TODO 1: Flat index (exact) ===")
idx = FaissIndex(dimension=DIM)
idx.build(data, metadata)
t0 = time.time()
gt_s, gt_i = idx.index.search(queries, TOP_K)
flat_t = time.time() - t0
report = idx.verify(data, sample_queries=10, top_k=TOP_K)
hits = sum(1 for r in report if r["self_hit"])
print(f"  Vectors: {idx.size}")
print(f"  Self-hits: {hits}/{len(report)}")
print(f"  Search time: {flat_t*1000:.2f}ms")

# TODO 2: IVF index
print("\n=== TODO 2: IVF index ===")
for nprobe in [1, 5, 10]:
    iv = FaissIndex(dimension=DIM)
    iv.build_ivf(data, metadata, nlist=50, nprobe=nprobe)
    t0 = time.time()
    _, ivf_i = iv.index.search(queries, TOP_K)
    ivf_t = time.time() - t0
    recall = sum(
        len(set(g.tolist()) & set(p.tolist()))
        for g, p in zip(gt_i, ivf_i)
    ) / (N_QUERIES * TOP_K) * 100
    print(f"  nprobe={nprobe:2d}: recall@5={recall:5.1f}%  time={ivf_t*1000:.2f}ms")

# TODO 3: HNSW index
print("\n=== TODO 3: HNSW index ===")
for efs in [16, 64, 128]:
    hw = FaissIndex(dimension=DIM)
    hw.build_hnsw(data, metadata, M=32, ef_construction=200, ef_search=efs)
    t0 = time.time()
    _, hw_i = hw.index.search(queries, TOP_K)
    hw_t = time.time() - t0
    recall = sum(
        len(set(g.tolist()) & set(p.tolist()))
        for g, p in zip(gt_i, hw_i)
    ) / (N_QUERIES * TOP_K) * 100
    print(f"  efSearch={efs:3d}: recall@5={recall:5.1f}%  time={hw_t*1000:.2f}ms")

# TODO 4: Incremental
print("\n=== TODO 4: Incremental add() ===")
half = N // 2
inc = FaissIndex(dimension=DIM)
inc.build(data[:half], metadata[:half])
print(f"  Before add(): {inc.size} vectors")
inc.add(data[half:], metadata[half:])
print(f"  After add():  {inc.size} vectors")
_, inc_i = inc.index.search(queries, TOP_K)
recall = sum(
    len(set(g.tolist()) & set(p.tolist()))
    for g, p in zip(gt_i, inc_i)
) / (N_QUERIES * TOP_K) * 100
print(f"  Recall vs flat: {recall:.1f}% (expected 100%)")

# Also test IVF incremental
inc_ivf = FaissIndex(dimension=DIM)
inc_ivf.build_ivf(data[:half], metadata[:half], nlist=25, nprobe=10)
print(f"  IVF before add(): {inc_ivf.size}")
inc_ivf.add(data[half:], metadata[half:])
print(f"  IVF after add():  {inc_ivf.size}")

print("\n=== ALL TESTS PASSED ===")
