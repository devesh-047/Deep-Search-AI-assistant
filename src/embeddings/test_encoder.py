from encoder import EmbeddingEncoder
import numpy as np

encoder = EmbeddingEncoder(device="cpu")
vectors = encoder.encode(["Hello world", "Another sentence"])
print("Vectors shape:", vectors.shape)
print("vector type", vectors.dtype)

norms = np.linalg.norm(vectors, axis=1)
print("Norms:", norms)

sim = np.dot(vectors[0], vectors[1])
print(f"Cosine similarity: {sim:.4f}")