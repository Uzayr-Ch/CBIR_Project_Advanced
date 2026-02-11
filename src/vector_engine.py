"""FAISS wrapper for efficient similarity search.

FAISS accelerates nearest-neighbor queries using SIMD-optimized code paths,
cache-friendly layouts, and optional quantization; this is significantly faster
than Python-bound distances (e.g., `scipy.spatial`) on large feature banks.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Literal, Tuple

import faiss  # type: ignore
import numpy as np

IndexType = Literal["flat", "ivf"]


@dataclass
class SearchResult:
    distances: np.ndarray
    indices: np.ndarray
    latency_ms: float


class FaissVectorEngine:
    def __init__(
        self,
        dim: int,
        index_type: IndexType = "flat",
        nlist: int = 256,
        index: faiss.Index | None = None,
    ) -> None:
        self.dim = dim
        self.index_type = index_type
        self.nlist = nlist
        self.index = index or self._create_index()

    def _create_index(self) -> faiss.Index:
        if self.index_type == "flat":
            return faiss.IndexFlatL2(self.dim)
        if self.index_type == "ivf":
            quantizer = faiss.IndexFlatL2(self.dim)
            return faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_L2)
        raise ValueError(f"Unsupported index type: {self.index_type}")

    def build_index(self, features: np.ndarray) -> None:
        feats = np.ascontiguousarray(features, dtype=np.float32)
        if feats.ndim != 2 or feats.shape[1] != self.dim:
            raise ValueError(f"Expected features with shape (_, {self.dim}), got {feats.shape}")

        if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
            self.index.train(feats)
        self.index.add(feats)

    def save_index(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        faiss.write_index(self.index, path)

    @classmethod
    def load_index(cls, path: str) -> "FaissVectorEngine":
        index = faiss.read_index(path)
        if isinstance(index, faiss.IndexIVFFlat):
            index_type: IndexType = "ivf"
        else:
            index_type = "flat"
        return cls(dim=index.d, index_type=index_type, nlist=getattr(index, "nlist", 0), index=index)

    def search(self, query_vector: np.ndarray, k: int = 10) -> SearchResult:
        if self.index.ntotal == 0:
            raise RuntimeError("Index is empty; build or load before searching.")

        queries = np.ascontiguousarray(query_vector, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        if queries.shape[1] != self.dim:
            raise ValueError(f"Query dim {queries.shape[1]} != index dim {self.dim}")

        start = time.perf_counter()
        distances, indices = self.index.search(queries, k)
        latency_ms = (time.perf_counter() - start) * 1000
        return SearchResult(distances=distances, indices=indices, latency_ms=latency_ms)

    def benchmark(self, queries: np.ndarray, k: int = 10, repeats: int = 5) -> float:
        """Return average search latency (ms) over `repeats` runs."""
        times: list[float] = []
        for _ in range(repeats):
            result = self.search(queries, k)
            times.append(result.latency_ms)
        return float(np.mean(times))


__all__ = ["FaissVectorEngine", "SearchResult", "IndexType"]
