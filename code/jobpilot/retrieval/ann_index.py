"""Unified nearest-neighbor retrieval interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np


class ANNRetriever:
    """ANN-style retriever with FAISS, sklearn, and numpy fallbacks."""

    def __init__(self, embeddings: np.ndarray, *, backend: str = "auto", cache_dir: str | Path | None = None):
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.backend = "numpy"
        self.index: Any | None = None
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if backend in {"auto", "faiss"}:
            try:
                import faiss  # type: ignore

                self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
                self.index.add(self.embeddings)
                self.backend = "faiss"
                if self.cache_dir:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    faiss.write_index(self.index, str(self.cache_dir / "faiss.index"))
                return
            except Exception:
                if backend == "faiss":
                    raise

        if backend in {"auto", "sklearn"}:
            try:
                from sklearn.neighbors import NearestNeighbors

                sklearn_cache = self.cache_dir / "sklearn_ann_index.joblib" if self.cache_dir else None
                if sklearn_cache and sklearn_cache.exists():
                    cached = joblib.load(sklearn_cache)
                    if tuple(cached.get("shape", ())) == tuple(self.embeddings.shape):
                        self.index = cached["index"]
                        self.backend = "sklearn"
                        return
                self.index = NearestNeighbors(metric="cosine", algorithm="brute")
                self.index.fit(self.embeddings)
                if sklearn_cache:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    joblib.dump({"shape": tuple(self.embeddings.shape), "index": self.index}, sklearn_cache)
                self.backend = "sklearn"
                return
            except Exception:
                if backend == "sklearn":
                    raise

    def search(self, query_embedding: np.ndarray, top_k: int = 100) -> list[dict[str, float | int]]:
        """Return nearest row indexes with similarity scores."""

        if top_k <= 0:
            return []
        top_k = min(top_k, len(self.embeddings))
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)

        if self.backend == "faiss":
            scores, indexes = self.index.search(query, top_k)  # type: ignore[union-attr]
            return [
                {"index": int(index), "similarity": float(score)}
                for index, score in zip(indexes[0], scores[0])
                if int(index) >= 0
            ]

        if self.backend == "sklearn":
            distances, indexes = self.index.kneighbors(query, n_neighbors=top_k)  # type: ignore[union-attr]
            return [
                {"index": int(index), "similarity": float(1.0 - distance)}
                for index, distance in zip(indexes[0], distances[0])
            ]

        scores = self.embeddings @ query[0]
        if top_k >= len(scores):
            indexes = np.argsort(-scores)
        else:
            indexes = np.argpartition(-scores, top_k - 1)[:top_k]
            indexes = indexes[np.argsort(-scores[indexes])]
        return [{"index": int(index), "similarity": float(scores[index])} for index in indexes]
