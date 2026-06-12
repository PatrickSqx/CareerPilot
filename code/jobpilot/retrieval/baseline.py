"""Keyword/TF-IDF retrieval baseline for Phase 2 benchmarks."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from jobpilot.retrieval.embeddings import job_embedding_text


class TfidfBaselineRetriever:
    """Simple TF-IDF cosine baseline over the same job text."""

    def __init__(self, job_rows: list[dict[str, Any]], *, max_features: int = 20000):
        self.job_rows = job_rows
        self.texts = [job_embedding_text(row) for row in job_rows]
        start = time.perf_counter()
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,
            stop_words="english",
            ngram_range=(1, 2),
            dtype=np.float32,
        )
        self.matrix = self.vectorizer.fit_transform(self.texts)
        self.fit_seconds = round(time.perf_counter() - start, 3)

    def search(self, query_text: str, top_k: int = 100) -> list[dict[str, float | int]]:
        start = time.perf_counter()
        query = self.vectorizer.transform([query_text])
        scores = (self.matrix @ query.T).toarray().ravel()
        top_k = min(top_k, len(scores))
        if top_k >= len(scores):
            indexes = np.argsort(-scores)
        else:
            indexes = np.argpartition(-scores, top_k - 1)[:top_k]
            indexes = indexes[np.argsort(-scores[indexes])]
        latency = time.perf_counter() - start
        return [
            {"index": int(index), "similarity": float(scores[index]), "latency_seconds": latency}
            for index in indexes
        ]
