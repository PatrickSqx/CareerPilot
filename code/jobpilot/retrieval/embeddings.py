"""Local embedding generation and caching for JobPilot jobs and profiles."""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from jobpilot.config import EMBEDDINGS_DIR, OFFLINE_SNAPSHOT_CSV
from jobpilot.profile.profile_parser import profile_to_text
from jobpilot.utils.text import clean_text


DEFAULT_SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TFIDF_MODEL_NAME = "local-tfidf-svd-128"


def load_job_rows(snapshot_path: str | Path = OFFLINE_SNAPSHOT_CSV, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Load the offline snapshot into a list of dictionaries."""

    rows: list[dict[str, Any]] = []
    with Path(snapshot_path).open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
            if limit and len(rows) >= limit:
                break
    return rows


def job_embedding_text(row: dict[str, Any]) -> str:
    """Return the text used for retrieval."""

    text = clean_text(row.get("embedding_text"))
    if text:
        return text
    parts = [
        row.get("title", ""),
        row.get("company", ""),
        row.get("location", ""),
        row.get("extracted_skills", ""),
        row.get("description_text", ""),
    ]
    return clean_text(" | ".join(str(part) for part in parts if clean_text(part)))


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _snapshot_fingerprint(snapshot_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    stat = snapshot_path.stat()
    return {
        "snapshot_path": snapshot_path.as_posix(),
        "snapshot_size_bytes": stat.st_size,
        "snapshot_mtime": stat.st_mtime,
        "row_count": len(rows),
        "first_job_id": rows[0].get("job_id", "") if rows else "",
        "last_job_id": rows[-1].get("job_id", "") if rows else "",
    }


@dataclass
class EmbeddingStore:
    """Embeddings plus enough metadata to embed a new profile/query."""

    job_rows: list[dict[str, Any]]
    job_ids: list[str]
    embeddings: np.ndarray
    metadata: dict[str, Any]
    cache_dir: Path
    transformer: Any | None = None
    vectorizer: Any | None = None
    svd: Any | None = None

    def embed_text(self, text: str) -> np.ndarray:
        cleaned = clean_text(text)
        backend = self.metadata.get("backend")
        if backend == "sentence-transformers":
            if self.transformer is None:
                self.transformer = _load_sentence_transformer(self.metadata.get("model_name", DEFAULT_SENTENCE_MODEL))
            vector = self.transformer.encode([cleaned], normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vector, dtype=np.float32)[0]
        if backend == "tfidf-svd":
            if self.vectorizer is None or self.svd is None:
                self.vectorizer, self.svd = _load_tfidf_artifacts(self.cache_dir)
            sparse = self.vectorizer.transform([cleaned])
            dense = self.svd.transform(sparse)
            return _normalize_matrix(dense)[0]
        raise ValueError(f"Unsupported embedding backend: {backend}")

    def embed_profile(self, profile: dict[str, Any]) -> np.ndarray:
        return self.embed_text(profile_to_text(profile))


def _load_sentence_transformer(model_name: str):
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except TypeError:
        return SentenceTransformer(model_name)


def _load_tfidf_artifacts(cache_dir: Path) -> tuple[Any, Any]:
    artifacts = joblib.load(cache_dir / "tfidf_svd_artifacts.joblib")
    return artifacts["vectorizer"], artifacts["svd"]


def _cache_is_valid(metadata: dict[str, Any], fingerprint: dict[str, Any], requested_backend: str) -> bool:
    if requested_backend != "auto" and metadata.get("backend") != requested_backend:
        return False
    for key, value in fingerprint.items():
        if metadata.get(key) != value:
            return False
    return True


def _try_build_sentence_embeddings(texts: list[str], model_name: str, batch_size: int) -> tuple[np.ndarray, Any]:
    model = _load_sentence_transformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32), model


def _build_tfidf_svd_embeddings(texts: list[str], cache_dir: Path, n_components: int = 128) -> tuple[np.ndarray, Any, Any]:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        max_features=20000,
        min_df=2,
        stop_words="english",
        ngram_range=(1, 2),
        dtype=np.float32,
    )
    sparse = vectorizer.fit_transform(texts)
    components = max(2, min(n_components, sparse.shape[1] - 1, sparse.shape[0] - 1))
    svd = TruncatedSVD(n_components=components, random_state=42)
    dense = svd.fit_transform(sparse)
    embeddings = _normalize_matrix(dense)
    joblib.dump({"vectorizer": vectorizer, "svd": svd}, cache_dir / "tfidf_svd_artifacts.joblib")
    return embeddings, vectorizer, svd


def build_or_load_job_embeddings(
    snapshot_path: str | Path = OFFLINE_SNAPSHOT_CSV,
    cache_dir: str | Path = EMBEDDINGS_DIR,
    *,
    backend: str = "auto",
    rebuild: bool = False,
    sentence_model_name: str = DEFAULT_SENTENCE_MODEL,
    batch_size: int = 64,
) -> EmbeddingStore:
    """Build or load cached job embeddings.

    backend may be "auto", "sentence-transformers", or "tfidf-svd". Auto tries
    a locally cached sentence-transformer first and falls back to TF-IDF + SVD.
    """

    if backend not in {"auto", "sentence-transformers", "tfidf-svd"}:
        raise ValueError("backend must be auto, sentence-transformers, or tfidf-svd")

    snapshot = Path(snapshot_path)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    rows = load_job_rows(snapshot)
    texts = [job_embedding_text(row) for row in rows]
    job_ids = [str(row.get("job_id", "")) for row in rows]
    fingerprint = _snapshot_fingerprint(snapshot, rows)
    embeddings_path = cache / "job_embeddings.npy"
    job_ids_path = cache / "job_ids.json"
    metadata_path = cache / "embedding_metadata.json"

    if not rebuild and embeddings_path.exists() and job_ids_path.exists() and metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if _cache_is_valid(metadata, fingerprint, backend):
            embeddings = np.load(embeddings_path)
            with job_ids_path.open("r", encoding="utf-8") as handle:
                cached_job_ids = json.load(handle)
            if len(cached_job_ids) == len(rows):
                store = EmbeddingStore(rows, cached_job_ids, embeddings, metadata, cache)
                if metadata.get("backend") == "tfidf-svd":
                    store.vectorizer, store.svd = _load_tfidf_artifacts(cache)
                elif metadata.get("backend") == "sentence-transformers":
                    try:
                        store.transformer = _load_sentence_transformer(metadata.get("model_name", DEFAULT_SENTENCE_MODEL))
                    except Exception as exc:
                        if backend == "sentence-transformers":
                            raise RuntimeError(
                                "Cached sentence-transformer embeddings exist, but the local model cannot be loaded."
                            ) from exc
                        # Auto mode should remain runnable without the optional model.
                        rebuild = True
                    else:
                        return store
                if not rebuild:
                    return store

    start = time.perf_counter()
    transformer = None
    vectorizer = None
    svd = None
    errors: list[str] = []
    selected_backend = backend
    model_name = sentence_model_name

    if backend in {"auto", "sentence-transformers"}:
        try:
            embeddings, transformer = _try_build_sentence_embeddings(texts, sentence_model_name, batch_size)
            selected_backend = "sentence-transformers"
        except Exception as exc:
            errors.append(f"sentence-transformers unavailable locally: {type(exc).__name__}: {exc}")
            if backend == "sentence-transformers":
                raise RuntimeError(errors[-1]) from exc

    if selected_backend != "sentence-transformers":
        embeddings, vectorizer, svd = _build_tfidf_svd_embeddings(texts, cache)
        selected_backend = "tfidf-svd"
        model_name = TFIDF_MODEL_NAME

    metadata = {
        **fingerprint,
        "backend": selected_backend,
        "model_name": model_name,
        "embedding_dimension": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "number_embedded": int(embeddings.shape[0]),
        "runtime_seconds": round(time.perf_counter() - start, 3),
        "cache_files": {
            "embeddings": (cache / "job_embeddings.npy").as_posix(),
            "job_ids": (cache / "job_ids.json").as_posix(),
            "metadata": (cache / "embedding_metadata.json").as_posix(),
        },
        "fallback_errors": errors,
    }

    np.save(embeddings_path, embeddings.astype(np.float32))
    with job_ids_path.open("w", encoding="utf-8") as handle:
        json.dump(job_ids, handle, indent=2)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return EmbeddingStore(rows, job_ids, embeddings.astype(np.float32), metadata, cache, transformer, vectorizer, svd)
