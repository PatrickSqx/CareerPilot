"""Local embedding generation and caching for JobPilot jobs and profiles."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
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
from jobpilot.retrieval.model_contract import (
    MODEL_EXPECTED_SHA256,
    MODEL_MANIFEST_FILENAME,
    MODEL_MANIFEST_SCHEMA_VERSION,
    MODEL_REQUIRED_FILES,
    SENTENCE_EMBEDDING_DIMENSION,
    SENTENCE_EMBEDDINGS_NORMALIZED,
    SENTENCE_MODEL_ID,
    SENTENCE_MODEL_REVISION,
)
from jobpilot.utils.text import clean_text


DEFAULT_SENTENCE_MODEL = SENTENCE_MODEL_ID
TFIDF_MODEL_NAME = "local-tfidf-svd-128"
EMBEDDING_CACHE_SCHEMA_VERSION = 2
EMBEDDING_TEXT_SCHEMA_VERSION = "jobpilot-job-embedding-text-v1"


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _truthy_env(name: str) -> bool:
    return clean_text(os.getenv(name)).lower() in {"1", "true", "yes", "on"}


def _snapshot_fingerprint(snapshot_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    job_ids = [str(row.get("job_id", "")) for row in rows]
    embedding_texts = [job_embedding_text(row) for row in rows]
    return {
        "embedding_cache_schema_version": EMBEDDING_CACHE_SCHEMA_VERSION,
        "embedding_text_schema_version": EMBEDDING_TEXT_SCHEMA_VERSION,
        "snapshot_sha256": _sha256_file(snapshot_path),
        "row_count": len(rows),
        "ordered_job_ids_sha256": _sha256_json(job_ids),
        "embedding_texts_sha256": _sha256_json(embedding_texts),
    }


def _model_identity(model_name: str) -> dict[str, Any]:
    """Return and, when present, verify the immutable local model manifest."""

    model_path = Path(model_name)
    manifest_path = model_path / MODEL_MANIFEST_FILENAME
    require_manifest = _truthy_env("JOBPILOT_REQUIRE_MODEL_MANIFEST")

    identity: dict[str, Any] = {
        "model_id": model_name,
        "model_revision": "",
        "model_manifest_sha256": "",
        "model_files_verified": False,
        "sentence_transformers_version": _package_version("sentence-transformers"),
        "transformers_version": _package_version("transformers"),
        "torch_version": _package_version("torch"),
        "tokenizers_version": _package_version("tokenizers"),
    }
    if not manifest_path.is_file():
        if require_manifest:
            raise RuntimeError(
                f"Required local model manifest is missing: {manifest_path.as_posix()}"
            )
        return identity

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("Local model manifest has no file hashes.")
    if require_manifest:
        expected_contract = {
            "schema_version": MODEL_MANIFEST_SCHEMA_VERSION,
            "model_id": SENTENCE_MODEL_ID,
            "revision": SENTENCE_MODEL_REVISION,
            "embedding_dimension": SENTENCE_EMBEDDING_DIMENSION,
            "normalized": SENTENCE_EMBEDDINGS_NORMALIZED,
        }
        for key, expected_value in expected_contract.items():
            if manifest.get(key) != expected_value:
                raise RuntimeError(f"Local model manifest contract mismatch: {key}")
        if set(files) != set(MODEL_REQUIRED_FILES):
            raise RuntimeError("Local model manifest required-file set mismatch.")

    resolved_root = model_path.resolve()
    for relative_path, expected_hash in sorted(files.items()):
        candidate = (model_path / str(relative_path)).resolve()
        if not candidate.is_relative_to(resolved_root):
            raise RuntimeError(f"Unsafe model manifest path: {relative_path}")
        if not candidate.is_file():
            raise RuntimeError(f"Model file is missing: {relative_path}")
        actual_hash = _sha256_file(candidate)
        if actual_hash != str(expected_hash):
            raise RuntimeError(f"Model file hash mismatch: {relative_path}")
        independent_hash = MODEL_EXPECTED_SHA256.get(str(relative_path))
        if require_manifest and independent_hash and actual_hash != independent_hash:
            raise RuntimeError(f"Model file does not match the pinned upstream hash: {relative_path}")

    identity.update(
        {
            "model_id": str(manifest.get("model_id") or model_name),
            "model_revision": str(manifest.get("revision") or ""),
            "model_manifest_sha256": _sha256_file(manifest_path),
            "model_files_verified": True,
        }
    )
    return identity


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
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, local_files_only=True)


def _load_tfidf_artifacts(cache_dir: Path) -> tuple[Any, Any]:
    artifacts = joblib.load(cache_dir / "tfidf_svd_artifacts.joblib")
    return artifacts["vectorizer"], artifacts["svd"]


def _cache_is_valid(
    metadata: dict[str, Any],
    fingerprint: dict[str, Any],
    requested_backend: str,
    sentence_model_identity: dict[str, Any] | None,
) -> bool:
    if requested_backend != "auto" and metadata.get("backend") != requested_backend:
        return False
    for key, value in fingerprint.items():
        if metadata.get(key) != value:
            return False
    if metadata.get("backend") == "sentence-transformers":
        if not sentence_model_identity or metadata.get("model_identity") != sentence_model_identity:
            return False
    return True


def _cache_files_are_valid(
    metadata: dict[str, Any],
    embeddings_path: Path,
    job_ids_path: Path,
    expected_job_ids: list[str],
) -> bool:
    if metadata.get("cache_embeddings_sha256") != _sha256_file(embeddings_path):
        return False
    if metadata.get("cache_job_ids_sha256") != _sha256_file(job_ids_path):
        return False
    with job_ids_path.open("r", encoding="utf-8") as handle:
        cached_job_ids = json.load(handle)
    if cached_job_ids != expected_job_ids:
        return False
    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(expected_job_ids):
        return False
    if embeddings.dtype != np.dtype(np.float32) or not bool(np.isfinite(embeddings).all()):
        return False
    if metadata.get("embedding_dimension") != embeddings.shape[1]:
        return False
    if metadata.get("number_embedded") != embeddings.shape[0]:
        return False
    if metadata.get("backend") == "sentence-transformers":
        if embeddings.shape[1] != SENTENCE_EMBEDDING_DIMENSION:
            return False
        norms = np.linalg.norm(embeddings, axis=1)
        if not bool(np.allclose(norms, 1.0, atol=1e-3, rtol=1e-3)):
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
    sentence_model_identity = None
    if backend in {"auto", "sentence-transformers"}:
        sentence_model_identity = _model_identity(sentence_model_name)
    embeddings_path = cache / "job_embeddings.npy"
    job_ids_path = cache / "job_ids.json"
    metadata_path = cache / "embedding_metadata.json"
    require_prebuilt = _truthy_env("JOBPILOT_REQUIRE_PREBUILT_EMBEDDINGS")

    if not rebuild and embeddings_path.exists() and job_ids_path.exists() and metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if _cache_is_valid(metadata, fingerprint, backend, sentence_model_identity) and _cache_files_are_valid(
            metadata, embeddings_path, job_ids_path, job_ids
        ):
            embeddings = np.load(embeddings_path, allow_pickle=False)
            with job_ids_path.open("r", encoding="utf-8") as handle:
                cached_job_ids = json.load(handle)
            if cached_job_ids == job_ids:
                loaded_metadata = dict(metadata)
                loaded_metadata["cache_hit"] = True
                store = EmbeddingStore(rows, cached_job_ids, embeddings, loaded_metadata, cache)
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

    if require_prebuilt:
        raise RuntimeError("Required prebuilt embedding cache is missing or failed integrity validation.")

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

    if selected_backend == "sentence-transformers" and sentence_model_identity is None:
        sentence_model_identity = _model_identity(sentence_model_name)

    np.save(embeddings_path, embeddings.astype(np.float32))
    with job_ids_path.open("w", encoding="utf-8") as handle:
        json.dump(job_ids, handle, indent=2)

    metadata = {
        **fingerprint,
        "backend": selected_backend,
        "model_name": model_name,
        "model_identity": sentence_model_identity if selected_backend == "sentence-transformers" else None,
        "model_revision": (
            sentence_model_identity.get("model_revision", "")
            if selected_backend == "sentence-transformers" and sentence_model_identity
            else ""
        ),
        "embedding_dimension": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "number_embedded": int(embeddings.shape[0]),
        "runtime_seconds": round(time.perf_counter() - start, 3),
        "cache_hit": False,
        "cache_embeddings_sha256": _sha256_file(embeddings_path),
        "cache_job_ids_sha256": _sha256_file(job_ids_path),
        "cache_files": {
            "embeddings": embeddings_path.name,
            "job_ids": job_ids_path.name,
            "metadata": metadata_path.name,
        },
        "fallback_errors": errors,
    }

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return EmbeddingStore(rows, job_ids, embeddings.astype(np.float32), metadata, cache, transformer, vectorizer, svd)
