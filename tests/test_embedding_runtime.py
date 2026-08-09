from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from app.services import matching_service
from jobpilot.retrieval import embeddings as embedding_module


def _write_snapshot(path: Path, descriptions: list[str]) -> None:
    rows = ["job_id,title,description_text"]
    rows.extend(f"job-{index},Role {index},{description}" for index, description in enumerate(descriptions, start=1))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_snapshot_fingerprint_is_content_based(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "nested" / "second.csv"
    second.parent.mkdir()
    _write_snapshot(first, ["alpha", "middle", "omega"])
    second.write_bytes(first.read_bytes())
    os.utime(second, (first.stat().st_atime + 20, first.stat().st_mtime + 20))

    first_rows = embedding_module.load_job_rows(first)
    second_rows = embedding_module.load_job_rows(second)
    assert embedding_module._snapshot_fingerprint(first, first_rows) == embedding_module._snapshot_fingerprint(
        second, second_rows
    )

    _write_snapshot(second, ["alpha", "changed", "omega"])
    changed = embedding_module._snapshot_fingerprint(second, embedding_module.load_job_rows(second))
    original = embedding_module._snapshot_fingerprint(first, first_rows)
    assert changed["snapshot_sha256"] != original["snapshot_sha256"]
    assert changed["embedding_texts_sha256"] != original["embedding_texts_sha256"]


def test_required_model_manifest_verifies_and_detects_tampering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_file = model_dir / "config.json"
    model_file.write_text('{"hidden_size": 384}\n', encoding="utf-8")
    file_hash = hashlib.sha256(model_file.read_bytes()).hexdigest()
    manifest = {
        "schema_version": embedding_module.MODEL_MANIFEST_SCHEMA_VERSION,
        "model_id": embedding_module.SENTENCE_MODEL_ID,
        "revision": embedding_module.SENTENCE_MODEL_REVISION,
        "embedding_dimension": embedding_module.SENTENCE_EMBEDDING_DIMENSION,
        "normalized": embedding_module.SENTENCE_EMBEDDINGS_NORMALIZED,
        "files": {"config.json": file_hash},
    }
    (model_dir / embedding_module.MODEL_MANIFEST_FILENAME).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    monkeypatch.setenv("JOBPILOT_REQUIRE_MODEL_MANIFEST", "1")
    monkeypatch.setattr(embedding_module, "MODEL_REQUIRED_FILES", ("config.json",))
    monkeypatch.setattr(embedding_module, "MODEL_EXPECTED_SHA256", {})

    identity = embedding_module._model_identity(str(model_dir))
    assert identity["model_files_verified"] is True
    assert identity["model_revision"] == embedding_module.SENTENCE_MODEL_REVISION

    model_file.write_text('{"hidden_size": 768}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        embedding_module._model_identity(str(model_dir))


def test_cache_validation_binds_sentence_model_identity() -> None:
    fingerprint = {"snapshot_sha256": "snapshot"}
    model_identity = {"model_revision": "a" * 40}
    metadata = {
        **fingerprint,
        "backend": "sentence-transformers",
        "model_identity": model_identity,
    }
    assert embedding_module._cache_is_valid(
        metadata, fingerprint, "sentence-transformers", model_identity
    )
    assert not embedding_module._cache_is_valid(
        metadata,
        fingerprint,
        "sentence-transformers",
        {"model_revision": "b" * 40},
    )


def test_cache_file_validation_checks_job_order_and_vectors(tmp_path: Path) -> None:
    embeddings_path = tmp_path / "job_embeddings.npy"
    job_ids_path = tmp_path / "job_ids.json"
    matrix = np.zeros((2, 384), dtype=np.float32)
    matrix[0, 0] = 1.0
    matrix[1, 1] = 1.0
    np.save(embeddings_path, matrix)
    job_ids_path.write_text(json.dumps(["job-1", "job-2"]), encoding="utf-8")
    metadata = {
        "backend": "sentence-transformers",
        "embedding_dimension": 384,
        "number_embedded": 2,
        "cache_embeddings_sha256": embedding_module._sha256_file(embeddings_path),
        "cache_job_ids_sha256": embedding_module._sha256_file(job_ids_path),
    }
    assert embedding_module._cache_files_are_valid(
        metadata, embeddings_path, job_ids_path, ["job-1", "job-2"]
    )

    job_ids_path.write_text(json.dumps(["job-2", "job-1"]), encoding="utf-8")
    metadata["cache_job_ids_sha256"] = embedding_module._sha256_file(job_ids_path)
    assert not embedding_module._cache_files_are_valid(
        metadata, embeddings_path, job_ids_path, ["job-1", "job-2"]
    )

    job_ids_path.write_text(json.dumps(["job-1", "job-2"]), encoding="utf-8")
    metadata["cache_job_ids_sha256"] = embedding_module._sha256_file(job_ids_path)
    matrix[0, 0] = np.nan
    np.save(embeddings_path, matrix)
    metadata["cache_embeddings_sha256"] = embedding_module._sha256_file(embeddings_path)
    assert not embedding_module._cache_files_are_valid(
        metadata, embeddings_path, job_ids_path, ["job-1", "job-2"]
    )


def test_required_prebuilt_cache_fails_instead_of_rebuilding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "jobs.csv"
    _write_snapshot(snapshot, ["alpha", "beta", "gamma"])
    monkeypatch.setenv("JOBPILOT_REQUIRE_PREBUILT_EMBEDDINGS", "1")
    with pytest.raises(RuntimeError, match="Required prebuilt embedding cache"):
        embedding_module.build_or_load_job_embeddings(
            snapshot_path=snapshot,
            cache_dir=tmp_path / "missing-cache",
            backend="tfidf-svd",
        )


def test_runtime_backend_and_model_are_shared_by_warmup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snapshot = tmp_path / "jobs.csv"
    snapshot.write_text("job_id,title\njob-1,Analyst\n", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    model_dir = tmp_path / "model"
    monkeypatch.setenv("JOBPILOT_EMBEDDING_BACKEND", "sentence-transformers")
    monkeypatch.setenv("JOBPILOT_SENTENCE_MODEL", str(model_dir))
    monkeypatch.setattr(matching_service, "resolve_snapshot_path", lambda: (snapshot, []))
    monkeypatch.setattr(matching_service, "EMBEDDINGS_DIR", cache_dir)
    calls: list[tuple[str, str, str, str]] = []

    def fake_ranker(snapshot_path: str, cache_path: str, backend: str, model_name: str):
        calls.append((snapshot_path, cache_path, backend, model_name))
        metadata = {
            "backend": backend,
            "model_name": model_name,
            "model_revision": "a" * 40,
            "embedding_dimension": 384,
            "number_embedded": 1,
            "cache_hit": True,
            "fallback_errors": [],
        }
        probe = np.zeros(384, dtype=np.float32)
        probe[0] = 1.0
        return SimpleNamespace(store=SimpleNamespace(metadata=metadata, embed_text=lambda _text: probe))

    monkeypatch.setattr(matching_service, "_ranker_for", fake_ranker)
    status = matching_service.warm_ranker_runtime()
    assert calls == [(str(snapshot.resolve()), str(cache_dir.resolve()), "sentence-transformers", str(model_dir))]
    assert status["embedding_backend"] == "sentence-transformers"
    assert status["embedding_cache_hit"] is True
    assert status["embedding_probe_warmed"] is True
    assert status["fallback_error_count"] == 0


def test_invalid_runtime_backend_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOBPILOT_EMBEDDING_BACKEND", "typo")
    with pytest.raises(ValueError, match="JOBPILOT_EMBEDDING_BACKEND"):
        matching_service.embedding_runtime_settings()
