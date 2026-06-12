"""Offline/live/hybrid candidate-pool helpers for Phase 2 ranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from jobpilot.config import EMBEDDINGS_DIR, OFFLINE_SNAPSHOT_CSV
from jobpilot.ingestion.dedup import add_dedup_fields
from jobpilot.retrieval.embeddings import (
    EmbeddingStore,
    build_or_load_job_embeddings,
    job_embedding_text,
)


@dataclass
class CandidatePool:
    mode_requested: str
    mode_used: str
    store: EmbeddingStore
    offline_rows: list[dict[str, Any]]
    live_rows: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


def _dedupe_live_rows(
    live_rows: list[dict[str, Any]],
    offline_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    offline_keys = {str(row.get("dedup_key", "")) for row in offline_rows if row.get("dedup_key")}
    seen_live: set[str] = set()
    unique: list[dict[str, Any]] = []
    duplicate_offline = 0
    duplicate_live = 0
    for row in live_rows:
        normalized = add_dedup_fields(row)
        key = str(normalized.get("dedup_key", ""))
        if key and key in offline_keys:
            duplicate_offline += 1
            continue
        if key and key in seen_live:
            duplicate_live += 1
            continue
        if key:
            seen_live.add(key)
        unique.append(normalized)
    return unique, {
        "duplicates_against_offline": duplicate_offline,
        "duplicates_within_live": duplicate_live,
        "live_records_retained": len(unique),
    }


def _embed_rows(base_store: EmbeddingStore, rows: list[dict[str, Any]]) -> np.ndarray:
    if not rows:
        return np.zeros((0, int(base_store.embeddings.shape[1])), dtype=np.float32)
    vectors = [base_store.embed_text(job_embedding_text(row)) for row in rows]
    return np.asarray(vectors, dtype=np.float32)


def _store_from_rows(
    base_store: EmbeddingStore,
    rows: list[dict[str, Any]],
    embeddings: np.ndarray,
    *,
    mode_used: str,
) -> EmbeddingStore:
    metadata = dict(base_store.metadata)
    metadata.update(
        {
            "candidate_pool_mode": mode_used,
            "number_embedded": int(embeddings.shape[0]),
            "row_count": len(rows),
        }
    )
    return EmbeddingStore(
        job_rows=rows,
        job_ids=[str(row.get("job_id", "")) for row in rows],
        embeddings=embeddings.astype(np.float32),
        metadata=metadata,
        cache_dir=base_store.cache_dir,
        transformer=base_store.transformer,
        vectorizer=base_store.vectorizer,
        svd=base_store.svd,
    )


def build_candidate_store(
    *,
    mode: str = "offline",
    live_rows: list[dict[str, Any]] | None = None,
    snapshot_path: str | Path = OFFLINE_SNAPSHOT_CSV,
    cache_dir: str | Path = EMBEDDINGS_DIR,
    embedding_backend: str = "auto",
    fallback_to_offline: bool = True,
) -> CandidatePool:
    """Build an EmbeddingStore for offline, live, or hybrid ranking."""

    mode = mode.lower()
    if mode not in {"offline", "live", "hybrid"}:
        raise ValueError("mode must be offline, live, or hybrid")

    base_store = build_or_load_job_embeddings(snapshot_path=snapshot_path, cache_dir=cache_dir, backend=embedding_backend)
    offline_rows = base_store.job_rows
    unique_live, dedup_metadata = _dedupe_live_rows(live_rows or [], offline_rows)
    warnings: list[str] = []

    if mode == "offline":
        mode_used = "offline"
        store = base_store
        selected_live: list[dict[str, Any]] = []
    elif mode == "live":
        if not unique_live and fallback_to_offline:
            warnings.append("No live records available; using offline fallback.")
            mode_used = "offline_fallback"
            store = base_store
            selected_live = []
        else:
            live_embeddings = _embed_rows(base_store, unique_live)
            store = _store_from_rows(base_store, unique_live, live_embeddings, mode_used="live")
            mode_used = "live"
            selected_live = unique_live
    else:
        if not unique_live and fallback_to_offline:
            warnings.append("No live records available for hybrid mode; using offline fallback.")
            mode_used = "offline_fallback"
            store = base_store
            selected_live = []
        else:
            live_embeddings = _embed_rows(base_store, unique_live)
            combined_rows = [*offline_rows, *unique_live]
            combined_embeddings = np.vstack([base_store.embeddings.astype(np.float32), live_embeddings])
            store = _store_from_rows(base_store, combined_rows, combined_embeddings, mode_used="hybrid")
            mode_used = "hybrid"
            selected_live = unique_live

    metadata = {
        "mode_requested": mode,
        "mode_used": mode_used,
        "offline_rows": len(offline_rows),
        "live_rows_input": len(live_rows or []),
        **dedup_metadata,
        "candidate_rows": len(store.job_rows),
        "warnings": warnings,
    }
    return CandidatePool(mode, mode_used, store, offline_rows, selected_live, metadata)


def build_live_candidate_pool(**kwargs: Any) -> CandidatePool:
    """Alias kept for UI-facing call sites."""

    return build_candidate_store(**kwargs)
