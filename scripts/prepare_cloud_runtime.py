"""Prepare the immutable MiniLM model and semantic cache used by Cloud Run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

try:
    from jobpilot_model_contract import (  # type: ignore[import-not-found]
        MODEL_EXPECTED_SHA256,
        MODEL_HUB_FILES,
        MODEL_MANIFEST_FILENAME,
        MODEL_MANIFEST_SCHEMA_VERSION,
        MODEL_REQUIRED_FILES,
        SENTENCE_EMBEDDING_DIMENSION,
        SENTENCE_EMBEDDINGS_NORMALIZED,
        SENTENCE_MODEL_ID,
        SENTENCE_MODEL_REVISION,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
    from jobpilot.retrieval.model_contract import (
        MODEL_EXPECTED_SHA256,
        MODEL_HUB_FILES,
        MODEL_MANIFEST_FILENAME,
        MODEL_MANIFEST_SCHEMA_VERSION,
        MODEL_REQUIRED_FILES,
        SENTENCE_EMBEDDING_DIMENSION,
        SENTENCE_EMBEDDINGS_NORMALIZED,
        SENTENCE_MODEL_ID,
        SENTENCE_MODEL_REVISION,
    )
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(model_dir: Path, model_id: str, revision: str) -> dict[str, object]:
    """Download one pinned PyTorch model representation and record its hashes."""

    if model_id != SENTENCE_MODEL_ID or revision != SENTENCE_MODEL_REVISION:
        raise ValueError("The production MiniLM model id and revision are immutable.")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Model revision must be a full 40-character commit SHA.")
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    from huggingface_hub import snapshot_download

    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=model_dir,
        allow_patterns=list(MODEL_HUB_FILES),
        token=False,
    )
    license_source = Path("/usr/share/common-licenses/Apache-2.0")
    if not license_source.is_file():
        raise RuntimeError(f"Apache-2.0 license text is missing: {license_source}")
    shutil.copyfile(license_source, model_dir / "LICENSE")

    missing = [relative for relative in MODEL_REQUIRED_FILES if not (model_dir / relative).is_file()]
    if missing:
        raise RuntimeError(f"Downloaded model is incomplete: {', '.join(missing)}")

    files = {
        relative: _sha256_file(model_dir / relative)
        for relative in MODEL_REQUIRED_FILES
        if (model_dir / relative).is_file()
    }
    for relative, expected_hash in MODEL_EXPECTED_SHA256.items():
        if files.get(relative) != expected_hash:
            raise RuntimeError(f"Downloaded model does not match the pinned upstream hash: {relative}")
    manifest: dict[str, object] = {
        "schema_version": MODEL_MANIFEST_SCHEMA_VERSION,
        "model_id": model_id,
        "revision": revision,
        "embedding_dimension": SENTENCE_EMBEDDING_DIMENSION,
        "normalized": SENTENCE_EMBEDDINGS_NORMALIZED,
        "files": files,
    }
    manifest_path = model_dir / MODEL_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "model_id": model_id,
        "revision": revision,
        "manifest_sha256": _sha256_file(manifest_path),
        "file_count": len(files),
    }


def build_cache(project_root: Path, model_dir: Path) -> dict[str, object]:
    """Precompute the public 500-row semantic vectors inside the final image."""

    sys.path.insert(0, str(project_root / "code"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["JOBPILOT_REQUIRE_MODEL_MANIFEST"] = "1"

    from jobpilot.config import EMBEDDINGS_DIR, OFFLINE_SNAPSHOT_SAMPLE_CSV
    from jobpilot.retrieval.embeddings import build_or_load_job_embeddings

    store = build_or_load_job_embeddings(
        snapshot_path=OFFLINE_SNAPSHOT_SAMPLE_CSV,
        cache_dir=EMBEDDINGS_DIR,
        backend="sentence-transformers",
        rebuild=True,
        sentence_model_name=str(model_dir),
    )
    metadata = store.metadata
    if metadata.get("backend") != "sentence-transformers":
        raise RuntimeError("Cloud runtime cache did not use sentence-transformers.")
    if metadata.get("embedding_dimension") != SENTENCE_EMBEDDING_DIMENSION:
        raise RuntimeError(f"Unexpected MiniLM dimension: {metadata.get('embedding_dimension')}")
    if metadata.get("number_embedded") != 500:
        raise RuntimeError(f"Unexpected hosted row count: {metadata.get('number_embedded')}")
    if metadata.get("fallback_errors"):
        raise RuntimeError("Cloud runtime cache recorded an embedding fallback.")
    identity = metadata.get("model_identity") or {}
    if (
        identity.get("model_id") != SENTENCE_MODEL_ID
        or identity.get("model_revision") != SENTENCE_MODEL_REVISION
    ):
        raise RuntimeError("Cloud runtime cache used the wrong model identity.")
    matrix = store.embeddings
    if matrix.shape != (500, SENTENCE_EMBEDDING_DIMENSION) or matrix.dtype.name != "float32":
        raise RuntimeError(f"Unexpected semantic cache shape or dtype: {matrix.shape} {matrix.dtype}")
    if not bool(np.isfinite(matrix).all()):
        raise RuntimeError("Semantic cache contains non-finite values.")
    norms = np.linalg.norm(matrix, axis=1)
    if not bool(np.allclose(norms, 1.0, atol=1e-3, rtol=1e-3)):
        raise RuntimeError("Semantic cache rows are not L2-normalized.")
    return {
        "backend": metadata.get("backend"),
        "model_revision": metadata.get("model_revision"),
        "embedding_dimension": metadata.get("embedding_dimension"),
        "row_count": metadata.get("number_embedded"),
        "cache_embeddings_sha256": metadata.get("cache_embeddings_sha256"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download-model")
    download_parser.add_argument("--model-dir", type=Path, required=True)
    download_parser.add_argument("--model-id", default=SENTENCE_MODEL_ID)
    download_parser.add_argument("--revision", default=SENTENCE_MODEL_REVISION)

    cache_parser = subparsers.add_parser("build-cache")
    cache_parser.add_argument("--project-root", type=Path, required=True)
    cache_parser.add_argument("--model-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "download-model":
        receipt = download_model(args.model_dir, args.model_id, args.revision)
    else:
        receipt = build_cache(args.project_root.resolve(), args.model_dir.resolve())
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
