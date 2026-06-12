"""Build the Phase 2.18J runtime EBM reranker artifact.

This script consumes only the frozen Phase 2.18J interaction feature table and
feature manifest. It does not touch retrieval, embeddings, Phase 1 ingestion,
or any live API path.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from jobpilot.config import PROCESSED_DATA_DIR  # noqa: E402
from jobpilot.utils.text import stable_hash  # noqa: E402


DEFAULT_FEATURES = PROCESSED_DATA_DIR / "phase2_18j_interaction_features.csv"
DEFAULT_FEATURE_MANIFEST = PROCESSED_DATA_DIR / "phase2_18j_interaction_feature_manifest.json"
DEFAULT_MODEL_REPORT = PROCESSED_DATA_DIR / "phase2_18j_ebm_model_report.json"
DEFAULT_OUTPUT = PROCESSED_DATA_DIR / "phase2_18j_runtime_ebm_reranker.joblib"
DEFAULT_MANIFEST_OUTPUT = PROCESSED_DATA_DIR / "phase2_18j_runtime_ebm_reranker_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 2.18J runtime EBM reranker artifact.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--feature-manifest", type=Path, default=DEFAULT_FEATURE_MANIFEST)
    parser.add_argument("--model-report", type=Path, default=DEFAULT_MODEL_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def file_hash(path: Path) -> str:
    return stable_hash(path.read_text(encoding="utf-8", errors="replace"))


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    feature_manifest = read_json(args.feature_manifest)
    model_report = read_json(args.model_report) if args.model_report.exists() else {}
    feature_columns = [str(column) for column in feature_manifest["model_features"]]
    label_column = str(feature_manifest["label_column"])
    rows = read_rows(args.features)
    if not rows:
        raise RuntimeError(f"No feature rows found: {args.features}")

    try:
        import interpret  # type: ignore
        from interpret.glassbox import ExplainableBoostingRegressor  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"interpret EBM dependency is unavailable: {type(exc).__name__}: {exc}") from exc

    x = np.asarray([[float(row.get(column) or 0.0) for column in feature_columns] for row in rows], dtype=np.float64)
    y = np.asarray([float(row.get(label_column) or 0.0) for row in rows], dtype=np.float64)
    model = ExplainableBoostingRegressor(random_state=42, interactions=0, n_jobs=1)
    model.fit(x, y)

    artifact = {
        "phase": "phase2_18J_runtime_ebm_reranker",
        "model": model,
        "model_class": "ExplainableBoostingRegressor",
        "feature_columns": feature_columns,
        "label_column": label_column,
        "training_rows": len(rows),
        "feature_manifest": args.feature_manifest.relative_to(PROJECT_ROOT).as_posix(),
        "feature_manifest_hash": file_hash(args.feature_manifest),
        "features_source": args.features.relative_to(PROJECT_ROOT).as_posix(),
        "features_hash": file_hash(args.features),
        "source_model_report": args.model_report.relative_to(PROJECT_ROOT).as_posix() if args.model_report.exists() else "",
        "source_model_report_status": model_report.get("status", ""),
        "source_model_report_policy": model_report.get("phase_id", ""),
        "candidate_reservoir_size": 200,
        "display_top_k_policy": "current UI Top N",
        "runtime_default_mode": "learned_active",
        "label_warning": "Trained from Phase 2.18J simulated feedback, not real user feedback or gold labels.",
        "forbidden_inputs_excluded": feature_manifest.get("forbidden_same_row_predictors_excluded", []),
        "interpret_version": getattr(interpret, "__version__", "unknown"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    manifest = {
        "phase": "phase2_18J_runtime_ebm_reranker",
        "status": "completed",
        "runtime_seconds": round(time.perf_counter() - start, 4),
        "artifact": args.output.relative_to(PROJECT_ROOT).as_posix(),
        "artifact_bytes": args.output.stat().st_size,
        "model_class": artifact["model_class"],
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "training_rows": len(rows),
        "candidate_reservoir_size": 200,
        "display_top_k_policy": "current UI Top N",
        "default_runtime_mode": "learned_active",
        "fallback_policy": "old production/rule rerank if artifact cannot load",
        "boundaries": {
            "ann_query_modified": False,
            "embeddings_rebuilt": False,
            "embedding_text_modified": False,
            "phase1_ingestion_modified": False,
            "hard_filters_bypassed": False,
            "live_apis_called": False,
            "simulated_labels_treated_as_real_feedback_or_gold": False,
        },
        "feature_manifest": artifact["feature_manifest"],
        "feature_manifest_hash": artifact["feature_manifest_hash"],
        "features_source": artifact["features_source"],
        "features_hash": artifact["features_hash"],
        "label_warning": artifact["label_warning"],
        "interpret_version": artifact["interpret_version"],
    }
    args.manifest_output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "artifact": manifest["artifact"], "training_rows": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
