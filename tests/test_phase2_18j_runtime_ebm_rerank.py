from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from jobpilot.evidence.learned_rerank import Phase218JLearnedReranker, load_runtime_artifact  # noqa: E402
from jobpilot.evidence.production_rerank import ProductionEvidenceReranker  # noqa: E402
from jobpilot.ranking.ranker import JobRanker  # noqa: E402


class ConstantModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, rows: Any) -> np.ndarray:
        return np.asarray([self.value for _ in range(len(rows))], dtype=np.float64)


class FakeEmbeddingStore:
    def __init__(self, tmp_path: Path, job_rows: list[dict[str, Any]]):
        self.job_rows = job_rows
        self.embeddings = np.asarray([[1.0, 0.0], [0.95, 0.0], [0.8, 0.0]][: len(job_rows)], dtype=np.float32)
        self.cache_dir = tmp_path
        self.metadata = {"backend": "test", "model_name": "test-model", "embedding_dimension": 2}

    def embed_text(self, _text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


def write_runtime_artifact(tmp_path: Path, *, score: float = 0.7) -> tuple[Path, Path]:
    manifest = tmp_path / "phase2_18j_interaction_feature_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "model_features": ["matched_skill_count", "rank_shown", "round_index"],
                "label_column": "outcome_label",
                "categorical_or_encoded_features": {"persona_id_encoded": {"test_profile": 0}},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    artifact = tmp_path / "phase2_18j_runtime_ebm_reranker.joblib"
    joblib.dump(
        {
            "model": ConstantModel(score),
            "model_class": "ConstantModel",
            "feature_columns": ["matched_skill_count", "rank_shown", "round_index"],
            "training_rows": 3,
            "feature_manifest_hash": "test",
        },
        artifact,
    )
    load_runtime_artifact.cache_clear()
    return artifact, manifest


def profile() -> dict[str, Any]:
    return {
        "profile_id": "test_profile",
        "name": "Test Profile",
        "target_roles": ["data analyst"],
        "skills": ["python", "sql"],
        "employment_types": ["full-time"],
        "location_preferences": ["remote"],
        "dealbreakers": [],
    }


def job(job_id: str, company: str) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "title": "Data Analyst",
        "company": company,
        "employer": company,
        "location": "Remote, US",
        "description_text": "Build analytics dashboards with Python and SQL.",
        "extracted_skills": "python|sql",
        "employment_type": "full-time",
        "company_type": "",
        "sponsorship_signal": "unknown",
        "salary_min": "90000",
        "salary_max": "120000",
        "source": "test",
        "link": f"https://example.com/{job_id}",
    }


def write_safe_sidecar(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    path = tmp_path / "phase2_18_job_ranking_features.csv"
    fieldnames = [
        "job_id",
        "usable_employer_size_bucket",
        "company_size_usage_policy",
        "snapshot_sponsorship_signal",
        "lca_match_scope",
        "lca_activity_label",
        "llm_overlay_available",
        "llm_reviewed_overlay_status",
        "llm_reviewed_role_family_overlay_candidate",
        "llm_reviewed_suggested_soft_action",
        "llm_downstream_use_gate",
        "llm_evidence_count",
        "llm_evidence_spans",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def reranker(tmp_path: Path, *, score: float = 0.7, sidecar_path: Path | None = None) -> Phase218JLearnedReranker:
    artifact, manifest = write_runtime_artifact(tmp_path, score=score)
    return Phase218JLearnedReranker(
        model_path=artifact,
        feature_manifest_path=manifest,
        sidecar_path=sidecar_path or (tmp_path / "missing-sidecar.csv"),
        fallback_reranker=ProductionEvidenceReranker(tmp_path / "missing-sidecar.csv"),
        mode="learned_active",
    )


def r0_fallback_reranker(tmp_path: Path, *, sidecar_path: Path | None = None) -> Phase218JLearnedReranker:
    manifest = tmp_path / "phase2_18j_interaction_feature_manifest.json"
    manifest.write_text(json.dumps({"model_features": ["matched_skill_count"]}), encoding="utf-8")
    load_runtime_artifact.cache_clear()
    return Phase218JLearnedReranker(
        model_path=tmp_path / "missing-ebm.joblib",
        feature_manifest_path=manifest,
        sidecar_path=sidecar_path or (tmp_path / "missing-sidecar.csv"),
        fallback_reranker=ProductionEvidenceReranker(tmp_path / "missing-sidecar.csv"),
        mode="learned_active",
    )


def test_learned_active_suppresses_same_rejected_job_before_display(tmp_path: Path) -> None:
    ranker = JobRanker(
        FakeEmbeddingStore(tmp_path, [job("reject-me", "Acme"), job("keep-me", "Beta")]),
        ann_backend="numpy",
        evidence_reranker=reranker(tmp_path),
    )
    result = ranker.rank(
        profile(),
        top_k=2,
        candidate_k=2,
        session_feedback_events=[{"action": "reject", "job_id": "reject-me", "job_snapshot": job("reject-me", "Acme")}],
    )

    assert [item["job_id"] for item in result["top_jobs"]] == ["keep-me"]
    metadata = result["metadata"]["evidence_rerank"]
    assert metadata["mode"] == "learned_active"
    assert metadata["session_feedback_overlay"]["same_job_reject_suppressed_count"] == 1
    assert metadata["session_feedback_overlay"]["retraining_triggered"] is False
    assert metadata["retraining_policy"]["retrain_inside_match_request"] is False


def test_session_accept_boosts_skill_neighborhood_after_ebm_score(tmp_path: Path) -> None:
    scored = [
        {
            "job_id": "candidate",
            "title": "Data Analyst",
            "company": "Beta",
            "location": "Remote, US",
            "employment_type": "full-time",
            "salary_min": "90000",
            "salary_max": "120000",
            "matched_skills": ["python", "sql"],
            "why_ranked": {"positive_drivers": ["Matches skills: python, sql"], "negative_drivers": []},
            "hard_filter_passed": True,
            "final_score": 0.4,
            "embedding_similarity": 0.9,
            "base_rank": 1,
        }
    ]
    metadata = reranker(tmp_path, score=0.5).apply(
        profile(),
        scored,
        session_feedback_events=[
            {"action": "accept", "job_snapshot": {"job_id": "accepted", "company": "Other", "matched_skills": ["python"]}}
        ],
    )

    assert scored[0]["learned_ebm_score"] == 0.5
    assert scored[0]["final_score"] > scored[0]["learned_ebm_score"]
    assert "shares skills with an accepted job" in scored[0]["feedback_adjustment_explanation"]
    assert metadata["session_feedback_overlay"]["accept_neighborhood_boost_count"] == 1


def test_safe_sidecar_missing_is_neutral_and_shadowed(tmp_path: Path) -> None:
    scored = [
        {
            "job_id": "candidate",
            "title": "Data Analyst",
            "company": "Beta",
            "location": "Remote, US",
            "employment_type": "full-time",
            "matched_skills": ["python"],
            "why_ranked": {"positive_drivers": ["Matches skills: python"], "negative_drivers": []},
            "hard_filter_passed": True,
            "final_score": 0.4,
            "embedding_similarity": 0.9,
            "base_rank": 1,
        }
    ]

    metadata = reranker(tmp_path, score=0.5).apply(profile(), scored)

    assert scored[0]["learned_safe_sidecar_features"]["company_size_known_flag"] == 0
    assert scored[0]["learned_feature_availability"]["sidecar_row_available"] is False
    assert scored[0].get("safe_sidecar_adjustment", 0) == 0
    assert metadata["sidecar_safe_features_available"] is False
    assert metadata["safe_sidecar_overlay"]["enabled"] is False
    assert metadata["comparator"] == "phase2_18j_r0_round0"
    assert metadata["old_baseline_runtime_use"] == "fallback_only"
    assert metadata["old_baseline"]["status"] == "not_run"
    assert "old_baseline_score" not in scored[0]


def test_safe_sidecar_present_is_shadow_only_with_old_artifact_columns(tmp_path: Path) -> None:
    sidecar = write_safe_sidecar(
        tmp_path,
        [
            {
                "job_id": "candidate",
                "usable_employer_size_bucket": "enterprise_10001_plus",
                "company_size_usage_policy": "usable_employer_context",
                "snapshot_sponsorship_signal": "mentions_sponsorship_or_work_auth",
                "lca_match_scope": "role_family",
                "lca_activity_label": "recent_lca_activity_high",
                "llm_overlay_available": "true",
                "llm_reviewed_overlay_status": "reviewed_candidate",
                "llm_reviewed_role_family_overlay_candidate": "data_analytics",
                "llm_reviewed_suggested_soft_action": "boost_soft",
                "llm_downstream_use_gate": "requires_separate_offline_impact_audit_before_reranking_use",
                "llm_evidence_count": "7",
                "llm_evidence_spans": '["raw span that must not be exposed"]',
            }
        ],
    )
    test_profile = {
        **profile(),
        "needs_sponsorship": True,
        "preferred_company_types": ["large_company"],
        "preferred_role_families": ["data_analytics"],
    }
    scored = [
        {
            "job_id": "candidate",
            "title": "Data Analyst",
            "company": "Beta",
            "location": "Remote, US",
            "employment_type": "full-time",
            "matched_skills": ["python"],
            "why_ranked": {"positive_drivers": ["Matches skills: python"], "negative_drivers": []},
            "hard_filter_passed": True,
            "final_score": 0.4,
            "embedding_similarity": 0.9,
            "base_rank": 1,
        }
    ]

    metadata = reranker(tmp_path, score=0.5, sidecar_path=sidecar).apply(test_profile, scored)

    features = scored[0]["learned_safe_sidecar_features"]
    assert features["company_size_known_flag"] == 1
    assert features["company_size_matches_profile"] == 1
    assert features["lca_activity_visible_bucket"] == 3
    assert features["llm_evidence_count_bucket"] == 3
    assert scored[0]["safe_sidecar_adjustment"] == 0.05
    assert scored[0]["final_score"] == 0.55
    assert "llm_evidence_spans" not in scored[0]["learned_shadow_features"]
    assert "raw span" not in json.dumps(scored[0], ensure_ascii=False)
    assert metadata["sidecar_safe_features_used_by_model"] == []
    assert set(metadata["sidecar_safe_features_shadowed_only"]) >= {"company_size_known_flag", "lca_activity_visible_bucket"}
    assert metadata["sidecar_safe_features_current_model_claim"] == "shadow_only_not_learned_by_current_artifact"
    assert metadata["safe_sidecar_overlay"]["lca_boundary"].startswith("historical employer filing activity only")
    assert metadata["safe_sidecar_overlay"]["llm_evidence_spans_used"] is False


def test_missing_ebm_uses_r0_safe_sidecar_fallback_not_old_baseline(tmp_path: Path) -> None:
    sidecar = write_safe_sidecar(
        tmp_path,
        [
            {
                "job_id": "candidate",
                "usable_employer_size_bucket": "enterprise_10001_plus",
                "company_size_usage_policy": "usable_employer_context",
                "snapshot_sponsorship_signal": "mentions_sponsorship_or_work_auth",
                "lca_match_scope": "role_family",
                "lca_activity_label": "recent_lca_activity_high",
                "llm_overlay_available": "true",
                "llm_reviewed_overlay_status": "reviewed_candidate",
                "llm_reviewed_role_family_overlay_candidate": "data_analytics",
                "llm_reviewed_suggested_soft_action": "boost_soft",
            }
        ],
    )
    test_profile = {
        **profile(),
        "needs_sponsorship": True,
        "preferred_company_types": ["large_company"],
        "preferred_role_families": ["data_analytics"],
    }
    scored = [
        {
            "job_id": "candidate",
            "title": "Data Analyst",
            "company": "Beta",
            "location": "Remote, US",
            "employment_type": "full-time",
            "matched_skills": ["python"],
            "why_ranked": {"positive_drivers": ["Matches skills: python"], "negative_drivers": []},
            "hard_filter_passed": True,
            "final_score": 0.4,
            "embedding_similarity": 0.9,
            "base_rank": 1,
        }
    ]

    metadata = r0_fallback_reranker(tmp_path, sidecar_path=sidecar).apply(test_profile, scored)

    assert metadata["status"] == "fallback_r0_safe_sidecar"
    assert metadata["policy"] == "phase2_18j_r0_safe_sidecar_fallback_v1"
    assert metadata["old_baseline_runtime_use"] == "not_used_for_r0_safe_sidecar_fallback"
    assert metadata["old_baseline"]["status"] == "not_run"
    assert scored[0]["learned_rerank_applied"] is False
    assert scored[0]["r0_safe_sidecar_fallback_applied"] is True
    assert scored[0]["r0_base_score"] == 0.4
    assert scored[0]["safe_sidecar_adjustment"] == 0.05
    assert scored[0]["final_score"] == 0.45
    assert "old_baseline_score" not in scored[0]
    assert "learned_ebm_score" not in scored[0]


def test_session_overlay_uses_prior_events_and_caps_same_company_reject(tmp_path: Path) -> None:
    scored = [
        {
            "job_id": "candidate",
            "title": "Data Analyst",
            "company": "Beta",
            "location": "Remote, US",
            "employment_type": "full-time",
            "interaction_action": "reject",
            "matched_skills": ["python"],
            "why_ranked": {"positive_drivers": ["Matches skills: python"], "negative_drivers": []},
            "hard_filter_passed": True,
            "final_score": 0.4,
            "embedding_similarity": 0.9,
            "base_rank": 1,
        }
    ]
    metadata = reranker(tmp_path, score=0.8).apply(
        profile(),
        scored,
        session_feedback_events=[{"action": "reject", "job_snapshot": {"job_id": "old", "company": "Beta"}}],
    )

    assert scored[0]["session_feedback_adjustment"] == -0.06
    assert scored[0]["final_score"] == 0.45
    assert scored[0].get("session_feedback_suppressed") is None
    assert metadata["session_feedback_overlay"]["uses_prior_session_feedback_only"] is True


def test_current_row_action_does_not_affect_current_scoring_without_prior_event(tmp_path: Path) -> None:
    scored = [
        {
            "job_id": "candidate",
            "title": "Data Analyst",
            "company": "Beta",
            "location": "Remote, US",
            "employment_type": "full-time",
            "interaction_action": "reject",
            "matched_skills": ["python"],
            "why_ranked": {"positive_drivers": ["Matches skills: python"], "negative_drivers": []},
            "hard_filter_passed": True,
            "final_score": 0.4,
            "embedding_similarity": 0.9,
            "base_rank": 1,
        }
    ]

    reranker(tmp_path, score=0.5).apply(profile(), scored, session_feedback_events=[])

    assert scored[0].get("session_feedback_suppressed") is None
    assert scored[0].get("session_feedback_adjustment") is None
    assert scored[0]["final_score"] == 0.5


def test_hard_filtered_candidate_is_not_resurrected_by_sidecar_overlay(tmp_path: Path) -> None:
    sidecar = write_safe_sidecar(
        tmp_path,
        [
            {
                "job_id": "blocked",
                "usable_employer_size_bucket": "enterprise_10001_plus",
                "company_size_usage_policy": "usable_employer_context",
                "snapshot_sponsorship_signal": "mentions_sponsorship_or_work_auth",
                "lca_match_scope": "role_family",
                "lca_activity_label": "recent_lca_activity_high",
            }
        ],
    )
    scored = [
        {
            "job_id": "blocked",
            "title": "Data Analyst",
            "company": "Beta",
            "location": "Remote, US",
            "employment_type": "full-time",
            "matched_skills": ["python"],
            "why_ranked": {"positive_drivers": ["Matches skills: python"], "negative_drivers": []},
            "hard_filter_passed": False,
            "hard_filter_violations": ["no_sponsorship"],
            "final_score": 0.99,
            "embedding_similarity": 0.99,
            "base_rank": 1,
        }
    ]

    reranker(tmp_path, score=0.9, sidecar_path=sidecar).apply(profile(), scored)

    assert scored[0]["learned_rerank_applied"] is False
    assert scored[0]["ranking_sort_group"] == 0
    assert "learned_ebm_score" not in scored[0]
