from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from jobpilot.evidence.production_rerank import ProductionEvidenceReranker  # noqa: E402
from jobpilot.ranking.ranker import JobRanker  # noqa: E402


class FakeEmbeddingStore:
    def __init__(self, tmp_path: Path, job_rows: list[dict[str, Any]]):
        self.job_rows = job_rows
        self.embeddings = np.array(
            [[max(0.1, 1.0 - index * 0.01), 0.0] for index in range(len(job_rows))],
            dtype=np.float32,
        )
        self.cache_dir = tmp_path
        self.metadata = {"backend": "test", "model_name": "test-model", "embedding_dimension": 2}

    def embed_text(self, _text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


def profile(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "profile_id": "display_diversity",
        "target_roles": ["data scientist"],
        "skills": ["python", "sql"],
        "location_preferences": [],
        "employment_types": ["full-time"],
        "dealbreakers": [],
    }
    payload.update(overrides)
    return payload


def job(
    job_id: str,
    *,
    title: str = "Data Scientist",
    company: str | None = None,
    location: str = "Atlanta, GA, US",
    description: str = "Build machine learning and analytics systems with Python and SQL.",
    salary_min: str = "100000",
    salary_max: str = "130000",
    link: str | None = None,
) -> dict[str, Any]:
    company_name = company or f"{job_id} Corp"
    return {
        "job_id": job_id,
        "dedup_key": f"dedup-{job_id}",
        "title": title,
        "company": company_name,
        "employer": company_name,
        "location": location,
        "description_text": description,
        "extracted_skills": "python|sql",
        "employment_type": "full-time",
        "company_type": "",
        "sponsorship_signal": "unknown",
        "salary_min": salary_min,
        "salary_max": salary_max,
        "source": "test",
        "link": link if link is not None else f"https://example.com/{job_id}",
    }


def rank(tmp_path: Path, jobs: list[dict[str, Any]], *, top_k: int, test_profile: dict[str, Any] | None = None):
    store = FakeEmbeddingStore(tmp_path, jobs)
    ranker = JobRanker(
        store,
        ann_backend="numpy",
        evidence_reranker=ProductionEvidenceReranker(tmp_path / "missing-sidecar.csv"),
    )
    return ranker.rank(test_profile or profile(), top_k=top_k, candidate_k=len(jobs))


def test_near_duplicate_roles_are_suppressed_and_backfilled(tmp_path: Path) -> None:
    jobs = [
        job("same_1", company="The Judge Group"),
        job("same_2", company="The Judge Group"),
        job("same_3", company="The Judge Group", title="Data Scientist II"),
        job("unique_1", company="Northstar Analytics"),
        job("unique_2", company="Summit Data"),
        job("unique_3", company="Blue River Labs"),
    ]

    result = rank(tmp_path, jobs, top_k=4)
    top_jobs = result["top_jobs"]
    metadata = result["metadata"]["display_diversification"]

    assert len(top_jobs) == 4
    assert sum(item["company"] == "The Judge Group" for item in top_jobs) == 1
    assert {item["job_id"] for item in top_jobs} >= {"unique_1", "unique_2", "unique_3"}
    assert metadata["suppressed_near_duplicate_count"] >= 2
    assert metadata["backfilled_job_count"] >= 2


def test_more_complete_duplicate_posting_wins_when_scores_are_close(tmp_path: Path) -> None:
    incomplete = job(
        "incomplete",
        company="SameCo",
        description="Python SQL.",
        salary_min="",
        salary_max="",
        link="",
    )
    complete = job(
        "complete",
        company="SameCo",
        description=(
            "Build machine learning and analytics systems with Python and SQL. "
            "Partner with product teams, deploy models, document requirements, and maintain data quality. "
            "This posting includes enough detail for a candidate to assess the role before applying."
        ),
        salary_min="110000",
        salary_max="140000",
        link="https://example.com/complete",
    )

    result = rank(tmp_path, [incomplete, complete], top_k=1)
    top = result["top_jobs"][0]
    metadata = result["metadata"]["display_diversification"]

    assert top["job_id"] == "complete"
    assert top["display_completeness_score"] > incomplete.get("display_completeness_score", 0)
    assert metadata["replacement_count"] == 1
    assert "incomplete" in metadata["suppressed_job_ids_sample"]


def test_display_backfill_does_not_resurrect_hard_filtered_jobs(tmp_path: Path) -> None:
    jobs = [
        job("blocked", company="BlockedCo", description="This blocked role uses Python and SQL."),
        job("same_1", company="The Judge Group"),
        job("same_2", company="The Judge Group"),
        job("unique", company="ClearPath Data"),
    ]

    result = rank(tmp_path, jobs, top_k=2, test_profile=profile(dealbreakers=["blocked"]))
    top_ids = [item["job_id"] for item in result["top_jobs"]]

    assert top_ids == ["same_1", "unique"]
    assert "blocked" not in top_ids
    assert result["metadata"]["filtered_out"] == 1
    assert result["metadata"]["display_diversification"]["returned_jobs"] == 2
