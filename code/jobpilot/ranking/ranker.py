"""High-level multi-stage ranking pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from jobpilot.config import EMBEDDINGS_DIR, OFFLINE_SNAPSHOT_CSV
from jobpilot.evidence.learned_rerank import Phase218JLearnedReranker
from jobpilot.profile.profile_parser import profile_to_text
from jobpilot.ranking.application_strategy import apply_application_strategy
from jobpilot.ranking.company_signals import detect_company_signals
from jobpilot.ranking.explanations import build_why_ranked
from jobpilot.ranking.filters import apply_hard_filters
from jobpilot.ranking.location_signals import detect_location_signals
from jobpilot.ranking.role_signals import matches_required_role_family, role_family_match_details
from jobpilot.ranking.scoring import compute_score
from jobpilot.retrieval.ann_index import ANNRetriever
from jobpilot.retrieval.embeddings import EmbeddingStore, build_or_load_job_embeddings
from jobpilot.utils.text import clean_text, normalize_for_key


DISPLAY_TITLE_STOPWORDS = {
    "i",
    "ii",
    "iii",
    "iv",
    "jr",
    "junior",
    "sr",
    "senior",
    "staff",
    "principal",
    "lead",
    "remote",
    "hybrid",
    "onsite",
}
DISPLAY_REPLACEMENT_SCORE_TOLERANCE = 0.03
DISPLAY_REPLACEMENT_MAX_SCORE_GAP = 0.08
DISPLAY_REPLACEMENT_COMPLETENESS_MARGIN = 2


class JobRanker:
    """Candidate generation, hard filtering, scoring, and explanation."""

    def __init__(
        self,
        embedding_store: EmbeddingStore,
        *,
        ann_backend: str = "auto",
        evidence_reranker: Any | None = None,
    ):
        self.store = embedding_store
        self.retriever = ANNRetriever(embedding_store.embeddings, backend=ann_backend, cache_dir=embedding_store.cache_dir)
        self.evidence_reranker = evidence_reranker or Phase218JLearnedReranker()

    def _candidate_sizes(self, candidate_k: int, *, strict_role_family: bool) -> list[int]:
        dataset_size = len(self.store.job_rows)
        initial = min(max(candidate_k, 1), dataset_size)
        if not strict_role_family:
            return [initial]
        sizes = [initial, 3000, 5000, 10000]
        unique: list[int] = []
        for size in sizes:
            bounded = min(max(size, 1), dataset_size)
            if bounded not in unique:
                unique.append(bounded)
        return unique

    def _output_key(self, job: dict[str, Any]) -> tuple[str, ...]:
        dedup_key = clean_text(job.get("dedup_key")).lower()
        if dedup_key:
            return ("dedup_key", dedup_key)
        title = clean_text(job.get("title")).lower()
        company = clean_text(job.get("company") or job.get("employer")).lower()
        location = clean_text(job.get("location")).lower()
        description_hash = clean_text(job.get("description_hash")).lower()
        if description_hash:
            return ("description_hash", title, company, location, description_hash)
        link = clean_text(job.get("link")).lower()
        return ("link", title, company, location, link)

    def _display_title_key(self, job: dict[str, Any]) -> str:
        title = normalize_for_key(job.get("title"))
        tokens = [token for token in title.split() if token not in DISPLAY_TITLE_STOPWORDS and len(token) >= 2]
        return " ".join(tokens)

    def _display_location_key(self, job: dict[str, Any]) -> str:
        location = normalize_for_key(job.get("location"))
        if "remote" in location:
            return "remote"
        parts = [part.strip() for part in location.split() if part.strip()]
        return " ".join(parts[:4])

    def _display_cluster_key(self, job: dict[str, Any]) -> tuple[str, str, str, str]:
        company = normalize_for_key(job.get("company") or job.get("employer"))
        title = self._display_title_key(job)
        if not company or not title:
            return ("unique", clean_text(job.get("job_id")) or clean_text(job.get("link")), "", "")
        return ("company_title", company, title, "")

    def _display_completeness_score(self, job: dict[str, Any]) -> int:
        score = 0
        description = clean_text(job.get("description_text"))
        if clean_text(job.get("link")):
            score += 1
        if clean_text(job.get("salary_raw")) or clean_text(job.get("salary_min")) or clean_text(job.get("salary_max")):
            score += 2
        if clean_text(job.get("employment_type")):
            score += 1
        if clean_text(job.get("years_required")):
            score += 1
        if clean_text(job.get("sponsorship_signal")) and clean_text(job.get("sponsorship_signal")).lower() != "unknown":
            score += 1
        if len(description) >= 300:
            score += 2
        elif len(description) >= 100:
            score += 1
        if job.get("matched_skills"):
            score += 1
        if clean_text(job.get("source")) or clean_text(job.get("raw_source")):
            score += 1
        return score

    def _should_replace_display_representative(self, candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
        candidate_completeness = int(candidate.get("display_completeness_score") or 0)
        incumbent_completeness = int(incumbent.get("display_completeness_score") or 0)
        if candidate_completeness <= incumbent_completeness:
            return False
        score_gap = float(incumbent.get("final_score") or 0.0) - float(candidate.get("final_score") or 0.0)
        if score_gap > DISPLAY_REPLACEMENT_MAX_SCORE_GAP:
            return False
        if score_gap <= DISPLAY_REPLACEMENT_SCORE_TOLERANCE:
            return True
        return candidate_completeness >= incumbent_completeness + DISPLAY_REPLACEMENT_COMPLETENESS_MARGIN

    def _select_display_top_jobs(
        self,
        scored: list[dict[str, Any]],
        *,
        top_k: int,
        include_filtered: bool,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if top_k <= 0:
            return [], {
                "enabled": True,
                "requested_top_k": top_k,
                "returned_jobs": 0,
                "suppressed_near_duplicate_count": 0,
                "backfilled_job_count": 0,
                "replacement_count": 0,
                "clusters_represented": 0,
                "suppressed_job_ids_sample": [],
            }

        raw_top_ids = [clean_text(item.get("job_id")) for item in scored[:top_k]]
        raw_top_id_set = {job_id for job_id in raw_top_ids if job_id}
        selected: list[dict[str, Any]] = []
        cluster_to_position: dict[tuple[str, str, str, str], int] = {}
        suppressed_job_ids: list[str] = []
        replacement_count = 0

        for item in scored:
            if item.get("session_feedback_suppressed"):
                suppressed_job_ids.append(clean_text(item.get("job_id")))
                continue
            if not include_filtered and not item.get("hard_filter_passed", True):
                continue

            cluster_key = self._display_cluster_key(item)
            item["display_cluster_key"] = "|".join(part for part in cluster_key if part)
            item["display_completeness_score"] = self._display_completeness_score(item)

            existing_position = cluster_to_position.get(cluster_key)
            if existing_position is None:
                if len(selected) >= top_k:
                    continue
                item["display_diversity_status"] = "primary"
                selected.append(item)
                cluster_to_position[cluster_key] = len(selected) - 1
                continue

            incumbent = selected[existing_position]
            if self._should_replace_display_representative(item, incumbent):
                incumbent["display_diversity_status"] = "same_company_near_duplicate_suppressed"
                suppressed_job_ids.append(clean_text(incumbent.get("job_id")))
                item["display_diversity_status"] = "primary"
                selected[existing_position] = item
                replacement_count += 1
            else:
                item["display_diversity_status"] = "same_company_near_duplicate_suppressed"
                suppressed_job_ids.append(clean_text(item.get("job_id")))

        backfilled_job_count = sum(1 for item in selected if clean_text(item.get("job_id")) not in raw_top_id_set)
        return selected, {
            "enabled": True,
            "requested_top_k": top_k,
            "returned_jobs": len(selected),
            "suppressed_near_duplicate_count": len([job_id for job_id in suppressed_job_ids if job_id]),
            "backfilled_job_count": backfilled_job_count,
            "replacement_count": replacement_count,
            "clusters_represented": len(cluster_to_position),
            "suppressed_job_ids_sample": [job_id for job_id in suppressed_job_ids if job_id][:10],
        }

    def _build_scored_job(
        self,
        profile: dict[str, Any],
        job: dict[str, Any],
        *,
        similarity: float,
        filter_result: Any,
    ) -> dict[str, Any]:
        score_payload = compute_score(profile, job, embedding_similarity=similarity, filter_result=filter_result)
        why_ranked = build_why_ranked(profile, job, filter_result, score_payload)
        company_signals = detect_company_signals(job)
        location_signals = detect_location_signals(job, profile)
        role_signals = role_family_match_details(profile, job)
        return {
            "job_id": job.get("job_id", ""),
            "dedup_key": clean_text(job.get("dedup_key")),
            "description_hash": clean_text(job.get("description_hash")),
            "title": clean_text(job.get("title")),
            "company": clean_text(job.get("company") or job.get("employer")),
            "employer": clean_text(job.get("employer") or job.get("company")),
            "location": clean_text(job.get("location")),
            "salary_min": clean_text(job.get("salary_min")),
            "salary_max": clean_text(job.get("salary_max")),
            "salary_raw": clean_text(job.get("salary_raw")),
            "link": clean_text(job.get("link")),
            "description_text": clean_text(job.get("description_text")),
            "source": clean_text(job.get("source")),
            "raw_source": clean_text(job.get("raw_source")),
            "employment_type": clean_text(job.get("employment_type")),
            "seniority": clean_text(job.get("seniority")),
            "years_required": clean_text(job.get("years_required")),
            "is_remote": clean_text(job.get("is_remote")),
            "location_signals": location_signals,
            "role_signals": role_signals,
            "company_type": clean_text(job.get("company_type")),
            "company_signals": company_signals,
            "sponsorship_signal": clean_text(job.get("sponsorship_signal")),
            "embedding_similarity": round(similarity, 6),
            "final_score": score_payload["final_score"],
            "score_components": score_payload["score_components"],
            "penalties": score_payload["penalties"],
            "matched_skills": score_payload["matched_skills"],
            "hard_filter_passed": filter_result.passed,
            "hard_filter_violations": filter_result.violations,
            "why_ranked": why_ranked,
        }

    def rank(
        self,
        profile: dict[str, Any],
        *,
        top_k: int = 10,
        candidate_k: int = 1000,
        include_filtered: bool = False,
        session_feedback_events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        profile_text = profile_to_text(profile)
        query_embedding = self.store.embed_text(profile_text)

        scored: list[dict[str, Any]] = []
        seen_indices: set[int] = set()
        seen_output_keys: set[tuple[str, ...]] = set()
        filtered_out = 0
        exact_duplicates_removed = 0
        ann_candidates_seen = 0
        expansion_attempts: list[dict[str, Any]] = []
        strict_role_family = bool(profile.get("strict_role_family"))

        def process_candidate(row_index: int, similarity: float) -> None:
            nonlocal exact_duplicates_removed, filtered_out, ann_candidates_seen
            if row_index in seen_indices:
                return
            seen_indices.add(row_index)
            job = self.store.job_rows[row_index]
            filter_result = apply_hard_filters(profile, job)
            if not filter_result.passed:
                filtered_out += 1
                if not include_filtered:
                    return
            output_key = self._output_key(job)
            if output_key in seen_output_keys:
                exact_duplicates_removed += 1
                return
            seen_output_keys.add(output_key)
            scored.append(self._build_scored_job(profile, job, similarity=similarity, filter_result=filter_result))

        for size in self._candidate_sizes(candidate_k, strict_role_family=strict_role_family):
            retrieval_results = self.retriever.search(query_embedding, top_k=size)
            expansion_attempts.append({"candidate_k": size, "retrieved": len(retrieval_results)})
            ann_candidates_seen = max(ann_candidates_seen, len(retrieval_results))
            for result in retrieval_results:
                process_candidate(int(result["index"]), float(result["similarity"]))
            if not strict_role_family or len(scored) >= top_k:
                break

        fallback_scan_used = False
        fallback_scan_rows = 0
        fallback_scan_matches = 0
        fallback_scan_added = 0
        if strict_role_family and len(scored) < top_k:
            fallback_scan_used = True
            fallback_scan_rows = len(self.store.job_rows)
            fallback_candidates: list[tuple[int, float]] = []
            embeddings = np.asarray(self.store.embeddings, dtype=np.float32)
            query = np.asarray(query_embedding, dtype=np.float32)
            for row_index, job in enumerate(self.store.job_rows):
                if row_index in seen_indices:
                    continue
                if not matches_required_role_family(profile, job):
                    continue
                fallback_scan_matches += 1
                fallback_candidates.append((row_index, float(embeddings[row_index] @ query)))
            fallback_candidates.sort(key=lambda item: item[1], reverse=True)
            for row_index, similarity in fallback_candidates:
                before = len(scored)
                process_candidate(row_index, similarity)
                if len(scored) > before:
                    fallback_scan_added += 1

        scored.sort(key=lambda item: (item["final_score"], item["embedding_similarity"]), reverse=True)
        for rank, item in enumerate(scored, start=1):
            item["base_rank"] = rank

        evidence_metadata = self.evidence_reranker.apply(
            profile,
            scored,
            session_feedback_events=session_feedback_events or [],
        )
        scored.sort(
            key=lambda item: (
                int(item.get("ranking_sort_group", 1)),
                float(item.get("ranking_primary_score", item.get("final_score") or 0.0)),
                -int(item.get("base_rank") or 0),
                float(item.get("embedding_similarity") or 0.0),
            ),
            reverse=True,
        )
        top_jobs, display_diversification = self._select_display_top_jobs(
            scored,
            top_k=top_k,
            include_filtered=include_filtered,
        )
        for rank, item in enumerate(top_jobs, start=1):
            item["rank"] = rank
            item["final_rank"] = rank
            base_rank = item.get("base_rank")
            item["rank_movement"] = int(base_rank) - rank if isinstance(base_rank, int) else 0
        application_strategy = apply_application_strategy(top_jobs)
        warnings: list[str] = []
        if strict_role_family and len(top_jobs) < top_k:
            warnings.append(
                f"strict_role_family returned {len(top_jobs)} of requested {top_k} jobs after candidate expansion and fallback scan"
            )
        retrieval_seconds = time.perf_counter() - start

        return {
            "profile_id": profile.get("profile_id", ""),
            "profile_text": profile_text,
            "top_jobs": top_jobs,
            "metadata": {
                "candidate_k": candidate_k,
                "candidate_expansion_attempts": expansion_attempts,
                "retrieved_candidates": ann_candidates_seen,
                "evaluated_candidates": len(seen_indices),
                "filtered_out": filtered_out,
                "exact_duplicate_postings_removed": exact_duplicates_removed,
                "returned_jobs": len(top_jobs),
                "requested_top_k": top_k,
                "topk_completion_rate": round(len(top_jobs) / max(top_k, 1), 4),
                "fallback_scan_used": fallback_scan_used,
                "fallback_scan_rows": fallback_scan_rows,
                "fallback_scan_matches": fallback_scan_matches,
                "fallback_scan_added": fallback_scan_added,
                "embedding_backend": self.store.metadata.get("backend"),
                "embedding_model": self.store.metadata.get("model_name"),
                "embedding_model_revision": self.store.metadata.get("model_revision", ""),
                "embedding_dimension": self.store.metadata.get("embedding_dimension"),
                "embedding_cache_hit": bool(self.store.metadata.get("cache_hit")),
                "ann_backend": self.retriever.backend,
                "evidence_rerank": evidence_metadata,
                "retrieval_seconds": round(retrieval_seconds, 6),
                "total_seconds": round(time.perf_counter() - start, 6),
                "display_diversification": display_diversification,
                "application_strategy": application_strategy,
                "warnings": warnings,
            },
        }


def rank_jobs_for_profile(
    profile: dict[str, Any],
    *,
    snapshot_path: str | Path = OFFLINE_SNAPSHOT_CSV,
    cache_dir: str | Path = EMBEDDINGS_DIR,
    top_k: int = 10,
    candidate_k: int = 1000,
    embedding_backend: str = "auto",
    rebuild_embeddings: bool = False,
) -> dict[str, Any]:
    store = build_or_load_job_embeddings(
        snapshot_path=snapshot_path,
        cache_dir=cache_dir,
        backend=embedding_backend,
        rebuild=rebuild_embeddings,
    )
    ranker = JobRanker(store)
    return ranker.rank(profile, top_k=top_k, candidate_k=candidate_k)
