"""Run deterministic Phase 3 feedback-learning simulation.

The simulator does not use random feedback. It converts persona fit signals from
the Phase 2 outputs into accept/reject/skip events, applies the Phase 3 feedback
re-ranker, and records measurable score movement.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.rerank_service import apply_feedback_rerank  # noqa: E402
from jobpilot.utils.text import clean_text, normalize_for_key  # noqa: E402


SOURCE = PROJECT_ROOT / "data/processed/persona_phase2_results.json"
OUTPUT = PROJECT_ROOT / "data/processed/phase3_feedback_simulation.json"


def _risk_terms(job: dict[str, Any]) -> bool:
    title = normalize_for_key(clean_text(job.get("title")))
    company_signals = job.get("company_signals") if isinstance(job.get("company_signals"), dict) else {}
    role_signals = job.get("role_signals") if isinstance(job.get("role_signals"), dict) else {}
    weak_role = not role_signals.get("title_families")
    terms = {"senior", "principal", "director", "manager", "contract", "temporary", "unpaid"}
    return bool(set(title.split()) & terms) or bool(company_signals.get("defense_government_contractor")) or weak_role


def _score(job: dict[str, Any]) -> float:
    return float(job.get("final_score") or job.get("adjusted_score") or 0.0)


def _decide_events(jobs: list[dict[str, Any]], profile: dict[str, Any]) -> list[dict[str, Any]]:
    ranked = sorted(jobs, key=_score, reverse=True)
    events: list[dict[str, Any]] = []
    used_job_ids: set[str] = set()
    accepted = 0
    skipped = 0
    rejected = 0
    if profile.get("needs_sponsorship"):
        for job in ranked:
            job_id = clean_text(job.get("job_id"))
            if not job_id or job_id in used_job_ids:
                continue
            if _unknown_sponsor_without_proxy(job) and rejected < 3:
                events.append({"action": "reject", "job_snapshot": job})
                used_job_ids.add(job_id)
                rejected += 1
            if rejected >= 3:
                break
    for job in ranked:
        job_id = clean_text(job.get("job_id"))
        if job_id in used_job_ids:
            continue
        violations = job.get("hard_filter_violations") or []
        sponsorship = clean_text(job.get("sponsorship_signal")).lower()
        company_signals = job.get("company_signals") if isinstance(job.get("company_signals"), dict) else {}
        unknown_sponsor_without_proxy = (
            bool(profile.get("needs_sponsorship"))
            and sponsorship == "unknown"
            and not company_signals.get("sponsor_friendly_proxy")
        )
        if _score(job) >= 0.65 and len(job.get("matched_skills") or []) >= 2 and not violations and not _risk_terms(job) and accepted < 3:
            action = "accept"
            accepted += 1
        elif (violations or _risk_terms(job) or unknown_sponsor_without_proxy or _score(job) < 0.58) and rejected < 3:
            action = "reject"
            rejected += 1
        elif skipped < 3:
            action = "skip"
            skipped += 1
        else:
            continue
        events.append({"action": action, "job_snapshot": job})
        used_job_ids.add(job_id)
        if accepted >= 3 and rejected >= 2 and skipped >= 2:
            break
    if rejected < 2:
        for job in reversed(ranked):
            job_id = clean_text(job.get("job_id"))
            if job_id in used_job_ids:
                continue
            events.append({"action": "reject", "job_snapshot": job})
            used_job_ids.add(job_id)
            rejected += 1
            if rejected >= 2:
                break
    if skipped < 2:
        for job in ranked:
            job_id = clean_text(job.get("job_id"))
            if job_id in used_job_ids:
                continue
            events.append({"action": "skip", "job_snapshot": job})
            used_job_ids.add(job_id)
            skipped += 1
            if skipped >= 2:
                break
    return events


def _job_by_id(jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {clean_text(job.get("job_id")): job for job in jobs}


def _rank_by_id(jobs: list[dict[str, Any]]) -> dict[str, int]:
    return {clean_text(job.get("job_id")): rank for rank, job in enumerate(jobs, start=1)}


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _mean_rank(job_ids: set[str], jobs: list[dict[str, Any]]) -> float | None:
    ranks = _rank_by_id(jobs)
    values = [ranks[job_id] for job_id in job_ids if job_id in ranks]
    return round(sum(values) / len(values), 4) if values else None


def _rank_moved_down(job_ids: set[str], before: list[dict[str, Any]], after: list[dict[str, Any]]) -> float | None:
    before_rank = _mean_rank(job_ids, before)
    after_rank = _mean_rank(job_ids, after)
    if before_rank is None or after_rank is None:
        return None
    return round(after_rank - before_rank, 4)


def _weighted_exposure(job_ids: set[str], jobs: list[dict[str, Any]], top_k: int = 10) -> int:
    exposure = 0
    for rank, job in enumerate(jobs[:top_k], start=1):
        if clean_text(job.get("job_id")) in job_ids:
            exposure += top_k + 1 - rank
    return exposure


def _exposure_improvement(job_ids: set[str], before: list[dict[str, Any]], after: list[dict[str, Any]], top_k: int = 10) -> int:
    return _weighted_exposure(job_ids, before, top_k) - _weighted_exposure(job_ids, after, top_k)


def _action_job_ids(events: list[dict[str, Any]], action: str) -> set[str]:
    return {
        clean_text(event.get("job_snapshot", {}).get("job_id"))
        for event in events
        if event.get("action") == action and clean_text(event.get("job_snapshot", {}).get("job_id"))
    }


def _count_ids_in_top(job_ids: set[str], jobs: list[dict[str, Any]], top_k: int) -> int:
    return sum(1 for job in jobs[:top_k] if clean_text(job.get("job_id")) in job_ids)


def _unknown_sponsor_without_proxy(job: dict[str, Any]) -> bool:
    sponsorship = clean_text(job.get("sponsorship_signal")).lower()
    company_signals = job.get("company_signals") if isinstance(job.get("company_signals"), dict) else {}
    return sponsorship == "unknown" and not bool(company_signals.get("sponsor_friendly_proxy"))


def _small_company_or_unknown_sponsor(job: dict[str, Any]) -> bool:
    company_type = clean_text(job.get("company_type")).lower()
    small_company = company_type in {"startup", "small_company", "small", "unknown"}
    return small_company or _unknown_sponsor_without_proxy(job)


def _action_score_shift(events: list[dict[str, Any]], before: list[dict[str, Any]], after: list[dict[str, Any]], action: str) -> float:
    before_map = _job_by_id(before)
    after_map = _job_by_id(after)
    deltas: list[float] = []
    for event in events:
        if event["action"] != action:
            continue
        job_id = clean_text(event["job_snapshot"].get("job_id"))
        if job_id in before_map and job_id in after_map:
            deltas.append(float(after_map[job_id].get("adjusted_score") or 0) - _score(before_map[job_id]))
    return _mean(deltas)


def run() -> dict[str, Any]:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing {SOURCE}")
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    results: dict[str, Any] = {}
    for persona, data in payload.items():
        jobs = list(data.get("top_jobs") or [])
        events = _decide_events(jobs, data.get("profile", {}))
        adjusted = apply_feedback_rerank(jobs, events)
        counts = Counter(event["action"] for event in events)
        accepted_ids = _action_job_ids(events, "accept")
        rejected_ids = _action_job_ids(events, "reject")
        accepted_rank_before = _mean_rank(accepted_ids, jobs)
        accepted_rank_after = _mean_rank(accepted_ids, adjusted)
        accepted_rank_improvement = (
            round(accepted_rank_before - accepted_rank_after, 4)
            if accepted_rank_before is not None and accepted_rank_after is not None
            else None
        )
        rejected_rank_before = _mean_rank(rejected_ids, jobs)
        rejected_rank_after = _mean_rank(rejected_ids, adjusted)
        persona_result: dict[str, Any] = {
            "events_by_action": dict(sorted(counts.items())),
            "feedback_events_simulated": len(events),
            "rounds_simulated": min(3, len(events)),
            "mean_accept_score_lift": _action_score_shift(events, jobs, adjusted, "accept"),
            "mean_reject_score_drop": _action_score_shift(events, jobs, adjusted, "reject"),
            "mean_skip_score_shift": _action_score_shift(events, jobs, adjusted, "skip"),
            "rejected_jobs_remaining_in_top5_before": _count_ids_in_top(rejected_ids, jobs, 5),
            "rejected_jobs_remaining_in_top5_after": _count_ids_in_top(rejected_ids, adjusted, 5),
            "accepted_jobs_average_rank_before": accepted_rank_before,
            "accepted_jobs_average_rank_after": accepted_rank_after,
            "accepted_jobs_average_rank_improvement": accepted_rank_improvement,
            "rejected_jobs_average_rank_before": rejected_rank_before,
            "rejected_jobs_average_rank_after": rejected_rank_after,
            "rejected_jobs_average_rank_moved_down": _rank_moved_down(rejected_ids, jobs, adjusted),
            "rejected_jobs_weighted_exposure_before": _weighted_exposure(rejected_ids, jobs),
            "rejected_jobs_weighted_exposure_after": _weighted_exposure(rejected_ids, adjusted),
            "rejected_jobs_weighted_exposure_improvement": _exposure_improvement(rejected_ids, jobs, adjusted),
            "top5_before": [job.get("job_id") for job in jobs[:5]],
            "top5_after": [job.get("job_id") for job in adjusted[:5]],
        }
        if persona == "kenji":
            rejected_sponsorship_risk_ids = {
                clean_text(event.get("job_snapshot", {}).get("job_id"))
                for event in events
                if event.get("action") == "reject"
                and _unknown_sponsor_without_proxy(event.get("job_snapshot", {}))
                and clean_text(event.get("job_snapshot", {}).get("job_id"))
            }
            persona_result.update(
                {
                    "unknown_sponsor_without_proxy_top10_before": sum(1 for job in jobs[:10] if _unknown_sponsor_without_proxy(job)),
                    "unknown_sponsor_without_proxy_top10_after": sum(1 for job in adjusted[:10] if _unknown_sponsor_without_proxy(job)),
                    "small_company_or_unknown_sponsor_top10_before": sum(1 for job in jobs[:10] if _small_company_or_unknown_sponsor(job)),
                    "small_company_or_unknown_sponsor_top10_after": sum(1 for job in adjusted[:10] if _small_company_or_unknown_sponsor(job)),
                    "kenji_rejected_sponsorship_risk_average_rank_before": _mean_rank(rejected_sponsorship_risk_ids, jobs),
                    "kenji_rejected_sponsorship_risk_average_rank_after": _mean_rank(rejected_sponsorship_risk_ids, adjusted),
                    "kenji_rejected_sponsorship_risk_moved_down": _rank_moved_down(rejected_sponsorship_risk_ids, jobs, adjusted),
                    "kenji_sponsorship_risk_weighted_exposure_before": _weighted_exposure(rejected_sponsorship_risk_ids, jobs),
                    "kenji_sponsorship_risk_weighted_exposure_after": _weighted_exposure(rejected_sponsorship_risk_ids, adjusted),
                    "kenji_sponsorship_risk_weighted_exposure_improvement": _exposure_improvement(rejected_sponsorship_risk_ids, jobs, adjusted),
                }
            )
        results[persona] = {
            **persona_result,
        }
    report = {
        "status": "completed",
        "source": "data/processed/persona_phase2_results.json",
        "method": "rule-based persona simulator using Phase 2 fit, risk, matched skills, and sponsorship/company signals",
        "random_feedback_used": False,
        "personas": results,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2, ensure_ascii=False))
