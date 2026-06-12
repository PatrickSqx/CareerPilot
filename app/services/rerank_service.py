"""Deterministic feedback-adjusted re-ranking for Phase 3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.services.paths import PROJECT_ROOT
from jobpilot.utils.text import clean_text, normalize_for_key


SIMULATION_OUTPUT = PROJECT_ROOT / "data/processed/phase3_feedback_simulation.json"


def _tokens(value: str) -> set[str]:
    return {token for token in normalize_for_key(value).split() if len(token) >= 2}


def _skill_set(job: dict[str, Any]) -> set[str]:
    skills = job.get("matched_skills") or []
    if isinstance(skills, str):
        skills = [part.strip() for part in skills.split("|") if part.strip()]
    return {normalize_for_key(str(skill)) for skill in skills if str(skill).strip()}


def _families(job: dict[str, Any]) -> set[str]:
    role_signals = job.get("role_signals") if isinstance(job.get("role_signals"), dict) else {}
    values = role_signals.get("detected_families") or role_signals.get("title_families") or []
    return {str(item).lower() for item in values}


def _company_type(job: dict[str, Any]) -> str:
    return clean_text(job.get("company_type")).lower() or "unknown"


def _sponsorship(job: dict[str, Any]) -> str:
    return clean_text(job.get("sponsorship_signal")).lower() or "unknown"


def _job_title_terms(job: dict[str, Any]) -> set[str]:
    return _tokens(clean_text(job.get("title")))


def _adjustment_for_event(job: dict[str, Any], event: dict[str, Any]) -> tuple[float, list[str]]:
    source_job = event.get("job_snapshot") or {}
    action = str(event.get("action", "")).lower()
    notes: list[str] = []
    adjustment = 0.0
    same_job = clean_text(job.get("job_id")) == clean_text(source_job.get("job_id"))
    skill_overlap = _skill_set(job) & _skill_set(source_job)
    family_overlap = _families(job) & _families(source_job)
    title_overlap = _job_title_terms(job) & _job_title_terms(source_job)
    same_company = normalize_for_key(clean_text(job.get("company"))) == normalize_for_key(clean_text(source_job.get("company")))

    if action == "accept":
        if same_job:
            adjustment += 0.10
            notes.append("accepted job")
        if skill_overlap:
            adjustment += min(0.05, 0.015 * len(skill_overlap))
            notes.append("shares accepted skills")
        if family_overlap:
            adjustment += 0.04
            notes.append("same accepted role family")
        if same_company:
            adjustment += 0.02
            notes.append("same accepted company")
    elif action == "reject":
        if same_job:
            adjustment -= 0.30
            notes.append("rejected job")
        if same_company:
            adjustment -= 0.08
            notes.append("same rejected company")
        if title_overlap:
            adjustment -= min(0.07, 0.012 * len(title_overlap))
            notes.append("similar rejected title")
        if _company_type(job) == _company_type(source_job) and _company_type(job) != "unknown":
            adjustment -= 0.04
            notes.append("same rejected company type")
        if _sponsorship(job) == _sponsorship(source_job) and _sponsorship(job) in {"unknown", "no_sponsorship"}:
            adjustment -= 0.05
            notes.append("same rejected sponsorship risk")
    elif action == "skip":
        if same_job:
            adjustment -= 0.03
            notes.append("skipped job")
    return adjustment, notes


def apply_feedback_rerank(jobs: list[dict[str, Any]], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    adjusted: list[dict[str, Any]] = []
    for job in jobs:
        item = dict(job)
        base = float(item.get("final_score") or 0)
        total_adjustment = 0.0
        notes: list[str] = []
        for event in events:
            delta, delta_notes = _adjustment_for_event(item, event)
            total_adjustment += delta
            notes.extend(delta_notes)
        total_adjustment = max(-0.45, min(0.25, total_adjustment))
        item["base_score"] = round(base, 6)
        item["feedback_adjustment"] = round(total_adjustment, 6)
        item["adjusted_score"] = round(max(0.0, min(1.0, base + total_adjustment)), 6)
        item["feedback_adjustment_explanation"] = "; ".join(dict.fromkeys(notes)) or "No direct feedback pattern match"
        adjusted.append(item)
    adjusted.sort(key=lambda row: (float(row.get("adjusted_score") or 0), float(row.get("final_score") or 0)), reverse=True)
    for rank, item in enumerate(adjusted, start=1):
        item["adjusted_rank"] = rank
    return adjusted


def _kenji_reject_pattern(job: dict[str, Any]) -> bool:
    sponsorship = _sponsorship(job)
    company_signals = job.get("company_signals") if isinstance(job.get("company_signals"), dict) else {}
    sponsor_proxy = bool(company_signals.get("sponsor_friendly_proxy"))
    return sponsorship == "unknown" and not sponsor_proxy


def write_kenji_feedback_simulation() -> dict[str, Any]:
    """Create deterministic feedback simulation evidence from existing Kenji output."""

    source = PROJECT_ROOT / "data/processed/phase2_matching_kenji.json"
    if not source.exists():
        payload = {"status": "skipped", "reason": "phase2_matching_kenji.json not found"}
    else:
        data = json.loads(source.read_text(encoding="utf-8"))
        jobs = data.get("top_jobs", [])
        feedback_events = [
            {
                "action": "reject",
                "job_snapshot": job,
            }
            for job in jobs
            if _kenji_reject_pattern(job)
        ][:3]
        adjusted = apply_feedback_rerank(jobs, feedback_events)
        before_top5_risks = sum(1 for job in jobs[:5] if _kenji_reject_pattern(job))
        after_top5_risks = sum(1 for job in adjusted[:5] if _kenji_reject_pattern(job))
        payload = {
            "status": "completed",
            "persona": "kenji",
            "simulation_rule": "Reject unknown-sponsorship jobs without sponsor-friendly company proxy.",
            "feedback_events_simulated": len(feedback_events),
            "before_top5_unknown_sponsor_without_proxy": before_top5_risks,
            "after_top5_unknown_sponsor_without_proxy": after_top5_risks,
            "improvement": before_top5_risks - after_top5_risks,
            "top_jobs_after_feedback": [
                {
                    "rank": job.get("adjusted_rank"),
                    "job_id": job.get("job_id"),
                    "title": job.get("title"),
                    "company": job.get("company"),
                    "base_score": job.get("base_score"),
                    "adjusted_score": job.get("adjusted_score"),
                    "feedback_adjustment": job.get("feedback_adjustment"),
                    "feedback_adjustment_explanation": job.get("feedback_adjustment_explanation"),
                }
                for job in adjusted[:10]
            ],
        }
    SIMULATION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SIMULATION_OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload

