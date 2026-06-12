"""Application strategy labels for ranked job recommendations."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from jobpilot.utils.text import clean_text, normalize_for_key


SAME_COMPANY_WARNING = (
    "Several strong matches come from the same company. Apply to the best 1-2 first rather than applying to all at once."
)


def _company_key(job: dict[str, Any]) -> str:
    return normalize_for_key(clean_text(job.get("company") or job.get("employer")))


def _title_key(job: dict[str, Any]) -> str:
    title = normalize_for_key(clean_text(job.get("title")))
    title = re.sub(r"\b(?:senior|sr|jr|junior|iii|ii|i)\b", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def _location_key(job: dict[str, Any]) -> str:
    return normalize_for_key(clean_text(job.get("location")))


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", value) if len(token) >= 2}


def _similar_title(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    a_tokens = _tokens(a)
    b_tokens = _tokens(b)
    if not a_tokens or not b_tokens:
        return False
    return len(a_tokens & b_tokens) / max(len(a_tokens | b_tokens), 1) >= 0.8


def _is_possible_near_duplicate(job: dict[str, Any], previous_company_jobs: list[dict[str, Any]]) -> bool:
    title = _title_key(job)
    location = _location_key(job)
    for previous in previous_company_jobs:
        if _location_key(previous) and location and _location_key(previous) != location:
            continue
        if _similar_title(title, _title_key(previous)):
            return True
    return False


def _append_strategy_note(job: dict[str, Any], note: str) -> None:
    why = job.get("why_ranked")
    if not isinstance(why, dict):
        return
    positives = why.setdefault("positive_drivers", [])
    if isinstance(positives, list) and note not in positives:
        positives.append(note)
    summary = clean_text(why.get("summary"))
    if note and note not in summary:
        why["summary"] = clean_text(f"{summary}. {note}" if summary else note)


def apply_application_strategy(
    top_jobs: list[dict[str, Any]],
    *,
    max_apply_now_per_company: int = 2,
) -> dict[str, Any]:
    """Annotate ranked jobs without changing final ranking order."""

    company_counts = Counter(_company_key(job) for job in top_jobs if _company_key(job))
    dominated_companies = {company for company, count in company_counts.items() if count > max_apply_now_per_company}
    company_seen: defaultdict[str, int] = defaultdict(int)
    previous_by_company: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    label_counts: Counter[str] = Counter()
    near_duplicate_count = 0

    for job in top_jobs:
        company = _company_key(job)
        company_seen[company] += 1
        same_company_rank = company_seen[company]
        possible_near_duplicate = _is_possible_near_duplicate(job, previous_by_company[company]) if company else False
        if possible_near_duplicate:
            label = "Potential duplicate role"
            recommended = False
            reason = "Similar title/location appears earlier for the same company; review before applying twice."
            near_duplicate_count += 1
        elif company and same_company_rank > max_apply_now_per_company:
            label = "Same-company alternative"
            recommended = False
            reason = "Same company already has higher-ranked recommendations; keep as a backup option."
        else:
            label = "Apply Now"
            recommended = True
            reason = ""

        warning = SAME_COMPANY_WARNING if company in dominated_companies else ""
        job["application_strategy_label"] = label
        job["same_company_rank"] = same_company_rank if company else 0
        job["company_application_warning"] = warning
        job["possible_near_duplicate_role"] = possible_near_duplicate
        job["recommended_apply_now"] = recommended
        job["also_consider_reason"] = reason
        label_counts[label] += 1

        if label == "Apply Now":
            _append_strategy_note(job, "Application strategy: prioritized apply-now recommendation")
        elif label == "Same-company alternative":
            _append_strategy_note(job, "Application strategy: same-company alternative, not a duplicate removal")
        else:
            _append_strategy_note(job, "Application strategy: possible near-duplicate role")
        previous_by_company[company].append(job)

    return {
        "max_apply_now_per_company": max_apply_now_per_company,
        "label_counts": dict(sorted(label_counts.items())),
        "same_company_counts": dict(sorted((company, count) for company, count in company_counts.items() if company)),
        "companies_with_concentration": sorted(dominated_companies),
        "company_application_warning": SAME_COMPANY_WARNING if dominated_companies else "",
        "possible_near_duplicate_role_count": near_duplicate_count,
    }
