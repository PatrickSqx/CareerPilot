"""Company-name and posting-text signals for sparse job metadata."""

from __future__ import annotations

import re
from typing import Any

from jobpilot.utils.text import clean_text


DEFENSE_CONTRACTOR_KEYWORDS = (
    "govcio",
    "mantech",
    "booz allen",
    "raytheon",
    "lockheed",
    "northrop",
    "general dynamics",
    "leidos",
    "saic",
    "l3harris",
    "bae systems",
    "caci",
    "defense",
    "department of defense",
    "dod",
    "defense contractor",
    "military",
    "military contractor",
    "federal contractor",
    "government contractor",
    "clearance",
    "security clearance",
    "active clearance",
    "top secret",
    "top secret sci",
    "ts sci",
    "ts/sci",
    "sci",
)

SPONSOR_FRIENDLY_COMPANY_KEYWORDS = (
    "google",
    "amazon",
    "microsoft",
    "meta",
    "apple",
    "nvidia",
    "oracle",
    "salesforce",
    "ibm",
    "intel",
    "adobe",
    "servicenow",
    "indeed",
    "general motors",
    "capital one",
    "jpmorgan",
    "bloomberg",
    "databricks",
    "openai",
    "anthropic",
    "deepmind",
)

RESEARCH_LAB_KEYWORDS = (
    "deepmind",
    "openai",
    "anthropic",
    "google brain",
    "microsoft research",
    "meta ai",
    "research lab",
    "ai lab",
)


def _normal_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean_text(value).lower()).strip()


def _keyword_hits(text: str, keywords: tuple[str, ...]) -> list[str]:
    normal_text = f" {_normal_text(text)} "
    hits: list[str] = []
    for keyword in keywords:
        normal_keyword = _normal_text(keyword)
        if not normal_keyword:
            continue
        normal_hit = re.search(rf"(?<![a-z0-9]){re.escape(normal_keyword)}(?![a-z0-9])", normal_text)
        if normal_hit:
            hits.append(keyword)
    return hits


def _company_text(job: dict[str, Any]) -> str:
    return " ".join(
        clean_text(job.get(key))
        for key in ["company", "employer"]
        if clean_text(job.get(key))
    )


def _posting_text(job: dict[str, Any]) -> str:
    return " ".join(
        [
            _company_text(job),
            clean_text(job.get("title")),
            clean_text(job.get("company_type")),
            clean_text(job.get("description_text"))[:2500],
        ]
    )


def detect_company_signals(job: dict[str, Any]) -> dict[str, Any]:
    """Return shared company/posting proxies used by filters, scoring, and evaluation."""

    company_text = _company_text(job)
    posting_text = _posting_text(job)
    defense_hits = _keyword_hits(posting_text, DEFENSE_CONTRACTOR_KEYWORDS)
    sponsor_hits = _keyword_hits(company_text, SPONSOR_FRIENDLY_COMPANY_KEYWORDS)
    research_hits = _keyword_hits(posting_text, RESEARCH_LAB_KEYWORDS)
    company_type = clean_text(job.get("company_type")).lower()

    large_company_proxy = bool(sponsor_hits) or company_type == "large_company"
    research_lab_proxy = bool(research_hits) or company_type == "research_lab"
    sponsor_friendly_proxy = large_company_proxy or research_lab_proxy
    return {
        "defense_government_contractor": bool(defense_hits) or company_type == "defense_military",
        "large_company_proxy": large_company_proxy,
        "research_lab_proxy": research_lab_proxy,
        "sponsor_friendly_proxy": sponsor_friendly_proxy,
        "defense_terms": defense_hits,
        "sponsor_proxy_terms": sponsor_hits,
        "research_lab_terms": research_hits,
    }
