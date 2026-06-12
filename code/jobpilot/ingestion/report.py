"""Report and lightweight analytics generation."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from jobpilot.schemas import CANONICAL_COLUMNS
from jobpilot.utils.io import write_json

PHASE18_STRUCTURED_FIELDS = [
    "raw_skills",
    "raw_industries",
    "raw_jobnames",
    "raw_keywords",
    "raw_requirements",
    "raw_languages",
    "raw_benefits",
    "normalized_skills",
    "normalized_industries",
    "normalized_role_terms",
    "normalized_keywords",
    "position_career_level_raw",
    "position_contract_type_raw",
    "position_department_raw",
    "company_description_raw",
    "company_size_raw",
    "schema_org_industry",
    "schema_org_occupational_category",
    "schema_org_education_requirements",
    "schema_org_valid_through",
    "reference_id",
    "schema_org_identifier",
    "postal_code",
    "county",
    "latitude",
    "longitude",
    "posting_date_raw",
    "expiration_date_raw",
    "direct_apply_raw",
    "salary_currency",
    "salary_is_predicted",
    "structured_signal_sources",
    "structured_signal_confidence",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_missing_field_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in CANONICAL_COLUMNS:
        counts[column] = sum(1 for row in rows if row.get(column) is None or str(row.get(column, "")).strip() == "")
    return counts


def coverage_rate(rows: list[dict[str, Any]], *columns: str) -> float:
    if not rows:
        return 0.0
    covered = 0
    for row in rows:
        if any(str(row.get(column, "") or "").strip() for column in columns):
            covered += 1
    return round(covered / len(rows), 4)


def field_coverage(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, float]:
    return {field: coverage_rate(rows, field) for field in fields}


def source_level_coverage(rows: list[dict[str, Any]], fields: list[str]) -> dict[str, dict[str, Any]]:
    groups = {
        "kaggle": [
            row
            for row in rows
            if str(row.get("raw_source", "") or "").lower() == "kaggle"
            or str(row.get("is_current_api", "")).lower() != "true"
        ],
        "adzuna": [row for row in rows if str(row.get("source", "") or "").lower() == "adzuna"],
        "jsearch": [row for row in rows if str(row.get("source", "") or "").lower() == "jsearch"],
    }
    return {
        name: {
            "row_count": len(group_rows),
            "coverage": field_coverage(group_rows, fields),
        }
        for name, group_rows in groups.items()
    }


def source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("source", "") or "unknown") for row in rows))


def query_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("query", "") or "(blank)") for row in rows))


def current_api_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    current = [row for row in rows if str(row.get("is_current_api")).lower() == "true"]
    return source_counts(current)


def build_market_analytics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    titles = Counter(str(row.get("title", "")).lower() for row in rows if row.get("title"))
    locations = Counter(str(row.get("location", "")) for row in rows if row.get("location"))
    remote = Counter(str(row.get("is_remote", "") or "unknown") for row in rows)
    employment = Counter(str(row.get("employment_type", "") or "unknown") for row in rows)
    categories = Counter(str(row.get("raw_categories", "") or "unknown") for row in rows)
    work_types = Counter(str(row.get("raw_work_types", "") or "unknown") for row in rows)
    skills: Counter[str] = Counter()
    normalized_skills: Counter[str] = Counter()
    normalized_industries: Counter[str] = Counter()
    normalized_role_terms: Counter[str] = Counter()
    salaries: list[float] = []
    for row in rows:
        for skill in str(row.get("extracted_skills", "")).split("|"):
            if skill:
                skills[skill] += 1
        for skill in str(row.get("normalized_skills", "")).split("|"):
            if skill:
                normalized_skills[skill] += 1
        for industry in str(row.get("normalized_industries", "")).split("|"):
            if industry:
                normalized_industries[industry] += 1
        for role_term in str(row.get("normalized_role_terms", "")).split("|"):
            if role_term:
                normalized_role_terms[role_term] += 1
        for column in ["salary_min", "salary_max"]:
            try:
                value = float(row.get(column) or 0)
            except ValueError:
                value = 0
            if value:
                salaries.append(value)

    salary_summary = {
        "count": len(salaries),
        "average": round(mean(salaries), 2) if salaries else None,
        "min": min(salaries) if salaries else None,
        "max": max(salaries) if salaries else None,
    }
    return {
        "row_count": len(rows),
        "source_counts": source_counts(rows),
        "top_titles": dict(titles.most_common(25)),
        "top_locations": dict(locations.most_common(25)),
        "top_skills": dict(skills.most_common(30)),
        "top_normalized_skills": dict(normalized_skills.most_common(30)),
        "top_normalized_industries": dict(normalized_industries.most_common(30)),
        "top_normalized_role_terms": dict(normalized_role_terms.most_common(30)),
        "remote_distribution": dict(remote),
        "employment_type_distribution": dict(employment),
        "raw_category_distribution_top25": dict(categories.most_common(25)),
        "raw_work_type_distribution_top25": dict(work_types.most_common(25)),
        "structured_signal_confidence_distribution": dict(
            Counter(str(row.get("structured_signal_confidence", "") or "unknown") for row in rows)
        ),
        "salary_summary": salary_summary,
    }


def write_data_dictionary(path: Path) -> None:
    rows = [
        ("job_id", "Stable hash ID for the normalized posting."),
        ("source", "Provider or source label, such as Kaggle source name, adzuna, or jsearch."),
        ("source_record_id", "Record identifier from the source system when available."),
        ("reference_id", "Provider reference identifier preserved when available, mainly Kaggle referenceID."),
        ("schema_org_identifier", "Compact schema.org identifier object/text when available."),
        ("company_id", "Raw Kaggle company identifier when available."),
        ("location_id", "Raw Kaggle location identifier when available."),
        ("is_current_api", "True when the record came from live, cached, or saved-current-posting ingestion."),
        ("ingested_at", "UTC timestamp when the pipeline normalized the record."),
        ("date_posted_or_scraped", "Original posting/scrape date when available."),
        ("posting_date_raw", "Provider posting/create date fallback, preserving source-backed date evidence."),
        ("expiration_date_raw", "Provider expiration/valid-through date evidence when available."),
        ("query", "Current-posting query term; blank for Kaggle records."),
        ("title", "Cleaned job title."),
        ("company", "Cleaned company name or Unknown Employer."),
        ("employer", "Alias for company used by later CSV export/UI."),
        ("location", "Cleaned display location."),
        ("country", "Country if available."),
        ("state", "State/region if available."),
        ("city", "City if available."),
        ("postal_code", "Postal code when exposed by the provider."),
        ("county", "County or subregion when exposed by the provider."),
        ("latitude", "Provider latitude when exposed."),
        ("longitude", "Provider longitude when exposed."),
        ("raw_source_country", "Provider/source country code preserved from raw data when available."),
        ("raw_locale", "Raw locale value preserved from the Kaggle source when available."),
        ("salary_min", "Parsed minimum salary when available."),
        ("salary_max", "Parsed maximum salary when available."),
        ("salary_raw", "Readable salary text or normalized salary range when available."),
        ("salary_period", "Detected source salary period such as hour, month, or year when available."),
        ("salary_normalization_method", "Whether salary came from source fields, annualization, or text fallback."),
        ("raw_salary_text", "Raw salary text preserved from the source when available."),
        ("raw_salary_value", "Raw scalar salary value preserved from the source when available."),
        ("raw_salary_period", "Raw salary period value preserved from the source when available."),
        ("schema_org_salary_min", "schema.org baseSalary minValue before annualization when available."),
        ("schema_org_salary_max", "schema.org baseSalary maxValue before annualization when available."),
        ("schema_org_salary_currency", "schema.org salary currency when available."),
        ("schema_org_salary_unit", "schema.org salary unitText when available."),
        ("salary_currency", "Provider salary currency normalized across schema.org, Adzuna, and JSearch when available."),
        ("salary_is_predicted", "Provider salary prediction flag when available, mainly Adzuna."),
        ("employment_type", "Inferred or source-provided employment type."),
        ("position_work_type_raw", "Raw position/API employment type field preserved when available."),
        ("position_career_level_raw", "Raw provider career-level signal when available."),
        ("position_contract_type_raw", "Raw provider contract/employment-type signal when available."),
        ("position_department_raw", "Raw provider department signal when available."),
        ("schema_org_employment_type", "schema.org employmentType value preserved when available."),
        ("description", "Cleaned full job description text."),
        ("description_text", "Same full job description text, preserved for export and matching."),
        ("link", "Posting URL when available."),
        ("company_url", "Company URL fields preserved from raw orgCompany/schema.org data when available."),
        ("company_description_raw", "Compact raw provider company description when available."),
        ("company_size_raw", "Raw provider company-size signal when available."),
        ("raw_source", "Raw provider family, such as kaggle, adzuna_api, jsearch_api."),
        ("raw_categories", "Pipe-separated source category tags preserved from raw data when available."),
        ("raw_work_types", "Pipe-separated source work-type tags preserved from raw data when available."),
        ("raw_qualifications", "Pipe-separated source qualification tags preserved from raw data when available."),
        ("raw_skills", "Pipe-separated provider structured skills when available."),
        ("raw_industries", "Pipe-separated provider structured industry/category signals when available."),
        ("raw_jobnames", "Pipe-separated provider structured role-name terms when available."),
        ("raw_keywords", "Pipe-separated provider structured keywords when available."),
        ("raw_requirements", "Compact provider structured requirements/highlights evidence when available."),
        ("raw_languages", "Pipe-separated provider language tags when available."),
        ("raw_benefits", "Pipe-separated provider benefits tags when available."),
        ("schema_org_skills", "schema.org skills field preserved from raw data when available."),
        ("schema_org_experience_requirements", "schema.org experienceRequirements preserved as compact JSON/text when available."),
        ("schema_org_industry", "schema.org industry signal when available."),
        ("schema_org_occupational_category", "schema.org occupationalCategory signal when available."),
        ("schema_org_education_requirements", "schema.org or provider education requirements as compact JSON/text when available."),
        ("schema_org_valid_through", "schema.org validThrough value when available."),
        ("direct_apply_raw", "Provider direct-apply flag when available."),
        ("normalized_skills", "Usage-layer skills combined from provider skills, schema.org skills, and parser fallback."),
        ("normalized_industries", "Usage-layer industry/category terms combined across providers."),
        ("normalized_role_terms", "Usage-layer role terms from structured role fields and title fallback."),
        ("normalized_keywords", "Usage-layer keywords from structured keywords, requirements, benefits, and parser fallback."),
        ("structured_signal_sources", "Pipe-separated source groups that contributed structured or fallback signals."),
        ("structured_signal_confidence", "Deterministic high/medium/low/parser_only/none label based on source-backed signal groups."),
        ("dedup_key", "Stable SHA-256 deduplication hash used by the Bloom pre-check and exact hash-set verification."),
        ("description_hash", "SHA-256 hash of the cleaned description."),
        ("extracted_skills", "Pipe-separated lightweight skill keyword hits."),
        ("seniority", "Lightweight inferred seniority label."),
        ("years_required", "Minimum years required inferred from description when available."),
        ("is_remote", "remote, hybrid, onsite, or unknown inference."),
        ("company_type", "Lightweight company type signal."),
        ("sponsorship_signal", "Lightweight work authorization/sponsorship signal."),
        ("embedding_text", "Concatenated text field for later embedding generation."),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# JobPilot Phase 1 Data Dictionary\n\n")
        handle.write("| Column | Description |\n|---|---|\n")
        for column, description in rows:
            handle.write(f"| `{column}` | {description} |\n")


def write_report(path: Path, report: dict[str, Any]) -> None:
    write_json(path, report)
