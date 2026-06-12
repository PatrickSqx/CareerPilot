"""Canonical job-posting schema used across Phase 1."""

from __future__ import annotations

from typing import Any


CANONICAL_COLUMNS = [
    "job_id",
    "source",
    "source_record_id",
    "reference_id",
    "schema_org_identifier",
    "company_id",
    "location_id",
    "is_current_api",
    "ingested_at",
    "date_posted_or_scraped",
    "posting_date_raw",
    "expiration_date_raw",
    "query",
    "title",
    "company",
    "employer",
    "location",
    "country",
    "state",
    "city",
    "postal_code",
    "county",
    "latitude",
    "longitude",
    "raw_source_country",
    "raw_locale",
    "salary_min",
    "salary_max",
    "salary_raw",
    "salary_period",
    "salary_normalization_method",
    "raw_salary_text",
    "raw_salary_value",
    "raw_salary_period",
    "schema_org_salary_min",
    "schema_org_salary_max",
    "schema_org_salary_currency",
    "schema_org_salary_unit",
    "salary_currency",
    "salary_is_predicted",
    "employment_type",
    "position_work_type_raw",
    "position_career_level_raw",
    "position_contract_type_raw",
    "position_department_raw",
    "schema_org_employment_type",
    "description",
    "description_text",
    "link",
    "company_url",
    "company_description_raw",
    "company_size_raw",
    "raw_source",
    "raw_categories",
    "raw_work_types",
    "raw_qualifications",
    "raw_skills",
    "raw_industries",
    "raw_jobnames",
    "raw_keywords",
    "raw_requirements",
    "raw_languages",
    "raw_benefits",
    "schema_org_skills",
    "schema_org_experience_requirements",
    "schema_org_industry",
    "schema_org_occupational_category",
    "schema_org_education_requirements",
    "schema_org_valid_through",
    "direct_apply_raw",
    "normalized_skills",
    "normalized_industries",
    "normalized_role_terms",
    "normalized_keywords",
    "structured_signal_sources",
    "structured_signal_confidence",
    "dedup_key",
    "description_hash",
    "extracted_skills",
    "seniority",
    "years_required",
    "is_remote",
    "company_type",
    "sponsorship_signal",
    "embedding_text",
]

REQUIRED_FOR_VALID_RECORD = ["title", "description_text"]


def empty_job_record() -> dict[str, Any]:
    """Return an empty record with every canonical column present."""

    return {column: "" for column in CANONICAL_COLUMNS}


def order_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a dict ordered and limited to canonical columns."""

    return {column: record.get(column, "") for column in CANONICAL_COLUMNS}


def validation_errors(record: dict[str, Any]) -> list[str]:
    """Return validation errors for fields required by the ingestion pipeline."""

    errors: list[str] = []
    for column in REQUIRED_FOR_VALID_RECORD:
        value = str(record.get(column, "") or "").strip()
        if not value:
            errors.append(f"missing_{column}")
    return errors
