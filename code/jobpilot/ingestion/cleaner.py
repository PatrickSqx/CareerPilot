"""Cleaning and validation for normalized job records."""

from __future__ import annotations

from typing import Any

from jobpilot.schemas import order_record, validation_errors
from jobpilot.utils.text import clean_text, first_nonempty, make_embedding_text, parse_salary_from_text


INVALID_TEXT_VALUES = {"", "false", "true", "none", "null", "nan", "n/a", "na"}


def clean_normalized_record(record: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    """Clean one normalized record and return `(record, errors)`."""

    cleaned = dict(record)
    for field in ["title", "company", "employer", "location", "country", "state", "city", "link"]:
        cleaned[field] = clean_text(cleaned.get(field))

    cleaned["description_text"] = clean_text(cleaned.get("description_text") or cleaned.get("description"))
    cleaned["description"] = cleaned["description_text"]
    cleaned["company"] = first_nonempty(cleaned.get("company"), "Unknown Employer")
    cleaned["employer"] = first_nonempty(cleaned.get("employer"), cleaned.get("company"))
    cleaned["location"] = first_nonempty(cleaned.get("location"), "Unknown Location")

    if not cleaned.get("salary_min") and not cleaned.get("salary_max"):
        salary_min, salary_max, salary_raw = parse_salary_from_text(cleaned.get("salary_raw"), cleaned.get("description_text"))
        cleaned["salary_min"] = salary_min
        cleaned["salary_max"] = salary_max
        cleaned["salary_raw"] = first_nonempty(cleaned.get("salary_raw"), salary_raw)

    cleaned["embedding_text"] = make_embedding_text(cleaned)
    errors = validation_errors(cleaned)
    if cleaned["title"].strip().lower() in INVALID_TEXT_VALUES:
        errors.append("invalid_title")
    if len(cleaned["description_text"]) < 40:
        errors.append("description_too_short")
    if errors:
        return None, errors
    return order_record(cleaned), []
