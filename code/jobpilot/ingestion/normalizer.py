"""Normalize raw provider records into the canonical JobPilot schema."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from jobpilot.schemas import empty_job_record, order_record
from jobpilot.utils.text import (
    clean_text,
    extract_skills,
    first_nonempty,
    infer_company_type,
    infer_employment_type,
    infer_remote,
    infer_seniority,
    infer_sponsorship_signal,
    infer_years_required,
    make_embedding_text,
    parse_salary_from_text,
    stable_hash,
)

INVALID_TITLE_VALUES = {"", "false", "true", "none", "null", "nan", "n/a", "na"}
ANNUALIZATION_FACTORS = {
    "year": 1.0,
    "annual": 1.0,
    "annually": 1.0,
    "yr": 1.0,
    "month": 12.0,
    "monthly": 12.0,
    "week": 52.0,
    "weekly": 52.0,
    "day": 260.0,
    "daily": 260.0,
    "hour": 2080.0,
    "hourly": 2080.0,
}
SALARY_CURRENCY_TOKEN = r"(?:\$|usd|dollars?|dollar|gbp|pounds?|pound|eur|euros?|euro)"
SIGNAL_SPLIT_RE = re.compile(r"[|;\n\r]+|,\s+(?=[A-Za-z0-9+#])")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _get_path(record: dict[str, Any], path: str) -> Any:
    value: Any = record
    for part in path.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value


def _first_nested_key(value: Any, *keys: str) -> Any:
    """Return the first non-empty value found for one of the keys in a nested object."""

    wanted = {key.lower() for key in keys}
    if isinstance(value, dict):
        for key, candidate in value.items():
            if str(key).lower() in wanted and clean_text(candidate):
                return candidate
        for candidate in value.values():
            found = _first_nested_key(candidate, *keys)
            if clean_text(found):
                return found
    elif isinstance(value, list):
        for candidate in value:
            found = _first_nested_key(candidate, *keys)
            if clean_text(found):
                return found
    return None


def _highlight_values(value: Any, *section_names: str) -> str:
    """Extract compact values from JSearch highlight sections."""

    if not isinstance(value, dict):
        return ""
    wanted = {name.lower() for name in section_names}
    values: list[Any] = []
    for key, candidate in value.items():
        normalized_key = str(key).lower().replace("_", " ").strip()
        if normalized_key in wanted:
            values.append(candidate)
    return _pipe_values(values)


def _source_id(value: Any) -> str:
    if isinstance(value, dict):
        return first_nonempty(value.get("$oid"), value.get("oid"), value.get("id"), value.get("value"))
    return clean_text(value)


def _mongo_number(value: Any) -> Any:
    if isinstance(value, dict):
        for key in ["$numberDecimal", "$numberDouble", "$numberInt", "$numberLong", "value"]:
            if key in value:
                return value.get(key)
    return value


def _float_or_none(value: Any) -> float | None:
    raw = clean_text(_mongo_number(value))
    if not raw:
        return None
    raw = raw.replace("$", "").replace(",", "").strip()
    try:
        return float(raw)
    except ValueError:
        return None


def _compact_number(value: Any) -> str:
    text = clean_text(_mongo_number(value))
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return f"{number:.8f}".rstrip("0").rstrip(".")


def _bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = clean_text(value).strip().lower()
    if text in {"true", "false", "1", "0", "yes", "no"}:
        return {"1": "true", "0": "false", "yes": "true", "no": "false"}.get(text, text)
    return clean_text(value)


def _format_number(value: float | int | None) -> str:
    if value is None or value <= 0:
        return ""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _json_text(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        return clean_text(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _signal_items(*values: Any) -> list[str]:
    items: list[str] = []
    for value in values:
        for raw_item in _flatten_values(value):
            for part in SIGNAL_SPLIT_RE.split(raw_item):
                item = clean_text(part)
                if item and item.lower() not in {"none", "null", "n/a", "na", "__class__"}:
                    items.append(item)
    return items


def _dedupe_pipe(*values: Any, max_items: int = 40) -> str:
    seen: set[str] = set()
    output: list[str] = []
    for item in _signal_items(*values):
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            output.append(item)
        if len(output) >= max_items:
            break
    return "|".join(output)


def _flatten_values(value: Any) -> list[str]:
    values: list[str] = []
    if value is None:
        return values
    if isinstance(value, list):
        for item in value:
            values.extend(_flatten_values(item))
        return values
    if isinstance(value, dict):
        preferred = first_nonempty(
            value.get("name"),
            value.get("title"),
            value.get("label"),
            value.get("value"),
            value.get("description"),
            value.get("localizationValue"),
            value.get("type"),
            value.get("typename"),
        )
        if preferred:
            return [preferred]
        for item in value.values():
            values.extend(_flatten_values(item))
        return values
    text = clean_text(value)
    if text and text.lower() not in {"none", "null", "n/a", "na"}:
        values.append(text)
    return values


def _pipe_values(value: Any) -> str:
    seen: set[str] = set()
    output: list[str] = []
    for item in _flatten_values(value):
        key = item.lower()
        if key not in seen:
            seen.add(key)
            output.append(item)
    return "|".join(output)


def _company_urls(value: Any) -> str:
    urls: list[str] = []
    if isinstance(value, dict):
        candidates = [value.get("url"), value.get("sameAs"), value.get("website")]
        candidates.extend(value.values())
    elif isinstance(value, list):
        candidates = value
    else:
        candidates = [value]
    for candidate in candidates:
        for text in _flatten_values(candidate):
            if text.startswith(("http://", "https://", "www.")):
                urls.append(text)
    return "|".join(dict.fromkeys(urls))


def _format_salary_value(value: Any) -> str:
    cleaned = clean_text(value)
    if not cleaned:
        return ""
    try:
        number = float(cleaned)
    except ValueError:
        return cleaned
    return f"${number:,.0f}"


def _salary_range_text(salary_min: Any, salary_max: Any) -> str:
    min_text = _format_salary_value(salary_min)
    max_text = _format_salary_value(salary_max)
    if min_text and max_text:
        if min_text == max_text:
            return min_text
        return f"{min_text} - {max_text}"
    return min_text or max_text


def _salary_method(provider: str, salary_min: Any, salary_max: Any, period: str = "") -> str:
    if not clean_text(salary_min) and not clean_text(salary_max):
        return ""
    normalized_period = _normalize_salary_period(period)
    if normalized_period and normalized_period != "year":
        return f"{provider}_salary_annualized_from_{normalized_period}"
    return f"{provider}_salary"


def _normalize_salary_period(*values: Any) -> str:
    text = " ".join(clean_text(value).lower() for value in values if clean_text(value))
    if not text:
        return ""
    if any(token in text for token in ["hour", "hourly", "/hr", "/hour"]):
        return "hour"
    if any(token in text for token in ["day", "daily", "/day"]):
        return "day"
    if any(token in text for token in ["week", "weekly", "/week"]):
        return "week"
    if any(token in text for token in ["month", "monthly", "/month"]):
        return "month"
    if any(token in text for token in ["year", "annual", "annually", "/yr", "/year"]):
        return "year"
    return clean_text(values[0]).lower() if values else ""


def _annualize_salary(value: float | None, period: str) -> float | None:
    if value is None or value <= 0:
        return None
    normalized = _normalize_salary_period(period)
    factor = ANNUALIZATION_FACTORS.get(normalized)
    if factor:
        return value * factor
    if value >= 1000:
        return value
    return None


def _salary_text_number(raw: str) -> float | None:
    text = clean_text(raw).lower().replace(",", "").replace(" ", "")
    multiplier = 1.0
    if text.endswith("k"):
        text = text[:-1]
        multiplier = 1000.0
    try:
        return float(text) * multiplier
    except ValueError:
        return None


def _parse_source_salary_text(raw_text: str, period: str) -> tuple[str, str, str]:
    """Parse source salary text using source period hints before generic fallback."""

    text = clean_text(raw_text)
    if not text:
        return "", "", ""
    normalized_period = _normalize_salary_period(period, text)
    if not normalized_period:
        return "", "", ""

    number = r"(\d+(?:,\d{3})*(?:\.\d+)?\s*k?|\d+(?:\.\d+)?\s*k?)"
    range_pattern = re.compile(
        rf"(?:{SALARY_CURRENCY_TOKEN}\s*)?{number}\s*(?:-|to|through|\u2013|\u2014)\s*"
        rf"(?:{SALARY_CURRENCY_TOKEN}\s*)?{number}",
        re.IGNORECASE,
    )
    currency_single_pattern = re.compile(rf"{SALARY_CURRENCY_TOKEN}\s*{number}", re.IGNORECASE)
    period_single_pattern = re.compile(
        rf"{number}\s*(?:/|per\s+)(?:hour|hr|day|week|month|year|yr)",
        re.IGNORECASE,
    )

    match = range_pattern.search(text)
    values: list[float] = []
    if match:
        values = [value for value in [_salary_text_number(match.group(1)), _salary_text_number(match.group(2))] if value]
        raw_match = match.group(0)
    else:
        match = currency_single_pattern.search(text) or period_single_pattern.search(text)
        values = [_salary_text_number(match.group(1))] if match else []
        raw_match = match.group(0) if match else ""

    annualized = sorted(value for value in (_annualize_salary(value, normalized_period) for value in values) if value)
    if not annualized:
        return "", "", ""
    if len(annualized) == 1:
        return _format_number(annualized[0]), "", raw_match
    return _format_number(annualized[0]), _format_number(annualized[-1]), raw_match


def _format_salary_raw(raw_text: str, salary_min: str, salary_max: str, period: str) -> str:
    if raw_text:
        return raw_text
    range_text = _salary_range_text(salary_min, salary_max)
    if not range_text:
        return ""
    normalized = _normalize_salary_period(period)
    if normalized and normalized != "year":
        return f"{range_text} per {normalized}"
    return range_text


def _schema_salary_values(base_salary: Any) -> tuple[float | None, float | None, str, str]:
    if not isinstance(base_salary, dict):
        return None, None, "", ""
    value = base_salary.get("value")
    currency = first_nonempty(base_salary.get("currency"))
    if isinstance(value, dict):
        minimum = _float_or_none(value.get("minValue"))
        maximum = _float_or_none(value.get("maxValue"))
        exact = _float_or_none(value.get("value"))
        unit = first_nonempty(value.get("unitText"))
        if exact and not minimum and not maximum:
            minimum = exact
            maximum = exact
        return minimum, maximum, currency, unit
    exact = _float_or_none(value)
    return exact, exact, currency, ""


def _schema_org_company_urls(raw: dict[str, Any]) -> str:
    hiring_org = _get_path(raw, "json.schemaOrg.hiringOrganization")
    candidates: list[Any] = [_get_path(raw, "orgCompany.urls")]
    if isinstance(hiring_org, dict):
        candidates.extend([hiring_org.get("url"), hiring_org.get("sameAs")])
    urls: list[str] = []
    for candidate in candidates:
        urls.extend(_company_urls(candidate).split("|"))
    return "|".join(dict.fromkeys(url for url in urls if url))


def _extract_source_salary(raw: dict[str, Any]) -> dict[str, str]:
    """Extract source-backed salary fields and annualized ranking values."""

    salary = raw.get("salary") if isinstance(raw.get("salary"), dict) else {}
    schema_min, schema_max, schema_currency, schema_unit = _schema_salary_values(
        _get_path(raw, "json.schemaOrg.baseSalary")
    )
    salary_currency = first_nonempty(_get_path(raw, "json.schemaOrg.salaryCurrency"), schema_currency)
    raw_salary_text = first_nonempty(salary.get("text"))
    raw_salary_value = _float_or_none(salary.get("value"))
    raw_salary_period = first_nonempty(salary.get("period"))
    period = _normalize_salary_period(raw_salary_period, schema_unit, raw_salary_text)

    source_min = schema_min
    source_max = schema_max
    if raw_salary_value and not source_min and not source_max:
        source_min = raw_salary_value
        source_max = raw_salary_value

    annual_min = _annualize_salary(source_min, period)
    annual_max = _annualize_salary(source_max, period)
    if annual_min and annual_max and annual_min > annual_max:
        annual_min, annual_max = annual_max, annual_min

    method = ""
    if annual_min or annual_max:
        normalized_period = _normalize_salary_period(period)
        if normalized_period and normalized_period != "year":
            method = f"source_salary_annualized_from_{normalized_period}"
        else:
            method = "source_salary"

    return {
        "salary_min": _format_number(annual_min),
        "salary_max": _format_number(annual_max),
        "salary_raw": _format_salary_raw(raw_salary_text, _format_number(source_min), _format_number(source_max), period),
        "salary_period": period,
        "salary_normalization_method": method,
        "raw_salary_text": raw_salary_text,
        "raw_salary_value": _format_number(raw_salary_value),
        "raw_salary_period": raw_salary_period,
        "schema_org_salary_min": _format_number(schema_min),
        "schema_org_salary_max": _format_number(schema_max),
        "schema_org_salary_currency": schema_currency,
        "schema_org_salary_unit": schema_unit,
        "salary_currency": salary_currency,
    }


def _normalized_signal_fields(record: dict[str, Any]) -> None:
    page_data_keywords = record.get("_page_data_keywords", "")
    requirement_terms = first_nonempty(record.get("_requirement_terms"), record.get("raw_requirements"))
    title_terms = _dedupe_pipe(
        record.get("raw_jobnames"),
        record.get("schema_org_occupational_category"),
        record.get("position_department_raw"),
        record.get("title"),
        max_items=30,
    )
    record["normalized_skills"] = _dedupe_pipe(
        record.get("raw_skills"),
        record.get("schema_org_skills"),
        record.get("extracted_skills"),
        max_items=50,
    )
    record["normalized_industries"] = _dedupe_pipe(
        record.get("schema_org_industry"),
        record.get("raw_industries"),
        record.get("raw_categories"),
        max_items=30,
    )
    record["normalized_role_terms"] = title_terms
    record["normalized_keywords"] = _dedupe_pipe(
        record.get("raw_keywords"),
        page_data_keywords,
        requirement_terms,
        record.get("raw_benefits"),
        record.get("schema_org_education_requirements"),
        record.get("extracted_skills"),
        max_items=60,
    )


def _structured_signal_sources(record: dict[str, Any]) -> None:
    source_labels: list[str] = []
    checks = [
        ("raw_skills", "raw_skills"),
        ("schema_org_skills", "schema_org_skills"),
        ("raw_industries", "raw_industries"),
        ("schema_org_industry", "schema_org_industry"),
        ("raw_jobnames", "raw_jobnames"),
        ("schema_org_occupational_category", "schema_org_occupational_category"),
        ("position_department_raw", "position_department"),
        ("raw_keywords", "raw_keywords"),
        ("_page_data_keywords", "page_data_keywords"),
        ("raw_requirements", "raw_requirements"),
        ("raw_benefits", "raw_benefits"),
        ("schema_org_education_requirements", "schema_org_education_requirements"),
        ("raw_categories", "raw_categories"),
    ]
    for field, label in checks:
        if clean_text(record.get(field)):
            source_labels.append(label)
    raw_source = clean_text(record.get("raw_source")).lower()
    if raw_source == "adzuna_api" and clean_text(record.get("raw_categories")):
        source_labels.append("adzuna_category")
    if raw_source == "jsearch_api":
        if clean_text(record.get("raw_skills")):
            source_labels.append("jsearch_required_skills")
        if clean_text(record.get("raw_requirements")):
            source_labels.append("jsearch_highlights")
    if clean_text(record.get("extracted_skills")):
        source_labels.append("parser_fallback")

    source_labels = list(dict.fromkeys(source_labels))
    source_backed_count = sum(1 for label in source_labels if label != "parser_fallback")
    if source_backed_count >= 5:
        confidence = "high"
    elif source_backed_count >= 3:
        confidence = "medium"
    elif source_backed_count >= 1:
        confidence = "low"
    elif "parser_fallback" in source_labels:
        confidence = "parser_only"
    else:
        confidence = "none"
    record["structured_signal_sources"] = "|".join(source_labels)
    record["structured_signal_confidence"] = confidence


def _valid_title(value: Any) -> str:
    title = clean_text(value)
    if title.lower() in INVALID_TITLE_VALUES:
        return ""
    return title


def _choose_title(*candidates: Any) -> tuple[str, bool]:
    """Choose the first valid title and report whether a fallback was needed."""

    first_candidate = clean_text(candidates[0]) if candidates else ""
    for candidate in candidates:
        title = _valid_title(candidate)
        if title:
            return title, title != first_candidate
    return "", False


def _finalize(record: dict[str, Any]) -> dict[str, Any]:
    description = clean_text(record.get("description_text") or record.get("description"))
    record["description_text"] = description
    record["description"] = description
    if not record.get("salary_min") and not record.get("salary_max"):
        salary_min, salary_max, salary_raw = _parse_source_salary_text(
            record.get("salary_raw", ""), record.get("salary_period", "")
        )
        if salary_min or salary_max:
            record["salary_min"] = salary_min
            record["salary_max"] = salary_max
            record["salary_raw"] = first_nonempty(record.get("salary_raw"), salary_raw)
            normalized_period = _normalize_salary_period(record.get("salary_period"))
            method = "source_salary_text"
            if normalized_period and normalized_period != "year":
                method = f"source_salary_text_annualized_from_{normalized_period}"
            record["salary_normalization_method"] = first_nonempty(record.get("salary_normalization_method"), method)

    source_salary_hint = first_nonempty(
        record.get("raw_salary_text"),
        record.get("raw_salary_value"),
        record.get("raw_salary_period"),
        record.get("salary_period"),
    )
    if not record.get("salary_min") and not record.get("salary_max") and not source_salary_hint:
        salary_min, salary_max, salary_raw = parse_salary_from_text(description)
        record["salary_min"] = salary_min
        record["salary_max"] = salary_max
        record["salary_raw"] = first_nonempty(record.get("salary_raw"), salary_raw)
        if salary_min or salary_max:
            record["salary_normalization_method"] = first_nonempty(
                record.get("salary_normalization_method"), "text_regex_salary_fallback"
            )
    record["extracted_skills"] = extract_skills(record.get("title"), description)
    record["seniority"] = infer_seniority(record.get("title"), description)
    record["years_required"] = infer_years_required(description)
    record["is_remote"] = infer_remote(record.get("title"), description, record.get("location"))
    record["company_type"] = infer_company_type(record.get("company"), description)
    record["sponsorship_signal"] = infer_sponsorship_signal(description)
    _normalized_signal_fields(record)
    _structured_signal_sources(record)
    record["embedding_text"] = make_embedding_text(record)
    ordered = order_record(record)
    if record.get("_title_repaired"):
        ordered["_title_repaired"] = record["_title_repaired"]
    return ordered


def normalize_kaggle_record(raw: dict[str, Any], line_no: int, ingested_at: str | None = None) -> dict[str, Any]:
    """Normalize one raw Kaggle JSONL record."""

    ingested_at = ingested_at or utc_now_iso()
    record = empty_job_record()
    source = first_nonempty(raw.get("source"), "kaggle")
    source_record_id = first_nonempty(raw.get("idInSource"), _get_path(raw, "_id.$oid"), str(line_no))
    title, title_repaired = _choose_title(
        raw.get("name"),
        raw.get("title"),
        _get_path(raw, "position.name"),
        _get_path(raw, "position.title"),
        _get_path(raw, "json.schemaOrg.title"),
        _get_path(raw, "json.schemaOrg.name"),
        _get_path(raw, "json.title"),
        _get_path(raw, "json.name"),
    )
    company = first_nonempty(
        _get_path(raw, "orgCompany.name"),
        _get_path(raw, "orgCompany.nameOrg"),
        _get_path(raw, "json.schemaOrg.hiringOrganization.name"),
    )
    city = first_nonempty(_get_path(raw, "orgAddress.city"))
    state = first_nonempty(_get_path(raw, "orgAddress.state"))
    country = first_nonempty(_get_path(raw, "orgAddress.countryCode"), _get_path(raw, "orgAddress.country"))
    location = first_nonempty(
        _get_path(raw, "orgAddress.addressLine"),
        _get_path(raw, "orgAddress.formatted"),
        ", ".join(part for part in [city, state, country] if part),
    )
    description = first_nonempty(raw.get("text"), raw.get("html"), _get_path(raw, "json.schemaOrg.description"))
    raw_work_type = first_nonempty(_get_path(raw, "position.workType"))
    raw_contract_type = first_nonempty(_get_path(raw, "position.contractType"))
    schema_org_employment_type = _pipe_values(_get_path(raw, "json.schemaOrg.employmentType"))
    schema_job_location = _get_path(raw, "json.schemaOrg.jobLocation")
    schema_valid_through = first_nonempty(_get_path(raw, "json.schemaOrg.validThrough"))
    posting_date_raw = first_nonempty(
        _get_path(raw, "json.schemaOrg.datePosted"),
        _get_path(raw, "dateCreated.$date"),
        _get_path(raw, "dateScraped.$date"),
        _get_path(raw, "dateUploaded.$date"),
    )
    source_salary = _extract_source_salary(raw)
    record.update(
        {
            "job_id": stable_hash(f"kaggle|{source}|{source_record_id}", length=24),
            "source": source,
            "source_record_id": source_record_id,
            "reference_id": first_nonempty(raw.get("referenceID")),
            "schema_org_identifier": _json_text(_get_path(raw, "json.schemaOrg.identifier")),
            "company_id": _source_id(raw.get("companyID")),
            "location_id": _source_id(raw.get("locationID")),
            "is_current_api": False,
            "ingested_at": ingested_at,
            "date_posted_or_scraped": first_nonempty(
                _get_path(raw, "dateScraped.$date"),
                _get_path(raw, "dateCreated.$date"),
                _get_path(raw, "dateUploaded.$date"),
            ),
            "posting_date_raw": posting_date_raw,
            "expiration_date_raw": first_nonempty(schema_valid_through, _get_path(raw, "dateExpired.$date")),
            "query": "",
            "title": title,
            "company": company,
            "employer": company,
            "location": location,
            "country": country,
            "state": state,
            "city": city,
            "postal_code": first_nonempty(_get_path(raw, "orgAddress.postCode"), _first_nested_key(schema_job_location, "postalCode")),
            "county": first_nonempty(_get_path(raw, "orgAddress.county")),
            "latitude": _compact_number(first_nonempty(_get_path(raw, "orgAddress.geoPoint.lat"), _first_nested_key(schema_job_location, "latitude", "lat"))),
            "longitude": _compact_number(first_nonempty(_get_path(raw, "orgAddress.geoPoint.lng"), _first_nested_key(schema_job_location, "longitude", "lng"))),
            "raw_source_country": first_nonempty(raw.get("sourceCC")),
            "raw_locale": first_nonempty(raw.get("locale")),
            "salary_min": source_salary["salary_min"],
            "salary_max": source_salary["salary_max"],
            "salary_raw": source_salary["salary_raw"],
            "salary_period": source_salary["salary_period"],
            "salary_normalization_method": source_salary["salary_normalization_method"],
            "raw_salary_text": source_salary["raw_salary_text"],
            "raw_salary_value": source_salary["raw_salary_value"],
            "raw_salary_period": source_salary["raw_salary_period"],
            "schema_org_salary_min": source_salary["schema_org_salary_min"],
            "schema_org_salary_max": source_salary["schema_org_salary_max"],
            "schema_org_salary_currency": source_salary["schema_org_salary_currency"],
            "schema_org_salary_unit": source_salary["schema_org_salary_unit"],
            "salary_currency": source_salary["salary_currency"],
            "employment_type": infer_employment_type(
                title,
                description,
                first_nonempty(raw_contract_type, raw_work_type, schema_org_employment_type, _pipe_values(_get_path(raw, "orgTags.WORK_TYPES"))),
            ),
            "position_work_type_raw": raw_work_type,
            "position_career_level_raw": first_nonempty(_get_path(raw, "position.careerLevel")),
            "position_contract_type_raw": raw_contract_type,
            "position_department_raw": first_nonempty(_get_path(raw, "position.department")),
            "schema_org_employment_type": schema_org_employment_type,
            "description": description,
            "description_text": description,
            "link": first_nonempty(raw.get("url")),
            "company_url": _schema_org_company_urls(raw),
            "company_description_raw": first_nonempty(_get_path(raw, "orgCompany.description")),
            "company_size_raw": _pipe_values(_get_path(raw, "orgCompany.info.companySize")),
            "raw_source": "kaggle",
            "raw_categories": _pipe_values(_get_path(raw, "orgTags.CATEGORIES")),
            "raw_work_types": _pipe_values(_get_path(raw, "orgTags.WORK_TYPES")),
            "raw_qualifications": _pipe_values(_get_path(raw, "orgTags.QUALIFICATIONS")),
            "raw_skills": _pipe_values(_get_path(raw, "orgTags.SKILLS")),
            "raw_industries": _pipe_values(_get_path(raw, "orgTags.INDUSTRIES")),
            "raw_jobnames": _pipe_values(_get_path(raw, "orgTags.JOBNAMES")),
            "raw_keywords": _pipe_values(_get_path(raw, "orgTags.KEYWORDS")),
            "raw_requirements": _pipe_values(_get_path(raw, "orgTags.REQUIREMENTS")),
            "raw_languages": _pipe_values(_get_path(raw, "orgTags.LANGUAGES")),
            "raw_benefits": _pipe_values(_get_path(raw, "orgTags.COMPANY_BENEFITS")),
            "schema_org_skills": _pipe_values(_get_path(raw, "json.schemaOrg.skills")),
            "schema_org_experience_requirements": _json_text(_get_path(raw, "json.schemaOrg.experienceRequirements")),
            "schema_org_industry": _pipe_values(_get_path(raw, "json.schemaOrg.industry")),
            "schema_org_occupational_category": _pipe_values(_get_path(raw, "json.schemaOrg.occupationalCategory")),
            "schema_org_education_requirements": _json_text(_get_path(raw, "json.schemaOrg.educationRequirements")),
            "schema_org_valid_through": schema_valid_through,
            "direct_apply_raw": _bool_text(_get_path(raw, "json.schemaOrg.directApply")),
            "_page_data_keywords": _dedupe_pipe(
                _first_nested_key(_get_path(raw, "json.pageData"), "keywords", "keySkills", "skills"),
                max_items=40,
            ),
            "_title_repaired": title_repaired,
        }
    )
    return _finalize(record)


def normalize_adzuna_record(raw: dict[str, Any], query: str = "", ingested_at: str | None = None) -> dict[str, Any]:
    """Normalize one Adzuna API result."""

    ingested_at = ingested_at or utc_now_iso()
    company = first_nonempty(_get_path(raw, "company.display_name"))
    location = first_nonempty(_get_path(raw, "location.display_name"))
    title, title_repaired = _choose_title(raw.get("title"), raw.get("name"), raw.get("job_title"))
    description = first_nonempty(raw.get("description"))
    salary_min = clean_text(raw.get("salary_min"))
    salary_max = clean_text(raw.get("salary_max"))
    salary_period = "year" if salary_min or salary_max else ""
    salary_raw = _salary_range_text(salary_min, salary_max)
    location_area = raw.get("location", {}).get("area") if isinstance(raw.get("location"), dict) else []
    country = first_nonempty(location_area[0] if len(location_area) > 0 else "")
    state = first_nonempty(location_area[1] if len(location_area) > 1 else "")
    city = first_nonempty(location_area[-1] if len(location_area) > 2 else "")
    county = first_nonempty(location_area[2] if len(location_area) > 3 else "")
    category = _dedupe_pipe(_get_path(raw, "category.label"), _get_path(raw, "category.tag"))
    contract_raw = _dedupe_pipe(raw.get("contract_time"), raw.get("contract_type"))
    salary_currency = first_nonempty(raw.get("salary_currency"), raw.get("currency"), raw.get("salary_currency_code"))
    record = empty_job_record()
    record.update(
        {
            "job_id": stable_hash(f"adzuna|{raw.get('id')}|{title}|{company}", length=24),
            "source": "adzuna",
            "source_record_id": first_nonempty(raw.get("id")),
            "is_current_api": True,
            "ingested_at": ingested_at,
            "date_posted_or_scraped": first_nonempty(raw.get("created")),
            "posting_date_raw": first_nonempty(raw.get("created")),
            "query": query,
            "title": title,
            "company": company,
            "employer": company,
            "location": location,
            "country": country,
            "state": state,
            "city": city,
            "county": county,
            "latitude": _compact_number(raw.get("latitude")),
            "longitude": _compact_number(raw.get("longitude")),
            "raw_source_country": country,
            "salary_min": salary_min,
            "salary_max": salary_max,
            "salary_raw": salary_raw,
            "salary_period": salary_period,
            "salary_normalization_method": _salary_method("adzuna", salary_min, salary_max, salary_period),
            "raw_salary_text": salary_raw,
            "raw_salary_value": first_nonempty(raw.get("salary_min"), raw.get("salary_max")),
            "raw_salary_period": salary_period,
            "salary_currency": salary_currency,
            "salary_is_predicted": first_nonempty(raw.get("salary_is_predicted")),
            "employment_type": infer_employment_type(
                title,
                description,
                contract_raw,
            ),
            "position_work_type_raw": contract_raw,
            "position_contract_type_raw": contract_raw,
            "description": description,
            "description_text": description,
            "link": first_nonempty(raw.get("redirect_url"), raw.get("adref")),
            "raw_source": "adzuna_api",
            "raw_categories": category,
            "raw_industries": category,
            "_title_repaired": title_repaired,
        }
    )
    return _finalize(record)


def normalize_jsearch_record(raw: dict[str, Any], query: str = "", ingested_at: str | None = None) -> dict[str, Any]:
    """Normalize one JSearch/RapidAPI result."""

    ingested_at = ingested_at or utc_now_iso()
    title, title_repaired = _choose_title(raw.get("job_title"), raw.get("title"), raw.get("name"))
    company = first_nonempty(raw.get("employer_name"))
    city = first_nonempty(raw.get("job_city"))
    state = first_nonempty(raw.get("job_state"))
    country = first_nonempty(raw.get("job_country"))
    location = first_nonempty(", ".join(part for part in [city, state, country] if part), raw.get("job_location"))
    description = first_nonempty(raw.get("job_description"))
    salary_min = clean_text(raw.get("job_min_salary"))
    salary_max = clean_text(raw.get("job_max_salary"))
    salary_period = _normalize_salary_period(raw.get("job_salary_period"), raw.get("job_salary"))
    salary_raw = first_nonempty(raw.get("job_salary"), _salary_range_text(salary_min, salary_max))
    employment_raw = _dedupe_pipe(raw.get("job_employment_type"), raw.get("job_employment_types"))
    highlights = raw.get("job_highlights")
    raw_skills = _dedupe_pipe(raw.get("job_required_skills"), _highlight_values(highlights, "Skills", "Qualifications"))
    requirement_terms = _dedupe_pipe(
        _highlight_values(highlights, "Qualifications", "Requirements", "Responsibilities"),
        _flatten_values(raw.get("job_required_experience")),
        max_items=60,
    )
    requirement_payload: dict[str, Any] = {}
    if highlights not in (None, "", [], {}):
        requirement_payload["job_highlights"] = highlights
    if raw.get("job_required_experience") not in (None, "", [], {}):
        requirement_payload["job_required_experience"] = raw.get("job_required_experience")
    raw_requirements = _json_text(requirement_payload)
    raw_industries = _dedupe_pipe(raw.get("job_industry"), raw.get("job_industries"), raw.get("job_category"), raw.get("job_categories"))
    posting_date_raw = first_nonempty(raw.get("job_posted_at_datetime_utc"), raw.get("job_posted_at_timestamp"))
    record = empty_job_record()
    record.update(
        {
            "job_id": stable_hash(f"jsearch|{raw.get('job_id')}|{title}|{company}", length=24),
            "source": "jsearch",
            "source_record_id": first_nonempty(raw.get("job_id")),
            "is_current_api": True,
            "ingested_at": ingested_at,
            "date_posted_or_scraped": posting_date_raw,
            "posting_date_raw": posting_date_raw,
            "query": query,
            "title": title,
            "company": company,
            "employer": company,
            "location": location,
            "country": country,
            "state": state,
            "city": city,
            "latitude": _compact_number(raw.get("job_latitude")),
            "longitude": _compact_number(raw.get("job_longitude")),
            "salary_min": salary_min,
            "salary_max": salary_max,
            "salary_raw": salary_raw,
            "salary_period": salary_period,
            "salary_normalization_method": _salary_method("jsearch", salary_min, salary_max, salary_period),
            "raw_salary_text": salary_raw,
            "raw_salary_value": first_nonempty(raw.get("job_min_salary"), raw.get("job_max_salary")),
            "raw_salary_period": first_nonempty(raw.get("job_salary_period")),
            "salary_currency": first_nonempty(raw.get("job_salary_currency")),
            "employment_type": infer_employment_type(title, description, employment_raw),
            "position_work_type_raw": employment_raw,
            "position_contract_type_raw": employment_raw,
            "description": description,
            "description_text": description,
            "link": first_nonempty(raw.get("job_apply_link"), raw.get("job_google_link")),
            "raw_source": "jsearch_api",
            "raw_categories": _dedupe_pipe(raw.get("job_category"), raw.get("job_categories")),
            "raw_skills": raw_skills,
            "raw_industries": raw_industries,
            "raw_keywords": _dedupe_pipe(raw.get("job_keywords")),
            "raw_requirements": raw_requirements,
            "schema_org_education_requirements": _json_text(raw.get("job_required_education")),
            "direct_apply_raw": _bool_text(raw.get("job_apply_is_direct")),
            "_requirement_terms": requirement_terms,
            "_title_repaired": title_repaired,
        }
    )
    return _finalize(record)
