"""Phase 2.17 company metadata cache foundation.

This module intentionally does not modify Phase 1 ingestion or Phase 2 ranking.
It builds a company-level sidecar cache that can be bootstrapped privately later
and exported as a no-key frozen cache for the default offline path.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

from jobpilot.utils.io import ensure_parent, write_csv, write_json
from jobpilot.utils.text import clean_text, normalize_for_key


APIFY_COMPANY_ACTOR_ID = "harvestapi/linkedin-company"
APIFY_COMPANY_PROVIDER = "apify_harvestapi_linkedin_company"
NO_LIVE_CALLS_NOTE = (
    "Phase 2.17A creates the metadata cache foundation and a dry-run Apify plan only. "
    "It does not call Apify, scrape LinkedIn, or require API keys."
)

JOB_BOARD_OR_REFERRAL_DOMAINS = {
    "careerbuilder.com",
    "careerbuilder.co.uk",
    "careerbuilder.ca",
    "careerbuilder.vn",
    "careerbuilder.se",
    "snagajob.com",
    "efinancialcareers.co.uk",
    "jobs.aarp.org",
    "linkedin.com",
    "jobs.de",
    "applitrack.com",
    "indeed.com",
    "glassdoor.com",
    "ziprecruiter.com",
    "monster.com",
    "greenhouse.io",
    "lever.co",
    "workdayjobs.com",
    "myworkdayjobs.com",
}

STAFFING_TERMS = (
    "staffing",
    "recruiting",
    "recruitment",
    "talent solutions",
    "workforce solutions",
    "robert half",
    "manpower",
    "experis",
    "jobot",
    "nesco",
    "judge group",
    "pridestaff",
    "staffmark",
)

FRANCHISE_HEAVY_BRAND_KEYS = {
    "arbys",
    "burger king",
    "carls jr",
    "dairy queen",
    "dominos",
    "dunkin",
    "kfc",
    "little caesars",
    "mcdonalds",
    "papa johns",
    "pizza hut",
    "popeyes",
    "sonic drive in",
    "subway",
    "taco bell",
    "wendys",
}

FRANCHISE_OPERATOR_TERMS = (
    "franchisee",
    "franchise of",
    "franchise operator",
    "dba burger king",
    "dba popeyes",
    "dba taco bell",
    "dba kfc",
    "dba pizza hut",
    "restaurant group",
    "qsr",
)

UNIVERSE_COLUMNS = [
    "company_key",
    "canonical_company_name",
    "snapshot_job_count",
    "snapshot_raw_names_json",
    "official_domain_candidates_json",
    "job_board_domain_hits_json",
    "snapshot_company_types_json",
    "snapshot_industries_json",
    "source_snapshot_path",
    "updated_at",
]

CURRENT_COLUMNS = [
    "company_key",
    "canonical_company_name",
    "snapshot_job_count",
    "linkedin_company_id",
    "linkedin_universal_name",
    "linkedin_url",
    "website_url",
    "website_domain",
    "industry_primary",
    "industries_json",
    "employee_count_linkedin",
    "employee_range_start",
    "employee_range_end",
    "matched_entity_size_bucket",
    "entity_scope",
    "usable_employer_size_bucket",
    "size_usage_policy",
    "size_bucket",
    "company_type_provider",
    "hq_country",
    "hq_state",
    "hq_city",
    "is_staffing_or_recruiting",
    "match_confidence",
    "match_decision",
    "metadata_status",
    "source_provider",
    "source_evidence_id",
    "first_seen_at",
    "last_verified_at",
    "expires_at",
    "notes",
    "updated_at",
]

FROZEN_CURRENT_COLUMNS = [
    "company_key",
    "canonical_company_name",
    "snapshot_job_count",
    "linkedin_company_id",
    "linkedin_universal_name",
    "linkedin_url",
    "website_url",
    "website_domain",
    "industry_primary",
    "industries_json",
    "employee_count_linkedin",
    "employee_range_start",
    "employee_range_end",
    "matched_entity_size_bucket",
    "entity_scope",
    "usable_employer_size_bucket",
    "size_usage_policy",
    "size_bucket",
    "company_type_provider",
    "hq_country",
    "hq_state",
    "hq_city",
    "is_staffing_or_recruiting",
    "match_confidence",
    "match_decision",
    "metadata_status",
    "source_provider",
    "source_evidence_id",
    "last_verified_at",
    "expires_at",
    "notes",
]

EVIDENCE_COLUMNS = [
    "evidence_id",
    "company_key",
    "observed_at",
    "provider",
    "provider_actor_id",
    "provider_run_id",
    "input_mode",
    "input_value",
    "candidate_rank",
    "candidate_linkedin_company_id",
    "candidate_linkedin_universal_name",
    "candidate_name",
    "candidate_linkedin_url",
    "candidate_website",
    "candidate_website_domain",
    "candidate_employee_count",
    "candidate_employee_range_start",
    "candidate_employee_range_end",
    "candidate_industries_json",
    "candidate_company_type",
    "candidate_hq_country",
    "candidate_hq_state",
    "candidate_hq_city",
    "candidate_locations_json",
    "match_features_json",
    "match_score",
    "confidence_label",
    "decision",
    "error_code",
    "error_message",
    "cost_units",
    "raw_payload_hash",
    "raw_payload_json_private",
    "created_at",
]

FROZEN_EVIDENCE_COLUMNS = [column for column in EVIDENCE_COLUMNS if column != "raw_payload_json_private"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_counter(counter: Counter[str], limit: int = 20) -> str:
    rows = [{"value": value, "count": count} for value, count in counter.most_common(limit)]
    return json.dumps(rows, ensure_ascii=False, sort_keys=True)


def _json_loads(value: Any, default: Any) -> Any:
    if not clean_text(value):
        return default
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def _first_json_value(value: Any) -> str:
    payload = _json_loads(value, [])
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            return clean_text(first.get("value") or first.get("name"))
        return clean_text(first)
    return ""


def _domain_from_url(url: Any) -> str:
    text = clean_text(url)
    if not text:
        return ""
    if not re.match(r"^[a-z]+://", text, flags=re.IGNORECASE):
        text = f"https://{text}"
    parsed = urlparse(text)
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _domains_from_pipe_urls(value: Any) -> list[str]:
    domains: list[str] = []
    for part in clean_text(value).split("|"):
        domain = _domain_from_url(part)
        if domain:
            domains.append(domain)
    return domains


def is_job_board_or_referral_domain(domain: str) -> bool:
    domain = clean_text(domain).lower()
    if not domain:
        return False
    return any(domain == blocked or domain.endswith(f".{blocked}") for blocked in JOB_BOARD_OR_REFERRAL_DOMAINS)


def _raw_payload_hash(raw_payload: Any) -> str:
    text = clean_text(raw_payload)
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_evidence_id(row: Mapping[str, Any]) -> str:
    basis = "|".join(
        [
            clean_text(row.get("company_key")),
            clean_text(row.get("provider")),
            clean_text(row.get("observed_at")),
            clean_text(row.get("input_mode")),
            clean_text(row.get("input_value")),
            clean_text(row.get("candidate_rank")),
            clean_text(row.get("candidate_linkedin_url")),
            clean_text(row.get("raw_payload_hash")),
        ]
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]


def _parse_int(value: Any) -> int | None:
    text = clean_text(value).replace(",", "")
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def size_bucket_from_employee_range(start: Any, end: Any, employee_count: Any = "") -> str:
    """Map LinkedIn employee count/range fields into a conservative size bucket."""

    count = _parse_int(employee_count)
    start_int = _parse_int(start)
    end_int = _parse_int(end)
    value = count or start_int or end_int
    if value is None:
        return "unknown"
    if value <= 10:
        return "micro_1_10"
    if value <= 50:
        return "small_11_50"
    if value <= 200:
        return "mid_51_200"
    if value <= 500:
        return "mid_201_500"
    if value <= 1000:
        return "large_501_1000"
    if value <= 5000:
        return "large_1001_5000"
    if value <= 10000:
        return "enterprise_5001_10000"
    return "enterprise_10001_plus"


def is_staffing_or_recruiting(name: Any, industries_json: Any) -> bool:
    text_parts = [normalize_for_key(name)]
    industries = _json_loads(industries_json, [])
    if isinstance(industries, list):
        for item in industries:
            if isinstance(item, dict):
                text_parts.append(normalize_for_key(item.get("value") or item.get("name")))
            else:
                text_parts.append(normalize_for_key(item))
    haystack = " ".join(text_parts)
    return any(term in haystack for term in STAFFING_TERMS)


def infer_entity_scope(
    company_key: Any,
    canonical_name: Any,
    candidate_name: Any,
    industries_json: Any,
    match_features_json: Any = "",
) -> str:
    """Classify the matched company entity for safe size usage.

    The scope separates the entity that LinkedIn returned from the employer
    size that future ranking can safely use. Franchise-heavy brand pages are
    treated as brand context only unless evidence identifies a specific
    franchise operator.
    """

    features = _json_loads(match_features_json, {})
    if isinstance(features, dict):
        explicit = clean_text(features.get("entity_scope") or features.get("entity_scope_hint")).lower()
        if explicit in {
            "corporate_employer",
            "franchise_operator",
            "brand_or_parent",
            "single_location",
            "staffing_agency",
            "job_board_alias",
            "unknown",
        }:
            return explicit

    key = normalize_for_key(company_key)
    canonical = normalize_for_key(canonical_name)
    candidate = normalize_for_key(candidate_name)
    text = " ".join(part for part in [key, canonical, candidate] if part)
    if is_staffing_or_recruiting(text, industries_json):
        return "staffing_agency"
    if any(term in text for term in FRANCHISE_OPERATOR_TERMS):
        return "franchise_operator"
    if key in FRANCHISE_HEAVY_BRAND_KEYS or candidate in FRANCHISE_HEAVY_BRAND_KEYS:
        return "brand_or_parent"
    if re.search(r"\b(store|location|unit|restaurant\s+#|#\s*\d+)\b", text):
        return "single_location"
    return "corporate_employer"


def usable_size_bucket_for_scope(entity_scope: str, matched_entity_size_bucket: str) -> tuple[str, str]:
    """Return the ranking-safe size bucket and explanation policy for a scope."""

    scope = clean_text(entity_scope).lower() or "unknown"
    matched = clean_text(matched_entity_size_bucket) or "unknown"
    if scope == "corporate_employer":
        return matched, "usable_employer_context"
    if scope == "franchise_operator":
        return matched, "usable_franchise_operator_context"
    if scope == "single_location":
        return matched, "usable_single_location_context"
    if scope == "brand_or_parent":
        return "unknown", "brand_context_only"
    if scope == "staffing_agency":
        return "unknown", "staffing_context_only"
    if scope == "job_board_alias":
        return "unknown", "job_board_alias_ignore"
    return "unknown", "unknown_neutral"


def build_company_universe(snapshot_path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    """Build one company-level universe row per normalized snapshot company."""

    companies: dict[str, dict[str, Any]] = {}
    with Path(snapshot_path).open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            if max_rows is not None and index > max_rows:
                break
            raw_company = clean_text(row.get("company")) or clean_text(row.get("employer"))
            company_key = normalize_for_key(raw_company)
            if not company_key:
                continue
            entry = companies.setdefault(
                company_key,
                {
                    "company_key": company_key,
                    "raw_names": Counter(),
                    "official_domains": Counter(),
                    "job_board_domains": Counter(),
                    "company_types": Counter(),
                    "industries": Counter(),
                    "snapshot_job_count": 0,
                },
            )
            entry["snapshot_job_count"] += 1
            entry["raw_names"][raw_company] += 1
            for domain in _domains_from_pipe_urls(row.get("company_url")):
                if is_job_board_or_referral_domain(domain):
                    entry["job_board_domains"][domain] += 1
                else:
                    entry["official_domains"][domain] += 1
            company_type = clean_text(row.get("company_type")) or "unknown"
            entry["company_types"][company_type] += 1
            for industry in clean_text(row.get("raw_industries")).split("|"):
                industry = clean_text(industry)
                if industry:
                    entry["industries"][industry] += 1

    records: list[dict[str, Any]] = []
    for company_key, entry in sorted(companies.items()):
        canonical_name = entry["raw_names"].most_common(1)[0][0]
        records.append(
            {
                "company_key": company_key,
                "canonical_company_name": canonical_name,
                "snapshot_job_count": entry["snapshot_job_count"],
                "snapshot_raw_names_json": _json_counter(entry["raw_names"]),
                "official_domain_candidates_json": _json_counter(entry["official_domains"]),
                "job_board_domain_hits_json": _json_counter(entry["job_board_domains"]),
                "snapshot_company_types_json": _json_counter(entry["company_types"]),
                "snapshot_industries_json": _json_counter(entry["industries"]),
            }
        )
    return records


def connect_cache(sqlite_path: Path) -> sqlite3.Connection:
    ensure_parent(Path(sqlite_path))
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS company_universe (
            company_key TEXT PRIMARY KEY,
            canonical_company_name TEXT NOT NULL,
            snapshot_job_count INTEGER NOT NULL,
            snapshot_raw_names_json TEXT NOT NULL,
            official_domain_candidates_json TEXT NOT NULL,
            job_board_domain_hits_json TEXT NOT NULL,
            snapshot_company_types_json TEXT NOT NULL,
            snapshot_industries_json TEXT NOT NULL,
            source_snapshot_path TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS company_metadata_evidence (
            evidence_id TEXT PRIMARY KEY,
            company_key TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            provider TEXT NOT NULL,
            provider_actor_id TEXT,
            provider_run_id TEXT,
            input_mode TEXT,
            input_value TEXT,
            candidate_rank INTEGER,
            candidate_linkedin_company_id TEXT,
            candidate_linkedin_universal_name TEXT,
            candidate_name TEXT,
            candidate_linkedin_url TEXT,
            candidate_website TEXT,
            candidate_website_domain TEXT,
            candidate_employee_count INTEGER,
            candidate_employee_range_start INTEGER,
            candidate_employee_range_end INTEGER,
            candidate_industries_json TEXT,
            candidate_company_type TEXT,
            candidate_hq_country TEXT,
            candidate_hq_state TEXT,
            candidate_hq_city TEXT,
            candidate_locations_json TEXT,
            match_features_json TEXT,
            match_score REAL,
            confidence_label TEXT,
            decision TEXT NOT NULL,
            error_code TEXT,
            error_message TEXT,
            cost_units REAL,
            raw_payload_hash TEXT,
            raw_payload_json_private TEXT,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_company_metadata_evidence_company
            ON company_metadata_evidence(company_key, decision, observed_at);

        CREATE TABLE IF NOT EXISTS company_metadata_current (
            company_key TEXT PRIMARY KEY,
            canonical_company_name TEXT NOT NULL,
            snapshot_job_count INTEGER NOT NULL,
            linkedin_company_id TEXT,
            linkedin_universal_name TEXT,
            linkedin_url TEXT,
            website_url TEXT,
            website_domain TEXT,
            industry_primary TEXT,
            industries_json TEXT,
            employee_count_linkedin INTEGER,
            employee_range_start INTEGER,
            employee_range_end INTEGER,
            matched_entity_size_bucket TEXT DEFAULT 'unknown',
            entity_scope TEXT DEFAULT 'unknown',
            usable_employer_size_bucket TEXT DEFAULT 'unknown',
            size_usage_policy TEXT DEFAULT 'unknown_neutral',
            size_bucket TEXT NOT NULL,
            company_type_provider TEXT,
            hq_country TEXT,
            hq_state TEXT,
            hq_city TEXT,
            is_staffing_or_recruiting INTEGER NOT NULL DEFAULT 0,
            match_confidence TEXT NOT NULL,
            match_decision TEXT NOT NULL,
            metadata_status TEXT NOT NULL,
            source_provider TEXT,
            source_evidence_id TEXT,
            first_seen_at TEXT,
            last_verified_at TEXT,
            expires_at TEXT,
            notes TEXT,
            updated_at TEXT NOT NULL
        );
        """
    )
    _ensure_current_columns(conn)
    conn.executescript(
        """
        DROP VIEW IF EXISTS company_metadata_current_view;
        CREATE VIEW company_metadata_current_view AS
            SELECT
                u.company_key,
                COALESCE(c.canonical_company_name, u.canonical_company_name) AS canonical_company_name,
                u.snapshot_job_count,
                c.linkedin_company_id,
                c.linkedin_universal_name,
                c.linkedin_url,
                c.website_url,
                c.website_domain,
                c.industry_primary,
                c.industries_json,
                c.employee_count_linkedin,
                c.employee_range_start,
                c.employee_range_end,
                COALESCE(c.matched_entity_size_bucket, 'unknown') AS matched_entity_size_bucket,
                COALESCE(c.entity_scope, 'unknown') AS entity_scope,
                COALESCE(c.usable_employer_size_bucket, 'unknown') AS usable_employer_size_bucket,
                COALESCE(c.size_usage_policy, 'unknown_neutral') AS size_usage_policy,
                COALESCE(c.size_bucket, 'unknown') AS size_bucket,
                c.company_type_provider,
                c.hq_country,
                c.hq_state,
                c.hq_city,
                COALESCE(c.is_staffing_or_recruiting, 0) AS is_staffing_or_recruiting,
                COALESCE(c.match_confidence, 'none') AS match_confidence,
                COALESCE(c.match_decision, 'unknown') AS match_decision,
                COALESCE(c.metadata_status, 'unknown') AS metadata_status,
                c.source_provider,
                c.source_evidence_id,
                c.first_seen_at,
                c.last_verified_at,
                c.expires_at,
                c.notes,
                COALESCE(c.updated_at, u.updated_at) AS updated_at
            FROM company_universe u
            LEFT JOIN company_metadata_current c ON c.company_key = u.company_key;
        """
    )
    conn.commit()


def _ensure_current_columns(conn: sqlite3.Connection) -> None:
    existing = {
        row["name"] if isinstance(row, sqlite3.Row) else row[1]
        for row in conn.execute("PRAGMA table_info(company_metadata_current)").fetchall()
    }
    additions = {
        "matched_entity_size_bucket": "TEXT DEFAULT 'unknown'",
        "entity_scope": "TEXT DEFAULT 'unknown'",
        "usable_employer_size_bucket": "TEXT DEFAULT 'unknown'",
        "size_usage_policy": "TEXT DEFAULT 'unknown_neutral'",
    }
    for column, definition in additions.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE company_metadata_current ADD COLUMN {column} {definition}")


def replace_company_universe(
    conn: sqlite3.Connection,
    records: Iterable[Mapping[str, Any]],
    snapshot_path: Path,
    updated_at: str | None = None,
) -> int:
    timestamp = updated_at or utc_now_iso()
    rows = []
    for record in records:
        row = {column: record.get(column, "") for column in UNIVERSE_COLUMNS}
        row["source_snapshot_path"] = str(snapshot_path)
        row["updated_at"] = timestamp
        rows.append(row)
    conn.execute("DELETE FROM company_universe")
    conn.executemany(
        """
        INSERT INTO company_universe (
            company_key, canonical_company_name, snapshot_job_count, snapshot_raw_names_json,
            official_domain_candidates_json, job_board_domain_hits_json,
            snapshot_company_types_json, snapshot_industries_json, source_snapshot_path, updated_at
        ) VALUES (
            :company_key, :canonical_company_name, :snapshot_job_count, :snapshot_raw_names_json,
            :official_domain_candidates_json, :job_board_domain_hits_json,
            :snapshot_company_types_json, :snapshot_industries_json, :source_snapshot_path, :updated_at
        )
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def insert_evidence(conn: sqlite3.Connection, evidence_rows: Iterable[Mapping[str, Any]]) -> int:
    """Append provider/manual evidence rows without changing current rows."""

    rows = []
    timestamp = utc_now_iso()
    for source in evidence_rows:
        row = {column: source.get(column, "") for column in EVIDENCE_COLUMNS}
        row["observed_at"] = clean_text(row["observed_at"]) or timestamp
        row["provider"] = clean_text(row["provider"]) or APIFY_COMPANY_PROVIDER
        row["provider_actor_id"] = clean_text(row["provider_actor_id"]) or APIFY_COMPANY_ACTOR_ID
        row["decision"] = clean_text(row["decision"]) or "unknown"
        row["created_at"] = clean_text(row["created_at"]) or timestamp
        row["candidate_website_domain"] = clean_text(row["candidate_website_domain"]) or _domain_from_url(
            row["candidate_website"]
        )
        row["raw_payload_hash"] = clean_text(row["raw_payload_hash"]) or _raw_payload_hash(
            row["raw_payload_json_private"]
        )
        row["evidence_id"] = clean_text(row["evidence_id"]) or _make_evidence_id(row)
        rows.append(row)
    if not rows:
        return 0
    conn.executemany(
        f"""
        INSERT OR IGNORE INTO company_metadata_evidence ({", ".join(EVIDENCE_COLUMNS)})
        VALUES ({", ".join(":" + column for column in EVIDENCE_COLUMNS)})
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def _accepted_evidence_by_company(conn: sqlite3.Connection) -> dict[str, sqlite3.Row]:
    accepted = {}
    rows = conn.execute(
        """
        SELECT *
        FROM company_metadata_evidence
        WHERE decision IN ('accepted', 'manual_accepted')
        """
    ).fetchall()
    confidence_rank = {"high": 3, "medium": 2, "low": 1}

    def sort_key(row: sqlite3.Row) -> tuple[int, float, str]:
        confidence = confidence_rank.get(clean_text(row["confidence_label"]).lower(), 0)
        score = float(row["match_score"] or 0)
        return (confidence, score, clean_text(row["observed_at"]))

    for row in rows:
        company_key = row["company_key"]
        if company_key not in accepted or sort_key(row) > sort_key(accepted[company_key]):
            accepted[company_key] = row
    return accepted


def _expires_at(observed_at: str, ttl_days: int) -> str:
    text = clean_text(observed_at)
    try:
        observed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        observed = datetime.now(timezone.utc).replace(microsecond=0)
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=timezone.utc)
    return (observed + timedelta(days=ttl_days)).replace(microsecond=0).isoformat()


def _current_row_from_universe(universe: sqlite3.Row, updated_at: str) -> dict[str, Any]:
    return {
        "company_key": universe["company_key"],
        "canonical_company_name": universe["canonical_company_name"],
        "snapshot_job_count": universe["snapshot_job_count"],
        "linkedin_company_id": "",
        "linkedin_universal_name": "",
        "linkedin_url": "",
        "website_url": "",
        "website_domain": "",
        "industry_primary": "",
        "industries_json": "[]",
        "employee_count_linkedin": None,
        "employee_range_start": None,
        "employee_range_end": None,
        "matched_entity_size_bucket": "unknown",
        "entity_scope": "unknown",
        "usable_employer_size_bucket": "unknown",
        "size_usage_policy": "unknown_neutral",
        "size_bucket": "unknown",
        "company_type_provider": "",
        "hq_country": "",
        "hq_state": "",
        "hq_city": "",
        "is_staffing_or_recruiting": 0,
        "match_confidence": "none",
        "match_decision": "unknown",
        "metadata_status": "unknown",
        "source_provider": "",
        "source_evidence_id": "",
        "first_seen_at": "",
        "last_verified_at": "",
        "expires_at": "",
        "notes": "No accepted company metadata evidence yet; unknown remains neutral for ranking.",
        "updated_at": updated_at,
    }


def _current_row_from_evidence(universe: sqlite3.Row, evidence: sqlite3.Row, updated_at: str, ttl_days: int) -> dict[str, Any]:
    industries_json = clean_text(evidence["candidate_industries_json"]) or "[]"
    employee_count = evidence["candidate_employee_count"]
    range_start = evidence["candidate_employee_range_start"]
    range_end = evidence["candidate_employee_range_end"]
    website_url = clean_text(evidence["candidate_website"])
    website_domain = clean_text(evidence["candidate_website_domain"]) or _domain_from_url(website_url)
    canonical_name = clean_text(evidence["candidate_name"]) or universe["canonical_company_name"]
    matched_entity_size_bucket = size_bucket_from_employee_range(range_start, range_end, employee_count)
    entity_scope = infer_entity_scope(
        universe["company_key"],
        universe["canonical_company_name"],
        canonical_name,
        industries_json,
        evidence["match_features_json"],
    )
    usable_employer_size_bucket, size_usage_policy = usable_size_bucket_for_scope(
        entity_scope, matched_entity_size_bucket
    )
    staffing_flag = int(entity_scope == "staffing_agency" or is_staffing_or_recruiting(canonical_name, industries_json))
    return {
        "company_key": universe["company_key"],
        "canonical_company_name": canonical_name,
        "snapshot_job_count": universe["snapshot_job_count"],
        "linkedin_company_id": clean_text(evidence["candidate_linkedin_company_id"]),
        "linkedin_universal_name": clean_text(evidence["candidate_linkedin_universal_name"]),
        "linkedin_url": clean_text(evidence["candidate_linkedin_url"]),
        "website_url": website_url,
        "website_domain": website_domain,
        "industry_primary": _first_json_value(industries_json),
        "industries_json": industries_json,
        "employee_count_linkedin": employee_count,
        "employee_range_start": range_start,
        "employee_range_end": range_end,
        "matched_entity_size_bucket": matched_entity_size_bucket,
        "entity_scope": entity_scope,
        "usable_employer_size_bucket": usable_employer_size_bucket,
        "size_usage_policy": size_usage_policy,
        "size_bucket": usable_employer_size_bucket,
        "company_type_provider": clean_text(evidence["candidate_company_type"]),
        "hq_country": clean_text(evidence["candidate_hq_country"]),
        "hq_state": clean_text(evidence["candidate_hq_state"]),
        "hq_city": clean_text(evidence["candidate_hq_city"]),
        "is_staffing_or_recruiting": staffing_flag,
        "match_confidence": clean_text(evidence["confidence_label"]) or "none",
        "match_decision": clean_text(evidence["decision"]) or "unknown",
        "metadata_status": "matched",
        "source_provider": clean_text(evidence["provider"]),
        "source_evidence_id": clean_text(evidence["evidence_id"]),
        "first_seen_at": clean_text(evidence["observed_at"]),
        "last_verified_at": clean_text(evidence["observed_at"]),
        "expires_at": _expires_at(evidence["observed_at"], ttl_days),
        "notes": (
            "Accepted company metadata evidence; employee count is provider/profile-derived, not audited headcount. "
            f"Size usage policy: {size_usage_policy}."
        ),
        "updated_at": updated_at,
    }


def rebuild_current_from_evidence(conn: sqlite3.Connection, updated_at: str | None = None, ttl_days: int = 180) -> int:
    """Rebuild the current table from universe rows plus accepted append-only evidence."""

    timestamp = updated_at or utc_now_iso()
    accepted = _accepted_evidence_by_company(conn)
    current_rows = []
    for universe in conn.execute("SELECT * FROM company_universe ORDER BY company_key").fetchall():
        evidence = accepted.get(universe["company_key"])
        if evidence is None:
            current_rows.append(_current_row_from_universe(universe, timestamp))
        else:
            current_rows.append(_current_row_from_evidence(universe, evidence, timestamp, ttl_days))

    conn.execute("DELETE FROM company_metadata_current")
    conn.executemany(
        f"""
        INSERT INTO company_metadata_current ({", ".join(CURRENT_COLUMNS)})
        VALUES ({", ".join(":" + column for column in CURRENT_COLUMNS)})
        """,
        current_rows,
    )
    conn.commit()
    return len(current_rows)


def export_frozen_cache(conn: sqlite3.Connection, export_dir: Path) -> dict[str, Any]:
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    current_path = export_dir / "company_metadata_cache_frozen.csv"
    evidence_path = export_dir / "company_metadata_evidence_frozen.jsonl"

    current_rows = [
        {column: row[column] for column in FROZEN_CURRENT_COLUMNS}
        for row in conn.execute("SELECT * FROM company_metadata_current_view ORDER BY company_key").fetchall()
    ]
    current_count = write_csv(current_path, current_rows, FROZEN_CURRENT_COLUMNS)

    evidence_count = 0
    with evidence_path.open("w", encoding="utf-8") as handle:
        rows = conn.execute("SELECT * FROM company_metadata_evidence ORDER BY observed_at, evidence_id").fetchall()
        for row in rows:
            payload = {column: row[column] for column in FROZEN_EVIDENCE_COLUMNS}
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            evidence_count += 1

    return {
        "frozen_current_path": str(current_path),
        "frozen_evidence_path": str(evidence_path),
        "frozen_current_rows": current_count,
        "frozen_evidence_rows": evidence_count,
    }


def current_status_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT metadata_status, COUNT(*) AS count
        FROM company_metadata_current_view
        GROUP BY metadata_status
        ORDER BY metadata_status
        """
    ).fetchall()
    return {row["metadata_status"]: int(row["count"]) for row in rows}


def evidence_decision_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT decision, COUNT(*) AS count
        FROM company_metadata_evidence
        GROUP BY decision
        ORDER BY decision
        """
    ).fetchall()
    return {row["decision"]: int(row["count"]) for row in rows}


def write_cache_state_manifest(
    conn: sqlite3.Connection,
    export_dir: Path,
    *,
    generated_at: str | None = None,
    snapshot_path: Path | str | None = None,
    sqlite_path: Path | str | None = None,
    phase: str = "2.17_company_metadata_cache_state",
    apify_live_calls_made: bool = False,
    apify_bootstrap_manifest_path: Path | str | None = None,
    dry_run_plan_info: Mapping[str, Any] | None = None,
    estimated_company_cost_usd: float | None = None,
) -> dict[str, Any]:
    """Write a current cache-state manifest aligned with frozen exports."""

    timestamp = generated_at or utc_now_iso()
    export_dir = Path(export_dir)
    export_info = export_frozen_cache(conn, export_dir)
    source_snapshot_path = clean_text(snapshot_path)
    if not source_snapshot_path:
        row = conn.execute(
            """
            SELECT source_snapshot_path
            FROM company_universe
            WHERE source_snapshot_path != ''
            LIMIT 1
            """
        ).fetchone()
        source_snapshot_path = clean_text(row["source_snapshot_path"]) if row is not None else ""
    company_universe_count = conn.execute("SELECT COUNT(*) AS count FROM company_universe").fetchone()["count"]
    current_rows = conn.execute("SELECT COUNT(*) AS count FROM company_metadata_current").fetchone()["count"]
    evidence_rows = conn.execute("SELECT COUNT(*) AS count FROM company_metadata_evidence").fetchone()["count"]
    manifest = {
        "phase": phase,
        "generated_at": timestamp,
        "snapshot_path": source_snapshot_path,
        "sqlite_path": str(sqlite_path or ""),
        "export_dir": str(export_dir),
        "company_universe_count": int(company_universe_count),
        "current_rows": int(current_rows),
        "evidence_rows": int(evidence_rows),
        "current_status_counts": current_status_counts(conn),
        "evidence_decision_counts": evidence_decision_counts(conn),
        "phase1_ingestion_modified": False,
        "ranking_behavior_changed": False,
        "apify_live_calls_made": bool(apify_live_calls_made),
        "direct_linkedin_indeed_glassdoor_scrape": False,
        "default_offline_requires_api_key": False,
        "boundaries": [
            "SQLite cache is a company-level sidecar, not Phase 1 ingestion.",
            "Ranking is not changed by the company metadata cache.",
            "Default/offline use does not require API keys.",
            "Unknown company size remains neutral and must not be treated as small.",
            "Private raw provider payloads are not included in frozen exports.",
        ],
        **export_info,
    }
    if apify_bootstrap_manifest_path is not None:
        manifest["apify_bootstrap_manifest_path"] = str(apify_bootstrap_manifest_path)
    if estimated_company_cost_usd is not None:
        manifest["estimated_company_cost_usd"] = round(float(estimated_company_cost_usd), 2)
    if dry_run_plan_info:
        manifest.update(dict(dry_run_plan_info))
    manifest_path = export_dir / "company_metadata_cache_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    write_json(manifest_path, manifest)
    return manifest


def build_apify_bootstrap_plan(
    current_rows: Iterable[Mapping[str, Any]],
    batch_size: int = 100,
    estimated_company_cost_per_1000_usd: float = 4.0,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Return a dry-run private Apify bootstrap plan without making provider calls."""

    rows = [
        {
            "company_key": clean_text(row.get("company_key")),
            "search": clean_text(row.get("canonical_company_name")),
            "metadata_status": clean_text(row.get("metadata_status")) or "unknown",
        }
        for row in current_rows
        if clean_text(row.get("company_key")) and clean_text(row.get("canonical_company_name"))
    ]
    rows = [row for row in rows if row["metadata_status"] in {"unknown", "review", "stale"}]
    batches = []
    batch_size = max(1, int(batch_size))
    for index in range(0, len(rows), batch_size):
        batch = rows[index : index + batch_size]
        batches.append(
            {
                "batch_id": f"apify-company-bootstrap-{len(batches) + 1:04d}",
                "company_count": len(batch),
                "actor_id": APIFY_COMPANY_ACTOR_ID,
                "input": {"searches": [row["search"] for row in batch]},
                "company_keys": [row["company_key"] for row in batch],
            }
        )
    company_count = len(rows)
    return {
        "generated_at": generated_at or utc_now_iso(),
        "dry_run": True,
        "live_calls_made": False,
        "provider": APIFY_COMPANY_PROVIDER,
        "actor_id": APIFY_COMPANY_ACTOR_ID,
        "input_mode": "searches",
        "companies_planned": company_count,
        "batch_size": batch_size,
        "batch_count": len(batches),
        "estimated_company_cost_per_1000_usd": estimated_company_cost_per_1000_usd,
        "estimated_company_cost_usd": round((company_count / 1000.0) * estimated_company_cost_per_1000_usd, 2),
        "default_offline_requires_api_key": False,
        "note": NO_LIVE_CALLS_NOTE,
        "batches": batches,
    }


def write_apify_plan(conn: sqlite3.Connection, export_dir: Path, batch_size: int, generated_at: str) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in conn.execute(
            """
            SELECT company_key, canonical_company_name, metadata_status
            FROM company_metadata_current_view
            ORDER BY snapshot_job_count DESC, company_key
            """
        ).fetchall()
    ]
    plan = build_apify_bootstrap_plan(rows, batch_size=batch_size, generated_at=generated_at)
    plan_path = Path(export_dir) / "apify_bootstrap_plan_dry_run.json"
    write_json(plan_path, plan)
    return {
        "dry_run_plan_path": str(plan_path),
        "dry_run_plan_batches": plan["batch_count"],
        "dry_run_plan_companies": plan["companies_planned"],
        "estimated_company_cost_usd": plan["estimated_company_cost_usd"],
    }


def initialize_company_metadata_cache(
    snapshot_path: Path,
    sqlite_path: Path,
    export_dir: Path,
    batch_size: int = 100,
    max_snapshot_rows: int | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build Phase 2.17A SQLite foundation, frozen exports, and Apify dry-run plan."""

    timestamp = generated_at or utc_now_iso()
    universe = build_company_universe(Path(snapshot_path), max_rows=max_snapshot_rows)
    with connect_cache(Path(sqlite_path)) as conn:
        initialize_schema(conn)
        universe_count = replace_company_universe(conn, universe, Path(snapshot_path), updated_at=timestamp)
        current_rows = rebuild_current_from_evidence(conn, updated_at=timestamp)
        export_info = export_frozen_cache(conn, Path(export_dir))
        plan_info = write_apify_plan(conn, Path(export_dir), batch_size=batch_size, generated_at=timestamp)
        evidence_rows = conn.execute("SELECT COUNT(*) AS count FROM company_metadata_evidence").fetchone()["count"]
        manifest = {
            "phase": "2.17A_company_metadata_cache_foundation",
            "generated_at": timestamp,
            "snapshot_path": str(snapshot_path),
            "sqlite_path": str(sqlite_path),
            "export_dir": str(export_dir),
            "company_universe_count": universe_count,
            "current_rows": current_rows,
            "evidence_rows": int(evidence_rows),
            "current_status_counts": current_status_counts(conn),
            "phase1_ingestion_modified": False,
            "ranking_behavior_changed": False,
            "apify_live_calls_made": False,
            "direct_linkedin_indeed_glassdoor_scrape": False,
            "default_offline_requires_api_key": False,
            "boundaries": [
                "SQLite cache is a company-level sidecar, not Phase 1 ingestion.",
                "Ranking is not changed in Phase 2.17A.",
                "Apify is represented only as a private bootstrap dry-run plan.",
                "Unknown company size remains neutral and must not be treated as small.",
            ],
            **export_info,
            **plan_info,
        }
        manifest_path = Path(export_dir) / "company_metadata_cache_manifest.json"
        write_json(manifest_path, manifest)
        manifest["manifest_path"] = str(manifest_path)
        return manifest
