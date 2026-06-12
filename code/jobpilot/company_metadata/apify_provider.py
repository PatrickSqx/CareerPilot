"""Private Apify bootstrap provider for company metadata enrichment.

Phase 2.17B uses HarvestAPI's ``harvestapi/linkedin-company`` actor as a
private bootstrap/refresh provider. The default path is dry-run only; live
calls require an explicit flag and token.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping

from jobpilot.company_metadata.cache import (
    APIFY_COMPANY_ACTOR_ID,
    APIFY_COMPANY_PROVIDER,
    connect_cache,
    current_status_counts,
    export_frozen_cache,
    initialize_schema,
    insert_evidence,
    rebuild_current_from_evidence,
    utc_now_iso,
    write_cache_state_manifest,
)
from jobpilot.utils.io import ensure_parent, write_json
from jobpilot.utils.text import clean_text, normalize_for_key


APIFY_API_BASE_URL = "https://api.apify.com"
APIFY_SYNC_DATASET_ENDPOINT = "/v2/acts/{actor_id}/run-sync-get-dataset-items"
DEFAULT_COMPANY_COST_PER_1000_USD = 4.0


class ApifyProviderError(RuntimeError):
    """Raised when the private Apify provider cannot complete a live run."""


class ApifyHttpClient:
    """Small stdlib-only Apify API client used by the private bootstrap path."""

    def __init__(self, token: str, base_url: str = APIFY_API_BASE_URL) -> None:
        self.token = token
        self.base_url = base_url.rstrip("/")

    def run_company_actor(
        self,
        actor_input: Mapping[str, Any],
        *,
        timeout_secs: int = 300,
        max_total_charge_usd: float | None = None,
        max_items: int | None = None,
        retries: int = 2,
    ) -> list[dict[str, Any]]:
        actor_api_id = urllib.parse.quote(APIFY_COMPANY_ACTOR_ID.replace("/", "~"), safe="~")
        params: dict[str, Any] = {
            "format": "json",
            "clean": "true",
            "timeout": int(timeout_secs),
            "restartOnError": "false",
        }
        if max_total_charge_usd is not None:
            params["maxTotalChargeUsd"] = f"{max_total_charge_usd:.2f}"
        if max_items is not None:
            params["maxItems"] = int(max_items)
        url = (
            f"{self.base_url}{APIFY_SYNC_DATASET_ENDPOINT.format(actor_id=actor_api_id)}"
            f"?{urllib.parse.urlencode(params)}"
        )
        body = json.dumps(dict(actor_input), ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        last_error: str | None = None
        for attempt in range(max(0, retries) + 1):
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=timeout_secs + 30) as response:
                    payload = response.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as exc:
                last_error = f"http_{exc.code}"
                if exc.code in {408, 429, 500, 502, 503, 504} and attempt < retries:
                    time.sleep(2**attempt)
                    continue
                raise ApifyProviderError(last_error) from exc
            except urllib.error.URLError as exc:
                last_error = "network_error"
                if attempt < retries:
                    time.sleep(2**attempt)
                    continue
                raise ApifyProviderError(f"{last_error}: {exc.reason}") from exc

            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ApifyProviderError("invalid_json_response") from exc
            if isinstance(decoded, list):
                return [item for item in decoded if isinstance(item, dict)]
            if isinstance(decoded, dict) and isinstance(decoded.get("items"), list):
                return [item for item in decoded["items"] if isinstance(item, dict)]
            raise ApifyProviderError("unexpected_response_shape")
        raise ApifyProviderError(last_error or "unknown_apify_error")


def actor_input_for_companies(company_rows: Iterable[Mapping[str, Any]], input_mode: str = "searches") -> dict[str, list[str]]:
    key = "companies" if input_mode == "companies" else "searches"
    values: list[str] = []
    for row in company_rows:
        if key == "companies":
            value = clean_text(row.get("linkedin_url"))
        else:
            value = clean_text(row.get("canonical_company_name"))
        if value:
            values.append(value)
    return {key: values}


def select_bootstrap_candidates(
    conn,
    statuses: Iterable[str] = ("unknown", "review", "stale"),
    limit: int | None = None,
) -> list[dict[str, Any]]:
    status_values = [clean_text(status) for status in statuses if clean_text(status)]
    if not status_values:
        status_values = ["unknown"]
    placeholders = ",".join("?" for _ in status_values)
    sql = f"""
        SELECT
            v.*,
            u.official_domain_candidates_json,
            u.job_board_domain_hits_json,
            u.snapshot_raw_names_json,
            u.snapshot_industries_json
        FROM company_metadata_current_view v
        JOIN company_universe u ON u.company_key = v.company_key
        WHERE v.metadata_status IN ({placeholders})
        ORDER BY v.snapshot_job_count DESC, v.company_key
    """
    rows = [dict(row) for row in conn.execute(sql, status_values).fetchall()]
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    return rows


def estimate_cost_usd(company_count: int, cost_per_1000_usd: float = DEFAULT_COMPANY_COST_PER_1000_USD) -> float:
    return round((max(0, int(company_count)) / 1000.0) * float(cost_per_1000_usd), 2)


def _json_loads(value: Any, default: Any) -> Any:
    if not clean_text(value):
        return default
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def _domain_from_url(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    if not text.lower().startswith(("http://", "https://")):
        text = f"https://{text}"
    parsed = urllib.parse.urlparse(text)
    domain = parsed.netloc.lower()
    return domain[4:] if domain.startswith("www.") else domain


def _same_or_related_domain(left: str, right: str) -> bool:
    left = clean_text(left).lower()
    right = clean_text(right).lower()
    if not left or not right:
        return False
    return left == right or left.endswith(f".{right}") or right.endswith(f".{left}")


def _auxiliary_snapshot_domain(domain: str) -> bool:
    normalized = normalize_for_key(domain)
    auxiliary_terms = {
        "audition",
        "auditions",
        "survey",
        "surveys",
        "culture",
        "assessment",
        "assessments",
        "recruiting",
        "recruitment",
        "careers",
        "jobs",
    }
    return any(term in normalized for term in auxiliary_terms)


def _official_domains(company_row: Mapping[str, Any]) -> list[str]:
    values = _json_loads(company_row.get("official_domain_candidates_json"), [])
    domains: list[str] = []
    if isinstance(values, list):
        for item in values:
            if isinstance(item, dict):
                domain = clean_text(item.get("value"))
            else:
                domain = clean_text(item)
            if domain:
                domains.append(domain)
    return domains


def _headquarter_location(item: Mapping[str, Any]) -> dict[str, Any]:
    locations = item.get("locations")
    if not isinstance(locations, list) or not locations:
        return {}
    hq = next((loc for loc in locations if isinstance(loc, dict) and loc.get("headquarter") is True), None)
    if not isinstance(hq, dict):
        hq = next((loc for loc in locations if isinstance(loc, dict)), {})
    parsed = hq.get("parsed") if isinstance(hq.get("parsed"), dict) else {}
    return {
        "country": clean_text(parsed.get("countryCode") or hq.get("country")),
        "state": clean_text(parsed.get("state") or hq.get("geographicArea")),
        "city": clean_text(parsed.get("city") or hq.get("city")),
    }


def _employee_range(item: Mapping[str, Any]) -> tuple[Any, Any]:
    payload = item.get("employeeCountRange")
    if isinstance(payload, dict):
        return payload.get("start", ""), payload.get("end", "")
    return "", ""


def _company_match_features(company_row: Mapping[str, Any], item: Mapping[str, Any]) -> dict[str, Any]:
    input_name = clean_text(company_row.get("canonical_company_name"))
    input_key = normalize_for_key(input_name)
    candidate_name = clean_text(item.get("name"))
    candidate_key = normalize_for_key(candidate_name)
    universal = normalize_for_key(str(item.get("universalName", "")).replace("-", " "))
    candidate_domain = _domain_from_url(item.get("website"))
    official_domains = _official_domains(company_row)
    domain_match = any(_same_or_related_domain(candidate_domain, domain) for domain in official_domains)
    name_exact = bool(input_key and (input_key == candidate_key or input_key == universal))
    auxiliary_domain_conflict = bool(
        official_domains
        and candidate_domain
        and not domain_match
        and name_exact
        and all(_auxiliary_snapshot_domain(domain) for domain in official_domains)
    )
    domain_conflict = bool(official_domains and candidate_domain and not domain_match and not auxiliary_domain_conflict)
    name_contains = bool(
        input_key
        and candidate_key
        and not name_exact
        and (input_key in candidate_key or candidate_key in input_key)
    )
    return {
        "input_name": input_name,
        "input_company_key": clean_text(company_row.get("company_key")),
        "candidate_name": candidate_name,
        "candidate_universal_name": clean_text(item.get("universalName")),
        "candidate_domain": candidate_domain,
        "official_domains": official_domains,
        "name_exact": name_exact,
        "name_contains": name_contains,
        "domain_match": domain_match,
        "domain_conflict": domain_conflict,
        "auxiliary_domain_conflict": auxiliary_domain_conflict,
        "has_linkedin_url": bool(clean_text(item.get("linkedinUrl"))),
        "has_employee_count": bool(clean_text(item.get("employeeCount"))),
        "has_industries": isinstance(item.get("industries"), list) and bool(item.get("industries")),
    }


def score_company_match(features: Mapping[str, Any]) -> float:
    score = 0.0
    if features.get("name_exact"):
        score += 0.58
    elif features.get("name_contains"):
        score += 0.35
    if features.get("has_linkedin_url"):
        score += 0.12
    if features.get("has_employee_count"):
        score += 0.08
    if features.get("has_industries"):
        score += 0.05
    if features.get("domain_match"):
        score += 0.17
    if features.get("domain_conflict"):
        score -= 0.35
    return round(max(0.0, min(1.0, score)), 4)


def confidence_label(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.6:
        return "medium"
    if score >= 0.4:
        return "low"
    return "none"


def decision_for_match(features: Mapping[str, Any], confidence: str) -> str:
    if features.get("domain_conflict"):
        return "review"
    if confidence in {"high", "medium"}:
        return "accepted"
    if confidence == "low":
        return "review"
    return "rejected"


def evidence_from_apify_item(
    company_row: Mapping[str, Any],
    item: Mapping[str, Any] | None,
    *,
    observed_at: str,
    provider_run_id: str,
    input_mode: str,
    candidate_rank: int = 1,
    cost_units: float = 1.0,
) -> dict[str, Any]:
    input_value = (
        clean_text(company_row.get("linkedin_url"))
        if input_mode == "companies"
        else clean_text(company_row.get("canonical_company_name"))
    )
    if not item or not any(clean_text(item.get(key)) for key in ("name", "linkedinUrl", "id", "universalName")):
        return {
            "company_key": clean_text(company_row.get("company_key")),
            "observed_at": observed_at,
            "provider": APIFY_COMPANY_PROVIDER,
            "provider_actor_id": APIFY_COMPANY_ACTOR_ID,
            "provider_run_id": provider_run_id,
            "input_mode": input_mode,
            "input_value": input_value,
            "candidate_rank": candidate_rank,
            "match_features_json": json.dumps({"input_name": input_value}, ensure_ascii=False, sort_keys=True),
            "match_score": 0.0,
            "confidence_label": "none",
            "decision": "no_result",
            "error_code": "no_result",
            "error_message": "Provider returned no usable company item for this input.",
            "cost_units": cost_units,
            "raw_payload_json_private": json.dumps(item or {}, ensure_ascii=False, sort_keys=True),
        }

    features = _company_match_features(company_row, item)
    score = score_company_match(features)
    confidence = confidence_label(score)
    decision = decision_for_match(features, confidence)
    hq = _headquarter_location(item)
    range_start, range_end = _employee_range(item)
    industries = item.get("industries") if isinstance(item.get("industries"), list) else []
    locations = item.get("locations") if isinstance(item.get("locations"), list) else []
    raw_payload = json.dumps(item, ensure_ascii=False, sort_keys=True)
    return {
        "company_key": clean_text(company_row.get("company_key")),
        "observed_at": observed_at,
        "provider": APIFY_COMPANY_PROVIDER,
        "provider_actor_id": APIFY_COMPANY_ACTOR_ID,
        "provider_run_id": provider_run_id,
        "input_mode": input_mode,
        "input_value": input_value,
        "candidate_rank": candidate_rank,
        "candidate_linkedin_company_id": clean_text(item.get("id")),
        "candidate_linkedin_universal_name": clean_text(item.get("universalName")),
        "candidate_name": clean_text(item.get("name")),
        "candidate_linkedin_url": clean_text(item.get("linkedinUrl")),
        "candidate_website": clean_text(item.get("website")),
        "candidate_website_domain": features["candidate_domain"],
        "candidate_employee_count": item.get("employeeCount") or "",
        "candidate_employee_range_start": range_start or "",
        "candidate_employee_range_end": range_end or "",
        "candidate_industries_json": json.dumps(industries, ensure_ascii=False, sort_keys=True),
        "candidate_company_type": clean_text(item.get("companyType")),
        "candidate_hq_country": hq.get("country", ""),
        "candidate_hq_state": hq.get("state", ""),
        "candidate_hq_city": hq.get("city", ""),
        "candidate_locations_json": json.dumps(locations, ensure_ascii=False, sort_keys=True),
        "match_features_json": json.dumps(features, ensure_ascii=False, sort_keys=True),
        "match_score": score,
        "confidence_label": confidence,
        "decision": decision,
        "error_code": "domain_conflict" if features.get("domain_conflict") else "",
        "error_message": "Official-domain candidate conflicts with provider website." if features.get("domain_conflict") else "",
        "cost_units": cost_units,
        "raw_payload_json_private": raw_payload,
    }


def map_items_to_evidence(
    company_rows: list[Mapping[str, Any]],
    items: list[Mapping[str, Any]],
    *,
    observed_at: str,
    provider_run_id: str,
    input_mode: str,
) -> list[dict[str, Any]]:
    # Actor dataset order is not a reliable input-to-output mapping. Always
    # score-match returned items back to inputs, even when counts are equal.
    remaining = list(items)
    evidence_rows: list[dict[str, Any]] = []
    for company_row in company_rows:
        best_index = -1
        best_score = -1.0
        for index, item in enumerate(remaining):
            features = _company_match_features(company_row, item)
            score = score_company_match(features)
            if score > best_score:
                best_score = score
                best_index = index
        if best_index >= 0 and best_score >= 0.4:
            item = remaining.pop(best_index)
        else:
            item = None
        evidence_rows.append(
            evidence_from_apify_item(
                company_row,
                item,
                observed_at=observed_at,
                provider_run_id=provider_run_id,
                input_mode=input_mode,
                candidate_rank=1,
            )
        )
    return evidence_rows


def _batches(rows: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    size = max(1, int(batch_size))
    for index in range(0, len(rows), size):
        yield rows[index : index + size]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def _default_private_raw_dir(sqlite_path: Path) -> Path:
    return Path(sqlite_path).parent / "company_metadata_apify_raw_batches"


def write_private_raw_batch_archive(
    raw_dir: Path,
    *,
    batch_id: str,
    provider_run_id: str,
    actor_id: str,
    input_mode: str,
    actor_input: Mapping[str, Any],
    company_rows: list[Mapping[str, Any]],
    items: list[Mapping[str, Any]],
    started_at: str,
    finished_at: str,
) -> Path:
    """Persist raw returned items privately before any matching decision."""

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{batch_id}.json"
    payload = {
        "batch_id": batch_id,
        "provider_run_id": provider_run_id,
        "actor_id": actor_id,
        "input_mode": input_mode,
        "started_at": started_at,
        "finished_at": finished_at,
        "company_count": len(company_rows),
        "items_returned": len(items),
        "company_keys": [clean_text(row.get("company_key")) for row in company_rows],
        "inputs": actor_input,
        "raw_items": list(items),
        "private_payload": True,
        "apify_live_call_made": True,
    }
    write_json(path, payload)
    return path


def run_private_apify_bootstrap(
    *,
    sqlite_path: Path,
    export_dir: Path,
    private_raw_dir: Path | None = None,
    token: str = "",
    run_live: bool = False,
    input_mode: str = "searches",
    statuses: Iterable[str] = ("unknown", "review", "stale"),
    batch_size: int = 25,
    max_companies: int | None = None,
    max_batches: int | None = None,
    budget_usd: float = 15.0,
    timeout_secs: int = 300,
    retries: int = 2,
    generated_at: str | None = None,
    client: ApifyHttpClient | None = None,
) -> dict[str, Any]:
    """Run or plan the private Apify bootstrap and write a manifest."""

    timestamp = generated_at or utc_now_iso()
    sqlite_path = Path(sqlite_path)
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    raw_archive_dir = Path(private_raw_dir) if private_raw_dir is not None else _default_private_raw_dir(sqlite_path)
    batch_log_rows: list[dict[str, Any]] = []
    raw_archive_paths: list[str] = []
    with connect_cache(sqlite_path) as conn:
        initialize_schema(conn)
        candidates = select_bootstrap_candidates(conn, statuses=statuses, limit=max_companies)
        batches = list(_batches(candidates, batch_size))
        if max_batches is not None:
            batches = batches[: max(0, int(max_batches))]
            candidates = [row for batch in batches for row in batch]
        estimated_cost = estimate_cost_usd(len(candidates))
        budget_ok = estimated_cost <= float(budget_usd)
        manifest = {
            "phase": "2.17B_private_apify_company_metadata_bootstrap",
            "generated_at": timestamp,
            "dry_run": not run_live,
            "live_calls_made": False,
            "provider": APIFY_COMPANY_PROVIDER,
            "actor_id": APIFY_COMPANY_ACTOR_ID,
            "input_mode": input_mode,
            "sqlite_path": str(sqlite_path),
            "export_dir": str(export_dir),
            "candidate_companies": len(candidates),
            "batch_count": len(batches),
            "batch_size": max(1, int(batch_size)),
            "estimated_company_cost_usd": estimated_cost,
            "budget_usd": float(budget_usd),
            "budget_ok": budget_ok,
            "phase1_ingestion_modified": False,
            "ranking_behavior_changed": False,
            "default_offline_requires_api_key": False,
            "direct_linkedin_indeed_glassdoor_scrape": False,
            "evidence_rows_inserted": 0,
            "decision_counts": {},
            "current_status_counts": current_status_counts(conn),
        }
        if not run_live:
            for batch_index, batch in enumerate(batches, start=1):
                batch_log_rows.append(
                    {
                        "batch_id": f"apify-company-bootstrap-{batch_index:04d}",
                        "dry_run": True,
                        "live_call_made": False,
                        "company_count": len(batch),
                        "actor_id": APIFY_COMPANY_ACTOR_ID,
                        "input_mode": input_mode,
                        "company_keys": [row["company_key"] for row in batch],
                        "inputs": actor_input_for_companies(batch, input_mode=input_mode).get(input_mode, []),
                    }
                )
        else:
            if not token:
                raise ApifyProviderError("APIFY_TOKEN is required for --run-live.")
            if not budget_ok:
                raise ApifyProviderError("Estimated run cost exceeds --budget-usd.")
            client = client or ApifyHttpClient(token=token)
            decision_counts: dict[str, int] = {}
            inserted_total = 0
            for batch_index, batch in enumerate(batches, start=1):
                actor_input = actor_input_for_companies(batch, input_mode=input_mode)
                provider_run_id = f"apify-sync-{timestamp}-batch-{batch_index:04d}"
                batch_id = f"apify-company-bootstrap-{batch_index:04d}"
                max_charge = max(0.01, min(float(budget_usd), estimate_cost_usd(len(batch)) + 0.05))
                started_at = utc_now_iso()
                items = client.run_company_actor(
                    actor_input,
                    timeout_secs=timeout_secs,
                    max_total_charge_usd=max_charge,
                    max_items=len(batch),
                    retries=retries,
                )
                finished_at = utc_now_iso()
                raw_archive_path = write_private_raw_batch_archive(
                    raw_archive_dir,
                    batch_id=batch_id,
                    provider_run_id=provider_run_id,
                    actor_id=APIFY_COMPANY_ACTOR_ID,
                    input_mode=input_mode,
                    actor_input=actor_input,
                    company_rows=batch,
                    items=items,
                    started_at=started_at,
                    finished_at=finished_at,
                )
                raw_archive_paths.append(str(raw_archive_path))
                evidence_rows = map_items_to_evidence(
                    batch,
                    items,
                    observed_at=utc_now_iso(),
                    provider_run_id=provider_run_id,
                    input_mode=input_mode,
                )
                inserted = insert_evidence(conn, evidence_rows)
                inserted_total += inserted
                for row in evidence_rows:
                    decision = clean_text(row.get("decision")) or "unknown"
                    decision_counts[decision] = decision_counts.get(decision, 0) + 1
                batch_log_rows.append(
                    {
                        "batch_id": batch_id,
                        "dry_run": False,
                        "live_call_made": True,
                        "started_at": started_at,
                        "finished_at": finished_at,
                        "company_count": len(batch),
                        "items_returned": len(items),
                        "evidence_rows_inserted": inserted,
                        "private_raw_archive_path": str(raw_archive_path),
                        "actor_id": APIFY_COMPANY_ACTOR_ID,
                        "input_mode": input_mode,
                        "company_keys": [row["company_key"] for row in batch],
                    }
                )
            current_rows = rebuild_current_from_evidence(conn, updated_at=utc_now_iso())
            export_info = export_frozen_cache(conn, export_dir)
            manifest.update(
                {
                    "live_calls_made": bool(batches),
                    "evidence_rows_inserted": inserted_total,
                    "decision_counts": decision_counts,
                    "current_rows": current_rows,
                    "current_status_counts": current_status_counts(conn),
                    "private_raw_archive_dir": str(raw_archive_dir),
                    "private_raw_archive_batches": len(raw_archive_paths),
                    "private_raw_archive_paths": raw_archive_paths,
                    "raw_returned_items_preserved_before_matching": True,
                    **export_info,
                }
            )

        batch_log_path = export_dir / "company_metadata_apify_bootstrap_batches.jsonl"
        batch_log_count = _write_jsonl(batch_log_path, batch_log_rows)
        manifest["batch_log_path"] = str(batch_log_path)
        manifest["batch_log_rows"] = batch_log_count
        manifest_path = export_dir / "company_metadata_apify_bootstrap_manifest.json"
        manifest["manifest_path"] = str(manifest_path)
        write_json(manifest_path, manifest)
        if run_live:
            cache_manifest = write_cache_state_manifest(
                conn,
                export_dir,
                generated_at=utc_now_iso(),
                sqlite_path=sqlite_path,
                phase="2.17B_company_metadata_cache_live_state",
                apify_live_calls_made=manifest["live_calls_made"],
                apify_bootstrap_manifest_path=manifest_path,
                estimated_company_cost_usd=estimated_cost,
            )
            manifest["cache_manifest_path"] = cache_manifest["manifest_path"]
            write_json(manifest_path, manifest)
        return manifest
