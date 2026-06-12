"""Build conservative H-1B/LCA employer activity evidence from OFLC disclosures.

This module intentionally does not change Phase 1 ingestion or Phase 2 ranking.
It produces an offline cache that can later be joined to ranked jobs by a
normalized employer key.
"""

from __future__ import annotations

import csv
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

from jobpilot.utils.io import write_csv, write_json
from jobpilot.utils.text import clean_text, normalize_for_key


DOL_OFLC_PERFORMANCE_URL = "https://www.dol.gov/agencies/eta/foreign-labor/performance"
DISCLOSURE_NOTE = (
    "LCA activity is historical or recent employer filing activity from DOL OFLC "
    "disclosure data. It is not confirmed sponsorship for any specific job posting."
)

DEFAULT_WINDOW_QUARTERS = {
    "recent_2q": 2,
    "recent_3q": 3,
    "historical_8q": 8,
}

TARGET_ROLE_FAMILIES = {
    "ml_ai",
    "data_analytics",
    "data_engineering",
    "software_engineering",
}

CORPORATE_SUFFIXES = {
    "inc",
    "incorporated",
    "llc",
    "l l c",
    "ltd",
    "limited",
    "corp",
    "corporation",
    "co",
    "com",
    "company",
    "plc",
    "lp",
    "llp",
    "services",
    "service",
    "technologies",
    "technology",
    "usa",
    "us",
}

COLUMN_ALIASES = {
    "case_number": ("CASE_NUMBER", "CASE_NO", "CASE_ID"),
    "case_status": ("CASE_STATUS", "STATUS"),
    "visa_class": ("VISA_CLASS", "CLASS_OF_ADMISSION"),
    "decision_date": ("DECISION_DATE", "DETERMINATION_DATE", "CASE_DECISION_DATE"),
    "employer_name": ("EMPLOYER_NAME", "EMPLOYER_BUSINESS_NAME", "EMPLOYER_LEGAL_BUSINESS_NAME"),
    "job_title": ("JOB_TITLE", "EMPLOYER_JOB_TITLE"),
    "soc_code": ("SOC_CODE", "SOC_CD"),
    "soc_title": ("SOC_TITLE", "SOC_NAME", "SOC_OCCUPATION_TITLE"),
    "total_worker_positions": ("TOTAL_WORKER_POSITIONS", "TOTAL_WORKERS", "WORKER_POSITIONS"),
    "h1b_dependent": ("H_1B_DEPENDENT", "H1B_DEPENDENT"),
    "willful_violator": ("WILLFUL_VIOLATOR",),
    "worksite_state": ("WORKSITE_STATE", "WORKSITE_STATE_1", "WORKSITE_LOCATION_STATE"),
}


@dataclass(frozen=True)
class LcaRecord:
    case_number: str
    case_status: str
    visa_class: str
    decision_date: date
    employer_name: str
    employer_key: str
    job_title: str
    soc_code: str
    soc_title: str
    role_family: str
    total_worker_positions: int
    h1b_dependent: str
    willful_violator: str
    worksite_state: str
    source_file: str


def canonical_header(value: Any) -> str:
    """Normalize a source column header for alias matching."""

    text = clean_text(value).upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    return text.strip("_")


def normalize_employer_key(value: Any) -> str:
    """Return a stable employer key for matching JobPilot company names."""

    key = normalize_for_key(value)
    if not key:
        return ""
    tokens = [token for token in key.split() if token not in CORPORATE_SUFFIXES]
    while tokens and tokens[-1] in CORPORATE_SUFFIXES:
        tokens.pop()
    return " ".join(tokens) or key


def parse_lca_date(value: Any) -> date | None:
    """Parse OFLC date text or Excel serial dates."""

    text = clean_text(value)
    if not text:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        try:
            serial = float(text)
        except ValueError:
            serial = 0
        if 20_000 <= serial <= 80_000:
            return date(1899, 12, 30) + timedelta(days=int(serial))

    text = text.replace("T00:00:00", "")
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d", "%d-%b-%Y", "%m-%d-%Y"):
        try:
            return datetime.strptime(text[:10], fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text[:10]).date()
    except ValueError:
        return None


def parse_worker_positions(value: Any) -> int:
    text = clean_text(value).replace(",", "")
    if not text:
        return 0
    try:
        return max(0, int(float(text)))
    except ValueError:
        return 0


def normalize_status(value: Any) -> str:
    return re.sub(r"\s+", " ", clean_text(value).upper().replace("_", " ")).strip()


def is_certified_status(status: str) -> bool:
    return normalize_for_key(status) == "certified"


def is_certified_or_withdrawn_status(status: str) -> bool:
    return normalize_for_key(status) in {"certified", "certified withdrawn"}


def normalize_visa_class(value: Any) -> str:
    return clean_text(value).upper().replace(" ", "")


def visa_class_allowed(value: Any, allowed: set[str]) -> bool:
    if not allowed or "ALL" in allowed:
        return True
    return normalize_visa_class(value) in allowed


def classify_role_family(job_title: Any, soc_title: Any = "", soc_code: Any = "") -> str:
    """Classify OFLC role text into broad JobPilot-relevant role families."""

    title = normalize_for_key(job_title)
    soc = normalize_for_key(soc_title)
    code = clean_text(soc_code)
    text = f"{title} {soc}"

    if re.search(r"\b(machine learning|ml engineer|mlops|artificial intelligence|ai engineer|deep learning)\b", text):
        return "ml_ai"
    if re.search(r"\b(computer vision|natural language processing|nlp|applied scientist|research scientist)\b", text):
        return "ml_ai"
    if re.search(r"\b(data engineer|analytics engineer|etl|data pipeline|big data|spark|kafka)\b", text):
        return "data_engineering"
    if re.search(r"\b(data scientist|data analyst|business intelligence|bi analyst|analytics|statistician)\b", text):
        return "data_analytics"
    if re.search(r"\b(software|developer|programmer|devops|site reliability|systems engineer|platform engineer)\b", text):
        return "software_engineering"

    if code.startswith("15-2051"):
        return "data_analytics"
    if code.startswith(("15-1252", "15-1253", "15-1254", "15-1256", "15-1257", "15-1258")):
        return "software_engineering"
    if code.startswith(("15-1243", "15-1244", "15-1241")):
        return "data_engineering"
    if code.startswith(("15-", "17-")):
        return "other_tech"
    return "other"


def discover_source_files(source_dir: Path, explicit_inputs: Iterable[Path] | None = None) -> list[Path]:
    """Find local OFLC disclosure files to process."""

    if explicit_inputs:
        return [Path(path) for path in explicit_inputs]
    if not source_dir.exists():
        return []
    files: list[Path] = []
    for suffix in ("*.csv", "*.xlsx", "*.xlsm"):
        files.extend(source_dir.glob(suffix))
    return sorted(files)


def _select_field(row: dict[str, str], canonical_name: str) -> str:
    for alias in COLUMN_ALIASES[canonical_name]:
        value = row.get(alias)
        if value is not None and clean_text(value):
            return clean_text(value)
    return ""


def _remap_headers(row: dict[str, str]) -> dict[str, str]:
    return {canonical_header(key): value for key, value in row.items()}


def _iter_csv_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield _remap_headers({str(key): clean_text(value) for key, value in row.items() if key is not None})


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(element: ET.Element, child_name: str) -> str:
    for child in element:
        if _xml_local_name(child.tag) == child_name:
            return child.text or ""
    return ""


def _load_shared_strings(workbook: zipfile.ZipFile) -> list[str]:
    try:
        source = workbook.open("xl/sharedStrings.xml")
    except KeyError:
        return []
    strings: list[str] = []
    with source:
        for event, element in ET.iterparse(source, events=("end",)):
            if _xml_local_name(element.tag) == "si":
                parts = [node.text or "" for node in element.iter() if _xml_local_name(node.tag) == "t"]
                strings.append("".join(parts))
                element.clear()
    return strings


def _first_worksheet_path(workbook: zipfile.ZipFile) -> str:
    try:
        workbook_xml = workbook.open("xl/workbook.xml")
        rels_xml = workbook.open("xl/_rels/workbook.xml.rels")
    except KeyError:
        return "xl/worksheets/sheet1.xml"

    with workbook_xml:
        root = ET.parse(workbook_xml).getroot()
    first_rid = ""
    for sheet in root.iter():
        if _xml_local_name(sheet.tag) == "sheet":
            first_rid = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id", "")
            break
    if not first_rid:
        return "xl/worksheets/sheet1.xml"

    with rels_xml:
        rel_root = ET.parse(rels_xml).getroot()
    for rel in rel_root:
        if rel.attrib.get("Id") == first_rid:
            target = rel.attrib.get("Target", "worksheets/sheet1.xml")
            if target.startswith("/"):
                return target.lstrip("/")
            return "xl/" + target.lstrip("/")
    return "xl/worksheets/sheet1.xml"


def _column_index(cell_ref: str, fallback: int) -> int:
    match = re.match(r"([A-Z]+)", cell_ref.upper())
    if not match:
        return fallback
    index = 0
    for char in match.group(1):
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def _xlsx_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iter() if _xml_local_name(node.tag) == "t")
    value = _child_text(cell, "v")
    if cell_type == "s":
        try:
            return shared_strings[int(value)]
        except (ValueError, IndexError):
            return ""
    return clean_text(value)


def _iter_xlsx_raw_rows(path: Path) -> Iterable[list[str]]:
    with zipfile.ZipFile(path) as workbook:
        shared_strings = _load_shared_strings(workbook)
        worksheet_path = _first_worksheet_path(workbook)
        with workbook.open(worksheet_path) as worksheet:
            for event, element in ET.iterparse(worksheet, events=("end",)):
                if _xml_local_name(element.tag) != "row":
                    continue
                cells: dict[int, str] = {}
                fallback_index = 0
                for cell in element:
                    if _xml_local_name(cell.tag) != "c":
                        continue
                    index = _column_index(cell.attrib.get("r", ""), fallback_index)
                    cells[index] = _xlsx_cell_value(cell, shared_strings)
                    fallback_index = index + 1
                if cells:
                    max_index = max(cells)
                    yield [cells.get(index, "") for index in range(max_index + 1)]
                element.clear()


def _iter_xlsx_rows(path: Path) -> Iterable[dict[str, str]]:
    header: list[str] = []
    for values in _iter_xlsx_raw_rows(path):
        if not header:
            header = [canonical_header(value) for value in values]
            continue
        if not any(clean_text(value) for value in values):
            continue
        row = {header[index]: clean_text(value) for index, value in enumerate(values) if index < len(header)}
        yield row


def iter_source_rows(path: Path) -> Iterable[dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        yield from _iter_csv_rows(path)
    elif suffix in {".xlsx", ".xlsm"}:
        yield from _iter_xlsx_rows(path)
    else:
        raise ValueError(f"Unsupported OFLC source format: {path}")


def record_from_row(row: dict[str, str], *, source_file: str, allowed_visa_classes: set[str]) -> LcaRecord | None:
    visa_class = _select_field(row, "visa_class")
    if not visa_class_allowed(visa_class, allowed_visa_classes):
        return None
    decision = parse_lca_date(_select_field(row, "decision_date"))
    employer = _select_field(row, "employer_name")
    if not decision or not employer:
        return None

    job_title = _select_field(row, "job_title")
    soc_title = _select_field(row, "soc_title")
    soc_code = _select_field(row, "soc_code")
    return LcaRecord(
        case_number=_select_field(row, "case_number"),
        case_status=normalize_status(_select_field(row, "case_status")),
        visa_class=normalize_visa_class(visa_class),
        decision_date=decision,
        employer_name=employer,
        employer_key=normalize_employer_key(employer),
        job_title=job_title,
        soc_code=soc_code,
        soc_title=soc_title,
        role_family=classify_role_family(job_title, soc_title, soc_code),
        total_worker_positions=parse_worker_positions(_select_field(row, "total_worker_positions")),
        h1b_dependent=_select_field(row, "h1b_dependent"),
        willful_violator=_select_field(row, "willful_violator"),
        worksite_state=_select_field(row, "worksite_state"),
        source_file=source_file,
    )


def dedupe_key(record: LcaRecord) -> tuple[str, ...]:
    if record.case_number:
        return ("case_number", normalize_for_key(record.case_number))
    return (
        "synthetic",
        record.employer_key,
        normalize_for_key(record.job_title),
        record.decision_date.isoformat(),
        normalize_for_key(record.case_status),
        normalize_for_key(record.soc_code),
        str(record.total_worker_positions),
    )


def load_lca_records(
    source_paths: Iterable[Path],
    *,
    allowed_visa_classes: set[str] | None = None,
    max_rows: int | None = None,
) -> tuple[list[LcaRecord], dict[str, Any]]:
    allowed = allowed_visa_classes or {"H-1B"}
    records_by_key: dict[tuple[str, ...], LcaRecord] = {}
    stats: dict[str, Any] = {
        "source_files": [],
        "rows_seen": 0,
        "rows_used": 0,
        "rows_skipped": 0,
        "duplicate_rows_replaced_or_ignored": 0,
    }

    for source_path in source_paths:
        path = Path(source_path)
        file_rows_seen = 0
        file_rows_used = 0
        for row in iter_source_rows(path):
            if max_rows is not None and stats["rows_seen"] >= max_rows:
                break
            stats["rows_seen"] += 1
            file_rows_seen += 1
            record = record_from_row(row, source_file=path.name, allowed_visa_classes=allowed)
            if record is None:
                stats["rows_skipped"] += 1
                continue
            key = dedupe_key(record)
            existing = records_by_key.get(key)
            if existing is not None:
                stats["duplicate_rows_replaced_or_ignored"] += 1
                if record.decision_date >= existing.decision_date:
                    records_by_key[key] = record
                continue
            records_by_key[key] = record
            stats["rows_used"] += 1
            file_rows_used += 1
        stats["source_files"].append(
            {
                "path": str(path),
                "rows_seen": file_rows_seen,
                "rows_used_before_cross_file_dedup": file_rows_used,
            }
        )
        if max_rows is not None and stats["rows_seen"] >= max_rows:
            break

    records = sorted(records_by_key.values(), key=lambda item: (item.employer_key, item.role_family, item.decision_date))
    return records, stats


def add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    month_lengths = [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day = min(value.day, month_lengths[month - 1])
    return date(year, month, day)


def window_start(as_of_date: date, quarters: int) -> date:
    return add_months(as_of_date, -3 * quarters) + timedelta(days=1)


def empty_metrics() -> dict[str, int]:
    return {
        "case_count": 0,
        "certified_case_count": 0,
        "certified_or_withdrawn_case_count": 0,
        "denied_case_count": 0,
        "total_worker_positions": 0,
        "certified_worker_positions": 0,
        "certified_or_withdrawn_worker_positions": 0,
    }


def update_metrics(metrics: dict[str, int], record: LcaRecord) -> None:
    positions = record.total_worker_positions
    certified = is_certified_status(record.case_status)
    certified_or_withdrawn = is_certified_or_withdrawn_status(record.case_status)
    metrics["case_count"] += 1
    metrics["total_worker_positions"] += positions
    if certified:
        metrics["certified_case_count"] += 1
        metrics["certified_worker_positions"] += positions
    if certified_or_withdrawn:
        metrics["certified_or_withdrawn_case_count"] += 1
        metrics["certified_or_withdrawn_worker_positions"] += positions
    if normalize_for_key(record.case_status) == "denied":
        metrics["denied_case_count"] += 1


def evidence_label(window_metrics: dict[str, dict[str, int]]) -> str:
    recent = window_metrics["recent_3q"]
    historical = window_metrics["historical_8q"]
    recent_cases = recent["certified_case_count"]
    recent_positions = recent["certified_worker_positions"]
    if recent_cases >= 10 or recent_positions >= 25:
        return "recent_lca_activity_high"
    if recent_cases >= 3 or recent_positions >= 5:
        return "recent_lca_activity_moderate"
    if recent_cases > 0 or recent_positions > 0:
        return "recent_lca_activity_low"
    if recent["case_count"] > 0:
        return "recent_lca_activity_no_certified_cases"
    if historical["certified_case_count"] > 0 or historical["case_count"] > 0:
        return "historical_lca_activity_only"
    return "no_lca_activity_in_cache"


def aggregate_records(
    records: list[LcaRecord],
    *,
    as_of_date: date | None = None,
    window_quarters: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not records:
        as_of = as_of_date or date.today()
    else:
        as_of = as_of_date or max(record.decision_date for record in records)
    windows = window_quarters or DEFAULT_WINDOW_QUARTERS
    window_ranges = {
        name: {
            "quarters": quarters,
            "start": window_start(as_of, quarters),
            "end": as_of,
        }
        for name, quarters in windows.items()
    }

    grouped: dict[tuple[str, str], list[LcaRecord]] = defaultdict(list)
    employer_grouped: dict[str, list[LcaRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.employer_key, record.role_family)].append(record)
        employer_grouped[record.employer_key].append(record)

    role_rows = [_summarize_group(employer_key, role_family, items, window_ranges) for (employer_key, role_family), items in grouped.items()]
    employer_rows = [_summarize_group(employer_key, "all", items, window_ranges) for employer_key, items in employer_grouped.items()]
    role_rows.sort(key=lambda row: (row["employer_key"], row["role_family"]))
    employer_rows.sort(key=lambda row: row["employer_key"])

    metadata = {
        "as_of_date": as_of.isoformat(),
        "windows": {
            name: {
                "quarters": detail["quarters"],
                "start": detail["start"].isoformat(),
                "end": detail["end"].isoformat(),
            }
            for name, detail in window_ranges.items()
        },
    }
    return role_rows, employer_rows, metadata


def _summarize_group(
    employer_key: str,
    role_family: str,
    records: list[LcaRecord],
    window_ranges: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    display_name = Counter(record.employer_name for record in records).most_common(1)[0][0]
    window_metrics = {name: empty_metrics() for name in window_ranges}
    for record in records:
        for name, detail in window_ranges.items():
            if detail["start"] <= record.decision_date <= detail["end"]:
                update_metrics(window_metrics[name], record)

    row: dict[str, Any] = {
        "employer_key": employer_key,
        "employer_display": display_name,
        "role_family": role_family,
        "is_focus_role_family": str(role_family in TARGET_ROLE_FAMILIES).lower(),
        "lca_activity_label": evidence_label(window_metrics),
        "latest_decision_date": max(record.decision_date for record in records).isoformat(),
        "source_file_count": len({record.source_file for record in records}),
        "unique_case_count": len({dedupe_key(record) for record in records}),
        "h1b_dependent_values": "|".join(sorted({clean_text(record.h1b_dependent) for record in records if clean_text(record.h1b_dependent)})),
        "willful_violator_values": "|".join(sorted({clean_text(record.willful_violator) for record in records if clean_text(record.willful_violator)})),
    }
    for window_name, metrics in window_metrics.items():
        for metric_name, value in metrics.items():
            row[f"{window_name}_{metric_name}"] = value
    return row


ROLE_SUMMARY_COLUMNS = [
    "employer_key",
    "employer_display",
    "role_family",
    "is_focus_role_family",
    "lca_activity_label",
    "latest_decision_date",
    "source_file_count",
    "unique_case_count",
    "h1b_dependent_values",
    "willful_violator_values",
]
for _window_name in DEFAULT_WINDOW_QUARTERS:
    for _metric_name in empty_metrics():
        ROLE_SUMMARY_COLUMNS.append(f"{_window_name}_{_metric_name}")


def build_lookup(role_rows: list[dict[str, Any]], employer_rows: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    employers: dict[str, Any] = {}
    for row in employer_rows:
        employer_key = str(row["employer_key"])
        employers[employer_key] = {
            "employer_display": row["employer_display"],
            "overall": row,
            "role_families": {},
        }
    for row in role_rows:
        employer_key = str(row["employer_key"])
        employers.setdefault(
            employer_key,
            {
                "employer_display": row["employer_display"],
                "overall": {},
                "role_families": {},
            },
        )
        employers[employer_key]["role_families"][row["role_family"]] = row
    return {
        "metadata": {
            **metadata,
            "source": DOL_OFLC_PERFORMANCE_URL,
            "interpretation_note": DISCLOSURE_NOTE,
        },
        "employers": employers,
    }


def write_report(path: Path, manifest: dict[str, Any], top_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# H-1B/LCA Employer Activity Cache",
        "",
        DISCLOSURE_NOTE,
        "",
        f"- Official source page: {DOL_OFLC_PERFORMANCE_URL}",
        f"- Generated at: {manifest['generated_at']}",
        f"- As-of date: {manifest['as_of_date']}",
        f"- Rows seen: {manifest['rows_seen']:,}",
        f"- Deduped records used: {manifest['deduped_records_used']:,}",
        f"- Visa classes included: {', '.join(manifest['visa_classes_included'])}",
        "",
        "## Rolling Windows",
        "",
    ]
    for name, detail in manifest["windows"].items():
        lines.append(f"- `{name}`: {detail['start']} through {detail['end']} ({detail['quarters']} quarters)")
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `employer_role_lca_summary.csv`: employer plus role-family rolling-window counts.",
            "- `employer_lca_summary.csv`: employer-level rolling-window counts across all roles.",
            "- `employer_lca_lookup.json`: employer-key lookup for future enrichment or diagnostics.",
            "- `lca_cache_manifest.json`: source files, windows, row counts, and limitations.",
            "",
            "## Top Recent Focus-Role Activity",
            "",
            "| Employer | Role family | Label | Recent 3Q certified cases | Recent 3Q certified positions |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in top_rows[:20]:
        lines.append(
            f"| {row['employer_display']} | {row['role_family']} | {row['lca_activity_label']} | "
            f"{row['recent_3q_certified_case_count']} | {row['recent_3q_certified_worker_positions']} |"
        )
    if not top_rows:
        lines.append("| No focus-role evidence rows generated |  |  | 0 | 0 |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_lca_cache(
    source_paths: Iterable[Path],
    *,
    output_dir: Path,
    as_of_date: date | None = None,
    visa_classes: Iterable[str] = ("H-1B",),
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Build LCA evidence summaries and write cache artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    allowed = {normalize_visa_class(value) for value in visa_classes}
    records, load_stats = load_lca_records(source_paths, allowed_visa_classes=allowed, max_rows=max_rows)
    role_rows, employer_rows, aggregate_metadata = aggregate_records(records, as_of_date=as_of_date)
    lookup = build_lookup(role_rows, employer_rows, aggregate_metadata)

    role_summary_path = output_dir / "employer_role_lca_summary.csv"
    employer_summary_path = output_dir / "employer_lca_summary.csv"
    lookup_path = output_dir / "employer_lca_lookup.json"
    manifest_path = output_dir / "lca_cache_manifest.json"
    report_path = output_dir / "lca_cache_report.md"

    write_csv(role_summary_path, role_rows, ROLE_SUMMARY_COLUMNS)
    write_csv(employer_summary_path, employer_rows, ROLE_SUMMARY_COLUMNS)
    write_json(lookup_path, lookup)

    generated_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    date_values = [record.decision_date for record in records]
    manifest: dict[str, Any] = {
        "generated_at": generated_at,
        "source": DOL_OFLC_PERFORMANCE_URL,
        "interpretation_note": DISCLOSURE_NOTE,
        "as_of_date": aggregate_metadata["as_of_date"],
        "windows": aggregate_metadata["windows"],
        "visa_classes_included": sorted(allowed),
        "source_files": load_stats["source_files"],
        "rows_seen": load_stats["rows_seen"],
        "rows_used_before_dedup": load_stats["rows_used"],
        "rows_skipped": load_stats["rows_skipped"],
        "duplicate_rows_replaced_or_ignored": load_stats["duplicate_rows_replaced_or_ignored"],
        "deduped_records_used": len(records),
        "min_decision_date": min(date_values).isoformat() if date_values else "",
        "max_decision_date": max(date_values).isoformat() if date_values else "",
        "employer_count": len(employer_rows),
        "employer_role_count": len(role_rows),
        "outputs": {
            "employer_role_lca_summary_csv": str(role_summary_path),
            "employer_lca_summary_csv": str(employer_summary_path),
            "employer_lca_lookup_json": str(lookup_path),
            "lca_cache_report_md": str(report_path),
        },
        "limitations": [
            DISCLOSURE_NOTE,
            "Counts are derived from disclosure records available in the local cache inputs.",
            "Overlapping quarterly or annual files are deduped by case number when present.",
            "Employer normalization is conservative and may not merge every affiliate or DBA variant.",
        ],
    }
    write_json(manifest_path, manifest)

    focus_rows = [
        row
        for row in role_rows
        if row["role_family"] in TARGET_ROLE_FAMILIES and row["recent_3q_case_count"] > 0
    ]
    focus_rows.sort(
        key=lambda row: (
            row["recent_3q_certified_worker_positions"],
            row["recent_3q_certified_case_count"],
            row["historical_8q_certified_worker_positions"],
        ),
        reverse=True,
    )
    write_report(report_path, manifest, focus_rows)
    return manifest
