"""Fast final-readiness checks for JobPilot deployment/submission.

This script does not build embeddings, run live APIs, or modify data. It checks
the files and imports needed before local smoke testing, Docker validation, and
Cloud Run deployment.
"""

from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def _rel(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


class Report:
    def __init__(self) -> None:
        self.hard_blockers: list[str] = []
        self.warnings: list[str] = []

    def ok(self, label: str, detail: str = "") -> None:
        print(f"[OK] {label}{': ' + detail if detail else ''}")

    def warn(self, label: str, detail: str = "") -> None:
        message = f"{label}{': ' + detail if detail else ''}"
        self.warnings.append(message)
        print(f"[WARN] {message}")

    def block(self, label: str, detail: str = "") -> None:
        message = f"{label}{': ' + detail if detail else ''}"
        self.hard_blockers.append(message)
        print(f"[BLOCK] {message}")


def _read_patterns(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _count_csv_rows(path: Path) -> tuple[int, list[str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        rows = sum(1 for _ in reader)
    return rows, header


def check_imports(report: Report) -> None:
    imports = [
        ("app.main", "FastAPI app"),
        ("jobpilot.ranking.ranker", "JobRanker module"),
        ("jobpilot.profile.personas", "persona fixtures"),
    ]
    for module, label in imports:
        try:
            importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - readiness diagnostics
            report.block(f"Import failed for {label}", f"{module}: {exc}")
        else:
            report.ok(f"Import {label}", module)


def check_required_files(report: Report) -> None:
    hard_required = [
        "app/main.py",
        "requirements.txt",
        "Dockerfile",
    ]
    for rel in hard_required:
        path = PROJECT_ROOT / rel
        if path.exists():
            report.ok("Required file exists", rel)
        else:
            report.block("Required file missing", rel)

    soft_required = [
        "README.md",
        "prompts.md",
        "brief_outline.md",
        ".dockerignore",
        ".gitignore",
        ".gcloudignore",
        "data/processed/market_analytics.json",
        "data/processed/phase2_benchmarks.json",
        "data/processed/persona_backend_diagnostics.json",
        "data/processed/persona_phase2_results.json",
        "data/processed/phase3_feedback_simulation.json",
    ]
    for rel in soft_required:
        path = PROJECT_ROOT / rel
        if path.exists():
            report.ok("Evidence/config file exists", rel)
        else:
            report.warn("Evidence/config file missing", rel)

    brief_pdf = PROJECT_ROOT / "brief.pdf"
    if brief_pdf.exists():
        report.ok("Final brief PDF exists", "brief.pdf")
    else:
        report.warn("Final brief PDF missing", "Create brief.pdf <= 4 pages before Canvas submission.")


def check_snapshot(report: Report) -> None:
    full = PROJECT_ROOT / "data/processed/jobs_offline_snapshot.csv"
    sample = PROJECT_ROOT / "data/processed/jobs_offline_snapshot_sample_500.csv"
    if full.exists():
        rows, header = _count_csv_rows(full)
        detail = f"{rows:,} rows, {len(header)} columns"
        if 20_000 <= rows <= 50_000:
            report.ok("Full offline snapshot ready", detail)
        else:
            report.block("Full offline snapshot outside required 20,000-50,000 range", detail)
        report.ok("Sample fallback active", "false")
        expected = {
            "job_id",
            "title",
            "company",
            "employer",
            "location",
            "salary_raw",
            "description_text",
            "link",
        }
        missing = sorted(expected - set(header))
        if missing:
            report.warn("Snapshot missing expected columns", ", ".join(missing))
        else:
            report.ok("Snapshot expected columns present", ", ".join(sorted(expected)))
    elif sample.exists():
        rows, header = _count_csv_rows(sample)
        report.warn("Full offline snapshot missing", "sample fallback is review/smoke-test only")
        report.ok("Sample fallback active", f"true; {rows:,} rows, {len(header)} columns")
    else:
        report.block("No usable offline snapshot", "Need jobs_offline_snapshot.csv or jobs_offline_snapshot_sample_500.csv")


def check_ignore_safety(report: Report) -> None:
    gitignore = _read_patterns(PROJECT_ROOT / ".gitignore")
    gcloudignore = _read_patterns(PROJECT_ROOT / ".gcloudignore")
    for rel in ["app/storage/", "data/raw/", "data/[KAGGLE]*/", "external_review_package/", "*.zip", "*.env", "API KEYS/"]:
        if rel in gitignore:
            report.ok(".gitignore excludes", rel)
        else:
            report.warn(".gitignore missing exclusion", rel)
        if rel in gcloudignore:
            report.ok(".gcloudignore excludes", rel)
        else:
            report.warn(".gcloudignore missing exclusion", rel)

    protected = [
        "data/processed/jobs_offline_snapshot.csv",
        "data/processed/market_analytics.json",
        "data/processed/phase2_benchmarks.json",
        "data/processed/persona_backend_diagnostics.json",
        "data/processed/persona_phase2_results.json",
        "data/processed/phase3_feedback_simulation.json",
    ]
    for rel in protected:
        if rel in gitignore or rel in gcloudignore:
            report.warn("Required processed file appears in ignore rules", rel)
        else:
            report.ok("Required processed file not directly ignored", rel)


def check_forbidden_file_presence(report: Report) -> None:
    checks = [
        ("API key directory present", PROJECT_ROOT / "API KEYS"),
        ("Runtime storage directory present", PROJECT_ROOT / "app/storage"),
        ("Raw data directory present", PROJECT_ROOT / "data/raw"),
        ("External review package directory present", PROJECT_ROOT / "external_review_package"),
    ]
    for label, path in checks:
        if path.exists():
            report.warn(label, f"{_rel(path)} is ignored but must not be submitted manually.")
        else:
            report.ok(label.replace(" present", " absent"))

    env_files = sorted(PROJECT_ROOT.rglob("*.env"))
    env_files.extend(path for path in PROJECT_ROOT.rglob(".env") if path.is_file())
    if env_files:
        report.warn("Env/API key files found", ", ".join(_rel(path) for path in env_files[:8]))
    else:
        report.ok("No env/API key files found")

    zip_files = sorted(PROJECT_ROOT.rglob("*.zip"))
    if zip_files:
        report.warn("ZIP files found", f"{len(zip_files)} ignored zip(s); do not include old review zips in final Canvas package.")
    else:
        report.ok("No ZIP files found")

    sqlite_files = sorted((PROJECT_ROOT / "app/storage").glob("*.sqlite")) if (PROJECT_ROOT / "app/storage").exists() else []
    if sqlite_files:
        report.warn("Runtime SQLite files found", ", ".join(_rel(path) for path in sqlite_files))
    else:
        report.ok("No runtime SQLite files found")


def main() -> int:
    report = Report()
    print("JobPilot final readiness check")
    print("=" * 34)
    check_required_files(report)
    check_imports(report)
    check_snapshot(report)
    check_ignore_safety(report)
    check_forbidden_file_presence(report)
    print("=" * 34)
    print(f"Warnings: {len(report.warnings)}")
    print(f"Hard blockers: {len(report.hard_blockers)}")
    if report.hard_blockers:
        return 1
    print("Status: ready for local smoke/deployment-readiness validation. Resolve warnings before final Canvas submission.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
