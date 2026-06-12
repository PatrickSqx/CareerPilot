"""Central configuration for the JobPilot Phase 1 pipeline."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

KAGGLE_JSONL = (
    DATA_DIR
    / "[KAGGLE]techmap-jobs-dump-2021-09.json"
    / "techmap-jobs-dump-2021-09.json"
)

CURRENT_RAW_DIR = RAW_DATA_DIR / "current_jobs"
CACHED_CURRENT_JSONL = RAW_DATA_DIR / "cached_current_jobs_sample.jsonl"

JOBS_CLEAN_CSV = PROCESSED_DATA_DIR / "jobs_clean.csv"
CURRENT_JOBS_CLEAN_CSV = PROCESSED_DATA_DIR / "current_jobs_clean.csv"
OFFLINE_SNAPSHOT_CSV = PROCESSED_DATA_DIR / "jobs_offline_snapshot.csv"
OFFLINE_SNAPSHOT_SAMPLE_CSV = PROCESSED_DATA_DIR / "jobs_offline_snapshot_sample_500.csv"
INGESTION_DEMO_CSV = PROCESSED_DATA_DIR / "ingestion_demo_500.csv"
INGESTION_REPORT_JSON = PROCESSED_DATA_DIR / "ingestion_report.json"
DATA_DICTIONARY_MD = PROCESSED_DATA_DIR / "data_dictionary.md"
MARKET_ANALYTICS_JSON = PROCESSED_DATA_DIR / "market_analytics.json"
TECH_MARKET_ANALYTICS_JSON = PROCESSED_DATA_DIR / "tech_market_analytics.json"
EMBEDDINGS_DIR = PROCESSED_DATA_DIR / "embeddings"
PHASE2_BENCHMARKS_JSON = PROCESSED_DATA_DIR / "phase2_benchmarks.json"
PERSONA_PHASE2_RESULTS_JSON = PROCESSED_DATA_DIR / "persona_phase2_results.json"

DEFAULT_TARGET_ROWS = 50_000
DEFAULT_DEMO_ROWS = 500
DEFAULT_SCHEMA_SAMPLE_ROWS = 200
DEFAULT_BATCH_SIZE = 100

CURRENT_JOB_QUERIES = [
    "data analyst",
    "business analyst",
    "machine learning engineer",
    "analytics engineer",
    "data scientist",
    "mlops engineer",
    "applied scientist",
]

ADZUNA_APP_ID_ENV = "ADZUNA_APP_ID"
ADZUNA_APP_KEY_ENV = "ADZUNA_APP_KEY"
RAPIDAPI_KEY_ENV = "RAPIDAPI_KEY"
JSEARCH_RAPIDAPI_HOST_ENV = "JSEARCH_RAPIDAPI_HOST"
JSEARCH_API_KEY_ENV = "JSEARCH_API_KEY"


def ensure_phase1_dirs() -> None:
    """Create the directory tree used by Phase 1 outputs."""

    for path in [RAW_DATA_DIR, CURRENT_RAW_DIR, PROCESSED_DATA_DIR]:
        path.mkdir(parents=True, exist_ok=True)
