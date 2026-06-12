"""Optional live-search backend for JobPilot.

The default project path remains offline-only. These helpers are used by the
Phase 2.11 CLI and are intended for future Streamlit refresh controls.
"""

from jobpilot.live.query_builder import build_live_queries
from jobpilot.live.hybrid_pool import build_candidate_store, build_live_candidate_pool
from jobpilot.live.jsearch_live import fetch_jsearch_live
from jobpilot.live.adzuna_live import fetch_adzuna_live

__all__ = [
    "build_live_queries",
    "build_candidate_store",
    "build_live_candidate_pool",
    "fetch_adzuna_live",
    "fetch_jsearch_live",
]
