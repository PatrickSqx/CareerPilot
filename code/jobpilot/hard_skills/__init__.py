"""Offline hard-skill sidecar extraction for JobPilot Phase 2.16B.

This package intentionally does not mutate Phase 1 snapshots or Phase 2
ranking. It produces separate job_id-level sidecar artifacts.
"""

from jobpilot.hard_skills.sidecar import build_hard_skill_sidecar

__all__ = ["build_hard_skill_sidecar"]
