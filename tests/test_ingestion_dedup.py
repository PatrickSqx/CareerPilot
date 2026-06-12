from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from jobpilot.ingestion.dedup import BloomFilter, ExactDeduplicator


def _record(title: str, company: str = "Acme", location: str = "Austin, TX") -> dict[str, str]:
    return {
        "title": title,
        "company": company,
        "location": location,
        "description_text": f"{title} role with clear job description.",
    }


def test_bloom_precheck_preserves_exact_duplicate_rejection() -> None:
    deduplicator = ExactDeduplicator(bloom_expected_items=100, bloom_false_positive_rate=0.01)

    keep_first, first = deduplicator.keep(_record("Data Analyst"))
    keep_duplicate, duplicate = deduplicator.keep(_record("Data Analyst"))

    assert keep_first is True
    assert keep_duplicate is False
    assert first["dedup_key"] == duplicate["dedup_key"]
    assert deduplicator.duplicates_removed == 1

    summary = deduplicator.summary()
    assert summary["method"] == "bloom_precheck_plus_exact_hash_set"
    assert summary["exact_verification"] is True
    assert summary["bloom_negative_checks"] == 1
    assert summary["bloom_positive_checks"] == 1
    assert summary["exact_membership_checks"] == 1
    assert summary["exact_membership_checks_skipped_by_bloom"] == 1


def test_bloom_false_positive_still_keeps_unique_record() -> None:
    tiny_bloom = BloomFilter(bit_size=1, hash_count=1)
    deduplicator = ExactDeduplicator(bloom_filter=tiny_bloom)

    keep_first, first = deduplicator.keep(_record("Data Analyst"))
    keep_second, second = deduplicator.keep(_record("Business Analyst"))

    assert keep_first is True
    assert keep_second is True
    assert first["dedup_key"] != second["dedup_key"]
    assert deduplicator.duplicates_removed == 0
    assert deduplicator.bloom_false_positive_checks == 1
    assert deduplicator.exact_membership_checks == 1
