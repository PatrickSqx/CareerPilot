"""Ingestion-time Bloom-assisted exact hash deduplication."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

from jobpilot.utils.text import normalize_for_key, stable_hash


def add_dedup_fields(record: dict[str, Any]) -> dict[str, Any]:
    """Attach description_hash and dedup_key to a normalized record."""

    description_hash = stable_hash(record.get("description_text", ""))
    basis = "|".join(
        [
            normalize_for_key(record.get("title")),
            normalize_for_key(record.get("company")),
            normalize_for_key(record.get("location")),
            description_hash,
        ]
    )
    updated = dict(record)
    updated["description_hash"] = description_hash
    updated["dedup_key"] = stable_hash(basis)
    return updated


@dataclass
class BloomFilter:
    """Small deterministic Bloom filter used as a streaming pre-check."""

    bit_size: int
    hash_count: int
    bits: bytearray = field(init=False)

    def __post_init__(self) -> None:
        if self.bit_size <= 0:
            raise ValueError("bit_size must be positive")
        if self.hash_count <= 0:
            raise ValueError("hash_count must be positive")
        self.bits = bytearray((self.bit_size + 7) // 8)

    @classmethod
    def from_capacity(cls, expected_items: int, false_positive_rate: float) -> "BloomFilter":
        """Create a Bloom filter sized for the expected stream cardinality."""

        expected = max(int(expected_items), 1)
        rate = min(max(float(false_positive_rate), 1e-9), 0.5)
        bit_size = max(8, int(-(expected * math.log(rate)) / (math.log(2) ** 2)))
        hash_count = max(1, int(round((bit_size / expected) * math.log(2))))
        return cls(bit_size=bit_size, hash_count=hash_count)

    def _indexes(self, key: str) -> list[int]:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        h1 = int.from_bytes(digest[:8], "big")
        h2 = int.from_bytes(digest[8:16], "big") or 1
        return [((h1 + index * h2 + index * index) % self.bit_size) for index in range(self.hash_count)]

    def might_contain(self, key: str) -> bool:
        for bit_index in self._indexes(key):
            byte_index, offset = divmod(bit_index, 8)
            if not (self.bits[byte_index] & (1 << offset)):
                return False
        return True

    def add(self, key: str) -> None:
        for bit_index in self._indexes(key):
            byte_index, offset = divmod(bit_index, 8)
            self.bits[byte_index] |= 1 << offset

    def metadata(self) -> dict[str, Any]:
        return {
            "bit_size": self.bit_size,
            "byte_size": len(self.bits),
            "hash_count": self.hash_count,
        }


@dataclass
class ExactDeduplicator:
    """Bloom pre-check plus exact hash-set verification.

    The Bloom filter is only used to avoid exact-set membership checks when it
    says a key has definitely not been seen. Records are never dropped based on
    the Bloom filter alone; possible duplicates are verified against ``seen``.
    """

    seen: set[str] = field(default_factory=set)
    duplicates_removed: int = 0
    bloom_enabled: bool = True
    bloom_expected_items: int = 500_000
    bloom_false_positive_rate: float = 0.01
    bloom_filter: BloomFilter | None = None
    records_checked: int = 0
    exact_membership_checks: int = 0
    bloom_negative_checks: int = 0
    bloom_positive_checks: int = 0
    bloom_false_positive_checks: int = 0
    keys_added: int = 0

    def __post_init__(self) -> None:
        if self.bloom_enabled and self.bloom_filter is None:
            self.bloom_filter = BloomFilter.from_capacity(
                self.bloom_expected_items,
                self.bloom_false_positive_rate,
            )

    def keep(self, record: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        record = add_dedup_fields(record)
        key = record["dedup_key"]
        self.records_checked += 1

        maybe_seen = True
        if self.bloom_filter is not None:
            maybe_seen = self.bloom_filter.might_contain(key)
            if maybe_seen:
                self.bloom_positive_checks += 1
            else:
                self.bloom_negative_checks += 1

        if maybe_seen:
            self.exact_membership_checks += 1
            if key in self.seen:
                self.duplicates_removed += 1
                return False, record
            if self.bloom_filter is not None:
                self.bloom_false_positive_checks += 1

        self.seen.add(key)
        self.keys_added += 1
        if self.bloom_filter is not None:
            self.bloom_filter.add(key)
        return True, record

    def summary(self) -> dict[str, Any]:
        exact_checks_skipped = self.records_checked - self.exact_membership_checks
        false_positive_rate_observed = (
            self.bloom_false_positive_checks / self.bloom_positive_checks
            if self.bloom_positive_checks
            else 0.0
        )
        return {
            "method": "bloom_precheck_plus_exact_hash_set",
            "exact_verification": True,
            "dedup_key_basis": "normalized_title + normalized_company + normalized_location + description_hash",
            "records_checked": self.records_checked,
            "unique_keys_added": self.keys_added,
            "duplicates_removed": self.duplicates_removed,
            "exact_membership_checks": self.exact_membership_checks,
            "exact_membership_checks_skipped_by_bloom": exact_checks_skipped,
            "bloom_enabled": self.bloom_filter is not None,
            "bloom_expected_items": self.bloom_expected_items,
            "bloom_false_positive_rate_target": self.bloom_false_positive_rate,
            "bloom_positive_checks": self.bloom_positive_checks,
            "bloom_negative_checks": self.bloom_negative_checks,
            "bloom_false_positive_checks": self.bloom_false_positive_checks,
            "bloom_false_positive_rate_observed": round(false_positive_rate_observed, 6),
            "bloom_filter": self.bloom_filter.metadata() if self.bloom_filter is not None else None,
            "safety_note": "Bloom filter is a pre-check only; duplicate rejection still requires exact hash-set verification.",
        }
