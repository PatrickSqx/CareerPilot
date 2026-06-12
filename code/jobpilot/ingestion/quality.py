"""Quality scoring and diversity guards for Phase 1 snapshot sampling."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


INVALID_VALUES = {
    "",
    "false",
    "true",
    "none",
    "null",
    "nan",
    "n/a",
    "na",
    "unknown",
    "unknown employer",
    "unknown location",
}

QUALITY_FEATURE_WEIGHTS = {
    "country": 8,
    "city": 4,
    "state": 4,
    "salary_any": 14,
    "employment_type_any": 12,
    "company_url": 8,
    "raw_categories": 12,
    "raw_work_types": 8,
    "raw_qualifications": 5,
    "schema_org_employment_type": 8,
    "schema_org_skills": 8,
    "schema_org_experience_requirements": 9,
}

QUALITY_SCORE_MAX = sum(QUALITY_FEATURE_WEIGHTS.values())
SAMPLING_ELIGIBLE_THRESHOLD = 60
STRONG_QUALITY_THRESHOLD = 75
SCORE_GT_85_THRESHOLD = 86
MEDIUM_QUALITY_THRESHOLD = 35

SOURCE_CAP_SHARE = 0.10
COUNTRY_CAP_SHARE = 0.30
CATEGORY_CAP_SHARE = 0.08
TITLE_CAP_SHARE = 0.03
COMPANY_CAP_SHARE = 0.10
MIN_SOURCE_CAP = 2_500
MIN_COUNTRY_CAP = 10_000
MIN_CATEGORY_CAP = 2_500
MIN_TITLE_CAP = 1_000
MIN_COMPANY_CAP = 3_000
NON_US_REMOTE_SOFT_TARGET_SHARE = 0.001
MIN_NON_US_REMOTE_SOFT_TARGET = 50
MAX_NON_US_REMOTE_SOFT_TARGET = 250

US_STATE_CODES = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "DC",
}

SOURCE_COUNTRY_SUFFIXES = {
    "us": "US",
    "usa": "US",
    "uk": "GB",
    "gb": "GB",
    "de": "DE",
    "au": "AU",
    "nz": "NZ",
    "ca": "CA",
    "ie": "IE",
    "nl": "NL",
    "se": "SE",
    "fr": "FR",
    "it": "IT",
    "es": "ES",
    "ch": "CH",
    "at": "AT",
    "in": "IN",
    "sg": "SG",
    "ph": "PH",
    "my": "MY",
    "ae": "AE",
    "cn": "CN",
    "hk": "HK",
    "lu": "LU",
    "ua": "UA",
    "ru": "RU",
    "vn": "VN",
}

REMOTE_SIGNAL_RE = re.compile(
    r"\b(remote|work\s+from\s+home|wfh|anywhere|distributed|virtual)\b",
    re.IGNORECASE,
)
GLOBAL_REMOTE_RE = re.compile(
    r"\b(remote\s+(?:anywhere|worldwide|globally)|anywhere\s+remote|work\s+from\s+anywhere|global\s+remote)\b",
    re.IGNORECASE,
)
NON_US_LOCATION_TOKEN = (
    r"(?:uk|u\.k\.|united kingdom|england|scotland|wales|ireland|dublin|london|"
    r"germany|deutschland|berlin|canada|toronto|ontario|australia|sydney|"
    r"new zealand|auckland|india|singapore|philippines|malaysia|europe|eu)"
)
REMOTE_RESTRICTION_RE = re.compile(
    rf"\b(?:{NON_US_LOCATION_TOKEN})\s+only\b|"
    rf"\b(?:{NON_US_LOCATION_TOKEN})\s+(?:applications|applicants|candidates)\s+only\b|"
    rf"\b(?:applications|applicants|candidates)\s+(?:from|in|within)\s+(?:the\s+)?{NON_US_LOCATION_TOKEN}\s+only\b|"
    rf"\bonly\s+(?:in|within|from)\s+(?:the\s+)?{NON_US_LOCATION_TOKEN}\b|"
    rf"\bmust\s+(?:be\s+)?(?:based|located|reside|live)\s+(?:in|within)\s+(?:the\s+)?{NON_US_LOCATION_TOKEN}\b|"
    rf"\b(?:candidate|applicant)s?\s+must\s+(?:be\s+)?(?:based|located|reside|live)\s+(?:in|within)\s+(?:the\s+)?{NON_US_LOCATION_TOKEN}\b|"
    rf"\bright\s+to\s+work\s+in\s+(?:the\s+)?{NON_US_LOCATION_TOKEN}\b|"
    rf"\bremote\s+(?:within|in|from|across)\s+(?:the\s+)?{NON_US_LOCATION_TOKEN}\b|"
    rf"\b(?:based|located)\s+in\s+{NON_US_LOCATION_TOKEN}\b",
    re.IGNORECASE,
)
KEY_NORMALIZER_RE = re.compile(r"[^a-z0-9]+")


def clean_value(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\x00", " ").split())


def has_value(value: Any) -> bool:
    text = clean_value(value)
    return bool(text) and text.strip().lower() not in INVALID_VALUES


def has_any(row: dict[str, Any], *columns: str) -> bool:
    return any(has_value(row.get(column, "")) for column in columns)


def normalized_country(row: dict[str, Any]) -> str:
    raw = clean_value(row.get("country") or row.get("raw_source_country")).strip()
    key = raw.lower()
    aliases = {
        "us": "US",
        "usa": "US",
        "united states": "US",
        "united states of america": "US",
        "gb": "GB",
        "uk": "GB",
        "united kingdom": "GB",
        "great britain": "GB",
        "england": "GB",
        "de": "DE",
        "germany": "DE",
        "au": "AU",
        "australia": "AU",
        "nz": "NZ",
        "new zealand": "NZ",
        "ca": "CA",
        "canada": "CA",
    }
    return aliases.get(key, raw.upper() if len(raw) <= 3 else raw)


def source_country_code(row: dict[str, Any]) -> str:
    """Infer a source-country code from provider names such as `careerbuilder_us`."""

    source = source_key(row).lower()
    for token in reversed(source.split("_")):
        if token in SOURCE_COUNTRY_SUFFIXES:
            return SOURCE_COUNTRY_SUFFIXES[token]
    return ""


def primary_category(row: dict[str, Any]) -> str:
    raw = clean_value(row.get("raw_categories"))
    if not raw:
        return "(blank)"
    return raw.split("|", 1)[0].strip() or "(blank)"


def source_key(row: dict[str, Any]) -> str:
    return clean_value(row.get("source")) or "unknown"


def country_key(row: dict[str, Any]) -> str:
    return normalized_country(row) or "(blank)"


def category_key(row: dict[str, Any]) -> str:
    return primary_category(row)


def normalized_key(value: Any, *, default: str = "(blank)", max_length: int = 120) -> str:
    text = clean_value(value).lower()
    text = KEY_NORMALIZER_RE.sub(" ", text).strip()
    return (text[:max_length].strip() if text else "") or default


def title_key(row: dict[str, Any]) -> str:
    return normalized_key(row.get("title"), default="(blank_title)", max_length=100)


def company_key(row: dict[str, Any]) -> str:
    return normalized_key(row.get("company") or row.get("employer"), default="unknown employer", max_length=100)


def state_key(row: dict[str, Any]) -> str:
    return clean_value(row.get("state")).upper() or "(blank)"


def is_us_eligible(row: dict[str, Any]) -> bool:
    """Return whether a record has a US market signal.

    The Kaggle dump is sparse, so this uses explicit country fields first and
    then source/locale/location signals. State alone is only used when no
    country/source-country contradicts it.
    """

    country = normalized_country(row).upper()
    source_country = source_country_code(row)
    locale = clean_value(row.get("raw_locale")).lower()
    location = clean_value(row.get("location")).lower()
    state = clean_value(row.get("state")).upper()
    if country == "US" or source_country == "US":
        return True
    if locale in {"en-us", "en_us"}:
        return True
    if re.search(r"\b(united states|usa|u\.s\.a\.|u\.s\.|us)\b", location):
        return True
    if not country and not source_country and state in US_STATE_CODES:
        return True
    return False


def has_remote_signal(row: dict[str, Any]) -> bool:
    fields = [
        row.get("is_remote"),
        row.get("title"),
        row.get("location"),
        row.get("position_work_type_raw"),
        row.get("raw_work_types"),
        clean_value(row.get("description_text") or row.get("description"))[:2000],
    ]
    text = " ".join(clean_value(value) for value in fields if clean_value(value))
    return bool(REMOTE_SIGNAL_RE.search(text))


def has_non_us_remote_restriction(row: dict[str, Any]) -> bool:
    fields = [
        row.get("title"),
        row.get("location"),
        row.get("country"),
        row.get("state"),
        row.get("raw_source_country"),
        clean_value(row.get("description_text") or row.get("description"))[:10000],
    ]
    text = " ".join(clean_value(value) for value in fields if clean_value(value))
    if GLOBAL_REMOTE_RE.search(text):
        return False
    return bool(REMOTE_RESTRICTION_RE.search(text))


def market_eligibility(row: dict[str, Any]) -> tuple[bool, str]:
    """Return final-market eligibility and a report label.

    Phase 1.7 is US-first. Non-US or unknown-market rows are retained only when
    they have an explicit remote-compatible signal and do not look restricted
    to a non-US local market.
    """

    if is_us_eligible(row):
        return True, "us"
    if not has_remote_signal(row):
        return False, "non_us_not_remote_compatible"
    if has_non_us_remote_restriction(row):
        return False, "non_us_remote_restricted"
    return True, "non_us_remote_compatible"


def required_ready(row: dict[str, Any]) -> bool:
    return (
        has_value(row.get("title"))
        and has_value(row.get("company"))
        and has_value(row.get("link"))
        and has_any(row, "description_text", "description")
        and has_value(row.get("source_record_id"))
        and has_value(row.get("source"))
        and has_any(row, "location", "city", "state", "country", "raw_source_country", "location_id")
    )


def row_quality_score(row: dict[str, Any]) -> int:
    features = {
        "country": has_any(row, "country", "raw_source_country"),
        "city": has_value(row.get("city")),
        "state": has_value(row.get("state")),
        "salary_any": has_any(row, "salary_raw", "salary_min", "salary_max", "raw_salary_text", "schema_org_salary_min"),
        "employment_type_any": has_any(row, "employment_type", "position_work_type_raw", "schema_org_employment_type"),
        "company_url": has_value(row.get("company_url")),
        "raw_categories": has_value(row.get("raw_categories")),
        "raw_work_types": has_value(row.get("raw_work_types")),
        "raw_qualifications": has_value(row.get("raw_qualifications")),
        "schema_org_employment_type": has_value(row.get("schema_org_employment_type")),
        "schema_org_skills": has_value(row.get("schema_org_skills")),
        "schema_org_experience_requirements": has_value(row.get("schema_org_experience_requirements")),
    }
    return sum(weight for feature, weight in QUALITY_FEATURE_WEIGHTS.items() if features[feature])


def selection_quality_score(row: dict[str, Any]) -> int:
    """Score used for Phase 1 selection.

    Phase 1.7 may carry a raw-audit manifest score for rows recovered from the
    score>85 manifest. That score is used only for selecting candidates; final
    CSV output still contains the canonical normalized columns only.
    """

    raw_score = clean_value(row.get("_phase1_manifest_quality_score"))
    if raw_score:
        try:
            return int(float(raw_score))
        except ValueError:
            pass
    return row_quality_score(row)


def quality_tier(row: dict[str, Any]) -> str:
    if not required_ready(row):
        return "not_required_ready"
    score = row_quality_score(row)
    if score >= STRONG_QUALITY_THRESHOLD:
        return "strong_quality"
    if score >= SAMPLING_ELIGIBLE_THRESHOLD:
        return "sampling_eligible"
    if score >= MEDIUM_QUALITY_THRESHOLD:
        return "medium_quality"
    return "required_ready"


def score_bucket(score: int) -> str:
    if score >= 80:
        return "80-100"
    if score >= 60:
        return "60-79"
    if score >= 40:
        return "40-59"
    if score >= 20:
        return "20-39"
    return "0-19"


def cap_limit(target_rows: int, share: float, minimum: int) -> int:
    return max(minimum, int(target_rows * share))


@dataclass
class QualityStratifiedSelector:
    """Select sampling-eligible rows while limiting source/country/category concentration."""

    target_rows: int
    min_quality_score: int = SAMPLING_ELIGIBLE_THRESHOLD
    source_cap_share: float = SOURCE_CAP_SHARE
    country_cap_share: float = COUNTRY_CAP_SHARE
    category_cap_share: float = CATEGORY_CAP_SHARE
    min_source_cap: int = MIN_SOURCE_CAP
    min_country_cap: int = MIN_COUNTRY_CAP
    min_category_cap: int = MIN_CATEGORY_CAP
    selected: list[dict[str, Any]] = field(default_factory=list)
    source_counts: Counter[str] = field(default_factory=Counter)
    country_counts: Counter[str] = field(default_factory=Counter)
    category_counts: Counter[str] = field(default_factory=Counter)
    rejection_counts: Counter[str] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        self.source_cap = cap_limit(self.target_rows, self.source_cap_share, self.min_source_cap)
        self.country_cap = cap_limit(self.target_rows, self.country_cap_share, self.min_country_cap)
        self.category_cap = cap_limit(self.target_rows, self.category_cap_share, self.min_category_cap)

    def complete(self) -> bool:
        return len(self.selected) >= self.target_rows

    def accept(self, row: dict[str, Any]) -> bool:
        if self.complete():
            self.rejection_counts["target_already_full"] += 1
            return False
        if not required_ready(row):
            self.rejection_counts["not_required_ready"] += 1
            return False
        score = row_quality_score(row)
        if score < self.min_quality_score:
            self.rejection_counts["below_quality_threshold"] += 1
            return False

        source = source_key(row)
        country = country_key(row)
        category = category_key(row)
        if self.source_counts[source] >= self.source_cap:
            self.rejection_counts["source_cap"] += 1
            return False
        if self.country_counts[country] >= self.country_cap:
            self.rejection_counts["country_cap"] += 1
            return False
        if self.category_counts[category] >= self.category_cap:
            self.rejection_counts["category_cap"] += 1
            return False

        self.selected.append(row)
        self.source_counts[source] += 1
        self.country_counts[country] += 1
        self.category_counts[category] += 1
        return True

    def policy(self) -> dict[str, Any]:
        return {
            "strategy": "quality_stratified_streaming",
            "target_rows": self.target_rows,
            "min_quality_score": self.min_quality_score,
            "min_quality_score_label": "sampling_eligible",
            "strong_quality_score_threshold": STRONG_QUALITY_THRESHOLD,
            "quality_score_max": QUALITY_SCORE_MAX,
            "source_cap": self.source_cap,
            "source_cap_share": self.source_cap_share,
            "country_cap": self.country_cap,
            "country_cap_share": self.country_cap_share,
            "category_cap": self.category_cap,
            "category_cap_share": self.category_cap_share,
            "quality_feature_weights": QUALITY_FEATURE_WEIGHTS,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "selected_rows": len(self.selected),
            "rejection_counts": dict(self.rejection_counts),
            "selected_source_counts": dict(self.source_counts.most_common()),
            "selected_country_counts": dict(self.country_counts.most_common()),
            "selected_category_counts": dict(self.category_counts.most_common()),
            "policy": self.policy(),
        }


@dataclass
class USFirstRemoteSelector:
    """Phase 1.7 selector: score>85, US-first, non-US only when remote-compatible.

    Source concentration is reported but no longer hard-capped. Category, title,
    and company guards remain to prevent the 50k sample from collapsing into a
    few repeated job families or employers.
    """

    target_rows: int
    min_quality_score: int = SCORE_GT_85_THRESHOLD
    category_cap_share: float = CATEGORY_CAP_SHARE
    title_cap_share: float = TITLE_CAP_SHARE
    company_cap_share: float = COMPANY_CAP_SHARE
    min_category_cap: int = MIN_CATEGORY_CAP
    min_title_cap: int = MIN_TITLE_CAP
    min_company_cap: int = MIN_COMPANY_CAP
    non_us_remote_soft_target_share: float = NON_US_REMOTE_SOFT_TARGET_SHARE
    min_non_us_remote_soft_target: int = MIN_NON_US_REMOTE_SOFT_TARGET
    max_non_us_remote_soft_target: int = MAX_NON_US_REMOTE_SOFT_TARGET
    selected: list[dict[str, Any]] = field(default_factory=list)
    source_counts: Counter[str] = field(default_factory=Counter)
    country_counts: Counter[str] = field(default_factory=Counter)
    category_counts: Counter[str] = field(default_factory=Counter)
    title_counts: Counter[str] = field(default_factory=Counter)
    company_counts: Counter[str] = field(default_factory=Counter)
    state_counts: Counter[str] = field(default_factory=Counter)
    market_counts: Counter[str] = field(default_factory=Counter)
    score_buckets: Counter[str] = field(default_factory=Counter)
    rejection_counts: Counter[str] = field(default_factory=Counter)
    replacement_counts: Counter[str] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        self.category_cap = cap_limit(self.target_rows, self.category_cap_share, self.min_category_cap)
        self.title_cap = cap_limit(self.target_rows, self.title_cap_share, self.min_title_cap)
        self.company_cap = cap_limit(self.target_rows, self.company_cap_share, self.min_company_cap)
        if self.target_rows < 1_000:
            self.non_us_remote_soft_target = 0
        else:
            target = int(self.target_rows * self.non_us_remote_soft_target_share)
            self.non_us_remote_soft_target = min(
                self.max_non_us_remote_soft_target,
                max(self.min_non_us_remote_soft_target, target),
                self.target_rows,
            )

    def complete(self) -> bool:
        if len(self.selected) < self.target_rows:
            return False
        if self.non_us_remote_soft_target <= 0:
            return True
        return self.market_counts["non_us_remote_compatible"] >= self.non_us_remote_soft_target

    def _row_keys(self, row: dict[str, Any]) -> dict[str, str]:
        return {
            "source": source_key(row),
            "country": country_key(row),
            "category": category_key(row),
            "title": title_key(row),
            "company": company_key(row),
            "state": state_key(row),
        }

    def _track(self, row: dict[str, Any], market_label: str | None = None) -> None:
        keys = self._row_keys(row)
        self.source_counts[keys["source"]] += 1
        self.country_counts[keys["country"]] += 1
        self.category_counts[keys["category"]] += 1
        self.title_counts[keys["title"]] += 1
        self.company_counts[keys["company"]] += 1
        self.state_counts[keys["state"]] += 1
        if market_label is None:
            _, market_label = market_eligibility(row)
        self.market_counts[market_label] += 1
        self.score_buckets[score_bucket(selection_quality_score(row))] += 1

    def _untrack(self, row: dict[str, Any]) -> None:
        keys = self._row_keys(row)
        for counter, key in [
            (self.source_counts, keys["source"]),
            (self.country_counts, keys["country"]),
            (self.category_counts, keys["category"]),
            (self.title_counts, keys["title"]),
            (self.company_counts, keys["company"]),
            (self.state_counts, keys["state"]),
        ]:
            counter[key] -= 1
            if counter[key] <= 0:
                del counter[key]
        _, market_label = market_eligibility(row)
        self.market_counts[market_label] -= 1
        if self.market_counts[market_label] <= 0:
            del self.market_counts[market_label]
        bucket = score_bucket(selection_quality_score(row))
        self.score_buckets[bucket] -= 1
        if self.score_buckets[bucket] <= 0:
            del self.score_buckets[bucket]

    def _guard_rejection(self, row: dict[str, Any]) -> str:
        keys = self._row_keys(row)
        if self.category_counts[keys["category"]] >= self.category_cap:
            return "category_guard"
        if self.title_counts[keys["title"]] >= self.title_cap:
            return "title_guard"
        if self.company_counts[keys["company"]] >= self.company_cap:
            return "company_guard"
        return ""

    def _find_us_replacement_index(self) -> int | None:
        candidates: list[tuple[int, int]] = []
        for index, row in enumerate(self.selected):
            _, market_label = market_eligibility(row)
            if market_label != "non_us_remote_compatible":
                candidates.append((selection_quality_score(row), index))
        if not candidates:
            return None
        return min(candidates)[1]

    def accept(self, row: dict[str, Any]) -> bool:
        if self.target_rows <= 0:
            self.rejection_counts["target_already_full"] += 1
            return False
        if not required_ready(row):
            self.rejection_counts["not_required_ready"] += 1
            return False
        score = selection_quality_score(row)
        if score < self.min_quality_score:
            self.rejection_counts["below_score_gt_85_threshold"] += 1
            return False

        eligible, market_label = market_eligibility(row)
        if not eligible:
            self.rejection_counts[market_label] += 1
            return False

        guard = self._guard_rejection(row)
        if guard:
            self.rejection_counts[guard] += 1
            return False

        if len(self.selected) < self.target_rows:
            self.selected.append(row)
            self._track(row, market_label)
            return True

        if (
            market_label == "non_us_remote_compatible"
            and self.market_counts["non_us_remote_compatible"] < self.non_us_remote_soft_target
        ):
            replace_index = self._find_us_replacement_index()
            if replace_index is None:
                self.rejection_counts["target_already_full"] += 1
                return False
            old_row = self.selected[replace_index]
            self._untrack(old_row)
            self.selected[replace_index] = row
            self._track(row, market_label)
            self.replacement_counts["us_replaced_by_non_us_remote"] += 1
            return True

        self.rejection_counts["target_already_full"] += 1
        return False

    def policy(self) -> dict[str, Any]:
        return {
            "strategy": "phase1_7_us_first_score_gt_85_remote_non_us_streaming",
            "target_rows": self.target_rows,
            "min_quality_score": self.min_quality_score,
            "min_quality_score_label": "score_gt_85",
            "strong_quality_score_threshold": STRONG_QUALITY_THRESHOLD,
            "quality_score_max": QUALITY_SCORE_MAX,
            "us_first": True,
            "non_us_policy": "remote_compatible_only",
            "non_us_remote_soft_target": self.non_us_remote_soft_target,
            "non_us_remote_soft_target_share": self.non_us_remote_soft_target_share,
            "source_hard_cap_enabled": False,
            "country_hard_cap_enabled": False,
            "category_guard_cap": self.category_cap,
            "category_guard_cap_share": self.category_cap_share,
            "title_guard_cap": self.title_cap,
            "title_guard_cap_share": self.title_cap_share,
            "company_guard_cap": self.company_cap,
            "company_guard_cap_share": self.company_cap_share,
            "quality_feature_weights": QUALITY_FEATURE_WEIGHTS,
            "selection_score_source": "raw_score85_manifest_when_available_else_normalized_quality_score",
        }

    def summary(self) -> dict[str, Any]:
        non_us_count = self.market_counts.get("non_us_remote_compatible", 0)
        return {
            "selected_rows": len(self.selected),
            "rejection_counts": dict(self.rejection_counts),
            "replacement_counts": dict(self.replacement_counts),
            "selected_market_eligibility_counts": dict(self.market_counts.most_common()),
            "selected_source_counts": dict(self.source_counts.most_common()),
            "selected_country_counts": dict(self.country_counts.most_common()),
            "selected_category_counts": dict(self.category_counts.most_common(50)),
            "selected_title_counts_top50": dict(self.title_counts.most_common(50)),
            "selected_company_counts_top50": dict(self.company_counts.most_common(50)),
            "selected_state_counts": dict(self.state_counts.most_common(50)),
            "selected_quality_score_buckets": dict(self.score_buckets),
            "non_us_remote_soft_target_met": non_us_count >= self.non_us_remote_soft_target,
            "policy": self.policy(),
        }
