"""User-facing label helpers for app/export surfaces."""

from __future__ import annotations

from typing import Any

from app.services.paths import PROJECT_ROOT  # noqa: F401
from jobpilot.utils.text import clean_text


APPLICATION_STRATEGY_LABELS = {
    "Apply Now": "Recommended",
    "Same-company alternative": "Additional posting",
    "Potential duplicate role": "Similar posting",
}


def application_strategy_display_label(value: Any) -> str:
    """Map internal application strategy labels to user-safe wording."""

    label = clean_text(value)
    return APPLICATION_STRATEGY_LABELS.get(label, label)
