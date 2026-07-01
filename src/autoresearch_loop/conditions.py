"""Search-condition config loading (search side).

Enforces the comparability contract: a valid conditions config must define every
known condition (so no condition is silently dropped from the comparison) and no
unknown ones. All conditions share a single `budget_stage`, which is what keeps
the scientific budget identical across the comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# The six comparable search conditions. Kept in sync with the candidate schema's
# `condition` enum (guarded by a test).
SEARCH_CONDITIONS: tuple[str, ...] = (
    "random_search",
    "human_ppia",
    "one_shot_llm",
    "loop_no_memory",
    "loop_with_memory",
    "loop_with_skill",
)


class ConditionsError(ValueError):
    """Raised when a conditions config would break condition comparability."""


def load_search_conditions(config_path: str | Path) -> dict[str, Any]:
    """Load and validate a search-conditions config."""
    data: dict[str, Any] = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if "budget_stage" not in data:
        raise ConditionsError("conditions config is missing 'budget_stage'")
    conditions = data.get("conditions") or {}

    defined = set(conditions)
    expected = set(SEARCH_CONDITIONS)
    missing = expected - defined
    if missing:
        raise ConditionsError(f"conditions config is missing: {', '.join(sorted(missing))}")
    unknown = defined - expected
    if unknown:
        raise ConditionsError(
            f"conditions config has unknown conditions: {', '.join(sorted(unknown))}"
        )
    return data
