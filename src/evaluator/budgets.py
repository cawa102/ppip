"""Evaluation-budget loading — part of the fixed evaluator.

A budget is the *scientific* unit of the experiment: candidates x task pairs x
seeds x rollouts. Selecting a stage (smoke/pilot/full) must never require code
changes, and `max_wall_clock_hours_per_candidate` is only a runaway guard, never
the scientific budget. There is deliberately no hardcoded 5-minute cap anywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class BudgetError(ValueError):
    """Raised when a budget config is missing a stage or a required field."""


# Every stage must define the full scientific budget explicitly; no field
# defaults, so a comparison can never silently run conditions on different
# budgets.
_REQUIRED_FIELDS = (
    "max_candidates_per_condition",
    "task_pairs",
    "seeds",
    "rollouts_per_candidate",
    "max_wall_clock_hours_per_candidate",
    "top_k_for_full_eval",
    "allow_async_jobs",
)


def load_evaluation_budget(config_path: str | Path, stage: str) -> dict[str, Any]:
    """Load the budget for `stage` from a YAML budget config."""
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if stage not in data:
        available = ", ".join(sorted(data)) or "<none>"
        raise BudgetError(f"unknown stage {stage!r}; available stages: {available}")
    budget: dict[str, Any] = data[stage]
    missing = [field for field in _REQUIRED_FIELDS if field not in budget]
    if missing:
        raise BudgetError(f"stage {stage!r} missing required fields: {', '.join(missing)}")
    return budget
