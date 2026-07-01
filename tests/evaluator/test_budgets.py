"""Behavior of evaluation-budget loading (stage selection, no hardcoded caps)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from evaluator.budgets import BudgetError, load_evaluation_budget

_CONFIG = textwrap.dedent(
    """
    smoke:
      max_candidates_per_condition: 2
      task_pairs:
        - user_task: put the bowl on the plate
          target_task: put the ketchup in the basket
      seeds: [0]
      rollouts_per_candidate: 1
      max_wall_clock_hours_per_candidate: 1.0
      top_k_for_full_eval: 0
      allow_async_jobs: false
    pilot:
      max_candidates_per_condition: 10
      task_pairs:
        - user_task: put the bowl on the plate
          target_task: put the ketchup in the basket
      seeds: [0, 1, 2]
      rollouts_per_candidate: 3
      max_wall_clock_hours_per_candidate: 4.0
      top_k_for_full_eval: 2
      allow_async_jobs: true
    """
)


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "budgets.yaml"
    path.write_text(_CONFIG, encoding="utf-8")
    return path


def test_loads_requested_stage(tmp_path):
    budget = load_evaluation_budget(_write_config(tmp_path), "smoke")

    assert budget["seeds"] == [0]
    assert budget["rollouts_per_candidate"] == 1
    assert budget["max_candidates_per_condition"] == 2


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHIPPED_BUDGETS = _REPO_ROOT / "experiments" / "configs" / "evaluation_budgets.yaml"


@pytest.mark.parametrize("stage", ["smoke", "pilot", "full"])
def test_shipped_budget_config_loads_every_stage(stage):
    budget = load_evaluation_budget(_SHIPPED_BUDGETS, stage)
    assert "seeds" in budget
    assert "rollouts_per_candidate" in budget


def test_unknown_stage_raises_budget_error(tmp_path):
    with pytest.raises(BudgetError):
        load_evaluation_budget(_write_config(tmp_path), "does_not_exist")


def test_stage_missing_required_field_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        textwrap.dedent(
            """
            smoke:
              seeds: [0]
              rollouts_per_candidate: 1
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(BudgetError):
        load_evaluation_budget(path, "smoke")
