"""Behavior of search-condition config loading (comparability enforcement)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoresearch_loop.conditions import (
    SEARCH_CONDITIONS,
    ConditionsError,
    load_search_conditions,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHIPPED = _REPO_ROOT / "experiments" / "configs" / "search_conditions.yaml"
_SCHEMA = _REPO_ROOT / "experiments" / "configs" / "attack_candidate.schema.json"


def test_loads_all_six_conditions_and_budget_stage():
    config = load_search_conditions(_SHIPPED)

    assert config["budget_stage"] == "pilot"
    assert set(config["conditions"]) == set(SEARCH_CONDITIONS)


def test_condition_constant_matches_schema_enum():
    # Guard: the known conditions must stay in sync with the candidate schema.
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    enum = schema["properties"]["condition"]["enum"]
    assert set(SEARCH_CONDITIONS) == set(enum)


def test_missing_budget_stage_is_rejected(tmp_path):
    path = tmp_path / "conditions.yaml"
    path.write_text("conditions: {}\n", encoding="utf-8")
    with pytest.raises(ConditionsError):
        load_search_conditions(path)


def test_missing_condition_breaks_comparability(tmp_path):
    path = tmp_path / "conditions.yaml"
    path.write_text(
        "budget_stage: pilot\nconditions:\n  random_search:\n    generator: random\n",
        encoding="utf-8",
    )
    with pytest.raises(ConditionsError):
        load_search_conditions(path)


def test_unknown_condition_is_rejected(tmp_path):
    path = tmp_path / "conditions.yaml"
    lines = ["budget_stage: pilot", "conditions:"]
    for name in list(SEARCH_CONDITIONS) + ["sneaky_extra"]:
        lines.append(f"  {name}:")
        lines.append("    generator: random")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ConditionsError):
        load_search_conditions(path)
