"""Panel selection for the held-out ASR figure.

The sweep produces 12 rollouts and a figure can show only a handful. Which ones it shows is a
claim about the sweep, so the choice is made by a tested rule rather than by eye: show one
panel per outcome class that actually occurred, so a reader sees the real spread instead of a
flattering subset. The rarest class must never be dropped -- the single hijack in 12 is the
whole reason the sweep has a non-zero attack rate.
"""

from __future__ import annotations

import os
import sys

import pytest

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "autoresearch/experiments/patch_attack"))

from make_asr_gif import representative_inits  # noqa: E402


def _rows(*pairs: tuple[int, str]) -> list[dict]:
    return [{"init": i, "outcome": o} for i, o in pairs]


def test_one_panel_per_outcome_class_that_occurred() -> None:
    rows = _rows((4, "dos"), (39, "commanded"), (46, "hijack"))

    picked = representative_inits(rows, limit=3)

    assert {r["outcome"] for r in picked} == {"dos", "commanded", "hijack"}


def test_the_single_hijack_is_never_dropped() -> None:
    # Ten DoS rollouts must not crowd out the one delivery; that would turn a 1/12 attack rate
    # into a figure showing 0/N.
    rows = _rows(*[(i, "dos") for i in range(10)], (46, "hijack"))

    picked = representative_inits(rows, limit=3)

    assert any(r["outcome"] == "hijack" for r in picked)


def test_classes_are_ordered_least_to_most_severe() -> None:
    rows = _rows((4, "dos"), (39, "commanded"), (46, "hijack"))

    picked = representative_inits(rows, limit=3)

    assert [r["outcome"] for r in picked] == ["commanded", "dos", "hijack"]


def test_remaining_slots_are_filled_from_the_majority_class() -> None:
    rows = _rows((4, "dos"), (7, "dos"), (22, "dos"), (46, "hijack"))

    picked = representative_inits(rows, limit=3)

    assert len(picked) == 3
    assert sum(r["outcome"] == "dos" for r in picked) == 2


def test_never_returns_more_than_the_limit() -> None:
    rows = _rows(*[(i, "dos") for i in range(20)])

    assert len(representative_inits(rows, limit=4)) == 4


def test_a_sweep_with_one_class_returns_only_that_class() -> None:
    rows = _rows((4, "dos"), (7, "dos"))

    picked = representative_inits(rows, limit=3)

    assert [r["outcome"] for r in picked] == ["dos", "dos"]


def test_rejects_an_empty_sweep() -> None:
    with pytest.raises(ValueError, match="no rollouts"):
        representative_inits([], limit=3)


def test_rejects_a_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="limit"):
        representative_inits(_rows((4, "dos")), limit=0)
