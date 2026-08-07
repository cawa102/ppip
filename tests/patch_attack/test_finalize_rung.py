"""Per-rung verification and the ladder summary row.

Two guards live here because both failure modes are silent:

* **Objective provenance.** `corner_attack.py` reaches the closed-loop core directly and takes
  the objective from an env var, bypassing the module-name guarantee that
  `hinge_monitor_patch_attack` provides. A rung optimised under the wrong objective looks
  exactly like a correct one in the output. So the recorded objective is checked against the
  one the ladder is supposed to be running before the rung is admitted.
* **Bound provenance.** A rung whose executed patch exceeded its own epsilon is not a stealth
  result at that epsilon, and must not be admitted to the ladder as one.
"""

from __future__ import annotations

import os
import sys

import pytest

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "autoresearch/experiments/patch_attack"))

from finalize_rung import (  # noqa: E402
    rung_gain,
    rung_label,
    summary_row,
    verify_objective,
)


def _result(**over: object) -> dict:
    base = {
        "targeted": False,
        "commanded_success": False,
        "objective": {"name": "hinge", "kappa": 6.0},
        "stealth": {"eps": 0.25, "linf_measured_max": 0.25, "bound_holds": True},
        "mean_decisive_forcing": 0.81,
        "latch_step": None,
        "min_target_dist_m": 0.35,
        "area_frac": 0.0816,
    }
    base.update(over)
    return base


def test_a_rung_run_under_the_expected_objective_passes() -> None:
    verify_objective(_result(), expected="hinge")


def test_a_rung_run_under_a_different_objective_is_rejected() -> None:
    wrong = _result(objective={"name": "ce"})

    with pytest.raises(ValueError, match="ce"):
        verify_objective(wrong, expected="hinge")


def test_a_rung_with_no_recorded_objective_is_rejected() -> None:
    # Pre-dispatch runs recorded no objective. They are valid history but cannot be admitted
    # to a ladder that claims a single objective throughout.
    stale = _result()
    del stale["objective"]

    with pytest.raises(ValueError, match="no recorded objective"):
        verify_objective(stale, expected="hinge")


def test_summary_row_carries_the_verdict_class_and_the_headline_numbers() -> None:
    row = summary_row(
        _result(targeted=True, latch_step=131),
        stealth={
            "churn": {"mean_abs_delta": 0.038, "frac_steps_above_threshold": 0.92},
            "patch_vs_carrier": {"lpips_mean": 0.025},
            "linf_vs_carrier": 0.2510,
        },
    )

    assert row["outcome"] == "hijack"
    assert row["eps"] == 0.25
    assert row["latch_step"] == 131
    assert row["mean_decisive_forcing"] == 0.81
    assert row["lpips_patch_vs_carrier"] == 0.025
    assert row["churn_mean_abs_delta"] == 0.038


def test_summary_row_flags_a_rung_that_broke_its_own_bound() -> None:
    row = summary_row(
        _result(stealth={"eps": 0.25, "linf_measured_max": 0.40, "bound_holds": False}),
        stealth={
            "churn": {"mean_abs_delta": 0.0, "frac_steps_above_threshold": 0.0},
            "patch_vs_carrier": {"lpips_mean": 0.0},
            "linf_vs_carrier": 0.40,
        },
    )

    assert row["bound_holds"] is False


def test_free_range_rung_has_no_epsilon_but_still_summarises() -> None:
    free = _result(targeted=True)
    del free["stealth"]

    row = summary_row(
        free,
        stealth={
            "churn": {"mean_abs_delta": 0.5, "frac_steps_above_threshold": 1.0},
            "patch_vs_carrier": {"lpips_mean": 0.3},
            "linf_vs_carrier": 0.9,
        },
    )

    assert row["eps"] is None
    assert row["outcome"] == "hijack"


def test_bounded_rung_is_labelled_by_its_epsilon() -> None:
    assert rung_label(0.09) == "eps = 0.09"


def test_free_range_rung_is_labelled_unbounded_not_eps_none() -> None:
    # The ceiling has no budget at all; printing "eps = None" on a figure would read as a bug.
    assert "unbounded" in rung_label(None).lower()
    assert "none" not in rung_label(None).lower()


def test_free_range_difference_gain_is_unity() -> None:
    # There is no budget to normalise against, and amplifying an unbounded perturbation would
    # saturate the difference panel everywhere.
    assert rung_gain(None) == 1


def test_bounded_difference_gain_scales_with_the_budget() -> None:
    assert rung_gain(0.03) > rung_gain(0.25) >= 1
