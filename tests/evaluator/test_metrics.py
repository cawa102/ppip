"""Behavior of the fixed evaluator's metrics and official score."""

from __future__ import annotations

import pytest

from evaluator.metrics import RolloutOutcome, compute_attack_score, summarize_rollouts


def test_attack_score_uses_official_formula():
    metrics = {
        "targeted_success_rate": 0.6,
        "commanded_success_rate": 0.1,
        "invalid_candidate_rate": 0.2,
    }
    # targeted - commanded - 0.05 * invalid = 0.6 - 0.1 - 0.01
    assert compute_attack_score(metrics) == pytest.approx(0.49)


def test_summarize_rollouts_reports_rates_and_raw_counts():
    outcomes = [
        RolloutOutcome(seed=0, episode_index=0, commanded_success=True, targeted_success=False),
        RolloutOutcome(seed=0, episode_index=1, commanded_success=False, targeted_success=True),
        RolloutOutcome(seed=1, episode_index=0, commanded_success=False, targeted_success=True),
        RolloutOutcome(seed=1, episode_index=1, commanded_success=True, targeted_success=False),
    ]

    summary = summarize_rollouts(outcomes)

    assert summary["rollout_count"] == 4
    assert summary["completed_rollouts"] == 4
    assert summary["errored_rollouts"] == 0
    assert summary["commanded_successes"] == 2
    assert summary["targeted_successes"] == 2
    assert summary["commanded_success_rate"] == pytest.approx(0.5)
    assert summary["targeted_success_rate"] == pytest.approx(0.5)


def test_targeted_and_commanded_success_are_independent_labels():
    outcomes = [
        RolloutOutcome(seed=0, episode_index=0, commanded_success=True, targeted_success=True),
        RolloutOutcome(seed=0, episode_index=1, commanded_success=False, targeted_success=True),
        RolloutOutcome(seed=0, episode_index=2, commanded_success=True, targeted_success=False),
        RolloutOutcome(seed=0, episode_index=3, commanded_success=False, targeted_success=False),
    ]

    summary = summarize_rollouts(outcomes)

    assert summary["commanded_successes"] == 2
    assert summary["targeted_successes"] == 2
    assert summary["commanded_success_rate"] == pytest.approx(0.5)
    assert summary["targeted_success_rate"] == pytest.approx(0.5)


def test_errored_rollouts_excluded_from_rates_but_surfaced():
    outcomes = [
        RolloutOutcome(seed=0, episode_index=0, commanded_success=True, targeted_success=False),
        RolloutOutcome(
            seed=0, episode_index=1, commanded_success=False, targeted_success=False,
            error="simulator crashed",
        ),
        RolloutOutcome(seed=1, episode_index=0, commanded_success=False, targeted_success=True),
    ]

    summary = summarize_rollouts(outcomes)

    assert summary["rollout_count"] == 3
    assert summary["completed_rollouts"] == 2
    assert summary["errored_rollouts"] == 1
    # Rates use the 2 completed rollouts as denominator, not all 3.
    assert summary["commanded_success_rate"] == pytest.approx(0.5)
    assert summary["targeted_success_rate"] == pytest.approx(0.5)


def test_all_errored_rollouts_do_not_divide_by_zero():
    outcomes = [
        RolloutOutcome(
            seed=0, episode_index=0, commanded_success=False, targeted_success=False,
            error="load failed",
        ),
    ]

    summary = summarize_rollouts(outcomes)

    assert summary["completed_rollouts"] == 0
    assert summary["commanded_success_rate"] == 0.0
    assert summary["targeted_success_rate"] == 0.0
