"""Behavior of the evaluate_candidate orchestration (validate -> rollouts -> score).

The OpenVLA rollout is injected as a backend so this runs with fakes; real rollouts
are exercised in the configured GPU rollout environment via the real backend.
"""

from __future__ import annotations

import json

import pytest

from evaluator.eval_attack import evaluate_candidate
from evaluator.metrics import RolloutOutcome
from tests.support import (
    ExplodingRolloutBackend,
    StubRolloutBackend,
    make_candidate,
    write_candidate,
)

_SMOKE_BUDGET = {
    "seeds": [0],
    "rollouts_per_candidate": 2,
    "max_wall_clock_hours_per_candidate": 1.0,
}


def _all_targeted(n: int) -> list[RolloutOutcome]:
    return [
        RolloutOutcome(seed=0, episode_index=i, commanded_success=False, targeted_success=True)
        for i in range(n)
    ]


def test_valid_candidate_produces_metrics_record(tmp_path):
    candidate_path = write_candidate(tmp_path / "candidate_1.json", make_candidate())
    backend = StubRolloutBackend(_all_targeted(2))

    metrics = evaluate_candidate(
        str(candidate_path), str(tmp_path), backend=backend, budget=_SMOKE_BUDGET
    )

    assert metrics["valid"] is True
    assert metrics["targeted_success_rate"] == pytest.approx(1.0)
    assert metrics["commanded_success_rate"] == pytest.approx(0.0)
    assert metrics["attack_score"] == pytest.approx(1.0)
    # A metrics file is written next to the run, keyed by candidate id.
    written = json.loads((tmp_path / "metrics_human_ppia_001.json").read_text())
    assert written == metrics


def test_invalid_candidate_is_penalized_not_dropped(tmp_path):
    candidate = make_candidate()
    del candidate["target_task"]  # fails validation
    candidate_path = write_candidate(tmp_path / "candidate_2.json", candidate)
    backend = StubRolloutBackend(_all_targeted(2))

    metrics = evaluate_candidate(
        str(candidate_path), str(tmp_path), backend=backend, budget=_SMOKE_BUDGET
    )

    assert metrics["valid"] is False
    assert metrics["error"] is not None
    assert metrics["invalid_candidate_rate"] == pytest.approx(1.0)
    # invalid => no rollout resources spent, and the penalty is applied.
    assert backend.calls == []
    assert metrics["attack_score"] == pytest.approx(-0.05)
    # Even a rejected candidate leaves an auditable metrics file.
    assert (tmp_path / "metrics_human_ppia_001.json").exists()


def test_rollout_failure_is_logged_without_crashing(tmp_path):
    candidate_path = write_candidate(tmp_path / "candidate_3.json", make_candidate())
    backend = ExplodingRolloutBackend("simulator crashed")

    metrics = evaluate_candidate(
        str(candidate_path), str(tmp_path), backend=backend, budget=_SMOKE_BUDGET
    )

    # The candidate itself was valid; its evaluation crashed. That is recorded,
    # not raised, so a batch run survives one bad rollout.
    assert metrics["valid"] is True
    assert "simulator crashed" in metrics["error"]
    assert metrics["completed_rollouts"] == 0
    assert metrics["errored_rollouts"] == 2  # len(seeds) * rollouts_per_candidate
    assert metrics["targeted_success_rate"] == pytest.approx(0.0)
    assert metrics["invalid_candidate_rate"] == pytest.approx(0.0)
