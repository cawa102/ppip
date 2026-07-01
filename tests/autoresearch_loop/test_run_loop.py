"""Behavior of the search loop: budget stopping and ledger-based resume."""

from __future__ import annotations

import random

from autoresearch_loop.candidate_writer import generate_random_candidate
from autoresearch_loop.ledger import read_ledger
from autoresearch_loop.run_loop import run_search_condition
from evaluator.metrics import RolloutOutcome
from tests.support import StubRolloutBackend

_BUDGET = {
    "max_candidates_per_condition": 3,
    "seeds": [0],
    "rollouts_per_candidate": 2,
    "max_wall_clock_hours_per_candidate": 1.0,
}


def _make_propose():
    def propose(index: int) -> dict:
        return generate_random_candidate(
            rng=random.Random(index),
            user_task="put the bowl on the plate",
            target_task="put the ketchup in the basket",
            candidate_id=f"random_{index:03d}",
        )

    return propose


def _backend():
    return StubRolloutBackend(
        [
            RolloutOutcome(seed=0, episode_index=0, commanded_success=False, targeted_success=True),
            RolloutOutcome(seed=0, episode_index=1, commanded_success=True, targeted_success=False),
        ]
    )


def test_loop_stops_at_candidate_budget(tmp_path):
    results = run_search_condition(
        budget=_BUDGET, run_dir=str(tmp_path), backend=_backend(), propose=_make_propose()
    )

    assert len(results) == 3
    rows = read_ledger(tmp_path / "ledger.jsonl")
    assert len(rows) == 3
    # Every recorded candidate has a unique id.
    ids = [r["candidate_id"] for r in rows]
    assert len(set(ids)) == 3


def test_loop_resumes_from_ledger_after_stop(tmp_path):
    small = {**_BUDGET, "max_candidates_per_condition": 1}

    first = run_search_condition(
        budget=small, run_dir=str(tmp_path), backend=_backend(), propose=_make_propose()
    )
    assert len(first) == 1

    # Resume with a larger budget in the same run dir: only the remaining two run.
    second = run_search_condition(
        budget=_BUDGET, run_dir=str(tmp_path), backend=_backend(), propose=_make_propose()
    )
    assert len(second) == 2

    rows = read_ledger(tmp_path / "ledger.jsonl")
    assert len(rows) == 3
    assert len({r["candidate_id"] for r in rows}) == 3


def test_loop_at_or_over_budget_generates_nothing(tmp_path):
    run_search_condition(
        budget=_BUDGET, run_dir=str(tmp_path), backend=_backend(), propose=_make_propose()
    )
    # Re-running an already-complete condition is a no-op (idempotent resume).
    again = run_search_condition(
        budget=_BUDGET, run_dir=str(tmp_path), backend=_backend(), propose=_make_propose()
    )
    assert again == []
    assert len(read_ledger(tmp_path / "ledger.jsonl")) == 3
