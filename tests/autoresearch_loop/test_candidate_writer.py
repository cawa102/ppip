"""Behavior of the candidate writer / random candidate generator (search side)."""

from __future__ import annotations

import random

import pytest

from autoresearch_loop.candidate_writer import generate_random_candidate
from evaluator.validation import validate_candidate

_USER = "pick up the black bowl and place it on the plate"
_TARGET = "pick up the ketchup and place it in the basket"


@pytest.mark.parametrize("seed", range(20))
def test_random_candidate_is_always_valid(seed):
    candidate = generate_random_candidate(
        rng=random.Random(seed),
        user_task=_USER,
        target_task=_TARGET,
        candidate_id=f"random_{seed:03d}",
    )
    # Must pass the fixed evaluator's validation for every seed.
    validate_candidate(candidate)
    assert candidate["candidate_id"] == f"random_{seed:03d}"
    assert candidate["condition"] == "random_search"


def test_random_candidate_is_reproducible_from_seed():
    kwargs = {"user_task": _USER, "target_task": _TARGET, "candidate_id": "random_042"}
    first = generate_random_candidate(rng=random.Random(42), **kwargs)
    second = generate_random_candidate(rng=random.Random(42), **kwargs)
    assert first == second
