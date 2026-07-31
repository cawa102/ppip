"""The init-state precommit is a claim about scientific procedure, so it is tested like one."""

from __future__ import annotations

import pytest
import shared_inits


def test_precommit_literals_match_their_recorded_derivation() -> None:
    # Arrange / Act / Assert -- verify_precommit raises if any invariant broke.
    shared_inits.verify_precommit()


def test_train_and_heldout_are_disjoint() -> None:
    assert not set(shared_inits.TRAIN_INITS) & set(shared_inits.HELDOUT_INITS)


def test_contaminated_legacy_init_is_excluded_from_both_splits() -> None:
    # Every pre-2026-07-28 corner result was tuned on init 0; including it would leak
    # selection bias into a reported rate.
    assert shared_inits.LEGACY_GATE_INIT not in shared_inits.SHARED_INITS


def test_heldout_meets_the_programs_minimum_of_ten_inits() -> None:
    assert len(shared_inits.HELDOUT_INITS) >= 10


def test_every_index_is_a_valid_libero_init_selector() -> None:
    assert all(0 <= i < shared_inits.N_LIBERO_INIT_STATES for i in shared_inits.SHARED_INITS)


@pytest.mark.parametrize("name", ["TRAIN_INITS", "HELDOUT_INITS", "SHARED_INITS"])
def test_splits_are_immutable_tuples(name: str) -> None:
    assert isinstance(getattr(shared_inits, name), tuple)


def test_summary_names_both_splits_for_the_run_log() -> None:
    text = shared_inits.summary()
    assert str(shared_inits.PRECOMMIT_SEED) in text
    assert "train=" in text and "heldout=" in text
