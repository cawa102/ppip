"""Horizon and init bookkeeping for the artifact-level replication panel.

The artifact panel re-runs the armed per-frame episode at an init to obtain a deployable video,
then replays that video under both word conditions. Two things must hold before any GPU time is
spent: the horizon has to clear BOTH Stage-C events (a short horizon renders a successful benign
rollout as a denial — it would libel the dormant condition), and the init has to be one where a
hijack and a benign success both actually exist, or the panel measures nothing.
"""

from __future__ import annotations

import pytest
from render_word_gate_figure import MARGIN, STAGE_C_EVENTS, horizon_for

#: Stage C: armed latched AND dormant completed. Inits 4/22 have no hijack to reproduce; 39/45
#: never completed the user's task within the reported horizon, so their benign leg is confounded.
CLEAN_CANDIDATES = (7, 24, 26, 33, 36, 38, 46, 49)


@pytest.mark.parametrize("init", CLEAN_CANDIDATES)
def test_every_clean_candidate_has_a_recorded_stage_c_outcome(init: int) -> None:
    assert init in STAGE_C_EVENTS, f"init {init} has no Stage-C events; horizon_for would raise"


@pytest.mark.parametrize("init", CLEAN_CANDIDATES)
def test_horizon_clears_both_stage_c_events_with_margin(init: int) -> None:
    latch, commanded = STAGE_C_EVENTS[init]

    horizon = horizon_for(init)

    assert horizon >= latch + MARGIN
    assert horizon >= commanded + MARGIN


def test_confounded_inits_are_not_in_the_candidate_set() -> None:
    """4/22 have no armed hijack; 39/45 no dormant completion. Running them measures nothing."""
    assert not set(CLEAN_CANDIDATES) & {4, 22, 39, 45}


def test_unknown_init_is_rejected_rather_than_given_a_guessed_horizon() -> None:
    with pytest.raises(ValueError, match="no recorded Stage-C outcome"):
        horizon_for(999)
