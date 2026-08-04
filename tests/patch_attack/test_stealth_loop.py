"""Loop plumbing: the three-way init split must hold, candidates must be resumable, and the
gate's verdict must never be confused with a score."""

from __future__ import annotations

import json

import pytest
import shared_inits
import stealth_loop
from stealth_optimize import MIN_DECISIVE_DIMS, Frame, parse_rect

from autoresearch_loop.ledger import LedgerError, append_ledger_row

# --- the separation the whole design rests on ------------------------------------------


def test_the_three_stages_use_three_disjoint_init_sets() -> None:
    optimize = set(shared_inits.OPTIMIZE_INITS)
    gate = set(shared_inits.GATE_INITS)
    score = set(shared_inits.HELDOUT_INITS)

    assert not optimize & gate, "the gate would measure fit, not transfer"
    assert not optimize & score, "the optimizer would have seen the evaluation set"
    assert not gate & score, "candidate selection would be informed by the evaluation set"


def test_the_loop_scores_only_on_heldout_inits() -> None:
    # A change that pointed evaluation at the training half would silently inflate every rate.
    source = (stealth_loop.__file__).replace(".pyc", ".py")
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    assert "init_indices=shared_inits.HELDOUT_INITS" in text


# --- candidate identity and resume ------------------------------------------------------


def test_candidate_ids_distinguish_every_axis_the_loop_varies() -> None:
    ids = {
        stealth_loop.candidate_id("aurora", 1.0, 0.0, "BL:64", 0),
        stealth_loop.candidate_id("vertex", 1.0, 0.0, "BL:64", 0),
        stealth_loop.candidate_id("aurora", 0.08, 0.0, "BL:64", 0),
        stealth_loop.candidate_id("aurora", 1.0, 0.5, "BL:64", 0),
        stealth_loop.candidate_id("aurora", 1.0, 0.0, "TR:80", 0),
        stealth_loop.candidate_id("aurora", 1.0, 0.0, "BL:64", 1),
    }
    assert len(ids) == 6


def test_candidate_id_has_no_colon_so_it_is_filesystem_safe() -> None:
    assert ":" not in stealth_loop.candidate_id("aurora", 1.0, 0.0, "BL:64", 0)


def test_completed_candidates_reads_back_what_the_ledger_recorded(tmp_path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    assert stealth_loop.completed_candidates(str(ledger)) == set()

    append_ledger_row(ledger, {"candidate_id": "aurora_eps1_tv0_BL64_s0", "eps": 1.0})

    assert stealth_loop.completed_candidates(str(ledger)) == {"aurora_eps1_tv0_BL64_s0"}


def test_a_recorded_candidate_cannot_be_overwritten(tmp_path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    append_ledger_row(ledger, {"candidate_id": "dup", "eps": 1.0})

    # Results are immutable once written -- a rerun must skip, not silently replace.
    with pytest.raises(LedgerError):
        append_ledger_row(ledger, {"candidate_id": "dup", "eps": 0.5})


def test_ledger_rows_survive_a_json_round_trip(tmp_path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    row = {
        "candidate_id": "aurora_eps1_tv0_BL64_s0",
        "eps": 1.0,
        "rect": [160, 0, 64, 64],
        "gate_passed": True,
        "attack_score": 0.25,
    }
    append_ledger_row(ledger, row)

    assert json.loads(ledger.read_text().strip()) == row


# --- the eps ladder ---------------------------------------------------------------------


def test_the_ladder_starts_at_the_capacity_ceiling_and_ends_at_the_pure_logo_control() -> None:
    # eps=1 first: if free-range static cannot hijack, no stealth budget will, and the
    # finding is about staticness rather than stealth. eps=0 is the pure-logo control.
    assert stealth_loop.DEFAULT_LADDER[0] == 1.0
    assert stealth_loop.DEFAULT_LADDER[-1] == 0.0


def test_the_ladder_descends_monotonically() -> None:
    ladder = stealth_loop.DEFAULT_LADDER
    assert all(a > b for a, b in zip(ladder[:-1], ladder[1:], strict=True))


# --- rect parsing -----------------------------------------------------------------------


def test_rect_spec_resolves_to_the_named_corner() -> None:
    assert parse_rect("BL:64") == (160, 0, 64, 64)
    assert parse_rect("TR:80") == (0, 144, 80, 80)
    assert parse_rect("bl:32") == (192, 0, 32, 32)


def test_the_default_rect_is_the_measured_non_occluding_one() -> None:
    # BL:80 clips the basket on 5/21 inits (occlusion_probe, 2026-07-28); BL:64 is clear.
    parser_default = "BL:64"
    assert parse_rect(parser_default) == (160, 0, 64, 64)


# --- decisive-frame selection -----------------------------------------------------------


def _frame(decisive_dims: tuple[int, ...]) -> Frame:
    import numpy as np

    return Frame(
        init=1, step=0, image=np.zeros((224, 224, 3), dtype=np.uint8),
        teacher=(0,) * 7, clean_user=(0,) * 7, decisive_dims=decisive_dims,
    )


def test_a_frame_is_decisive_only_when_the_instructions_disagree_enough() -> None:
    assert not _frame(()).is_decisive
    assert not _frame((3,)).is_decisive  # one dim: below MIN_DECISIVE_DIMS
    assert _frame((3, 5)).is_decisive
    assert MIN_DECISIVE_DIMS == 2
