"""Discovering ladder rungs from a run directory.

The ladder figure is rebuilt every time a rung lands, so it discovers rungs from the result
JSONs on disk rather than from a hand-maintained list -- a list would silently omit a rung that
finished after the last edit. Discovery parses the tag convention `corner_attack.py` writes,
and rungs are ordered by their epsilon so the figure reads left to right as increasing
perturbation budget.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "autoresearch/experiments/patch_attack"))

from make_ladder_gif import discover_rungs, parse_result_name  # noqa: E402


def test_parses_the_tag_convention_written_by_corner_attack() -> None:
    parsed = parse_result_name("result_corner_BL_64_seed0_eps025_hinge_trial0.json")

    assert parsed is not None
    assert parsed.corner == "BL"
    assert parsed.size == 64
    assert parsed.seed == 0
    assert parsed.suffix == "_eps025_hinge"
    assert parsed.trial == 0


def test_parses_a_result_with_no_tag_suffix() -> None:
    parsed = parse_result_name("result_corner_TR_95_seed0_trial0.json")

    assert parsed is not None
    assert parsed.suffix == ""


def test_ignores_files_that_are_not_result_jsons() -> None:
    for name in ("trace_corner_BL_64_seed0_x_trial0.json", "notes.txt", "corner_summary.json"):
        assert parse_result_name(name) is None


def _write_rung(directory: str, suffix: str, eps: float | None) -> None:
    result = {"targeted": False, "commanded_success": True}
    if eps is not None:
        result["stealth"] = {"eps": eps, "linf_measured_max": eps, "bound_holds": True}
    name = f"result_corner_BL_64_seed0{suffix}_trial0.json"
    with open(os.path.join(directory, name), "w") as fh:
        json.dump(result, fh)
    os.makedirs(os.path.join(directory, f"rec_BL_64{suffix}", "policy_input"), exist_ok=True)


def test_rungs_come_back_ordered_by_increasing_epsilon(tmp_path) -> None:
    for suffix, eps in (("_c", 0.25), ("_a", 0.06), ("_b", 0.12)):
        _write_rung(str(tmp_path), suffix, eps)

    rungs = discover_rungs(str(tmp_path))

    assert [r.eps for r in rungs] == [0.06, 0.12, 0.25]


def test_a_free_range_rung_sorts_last_as_the_ceiling(tmp_path) -> None:
    # The unbounded run has no stealth block; it is the top of the ladder, not the bottom.
    _write_rung(str(tmp_path), "_free", None)
    _write_rung(str(tmp_path), "_a", 0.25)

    rungs = discover_rungs(str(tmp_path))

    assert rungs[-1].eps is None
    assert rungs[0].eps == 0.25


def test_a_rung_whose_frames_were_never_recorded_is_skipped(tmp_path) -> None:
    # Without frames there is nothing to animate; including it would crash the render at the
    # end of a multi-hour figure build rather than at discovery.
    _write_rung(str(tmp_path), "_ok", 0.25)
    with open(os.path.join(tmp_path, "result_corner_BL_64_seed0_noframes_trial0.json"), "w") as f:
        json.dump({"targeted": False, "commanded_success": True}, f)

    rungs = discover_rungs(str(tmp_path))

    assert [r.suffix for r in rungs] == ["_ok"]


def test_an_empty_directory_yields_no_rungs(tmp_path) -> None:
    assert discover_rungs(str(tmp_path)) == []


def test_missing_directory_is_an_error_not_an_empty_ladder(tmp_path) -> None:
    with pytest.raises(SystemExit, match="no such"):
        discover_rungs(str(tmp_path / "does-not-exist"))
