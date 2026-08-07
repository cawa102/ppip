"""Path derivation for per-rung GIFs.

A rung's result JSON and its recorded frames are written under two different naming schemes by
``corner_attack.py`` (``result_corner_BL_64_seed0_<suffix>_trial0.json`` beside
``rec_BL_64_<suffix>/policy_input``). Deriving both from one rung spec keeps a figure from
being built out of one rung's frames and another rung's verdict -- which would look completely
normal and be completely wrong.
"""

from __future__ import annotations

import os
import sys

import pytest

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "autoresearch/experiments/patch_attack"))

from make_rung_gif import Rung, rung_paths  # noqa: E402


def test_derives_both_paths_from_one_spec() -> None:
    rung = Rung(run_dir="/runs/ladder", corner="BL", size=64, seed=0, suffix="_eps025_hinge")

    result, frames = rung_paths(rung)

    assert result == "/runs/ladder/result_corner_BL_64_seed0_eps025_hinge_trial0.json"
    assert frames == "/runs/ladder/rec_BL_64_eps025_hinge/policy_input"


def test_an_unsuffixed_rung_still_resolves() -> None:
    rung = Rung(run_dir="/runs/c", corner="TR", size=95, seed=0, suffix="")

    result, frames = rung_paths(rung)

    assert result.endswith("result_corner_TR_95_seed0_trial0.json")
    assert frames.endswith("rec_TR_95/policy_input")


def test_trial_index_reaches_both_names_correctly() -> None:
    # The trial index appears in the result filename but never in the recording dir.
    rung = Rung(run_dir="/r", corner="BL", size=64, seed=0, suffix="_x", trial=1)

    result, frames = rung_paths(rung)

    assert result.endswith("_trial1.json")
    assert "trial" not in os.path.basename(os.path.dirname(frames))


def test_seed_reaches_the_result_name_but_not_the_recording_dir() -> None:
    rung = Rung(run_dir="/r", corner="BL", size=64, seed=7, suffix="_x")

    result, frames = rung_paths(rung)

    assert "seed7" in result
    assert "seed" not in frames


@pytest.mark.parametrize("size", [0, -1])
def test_rejects_a_non_positive_patch_size(size: int) -> None:
    with pytest.raises(ValueError, match="size"):
        Rung(run_dir="/r", corner="BL", size=size, seed=0, suffix="")


def test_rejects_an_unknown_corner() -> None:
    with pytest.raises(ValueError, match="corner"):
        Rung(run_dir="/r", corner="MIDDLE", size=64, seed=0, suffix="")
