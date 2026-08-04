"""Pure logic of the per-frame objective probe (§4.2 of the ε-threshold design).

The probe answers one question before any multi-day ladder spend: optimised **per frame**, the
way the closed-loop attack actually runs, does the margin objective behave sanely at a tight ε?
Every earlier objective comparison used `stealth_optimize --max-frames N`, which fits ONE patch
to N frames — the *static* problem, not this one.

Only the aggregation and spec handling are tested here; the optimisation itself is a
`@requires_gpu` seam.
"""

from __future__ import annotations

import os
import sys

import pytest

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "autoresearch/experiments/patch_attack"))

from objective_probe import ObjectiveSpec, aggregate, spec_label  # noqa: E402


def _row(label: str, forced: float, n_dims: int = 4, fully: bool = False) -> dict[str, object]:
    return {"spec": label, "forced_fraction": forced, "n_decisive": n_dims, "fully_forced": fully}


def test_label_names_the_objective_and_its_margin() -> None:
    spec = ObjectiveSpec(objective="hinge", kappa=6.0)

    assert spec_label(spec) == "hinge@k6"


def test_label_omits_margin_for_objectives_that_do_not_use_it() -> None:
    # kappa is meaningless for cross-entropy; putting it in the label would imply it was varied.
    assert spec_label(ObjectiveSpec(objective="ce")) == "ce"
    assert spec_label(ObjectiveSpec(objective="ce_decisive")) == "ce_decisive"


def test_label_records_a_nonzero_anchor() -> None:
    spec = ObjectiveSpec(objective="directional", anchor=0.1)

    assert spec_label(spec) == "directional+a0.1"


def test_aggregate_means_forcing_per_spec() -> None:
    rows = [_row("hinge@k6", 1.0), _row("hinge@k6", 0.5), _row("ce", 0.0), _row("ce", 0.5)]

    got = aggregate(rows)

    assert got["hinge@k6"]["mean_forcing"] == pytest.approx(0.75)
    assert got["ce"]["mean_forcing"] == pytest.approx(0.25)


def test_aggregate_counts_frames_and_fully_forced() -> None:
    rows = [_row("ce", 1.0, fully=True), _row("ce", 0.5), _row("ce", 1.0, fully=True)]

    got = aggregate(rows)

    assert got["ce"]["n_frames"] == 3
    assert got["ce"]["fully_forced_fraction"] == pytest.approx(2 / 3)


def test_aggregate_weights_by_decisive_dims_as_well_as_by_frame() -> None:
    """A frame with 6 contested dims is more evidence than one with 2.

    The per-frame mean treats them equally; the dim-weighted mean is what the capacity tables
    elsewhere in this project report, so both are emitted and the reader is not left guessing
    which convention a number follows.
    """
    rows = [_row("ce", 1.0, n_dims=6), _row("ce", 0.0, n_dims=2)]

    got = aggregate(rows)

    assert got["ce"]["mean_forcing"] == pytest.approx(0.5)  # per-frame
    assert got["ce"]["dim_weighted_forcing"] == pytest.approx(6 / 8)  # per-dim


def test_aggregate_is_empty_for_no_rows() -> None:
    assert aggregate([]) == {}


def test_aggregate_ignores_frames_with_no_decisive_dims() -> None:
    """Nothing to force on such a frame, so scoring it would dilute every objective equally.

    They are excluded rather than counted as 0.0 — the same convention as `forced_fraction`.
    """
    rows = [_row("ce", 1.0, n_dims=3), _row("ce", 0.0, n_dims=0)]

    got = aggregate(rows)

    assert got["ce"]["n_frames"] == 1
    assert got["ce"]["mean_forcing"] == pytest.approx(1.0)


def test_specs_are_hashable_so_they_can_key_results() -> None:
    assert len({ObjectiveSpec(objective="ce"), ObjectiveSpec(objective="ce")}) == 1


def test_kappa_default_follows_forcing_loss() -> None:
    import forcing_loss as FL

    assert ObjectiveSpec(objective="hinge").kappa == FL.DEFAULT_KAPPA
