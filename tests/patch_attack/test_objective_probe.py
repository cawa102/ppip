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

from objective_probe import (  # noqa: E402
    DEFAULT_SPECS,
    ObjectiveSpec,
    aggregate,
    spec_label,
)


def _row(label: str, forced: float, n_dims: int = 4, fully: bool = False) -> dict[str, object]:
    return {"spec": label, "forced_fraction": forced, "n_decisive": n_dims, "fully_forced": fully}


def test_label_names_the_objective_and_its_margin() -> None:
    spec = ObjectiveSpec(objective="hinge", kappa=6.0)

    assert spec_label(spec) == "hinge@k6"


def test_label_omits_margin_for_objectives_that_do_not_use_it() -> None:
    # kappa is meaningless for cross-entropy; putting it in the label would imply it was varied.
    assert spec_label(ObjectiveSpec(objective="ce")) == "ce"
    assert spec_label(ObjectiveSpec(objective="ce_decisive")) == "ce_decisive"


def test_label_carries_the_margin_for_saturating_cross_entropy() -> None:
    """kappa *is* meaningful for `ce_saturating` — it decides when a dim is released.

    Omitting it (as for `ce`/`ce_decisive`) would let two runs at different margins collide
    under one label, which is the failure the labels exist to prevent.
    """
    assert spec_label(ObjectiveSpec(objective="ce_saturating", kappa=6.0)) == "ce_saturating@k6"


def test_the_probe_covers_all_four_cells_of_the_shape_by_saturation_grid() -> None:
    """Dropping any cell collapses the probe back to a confounded CE-vs-hinge comparison.

    `ce_decisive` -> `ce_saturating` isolates saturation at fixed shape; `ce_saturating` ->
    `hinge@k6` isolates shape at fixed saturation. Both saturating cells must sit at the SAME
    kappa or the second comparison varies the release threshold too.
    """
    labels = [spec_label(spec) for spec in DEFAULT_SPECS]

    assert "ce" in labels  # the reference every published result was produced with
    assert "ce_decisive" in labels
    assert "ce_saturating@k6" in labels
    assert "hinge@k6" in labels


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


# --- the distortion penalty spec (design section 4.5) ---------------------------------


def test_label_carries_the_distortion_weight() -> None:
    # lambda changes the gradient, so two runs at different weights must not collide.
    spec = ObjectiveSpec(objective="ce", distortion_weight=0.5)

    assert spec_label(spec) == "ce+m0.5"


def test_label_omits_the_distortion_weight_when_it_is_off() -> None:
    assert spec_label(ObjectiveSpec(objective="ce")) == "ce"


def test_distortion_specs_are_not_in_the_default_set() -> None:
    """lambda is unswept, and an unswept knob in the default set is the kappa=3 failure again.

    They ride the probe explicitly (`--with-mse`), so the default probe's cost and meaning are
    unchanged and a lambda value can never enter a comparison without someone choosing it.
    """
    from objective_probe import MSE_SPECS

    assert all(spec.distortion_weight == 0.0 for spec in DEFAULT_SPECS)
    assert len(MSE_SPECS) >= 2  # a sweep, not a guess
    assert {spec.distortion_weight for spec in MSE_SPECS} == {
        spec.distortion_weight for spec in MSE_SPECS
    } - {0.0}


def test_distortion_specs_hold_the_action_objective_fixed() -> None:
    """The lambda sweep varies ONE thing. Varying the action loss too would confound it."""
    from objective_probe import MSE_SPECS

    assert len({spec.objective for spec in MSE_SPECS}) == 1


# --- stratified frame sampling --------------------------------------------------------


class _Frame:
    def __init__(self, init: int, step: int) -> None:
        self.init, self.step = init, step

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return f"F({self.init},{self.step})"


def test_stratified_sampling_spreads_across_inits() -> None:
    """The 2026-08-04 probe took the first 8 frames — all init 1, steps 0-40.

    Consecutive steps of one episode are highly correlated, which is why that pass could clear
    pathology but could not rank. Sampling must cover the inits instead.
    """
    from objective_probe import stratified_sample

    frames = [_Frame(init, step) for init in (0, 1, 2, 3) for step in range(20)]

    chosen = stratified_sample(frames, 8)

    assert len(chosen) == 8
    assert {f.init for f in chosen} == {0, 1, 2, 3}


def test_stratified_sampling_spreads_within_an_init_too() -> None:
    # Covering four inits by taking steps 0-1 of each would still be four episode openings.
    from objective_probe import stratified_sample

    frames = [_Frame(init, step) for init in (0, 1) for step in range(100)]

    chosen = stratified_sample(frames, 8)
    steps = sorted(f.step for f in chosen if f.init == 0)

    assert max(steps) - min(steps) > 40


def test_stratified_sampling_returns_everything_when_asked_for_more_than_exists() -> None:
    from objective_probe import stratified_sample

    frames = [_Frame(0, step) for step in range(3)]

    assert len(stratified_sample(frames, 8)) == 3


def test_stratified_sampling_is_deterministic() -> None:
    """A probe that resampled between specs would compare objectives on different frames."""
    from objective_probe import stratified_sample

    frames = [_Frame(init, step) for init in (0, 1, 2) for step in range(30)]

    first = [(f.init, f.step) for f in stratified_sample(frames, 9)]
    second = [(f.init, f.step) for f in stratified_sample(frames, 9)]

    assert first == second
