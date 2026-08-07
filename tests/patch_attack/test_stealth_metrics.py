"""Perceptual + temporal stealth measures for an executed patch sequence.

These run on the recorded ``patch/f*.png`` crops, which are the uint8 images OpenVLA actually
consumed -- so the numbers describe the executed signal rather than a re-render of it.

Two properties are load-bearing for the paper and are therefore pinned here:

* **Churn** is a *temporal* measure. A per-frame patch that re-randomises every step is
  spatially invisible and temporally obvious, and the study claims spatial stealth only.
  Churn must therefore stay independent of how stealthy any single frame is.
* **Linf vs the carrier** is the empirical check on the epsilon bound. It is measured in uint8,
  where quantisation can push it up to half a level past the float bound, so the comparison
  carries an explicit tolerance instead of an exact `<=`.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "autoresearch/experiments/patch_attack"))

from stealth_metrics import (  # noqa: E402
    QUANT_TOLERANCE,
    bound_holds,
    churn_stats,
    linf_vs_carrier,
    measure,
)


def _frames(*values: float, size: int = 4) -> np.ndarray:
    return np.stack([np.full((size, size, 3), v, dtype=np.float32) for v in values])


def test_a_static_patch_has_zero_churn() -> None:
    stats = churn_stats(_frames(0.5, 0.5, 0.5, 0.5))

    assert stats.mean_abs_delta == pytest.approx(0.0)
    assert stats.frac_steps_above_threshold == pytest.approx(0.0)


def test_churn_is_the_mean_absolute_step_to_step_change() -> None:
    stats = churn_stats(_frames(0.0, 0.2, 0.4))

    assert stats.mean_abs_delta == pytest.approx(0.2)
    assert stats.n_transitions == 2


def test_churn_counts_the_fraction_of_steps_that_visibly_change() -> None:
    # Two transitions well above 5/255, one well below.
    stats = churn_stats(_frames(0.0, 0.3, 0.6, 0.6001))

    assert stats.frac_steps_above_threshold == pytest.approx(2 / 3)


def test_churn_ignores_how_stealthy_a_single_frame_is() -> None:
    # A sequence alternating within a narrow band around a carrier is near-invisible in any
    # single frame yet fully re-randomised every step -- which is exactly the regime the
    # measured epsilon=0.06 rollout sits in. Churn must report the second fact, not the first.
    low = churn_stats(_frames(0.50, 0.53, 0.50, 0.53))
    high = churn_stats(_frames(0.10, 0.90, 0.10, 0.90))

    assert low.mean_abs_delta < high.mean_abs_delta
    assert low.frac_steps_above_threshold == pytest.approx(1.0)


def test_churn_below_the_visibility_threshold_is_not_counted() -> None:
    # A 0.01 alternation is under 5/255 -- real motion, but nothing an observer could see.
    stats = churn_stats(_frames(0.50, 0.51, 0.50, 0.51))

    assert stats.mean_abs_delta > 0.0
    assert stats.frac_steps_above_threshold == pytest.approx(0.0)


def test_a_single_frame_sequence_has_no_transitions() -> None:
    stats = churn_stats(_frames(0.5))

    assert stats.n_transitions == 0
    assert stats.mean_abs_delta == pytest.approx(0.0)


def test_rejects_an_empty_sequence() -> None:
    with pytest.raises(ValueError, match="empty"):
        churn_stats(np.zeros((0, 4, 4, 3), dtype=np.float32))


def test_uint8_frames_are_scaled_into_zero_one() -> None:
    frames = np.stack(
        [np.full((4, 4, 3), 0, dtype=np.uint8), np.full((4, 4, 3), 255, dtype=np.uint8)]
    )

    assert churn_stats(frames).mean_abs_delta == pytest.approx(1.0)


def test_linf_vs_carrier_is_the_largest_deviation_over_the_sequence() -> None:
    carrier = np.full((4, 4, 3), 0.5, dtype=np.float32)
    frames = _frames(0.55, 0.40, 0.52)

    assert linf_vs_carrier(frames, carrier) == pytest.approx(0.10)


def test_a_patch_inside_its_budget_passes_the_bound_check() -> None:
    assert bound_holds(measured=0.0600, eps=0.06)


def test_quantisation_sized_overshoot_is_tolerated() -> None:
    # uint8 rounding alone can push the measured Linf a fraction of a level past the float
    # bound; that is not a broken attack and must not be reported as one.
    assert bound_holds(measured=0.06 + QUANT_TOLERANCE / 2, eps=0.06)


def test_a_real_overshoot_fails_the_bound_check() -> None:
    assert not bound_holds(measured=0.09, eps=0.06)


def test_carrier_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        linf_vs_carrier(_frames(0.5), np.full((8, 8, 3), 0.5, dtype=np.float32))


def test_measure_reports_both_perceptual_bases_and_the_churn_block(tmp_path) -> None:
    # The two LPIPS numbers answer different questions (is the logo intact / is the scene
    # changed) and get quoted for different claims, so both must survive into the output.
    from PIL import Image

    rec = tmp_path / "rec"
    for sub in ("patch", "policy_input", "clean_input"):
        (rec / sub).mkdir(parents=True)
    # LPIPS runs a convnet, so the fixtures use realistic patch/frame sizes rather than
    # toy ones -- the backbone cannot accept an 8px input.
    rng = np.random.default_rng(0)
    for i in range(3):
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(
            rec / "patch" / f"f{i:04d}.png"
        )
        Image.fromarray(rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)).save(
            rec / "policy_input" / f"f{i:04d}.png"
        )
        Image.fromarray(rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)).save(
            rec / "clean_input" / f"f{i:04d}.png"
        )
    carrier = tmp_path / "carrier.png"
    Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8)).save(carrier)

    stats = measure(str(rec), str(carrier))

    assert stats["n_frames"] == 3
    assert set(stats["churn"]) >= {"mean_abs_delta", "frac_steps_above_threshold"}
    assert "lpips_mean" in stats["patch_vs_carrier"]
    assert "lpips_mean" in stats["frame_vs_clean"]


def test_measure_omits_the_frame_comparison_when_clean_frames_were_not_recorded(
    tmp_path,
) -> None:
    from PIL import Image

    rec = tmp_path / "rec"
    (rec / "patch").mkdir(parents=True)
    Image.fromarray(np.full((64, 64, 3), 100, dtype=np.uint8)).save(
        rec / "patch" / "f0000.png"
    )
    carrier = tmp_path / "carrier.png"
    Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8)).save(carrier)

    stats = measure(str(rec), str(carrier))

    assert "frame_vs_clean" not in stats


# --- ball occupancy (design 2026-08-04 eps-threshold, section 4.5) --------------------
#
# The ladder's x-axis is the budget GRANTED; occupancy is how much of it was SPENT. The
# measured gap is large -- the eps=0.42 rung spends 5% -- so quoting eps alone overstates the
# perturbation actually applied. These pin the quantity that keeps that honest.


def test_a_patch_at_the_boundary_fully_occupies_its_ball() -> None:
    from stealth_metrics import ball_occupancy

    carrier = np.full((4, 4, 3), 0.5, dtype=np.float32)
    stats = ball_occupancy(_frames(0.56, 0.44), carrier, eps=0.06)

    assert stats.mean_ratio == pytest.approx(1.0, rel=1e-3)
    assert stats.frac_above_90pct == pytest.approx(1.0)
    assert stats.frac_below_10pct == pytest.approx(0.0)


def test_a_patch_that_never_moved_occupies_none_of_it() -> None:
    from stealth_metrics import ball_occupancy

    carrier = np.full((4, 4, 3), 0.5, dtype=np.float32)
    stats = ball_occupancy(_frames(0.5, 0.5), carrier, eps=0.42)

    assert stats.mean_ratio == pytest.approx(0.0)
    assert stats.frac_above_90pct == pytest.approx(0.0)
    assert stats.frac_below_10pct == pytest.approx(1.0)


def test_occupancy_is_the_deviation_as_a_fraction_of_the_budget() -> None:
    from stealth_metrics import ball_occupancy

    carrier = np.full((4, 4, 3), 0.5, dtype=np.float32)
    stats = ball_occupancy(_frames(0.53), carrier, eps=0.06)  # half the budget

    assert stats.mean_ratio == pytest.approx(0.5, rel=1e-3)
    assert stats.median_ratio == pytest.approx(0.5, rel=1e-3)
    assert stats.frac_above_50pct == pytest.approx(0.0)  # strictly above


def test_occupancy_reports_the_tail_not_only_the_mean() -> None:
    # Half the pixels pinned at the boundary and half untouched is a very different patch
    # from every pixel sitting at half budget, yet the two share a mean of 0.5.
    from stealth_metrics import ball_occupancy

    carrier = np.full((4, 4, 3), 0.5, dtype=np.float32)
    frame = np.full((1, 4, 4, 3), 0.5, dtype=np.float32)
    frame[:, :2, :, :] = 0.56  # boundary
    stats = ball_occupancy(frame, carrier, eps=0.06)

    assert stats.mean_ratio == pytest.approx(0.5, rel=1e-3)
    assert stats.frac_above_90pct == pytest.approx(0.5)
    assert stats.frac_below_10pct == pytest.approx(0.5)


def test_a_zero_budget_is_rejected_rather_than_divided_by() -> None:
    # The eps=0 pure-logo control has no ball to occupy. Its check is `linf_vs_carrier == 0`,
    # which is also what proves the carrier PNG matches the executed base bit-for-bit.
    from stealth_metrics import ball_occupancy

    carrier = np.full((4, 4, 3), 0.5, dtype=np.float32)

    with pytest.raises(ValueError, match="eps"):
        ball_occupancy(_frames(0.5), carrier, eps=0.0)


def test_occupancy_carrier_shape_mismatch_is_rejected() -> None:
    from stealth_metrics import ball_occupancy

    with pytest.raises(ValueError, match="shape"):
        ball_occupancy(_frames(0.5), np.full((8, 8, 3), 0.5, dtype=np.float32), eps=0.06)


def test_measure_includes_occupancy_only_when_a_budget_is_supplied(tmp_path) -> None:
    from PIL import Image

    rec = tmp_path / "rec"
    (rec / "patch").mkdir(parents=True)
    Image.fromarray(np.full((64, 64, 3), 130, dtype=np.uint8)).save(rec / "patch" / "f0000.png")
    carrier = tmp_path / "carrier.png"
    Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8)).save(carrier)

    assert "ball_occupancy" not in measure(str(rec), str(carrier))

    stats = measure(str(rec), str(carrier), eps=0.06)
    assert stats["ball_occupancy"]["eps"] == pytest.approx(0.06)
    assert stats["ball_occupancy"]["mean_ratio"] > 0.0
