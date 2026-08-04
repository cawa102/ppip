"""The policy crops 0.9 of the frame's *area* before the ViT sees it, so a patch flush to the
frame edge is partly thrown away before it can influence anything.

These tests pin that geometry against the real preprocessing function (`vla_diff`), not a
re-derivation of it — the whole point is that the reported area of a patch and the area the
model actually consumes are two different numbers.
"""

from __future__ import annotations

import pytest
import torch
from crop_geometry import (
    CROP_AREA,
    POLICY_SIDE,
    crop_safe_bounds,
    crop_window,
    input_area_fraction,
    nominal_area_fraction,
    rect_mask,
    retained_fraction,
    shift_rect_into_crop,
)


def _corner(corner: str, size: int) -> tuple[int, int, int, int]:
    n = POLICY_SIDE
    r0, c0 = {"TL": (0, 0), "TR": (0, n - size), "BL": (n - size, 0)}[corner]
    return r0, c0, size, size


# --- the window itself ------------------------------------------------------------------


def test_the_sampled_window_leaves_a_border_the_model_never_sees() -> None:
    lo, hi = crop_window()

    assert lo == pytest.approx(5.72, abs=0.01)
    assert hi == pytest.approx(217.28, abs=0.01)
    assert crop_safe_bounds() == (6, 217)


def test_the_window_is_symmetric_about_the_frame_centre() -> None:
    lo, hi = crop_window()
    assert lo + hi == pytest.approx(POLICY_SIDE - 1)


# --- nominal vs consumed area -----------------------------------------------------------


def test_a_rect_fully_inside_the_window_is_magnified_by_exactly_the_crop_factor() -> None:
    # Cropping 0.9 of the area and resizing back to 224 magnifies everything inside by 1/0.9.
    rect = (60, 60, 40, 40)

    assert input_area_fraction(rect) == pytest.approx(
        nominal_area_fraction(rect) / CROP_AREA, rel=1e-3
    )
    assert retained_fraction(rect) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize(
    ("corner", "size", "expected_retained"),
    [("BL", 64, 0.829), ("BL", 48, 0.775), ("BL", 40, 0.734), ("BL", 32, 0.673), ("TL", 80, 0.862)],
)
def test_a_flush_corner_patch_loses_a_measured_share_of_itself_to_the_crop(
    corner: str, size: int, expected_retained: float
) -> None:
    # These are the corner sizes the published sweeps use; the smaller the patch, the worse
    # the loss, because the ~5.7px border is a fixed cost paid on two of its four sides.
    assert retained_fraction(_corner(corner, size)) == pytest.approx(expected_retained, abs=0.005)


def test_the_smaller_the_flush_corner_the_larger_the_share_lost() -> None:
    retained = [retained_fraction(_corner("BL", s)) for s in (64, 48, 40, 32)]

    assert retained == sorted(retained, reverse=True)


def test_the_two_band_region_reported_as_37_9_percent_reaches_the_model_as_34_6() -> None:
    mask = torch.zeros(1, 1, POLICY_SIDE, POLICY_SIDE)
    mask[:, :, 0:22, :] = 1.0
    mask[:, :, 161:224, :] = 1.0

    assert nominal_area_fraction(mask) == pytest.approx(0.3795, abs=0.001)
    assert input_area_fraction(mask) == pytest.approx(0.3460, abs=0.001)


# --- the free fix -----------------------------------------------------------------------


@pytest.mark.parametrize("size", [64, 48, 40, 32])
def test_shifting_a_flush_corner_inside_the_window_buys_area_for_free(size: int) -> None:
    flush = _corner("BL", size)
    inset = shift_rect_into_crop(flush)

    # Same number of pixels — the patch is not made bigger, only moved off the dead border.
    assert nominal_area_fraction(inset) == pytest.approx(nominal_area_fraction(flush))
    assert input_area_fraction(inset) > input_area_fraction(flush)
    # Not exactly 1.0: resampling an indicator interpolates across the rect's own boundary, so
    # a few tenths of a percent leak whatever the placement. That slop is bounded and tiny.
    assert retained_fraction(inset) > 0.995


def test_shifting_moves_the_rect_by_the_minimum_needed_and_no_further() -> None:
    lo, hi = crop_safe_bounds()

    assert shift_rect_into_crop((160, 0, 64, 64)) == (hi + 1 - 64, lo, 64, 64)
    assert shift_rect_into_crop((60, 60, 40, 40)) == (60, 60, 40, 40)  # already inside: untouched


def test_a_rect_too_large_to_fit_inside_the_window_is_rejected_not_silently_shrunk() -> None:
    with pytest.raises(ValueError, match="cannot fit"):
        shift_rect_into_crop((0, 0, 220, 220))


# --- the mask helper --------------------------------------------------------------------


def test_rect_mask_marks_exactly_the_rect() -> None:
    mask = rect_mask((160, 0, 64, 64))

    assert mask.shape == (1, 1, POLICY_SIDE, POLICY_SIDE)
    assert mask.sum().item() == 64 * 64
    assert mask[0, 0, 160:224, 0:64].min() == 1.0
