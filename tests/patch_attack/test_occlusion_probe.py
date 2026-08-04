"""The non-occlusion claim is only as good as the mapping from a raw MuJoCo segmentation
into the policy's frame. A silent orientation error would swap which corner is being
measured and turn "provably off the objects" into its opposite, so the chain is pinned here."""

from __future__ import annotations

import numpy as np
import pytest
from occlusion_probe import (
    POLICY_SIDE,
    RENDER_SIDE,
    bbox,
    calibrate_orientation,
    corner_rect,
    dilate,
    entity_masks,
    measure,
    to_policy_space,
)

NAMES = {7: "alphabet_soup_1_g0", 9: "salad_dressing_1_g0", 11: "basket_1_g0"}


def _raw_with_block(rows: slice, cols: slice, value: int = 5) -> np.ndarray:
    raw = np.zeros((RENDER_SIDE, RENDER_SIDE), dtype=np.int32)
    raw[rows, cols] = value
    return raw


# --- geometry -------------------------------------------------------------------------


def test_corner_rects_sit_in_their_named_corners() -> None:
    assert corner_rect("TL", 32) == (0, 0, 32, 32)
    assert corner_rect("TR", 32) == (0, POLICY_SIDE - 32, 32, 32)
    assert corner_rect("BL", 32) == (POLICY_SIDE - 32, 0, 32, 32)
    assert corner_rect("BR", 32) == (POLICY_SIDE - 32, POLICY_SIDE - 32, 32, 32)


def test_the_contaminated_legacy_init_is_labelled_as_neither_split() -> None:
    import shared_inits
    from occlusion_probe import split_of

    assert split_of(shared_inits.LEGACY_GATE_INIT) == "legacy_gate"
    assert split_of(shared_inits.TRAIN_INITS[0]) == "train"
    assert split_of(shared_inits.HELDOUT_INITS[0]) == "heldout"


def test_bbox_bounds_a_mask_and_returns_none_when_empty() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:8] = True

    assert bbox(mask) == (2, 4, 3, 7)
    assert bbox(np.zeros((10, 10), dtype=bool)) is None


def test_dilation_grows_a_mask_by_one_pixel_in_each_direction() -> None:
    mask = np.zeros((7, 7), dtype=bool)
    mask[3, 3] = True

    grown = dilate(mask)

    assert grown.sum() == 5  # centre + 4 neighbours
    assert grown[2, 3] and grown[4, 3] and grown[3, 2] and grown[3, 4]
    assert not grown[2, 2]  # 4-neighbour, not 8
    assert mask.sum() == 1  # input untouched


# --- the orientation chain (the part that can silently invert a claim) ----------------


def test_the_180_degree_rotation_sends_a_raw_top_left_block_to_the_policy_bottom_right() -> None:
    raw = _raw_with_block(slice(0, 12), slice(0, 12))

    policy = to_policy_space(raw, "identity")

    assert policy[POLICY_SIDE - 1, POLICY_SIDE - 1] == 5
    assert policy[0, 0] == 0


def test_a_vertical_render_flip_lands_the_same_block_in_the_top_right() -> None:
    raw = _raw_with_block(slice(0, 12), slice(0, 12))

    policy = to_policy_space(raw, "vflip")

    assert policy[0, POLICY_SIDE - 1] == 5
    assert policy[POLICY_SIDE - 1, 0] == 0


def test_policy_space_output_is_the_policys_own_resolution() -> None:
    assert to_policy_space(_raw_with_block(slice(0, 4), slice(0, 4)), "identity").shape == (
        POLICY_SIDE,
        POLICY_SIDE,
    )


def test_calibration_recovers_a_known_flip() -> None:
    rng = np.random.default_rng(0)
    observed = rng.integers(0, 255, size=(64, 64, 3)).astype(np.uint8)

    assert calibrate_orientation(observed[::-1].copy(), observed) == "vflip"
    assert calibrate_orientation(observed.copy(), observed) == "identity"
    assert calibrate_orientation(observed[::-1, ::-1].copy(), observed) == "rot180"


def test_calibration_refuses_to_guess_when_no_flip_matches() -> None:
    rng = np.random.default_rng(1)
    rendered = rng.integers(0, 255, size=(64, 64, 3)).astype(np.uint8)
    observed = rng.integers(0, 255, size=(64, 64, 3)).astype(np.uint8)

    with pytest.raises(RuntimeError, match="could not calibrate"):
        calibrate_orientation(rendered, observed)


# --- entity masks and overlap ---------------------------------------------------------


def test_entities_match_their_geoms_and_the_union_covers_every_object() -> None:
    seg = np.zeros((16, 16), dtype=np.int32)
    seg[0:4, 0:4] = 7  # alphabet soup
    seg[8:12, 8:12] = 9  # salad dressing

    masks, matched = entity_masks(seg, NAMES)

    assert masks["user_object"].sum() == 16
    assert masks["target_object"].sum() == 16
    assert masks["any_task_object"].sum() == 32
    assert matched["user_object"] == ["alphabet_soup_1_g0"]
    assert masks["gripper"].sum() == 0  # no gripper geom in this fake model


def test_a_rect_over_an_object_is_reported_as_occluding_and_a_clear_corner_is_not() -> None:
    seg = np.zeros((POLICY_SIDE, POLICY_SIDE), dtype=np.int32)
    seg[200:224, 0:24] = 7  # an object sitting in the bottom-left

    result = measure(seg, NAMES)
    by_key = {(o["corner"], o["size"]): o for o in result["overlaps"]}

    assert by_key[("BL", 32)]["occludes_any_object"] is True
    assert by_key[("BL", 32)]["overlap_px"]["user_object"] > 0
    assert by_key[("TR", 32)]["occludes_any_object"] is False
    assert by_key[("TR", 32)]["overlap_px"]["any_task_object"] == 0


def test_overlap_is_measured_on_the_dilated_mask_so_boundaries_cannot_hide() -> None:
    # One object pixel exactly one row above the BL:32 rect (rect starts at row 192).
    seg = np.zeros((POLICY_SIDE, POLICY_SIDE), dtype=np.int32)
    seg[191, 5] = 7

    result = measure(seg, NAMES)
    by_key = {(o["corner"], o["size"]): o for o in result["overlaps"]}

    # Undilated it would miss by one pixel; dilation makes the probe err toward occlusion.
    assert by_key[("BL", 32)]["occludes_any_object"] is True


def test_area_fractions_match_the_previously_reported_corner_sizes() -> None:
    seg = np.zeros((POLICY_SIDE, POLICY_SIDE), dtype=np.int32)
    by_key = {(o["corner"], o["size"]): o for o in measure(seg, NAMES)["overlaps"]}

    assert by_key[("BL", 80)]["area_frac"] == pytest.approx(0.128, abs=5e-4)
    assert by_key[("BL", 64)]["area_frac"] == pytest.approx(0.082, abs=5e-4)
    assert by_key[("BL", 32)]["area_frac"] == pytest.approx(0.020, abs=5e-4)


def test_an_empty_scene_leaves_every_corner_clear() -> None:
    seg = np.zeros((POLICY_SIDE, POLICY_SIDE), dtype=np.int32)

    assert all(not o["occludes_any_object"] for o in measure(seg, NAMES)["overlaps"])
