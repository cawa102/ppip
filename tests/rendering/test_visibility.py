"""Contract of the prompt-visibility helpers (CPU-pure).

Given a MuJoCo segmentation mask (per-pixel geom id) and the injected prompt geom's
id, `prompt_pixel_fraction` reports how much of the camera frame the label occupies —
the objective "was the prompt actually in view?" gate. `visibility_overlay` tints
those pixels for presentation figures. Both are pure numpy and unit-tested here; the
segmentation render itself happens in the rollout backend in the configured GPU
rollout environment.
"""

from __future__ import annotations

import numpy as np

from rendering.visibility import prompt_pixel_fraction, visibility_overlay

_GEOM_ID = 7


def _mask_with_prompt(n_prompt_pixels: int, size: int = 8) -> np.ndarray:
    seg = np.zeros((size, size), dtype=np.int32)
    seg.flat[:n_prompt_pixels] = _GEOM_ID
    return seg


def test_pixel_fraction_counts_prompt_pixels():
    seg = _mask_with_prompt(16, size=8)  # 16 of 64 pixels

    assert prompt_pixel_fraction(seg, _GEOM_ID) == 0.25


def test_pixel_fraction_is_zero_when_prompt_absent():
    seg = np.zeros((8, 8), dtype=np.int32)

    assert prompt_pixel_fraction(seg, _GEOM_ID) == 0.0


def test_pixel_fraction_ignores_other_geoms():
    seg = _mask_with_prompt(8, size=8)
    seg[seg == 0] = 3  # background is some other geom id, not the prompt

    assert prompt_pixel_fraction(seg, _GEOM_ID) == 0.125


def test_overlay_tints_prompt_pixels_and_leaves_others_untouched():
    frame = np.full((8, 8, 3), 100, dtype=np.uint8)
    seg = _mask_with_prompt(8, size=8)
    color = (255, 0, 255)

    out = visibility_overlay(frame, seg, _GEOM_ID, color=color)

    assert out.shape == frame.shape and out.dtype == np.uint8
    prompt = seg == _GEOM_ID
    # tinted pixels moved toward the highlight colour; others are unchanged.
    assert (out[prompt] != 100).any()
    assert np.array_equal(out[~prompt], frame[~prompt])


def test_overlay_does_not_mutate_the_input_frame():
    frame = np.full((8, 8, 3), 100, dtype=np.uint8)
    seg = _mask_with_prompt(8, size=8)

    visibility_overlay(frame, seg, _GEOM_ID, color=(255, 0, 255))

    assert np.array_equal(frame, np.full((8, 8, 3), 100, dtype=np.uint8))
