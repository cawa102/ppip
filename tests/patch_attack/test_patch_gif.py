"""Amplified-difference rendering for the patch-evolution GIF.

The perturbation at a small epsilon is, by construction, near-invisible -- which makes an
honest figure of it look like a still image of the logo. The difference panel multiplies
`patch - carrier` by a stated gain around mid-grey so the structure and the frame-to-frame
churn become visible. The gain is printed on the figure, because an amplified difference shown
without its gain reads as the actual perturbation and overstates the attack.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "autoresearch/experiments/patch_attack"))

from make_patch_gif import amplify_difference, gain_for_eps  # noqa: E402


def test_no_difference_renders_as_flat_mid_grey() -> None:
    carrier = np.full((4, 4, 3), 0.5, dtype=np.float32)

    out = amplify_difference(carrier, carrier, gain=10)

    assert np.allclose(out, 128, atol=1)


def test_a_positive_difference_renders_brighter_than_mid_grey() -> None:
    carrier = np.full((4, 4, 3), 0.5, dtype=np.float32)
    patch = np.full((4, 4, 3), 0.52, dtype=np.float32)

    out = amplify_difference(patch, carrier, gain=10)

    assert out.mean() > 128


def test_a_negative_difference_renders_darker_than_mid_grey() -> None:
    carrier = np.full((4, 4, 3), 0.5, dtype=np.float32)
    patch = np.full((4, 4, 3), 0.48, dtype=np.float32)

    assert amplify_difference(patch, carrier, gain=10).mean() < 128


def test_gain_scales_the_rendered_deviation() -> None:
    carrier = np.full((4, 4, 3), 0.5, dtype=np.float32)
    patch = np.full((4, 4, 3), 0.51, dtype=np.float32)

    low = amplify_difference(patch, carrier, gain=2).mean()
    high = amplify_difference(patch, carrier, gain=8).mean()

    assert high - 128 > low - 128 > 0


def test_extreme_amplification_clips_instead_of_wrapping() -> None:
    # Wrapping would turn a saturated positive deviation into a black pixel -- the figure would
    # read as a large negative perturbation exactly where the attack pushed hardest.
    carrier = np.full((4, 4, 3), 0.5, dtype=np.float32)
    patch = np.full((4, 4, 3), 0.9, dtype=np.float32)

    out = amplify_difference(patch, carrier, gain=50)

    assert out.max() == 255
    assert out.min() >= 128


def test_output_is_uint8_and_shaped_like_the_input() -> None:
    carrier = np.full((6, 5, 3), 0.5, dtype=np.float32)
    patch = np.full((6, 5, 3), 0.55, dtype=np.float32)

    out = amplify_difference(patch, carrier, gain=4)

    assert out.dtype == np.uint8
    assert out.shape == (6, 5, 3)


def test_uint8_inputs_are_accepted_and_scaled() -> None:
    carrier = np.full((4, 4, 3), 128, dtype=np.uint8)
    patch = np.full((4, 4, 3), 128, dtype=np.uint8)

    assert np.allclose(amplify_difference(patch, carrier, gain=10), 128, atol=1)


def test_rejects_a_non_positive_gain() -> None:
    carrier = np.full((4, 4, 3), 0.5, dtype=np.float32)

    with pytest.raises(ValueError, match="gain"):
        amplify_difference(carrier, carrier, gain=0)


def test_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        amplify_difference(
            np.zeros((4, 4, 3), dtype=np.float32), np.zeros((8, 8, 3), dtype=np.float32), gain=4
        )


def test_gain_scales_inversely_with_the_epsilon_budget() -> None:
    # A fixed gain cannot serve the whole ladder: what makes a small eps legible saturates a
    # large one, and a saturated panel misreports the perturbation as uniformly maximal.
    assert gain_for_eps(0.06) > gain_for_eps(0.25) >= gain_for_eps(1.0)


def test_gain_maps_the_budget_near_the_target_deviation() -> None:
    for eps in (0.03, 0.06, 0.12, 0.25):
        rendered = eps * gain_for_eps(eps)

        assert 0.2 <= rendered <= 0.8


def test_gain_never_drops_below_one() -> None:
    # An unbounded (free-range) patch would otherwise round to a gain of 0 and render black.
    assert gain_for_eps(5.0) == 1


def test_gain_rejects_a_non_positive_epsilon() -> None:
    with pytest.raises(ValueError, match="eps"):
        gain_for_eps(0.0)
