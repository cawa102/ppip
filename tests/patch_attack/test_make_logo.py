"""Base images carry a published stealth bound, so determinism and range are contractual."""

from __future__ import annotations

import numpy as np
import pytest
from make_logo import ALL_BASES, DEFAULT_SIZE, LOGOS, build_base, save_png


@pytest.mark.parametrize("name", ALL_BASES)
def test_every_base_has_the_requested_shape_and_range(name: str) -> None:
    base = build_base(name, 80)

    assert base.shape == (80, 80, 3)
    assert base.dtype == np.float32
    assert base.min() >= 0.0 and base.max() <= 1.0


@pytest.mark.parametrize("name", ALL_BASES)
def test_bases_are_byte_identical_across_calls(name: str) -> None:
    # The published patch PNG is only checkable against the base if the base is reproducible.
    assert np.array_equal(build_base(name, 48), build_base(name, 48))


@pytest.mark.parametrize("name", list(LOGOS))
def test_scrambled_control_preserves_the_colour_histogram_but_not_the_layout(name: str) -> None:
    logo = build_base(name, 80)
    scrambled = build_base(f"scrambled:{name}", 80)

    # Same pixels, rearranged -> identical sorted values, different arrangement.
    assert np.array_equal(np.sort(logo, axis=None), np.sort(scrambled, axis=None))
    assert not np.array_equal(logo, scrambled)


@pytest.mark.parametrize("name", list(LOGOS))
def test_flat_control_is_uniform_at_the_logos_mean_colour(name: str) -> None:
    logo = build_base(name, 80)
    flat = build_base(f"flat:{name}", 80)

    assert np.allclose(flat, flat[0, 0], atol=1e-6)
    # atol is float32 summation slack over 6400 px, still ~40x tighter than 8-bit PNG
    # quantisation (1/255), which is the resolution the control is actually inspected at.
    assert np.allclose(flat.mean(axis=(0, 1)), logo.mean(axis=(0, 1)), atol=1e-4)


def test_gray_control_matches_the_existing_blank_patch_mode() -> None:
    # `run_confined_episode(patch_mode="blank")` fills the rect with 0.5.
    assert np.allclose(build_base("gray", 32), 0.5)


@pytest.mark.parametrize("name", list(LOGOS))
def test_logos_are_low_frequency(name: str) -> None:
    logo = build_base(name, DEFAULT_SIZE)
    gray = logo.mean(axis=2)
    edge_energy = np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()

    # A flat mark on a solid fill: nearly all neighbouring pixels are identical. Adversarial
    # noise in the same rect sits an order of magnitude above this.
    assert edge_energy < 0.05


def test_unknown_base_name_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unknown base"):
        build_base("not-a-logo")


def test_unknown_modifier_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unknown base modifier"):
        build_base("inverted:aurora")


def test_scramble_rejects_a_size_it_cannot_partition_exactly() -> None:
    with pytest.raises(ValueError, match="divisible"):
        build_base("scrambled:aurora", 50)


def test_save_png_round_trips_within_quantisation_error(tmp_path) -> None:
    from PIL import Image

    base = build_base("aurora", 64)
    path = save_png(base, str(tmp_path / "nested" / "aurora.png"))
    reloaded = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

    assert np.abs(reloaded - base).max() <= 1.0 / 255.0 + 1e-6
