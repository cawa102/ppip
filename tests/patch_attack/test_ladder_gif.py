"""Frame alignment for the 4-condition comparison GIF.

The four rollouts have different lengths (the free-range hijack ends at its latch, step 131;
the others run to 220/240). Panels must stay aligned *by step index* or the GIF would show
step 200 of one condition beside step 90 of another and silently misrepresent the comparison.
A finished rollout holds its last frame so its outcome stays on screen.
"""

from __future__ import annotations

import os
import sys

import pytest

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "autoresearch/experiments/patch_attack"))

from make_ladder_gif import aligned_index, sampled_steps  # noqa: E402


@pytest.mark.parametrize("step", [0, 5, 130])
def test_index_passes_through_while_the_rollout_is_running(step: int) -> None:
    assert aligned_index(step, n_frames=131) == step


def test_last_frame_is_held_after_the_rollout_ends() -> None:
    # The free-range run stops at 131 frames; at global step 200 it must still show its final
    # frame (the delivered object), not wrap around or blank out.
    assert aligned_index(200, n_frames=131) == 130


def test_final_frame_index_is_the_last_valid_one() -> None:
    assert aligned_index(130, n_frames=131) == 130


def test_rejects_an_empty_rollout() -> None:
    with pytest.raises(ValueError, match="no frames"):
        aligned_index(0, n_frames=0)


def test_sampled_steps_starts_at_zero_and_covers_the_longest_rollout() -> None:
    steps = sampled_steps(total=240, stride=2)

    assert steps[0] == 0
    assert steps[-1] == 238
    assert all(b - a == 2 for a, b in zip(steps, steps[1:]))


def test_stride_of_one_keeps_every_step() -> None:
    assert sampled_steps(total=5, stride=1) == [0, 1, 2, 3, 4]


def test_rejects_a_non_positive_stride() -> None:
    with pytest.raises(ValueError, match="stride"):
        sampled_steps(total=10, stride=0)
