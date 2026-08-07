"""Deploying a pre-recorded patch video (the artifact-level word-gate test).

The per-frame attack re-solves a patch every step, so the armed and dormant rollouts never
share pixels — which makes the "one video plays regardless of what the operator says" threat
model *untested*, not demonstrated. These cover the pure pieces of the replay path: reading a
recorded patch-crop sequence back in the order it was written, and pasting frame ``t`` into the
fixed rectangle of a live observation without disturbing anything outside it.

Frame SELECTION is deliberately not re-implemented here — ``monitor_replay.time_indexed_texture``
already provides a state-blind, strictly time-indexed selector (and its own tests); the last test
pins the integration so the replay path cannot quietly acquire state-conditioned selection.
"""

from __future__ import annotations

import imageio.v2 as imageio
import numpy as np
import pytest
from monitor_replay import time_indexed_texture
from patch_replay import composite_patch, load_patch_video

RECT = (160, 0, 64, 64)


def _write_frames(tmp_path, count: int, side: int = 64):
    """Write ``count`` distinguishable patch crops, as ``run_confined_episode`` records them."""
    for i in range(count):
        frame = np.full((side, side, 3), i, dtype=np.uint8)
        imageio.imwrite(str(tmp_path / f"f{i:04d}.png"), frame)
    return tmp_path


def test_load_patch_video_returns_frames_in_recorded_step_order(tmp_path):
    # Arrange
    _write_frames(tmp_path, 12)

    # Act
    video = load_patch_video(str(tmp_path))

    # Assert -- frame t must be the patch recorded at step t, or replay is silently misaligned
    assert len(video) == 12
    assert [int(f[0, 0, 0]) for f in video] == list(range(12))


def test_load_patch_video_ignores_non_frame_files(tmp_path):
    # Arrange
    _write_frames(tmp_path, 3)
    (tmp_path / "notes.txt").write_text("not a frame")

    # Act / Assert
    assert len(load_patch_video(str(tmp_path))) == 3


def test_load_patch_video_raises_when_no_frames_found(tmp_path):
    # A silently-empty video would run the whole episode as an unpatched control while the
    # result JSON claimed patch_mode='replay'.
    with pytest.raises(ValueError, match="no recorded patch frames"):
        load_patch_video(str(tmp_path))


def test_load_patch_video_raises_on_ragged_frame_shapes(tmp_path):
    # Arrange
    _write_frames(tmp_path, 2)
    imageio.imwrite(str(tmp_path / "f0002.png"), np.zeros((32, 32, 3), dtype=np.uint8))

    # Act / Assert
    with pytest.raises(ValueError, match="inconsistent"):
        load_patch_video(str(tmp_path))


def test_composite_patch_places_the_patch_at_the_rect():
    # Arrange
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    patch = np.full((64, 64, 3), 200, dtype=np.uint8)

    # Act
    out = composite_patch(image, patch, RECT)

    # Assert
    assert np.array_equal(out[160:224, 0:64], patch)
    assert out.sum() == patch.sum()  # nothing painted outside the rectangle


def test_composite_patch_leaves_the_live_observation_untouched():
    # The pixels OUTSIDE the monitor are the live render -- the whole point is that they differ
    # between the armed and dormant rollouts while the patch stays the same.
    image = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    before = image.copy()
    patch = np.full((64, 64, 3), 7, dtype=np.uint8)

    out = composite_patch(image, patch, RECT)

    assert np.array_equal(image, before), "composite_patch must not mutate its input"
    assert not np.array_equal(out, before)
    assert np.array_equal(out[0:160], before[0:160])


def test_composite_patch_rejects_a_patch_that_does_not_fit_the_rect():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="does not fit"):
        composite_patch(image, np.zeros((80, 80, 3), dtype=np.uint8), RECT)


def test_replay_selection_is_time_indexed_and_holds_the_last_frame():
    # Pins the deployment contract: the video is a function of the STEP INDEX alone. The dormant
    # rollout runs longer than the armed video it replays, so the tail must be defined.
    video = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(3)]

    assert int(time_indexed_texture(video, 0)[0, 0, 0]) == 0
    assert int(time_indexed_texture(video, 2)[0, 0, 0]) == 2
    assert int(time_indexed_texture(video, 99)[0, 0, 0]) == 2
