"""Replay a pre-recorded patch video instead of optimising a fresh patch each step.

**Why this exists.** ``ce_monitor_patch_attack`` re-solves the patch at every control step, and its
docstring asserts that "concatenated it is exactly a 'video' a monitor plays". That assertion has
never been tested on this track. It matters because the word-gate threat model is an *artifact*
one: the attacker leaves a video playing on a monitor, the video is physically there whether or not
the operator utters the trigger, and the frozen policy's own cross-modal routing is supposed to do
the gating. A live optimiser in the loop does not demonstrate that; replaying fixed pixels does.

**What replay is, precisely.** The deployed video is a function of the CONTROL STEP ALONE
(``monitor_replay.time_indexed_texture`` — reused rather than reimplemented, so no state-conditioned
frame selection is expressible here either). At step ``t`` the monitor shows ``v_t``; the camera
renders whatever the robot is actually looking at; the policy sees ``live_observation ⊕ v_t``. So
the two word conditions share the patch pixels exactly and differ in everything around them —
which is the point, and which is why the dormant leg is the real test: ``v_t`` was fitted against
the *armed* rollout's surroundings and must stay inert when it meets the dormant rollout's.

Pure and CPU-only by design (no torch, no policy): the replay path has no optimiser, so everything
except the forward pass is testable without a GPU.
"""

from __future__ import annotations

import os
import re
from typing import Final

import imageio.v2 as imageio
import numpy as np
from numpy.typing import NDArray

#: How ``run_confined_episode`` names recorded patch crops (``record_dir/patch/f0007.png``).
FRAME_RE: Final = re.compile(r"^f(\d+)\.png$")


def frame_paths(patch_dir: str) -> tuple[str, ...]:
    """Recorded patch-crop paths, ordered by their **frame index**, not lexically.

    Sorting on the parsed integer rather than the filename keeps replay aligned even if a future
    recorder changes the zero-padding width — a silent off-by-N misalignment would look exactly
    like a failed attack.
    """
    if not os.path.isdir(patch_dir):
        raise ValueError(f"no recorded patch frames: {patch_dir!r} is not a directory")
    indexed = [
        (int(m.group(1)), os.path.join(patch_dir, name))
        for name in os.listdir(patch_dir)
        if (m := FRAME_RE.match(name))
    ]
    if not indexed:
        raise ValueError(f"no recorded patch frames in {patch_dir!r}")
    return tuple(path for _idx, path in sorted(indexed))


def load_patch_video(patch_dir: str) -> list[NDArray[np.uint8]]:
    """Load a recorded patch sequence as uint8 crops in control-step order.

    Raises if the crops are not all the same shape: a ragged video cannot be pasted into one fixed
    rectangle, and discovering that mid-rollout would waste the whole episode.
    """
    video = [np.asarray(imageio.imread(p), dtype=np.uint8) for p in frame_paths(patch_dir)]
    shapes = {frame.shape for frame in video}
    if len(shapes) > 1:
        raise ValueError(f"inconsistent patch-frame shapes in {patch_dir!r}: {sorted(shapes)}")
    return video


def composite_patch(
    image: NDArray[np.uint8], patch: NDArray[np.uint8], rect: tuple[int, int, int, int]
) -> NDArray[np.uint8]:
    """Return a NEW observation with ``patch`` pasted into ``rect`` — the input is never mutated.

    ``rect`` is ``(r0, c0, ph, pw)``, matching ``ce_monitor_patch_attack``. The patch must fit the
    rectangle exactly as recorded; a mismatch means the video came from a different geometry, which
    must fail loudly rather than be resized into a different attack.
    """
    r0, c0, ph, pw = rect
    if patch.shape[:2] != (ph, pw):
        raise ValueError(
            f"patch {patch.shape[:2]} does not fit rect {(ph, pw)} — the recorded video "
            "was made at a different geometry"
        )
    if r0 + ph > image.shape[0] or c0 + pw > image.shape[1]:
        raise ValueError(f"rect {rect} does not fit an image of shape {image.shape[:2]}")
    out = image.copy()
    out[r0 : r0 + ph, c0 : c0 + pw] = patch
    return out
