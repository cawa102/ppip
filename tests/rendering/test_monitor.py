"""Contract of the in-scene monitor asset + in-place texture-upload seam (Task 1).

The CPU-pure parts are unit-tested here without MuJoCo:
  * `build_monitor_asset` allocates a single FIXED ``W x H`` texture (in-place upload
    cannot change texture dimensions, so the asset must be size-invariant to content).
  * `mask_local_hash` / `outside_mask_delta` are the change-detection primitives used to
    prove an upload changed the monitor region while leaving the rest of the frame stable.

The actual per-step in-place `mjr_uploadTexture` (no sim reset) is the GATE-A spike; it
runs only in the configured GPU rollout env behind `PPIP_GPU_TESTS` (see the bottom of
this file) and is exercised by `experiments/patch_attack/monitor_upload_probe.py`.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.support import make_candidate

# The in-place `mjr_uploadTexture` spike needs a live MuJoCo/EGL env, so it only runs in
# the configured GPU rollout env (GPU 1). Set PPIP_GPU_TESTS=1 with CUDA_VISIBLE_DEVICES=1,
# MUJOCO_GL=egl, PYTHONPATH=$HOME/LIBERO to enable it.
requires_gpu = pytest.mark.skipif(
    not os.environ.get("PPIP_GPU_TESTS"),
    reason="set PPIP_GPU_TESTS=1 in the GPU rollout env (GPU 1) to run the upload spike",
)

_TEX_HW = (128, 192)  # (H, W): a fixed monitor canvas, deliberately non-square


def _frame_and_mask() -> tuple[np.ndarray, np.ndarray]:
    """A 32x32 mid-grey frame with a rectangular 'monitor' mask in the top-left."""
    frame = np.full((32, 32, 3), 127, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=bool)
    mask[4:12, 4:20] = True
    return frame, mask


def test_build_monitor_asset_has_fixed_dims_regardless_of_content():
    from rendering.monitor import build_monitor_asset

    short = build_monitor_asset(
        make_candidate(visual_prompt={"text": "hi", "prompt_level": "semantic_visual_prompt"}),
        tex_hw=_TEX_HW,
    )
    verbose = build_monitor_asset(
        make_candidate(
            visual_prompt={
                "text": "A MUCH LONGER OVERRIDE LABEL THAT WOULD RESIZE A NORMAL PANEL",
                "prompt_level": "semantic_visual_prompt",
            },
            style={
                "foreground_color": "#ffffff",
                "background_color": "#000000",
                "font_size": 72,
                "font_family": "serif",
            },
        ),
        tex_hw=_TEX_HW,
    )

    assert short.texture.shape == (_TEX_HW[0], _TEX_HW[1], 3)
    assert verbose.texture.shape == (_TEX_HW[0], _TEX_HW[1], 3)


def test_mask_local_hash_tracks_only_in_mask_content():
    from rendering.monitor import mask_local_hash

    frame, mask = _frame_and_mask()

    # Changing a pixel INSIDE the mask changes the hash.
    changed_in = frame.copy()
    changed_in[5, 5] = (255, 0, 0)
    # Changing a pixel OUTSIDE the mask does NOT change the hash.
    changed_out = frame.copy()
    changed_out[20, 20] = (0, 255, 0)

    assert mask_local_hash(changed_in, mask) != mask_local_hash(frame, mask)
    assert mask_local_hash(changed_out, mask) == mask_local_hash(frame, mask)


def test_outside_mask_delta_ignores_in_mask_changes():
    from rendering.monitor import outside_mask_delta

    frame, mask = _frame_and_mask()

    # A big change INSIDE the monitor region must not register as outside drift.
    in_mask_change = frame.copy()
    in_mask_change[mask] = 0
    assert outside_mask_delta(frame, in_mask_change, mask) == 0.0

    # A change OUTSIDE the mask registers, reported on the 0..255 pixel scale.
    out_change = frame.copy()
    out_change[20, 20] = (127, 127, 227)  # +100 in one channel
    assert outside_mask_delta(frame, out_change, mask) == 100.0


def test_dilate_mask_grows_region_by_iterations_without_mutating_input():
    from rendering.monitor import dilate_mask

    mask = np.zeros((7, 7), dtype=bool)
    mask[3, 3] = True

    d1 = dilate_mask(mask, iterations=1)
    # 8-connectivity: one pixel grows to a 3x3 block (absorbs diagonal AA leaks too).
    assert d1[2:5, 2:5].all()
    assert d1.sum() == 9
    # Two iterations -> 5x5 block.
    assert dilate_mask(mask, iterations=2).sum() == 25
    # The input mask is never mutated (immutability).
    assert mask.sum() == 1


@requires_gpu
def test_in_place_upload_changes_monitor_without_reset_or_disturbing_sim():
    """GATE A: per-step texture upload with NO sim reset and no physics disturbance."""
    from experiments.patch_attack.monitor_upload_probe import run_upload_probe

    report = run_upload_probe(n_steps=20)

    # No reset after the one-time setup injection: the OSC controller is never re-init'd.
    assert report["reset_calls_after_setup"] == 0
    # Different uploads actually change the rendered monitor region.
    assert report["distinct_hashes"] > 1
    # The rest of the frame is stable within tolerance (AA/resolution, not bit-identity).
    assert report["max_outside_delta"] <= report["outside_tolerance"]
    # An in-place upload is visual-only: it must not perturb simulator state at all.
    assert report["max_eef_jump_m"] < 1e-6
    assert report["gate_a_pass"] is True
