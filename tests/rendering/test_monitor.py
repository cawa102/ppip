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


def test_homography_quad_to_texture_maps_image_corners_onto_texture_corners():
    from rendering.monitor import UVMap, homography_quad_to_texture

    # A 256x256 texture (TL, TR, BR, BL) seen as a perspective-warped quad in the image.
    texture_corners = np.array([[0, 0], [256, 0], [256, 256], [0, 256]], dtype=np.float64)
    image_corners = np.array(
        [[100, 50], [151, 54], [147, 112], [97, 104]], dtype=np.float64
    )
    uv_map = UVMap(texture_corners=texture_corners, image_corners=image_corners)

    homography = homography_quad_to_texture(uv_map)
    mapped = homography.apply(image_corners)

    # The fitted image->texture map sends each image corner onto its texture corner.
    assert np.allclose(mapped, texture_corners, atol=1e-6)
    # And the centroid maps to the texture centroid (interior consistency, not just corners).
    img_centroid = image_corners.mean(axis=0, keepdims=True)
    assert np.allclose(homography.apply(img_centroid), [[128, 128]], atol=8.0)


def test_center_crop_mask_drops_the_edge_and_magnifies_the_centre():
    from rendering.monitor import center_crop_mask

    # A block hard against the frame edge is removed by the 0.9-area center crop.
    edge = np.zeros((224, 224), dtype=bool)
    edge[0:4, 0:4] = True
    assert not center_crop_mask(edge).any()

    # A centred block survives and is magnified by the crop-and-resize-to-224 (the 0.9-area
    # crop zooms in, so central content covers more post-crop pixels). Use a large block so
    # the ~1.05x/axis magnification is well above nearest-neighbour rounding.
    centre = np.zeros((224, 224), dtype=bool)
    centre[62:162, 62:162] = True
    out = center_crop_mask(centre)
    assert out.any()
    assert out.sum() > centre.sum()

    # A full mask stays full (cropping all-True yields all-True).
    assert center_crop_mask(np.ones((224, 224), dtype=bool)).all()


def test_center_crop_mask_matches_the_region_vla_diff_samples():
    import pytest

    torch = pytest.importorskip("torch")
    import vla_diff

    from rendering.monitor import center_crop_mask

    rng = np.zeros((224, 224), dtype=bool)
    rng[60:150, 40:180] = True  # an off-centre rectangle

    ours = center_crop_mask(rng)

    # Push the mask as a float image through the REAL preprocessing crop and threshold it.
    img = torch.from_numpy(rng.astype("float32"))[None, None].repeat(1, 3, 1, 1)
    cropped = vla_diff.center_crop_resize(img)[0, 0].numpy() > 0.5

    # Nearest-neighbour vs bilinear can disagree by a thin boundary ring; require tight IoU.
    inter = (ours & cropped).sum()
    union = (ours | cropped).sum()
    assert union > 0 and inter / union > 0.98


def _polygon_area(corners):
    x, y = corners[:, 0], corners[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


@requires_gpu
def test_calibrate_uv_round_trips_and_mask_is_in_frame():
    """GATE-2: UV calibration is perspective-consistent and the mask lands in the frame."""
    import numpy as np
    from experiments.patch_attack.monitor_upload_probe import setup_monitor_env

    from rendering.monitor import (
        calibrate_uv,
        homography_from_correspondences,
        monitor_mask_224,
    )

    backend, env, geom, handle = setup_monitor_env()
    try:
        uv = calibrate_uv(env, handle)

        # 4 corners inside the 224 policy frame, forming a non-degenerate visible quad.
        assert uv.image_corners.min() >= 0.0 and uv.image_corners.max() <= 223.0
        assert _polygon_area(uv.image_corners) > 100.0

        # Perspective round-trip: the texture->image homography fit from the 4 corners must
        # also predict an INTERIOR marker's rendered location (generalisation, not just fit).
        h, w = handle.dims
        forward = homography_from_correspondences(uv.texture_corners, uv.image_corners)
        tex_pt = np.array([[w * 0.5, h * 0.5]])
        predicted = forward.apply(tex_pt)[0]

        marker = np.zeros((h, w, 3), dtype=np.uint8)
        p = max(4, min(h, w) // 6)
        marker[h // 2 - p // 2 : h // 2 + p // 2, w // 2 - p // 2 : w // 2 + p // 2] = 255
        handle.upload(np.zeros((h, w, 3), dtype=np.uint8))
        env.sim.forward()
        from rendering.monitor import _policy_input_frame

        base = _policy_input_frame(env).astype(np.int64)
        handle.upload(marker)
        env.sim.forward()
        lit = _policy_input_frame(env).astype(np.int64)
        ys, xs = np.where(np.abs(lit - base).max(axis=2) > 12)
        observed = np.array([xs.mean(), ys.mean()])
        assert np.linalg.norm(predicted - observed) < 12.0

        # Mask: right shape, non-empty, entirely in-frame.
        mask = monitor_mask_224(env, handle)
        assert mask.shape == (224, 224)
        assert mask.any()
    finally:
        env.close()


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
