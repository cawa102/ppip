"""Task 6 — masked-δ texture design + real-render surrogate.

The attacker may only change pixels the *monitor* controls, so the perturbation is
confined to the monitor mask, projected onto the texture via the calibrated homography,
and corrected for the render reality-gap by a surrogate measured against the real render.
The confinement / warp / surrogate math is CPU-pure and unit-tested here; the vla_diff
optimisation and stateless proxy scoring are GPU seams behind ``PPIP_GPU_TESTS``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

requires_gpu = pytest.mark.skipif(
    not os.environ.get("PPIP_GPU_TESTS"),
    reason="set PPIP_GPU_TESTS=1 in the GPU rollout env (GPU 1) to run the OpenVLA seam",
)


def test_apply_masked_delta_changes_only_inside_the_mask():
    from texture_surrogate import apply_masked_delta

    frame = np.full((224, 224, 3), 0.4, dtype=np.float64)
    delta = np.full((224, 224, 3), 0.2, dtype=np.float64)
    mask = np.zeros((224, 224), dtype=bool)
    mask[50:150, 60:160] = True

    out = apply_masked_delta(frame, delta, mask)

    # Outside the monitor mask the frame is byte-identical (the attacker can't touch it).
    assert np.array_equal(out[~mask], frame[~mask])
    # Inside the mask the delta is applied.
    assert np.allclose(out[mask], frame[mask] + delta[mask])
    # Never mutates the input frame.
    assert np.array_equal(frame, np.full((224, 224, 3), 0.4))


def test_warp_pattern_to_texture_places_the_image_pattern_at_the_monitor_quad():
    from texture_surrogate import warp_pattern_to_texture

    from rendering.monitor import UVMap

    tex_hw = (64, 64)
    # Texture corners span the full canvas; image corners are a projected quad (TL,TR,BR,BL
    # as (x=col, y=row)) -- the inverse warp must land each texture corner on its image one.
    texture_corners = np.array([[0, 0], [63, 0], [63, 63], [0, 63]], dtype=float)
    image_corners = np.array([[40, 30], [180, 50], [170, 190], [30, 170]], dtype=float)
    uv = UVMap(texture_corners=texture_corners, image_corners=image_corners)

    pattern = np.zeros((224, 224, 3), dtype=np.uint8)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for (x, y), color in zip(image_corners, colors, strict=True):
        pattern[int(y), int(x)] = color

    tex = warp_pattern_to_texture(pattern, uv, tex_hw)

    assert tex.shape == (64, 64, 3)
    # Each texture corner samples the image pattern at the corresponding image corner.
    for (tx, ty), color in zip([(0, 0), (63, 0), (63, 63), (0, 63)], colors, strict=True):
        assert tuple(int(v) for v in tex[ty, tx]) == color


def test_calibrate_surrogate_measures_gap_and_corrects_proposal_to_the_real_render():
    from texture_surrogate import calibrate_surrogate

    proposed = np.full((16, 16, 3), 0.5, dtype=np.float64)
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:10, 4:10] = True
    # The renderer shifts the masked region (additive-δ proposal != what it actually shows).
    real = proposed.copy()
    real[mask] = real[mask] + 0.1

    surrogate = calibrate_surrogate(proposed, real, mask)

    # The measured reality-gap magnitude inside the monitor.
    assert abs(surrogate.gap - 0.1) < 1e-9
    # Applying the surrogate corrects the proposal to the real render...
    corrected = surrogate.apply(proposed)
    assert np.allclose(corrected, real)
    # ...and never disturbs anything outside the monitor mask.
    assert np.allclose(corrected[~mask], proposed[~mask])


def _precrop_monitor_mask(env, handle):
    """Pre-crop-224 monitor region via a black<->white contrast (delta lives pre-crop)."""
    from rendering.monitor import _policy_input_frame, dilate_mask

    h, w = handle.dims
    handle.upload(np.zeros((h, w, 3), dtype=np.uint8))
    env.sim.forward()
    black = _policy_input_frame(env)
    handle.upload(np.full((h, w, 3), 255, dtype=np.uint8))
    env.sim.forward()
    white = _policy_input_frame(env)
    return dilate_mask(np.abs(white.astype(int) - black.astype(int)).max(axis=2) > 12, 2)


@requires_gpu
def test_optimize_masked_delta_reduces_teacher_ce_and_stays_inside_mask():
    from adaptive_attack import _prompt_ids
    from monitor_attack import USER_TASK, neutral_texture, teacher_tokens
    from monitor_hijack_backend import MonitorHijackBackend
    from monitor_upload_probe import setup_monitor_env
    from texture_surrogate import _target_token_ce, optimize_masked_delta

    from rendering.monitor import _policy_input_frame

    _base, env, _geom, handle = setup_monitor_env()
    try:
        backend = MonitorHijackBackend()
        backend._policy = backend._load_policy()
        model, processor, _cfg, _rs = backend._policy
        neutral = neutral_texture(handle.dims)

        mask = _precrop_monitor_mask(env, handle)
        teacher = teacher_tokens(backend, env, handle, neutral)  # uploads neutral, [1,7]
        user_ids = _prompt_ids(processor, USER_TASK)

        handle.upload(neutral)
        env.sim.forward()
        frame = _policy_input_frame(env)
        ce_before = _target_token_ce(model, frame, user_ids, teacher)

        delta = optimize_masked_delta(model, frame, mask, teacher, user_ids, k=10)

        # Confinement: the attacker only touched the monitor region.
        assert float(np.abs(delta[~mask]).max()) == 0.0
        # The monitor-confined delta moves the USER policy toward the TARGET tokens.
        perturbed = np.clip(frame.astype(np.float64) / 255.0 + delta, 0, 1)
        ce_after = _target_token_ce(
            model, (perturbed * 255).astype(np.uint8), user_ids, teacher
        )
        assert ce_after < ce_before
    finally:
        env.close()


@requires_gpu
def test_select_texture_is_stateless_and_scores_every_candidate():
    from adaptive_attack import _prompt_ids
    from monitor_attack import USER_TASK, neutral_texture, teacher_tokens
    from monitor_hijack_backend import MonitorHijackBackend
    from monitor_upload_probe import _eef_pos, setup_monitor_env
    from texture_surrogate import select_texture

    _base, env, _geom, handle = setup_monitor_env()
    try:
        backend = MonitorHijackBackend()
        backend._policy = backend._load_policy()
        _model, processor, _cfg, _rs = backend._policy
        h, w = handle.dims
        neutral = neutral_texture(handle.dims)
        teacher = teacher_tokens(backend, env, handle, neutral)
        user_ids = _prompt_ids(processor, USER_TASK)
        candidates = [
            neutral,
            np.full((h, w, 3), 255, dtype=np.uint8),
            np.zeros((h, w, 3), dtype=np.uint8),
        ]

        eef_before = _eef_pos(env)
        best, scores = select_texture(candidates, backend, env, handle, teacher, user_ids)
        eef_after = _eef_pos(env)

        assert len(scores) == 3
        assert best.shape == neutral.shape
        # Stateless: scoring N candidates never stepped the committed rollout (physics frozen).
        assert float(np.abs(eef_after - eef_before).max()) < 1e-9
        # The winner is the lowest-CE candidate.
        assert np.array_equal(best, candidates[int(np.argmin(scores))])
    finally:
        env.close()
