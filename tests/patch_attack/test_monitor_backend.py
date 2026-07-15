"""Contract of MonitorHijackBackend's canonical-stage invariant (Task 4).

The threat-model invariant: every scored rollout's policy input derives ONLY from a fresh
MuJoCo render taken AFTER the monitor texture upload; the camera image buffer is never
written (unlike hijack_backend/adaptive_attack, which execute from a perturbed `pu8`).

The pure invariant logic -- three canonical image-stage hashes and the freshness assertion
-- is unit-tested here on CPU with hand-built arrays. `step_with_texture` (the render/step
loop) is exercised in the GPU rollout env behind PPIP_GPU_TESTS.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

requires_gpu = pytest.mark.skipif(
    not os.environ.get("PPIP_GPU_TESTS"),
    reason="set PPIP_GPU_TESTS=1 in the GPU rollout env (GPU 1) to run the backend rollout",
)


def test_canonical_stage_hashes_are_stage_specific_and_stable():
    from monitor_hijack_backend import canonical_stage_hashes

    s1 = np.zeros((256, 256, 3), dtype=np.uint8)
    s2 = np.ones((224, 224, 3), dtype=np.uint8)
    s3 = np.full((6, 224, 224), 7, dtype=np.float32)

    hashes = canonical_stage_hashes(s1, s2, s3)

    assert len(hashes) == 3
    assert len(set(hashes)) == 3  # distinct stages -> distinct hashes
    # Stable: same arrays -> same hashes (byte-exact evidence for the invariant log).
    assert canonical_stage_hashes(s1, s2, s3) == hashes
    # A change in any stage changes only that stage's hash.
    s2b = s2.copy()
    s2b[0, 0] = 2
    assert canonical_stage_hashes(s1, s2b, s3)[1] != hashes[1]
    assert canonical_stage_hashes(s1, s2b, s3)[0] == hashes[0]


def test_assert_policy_input_fresh_accepts_transform_of_s1_and_rejects_stale():
    from monitor_hijack_backend import StalePolicyInputError, assert_policy_input_fresh

    s1 = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)

    def transform(raw):
        return raw[::-1, ::-1]  # stand-in for the real rot180+resize pipeline

    # The freshly-transformed post-upload render passes the guard.
    assert_policy_input_fresh(s1, transform(s1), transform)

    # A stale frame (not transform(S1)) -- e.g. a cached obs image -- is rejected.
    stale = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(StalePolicyInputError):
        assert_policy_input_fresh(s1, stale, transform)


def test_backend_is_not_the_camera_buffer_write_path():
    from monitor_hijack_backend import MonitorHijackBackend

    from evaluator.openvla_backend import OpenVLARolloutBackend

    backend = MonitorHijackBackend()

    # It IS the fixed evaluator backend (inherits the scorer untouched)...
    assert isinstance(backend, OpenVLARolloutBackend)
    # ...but NOT the camera-buffer-write attack path (hijack_backend/adaptive_attack), and
    # exposes none of its camera-write seams that overlay/add a delta to the policy image.
    assert type(backend).__mro__[1] is OpenVLARolloutBackend
    for camera_write_attr in ("_delta", "_patch", "set_delta", "set_patch", "_overlay"):
        assert not hasattr(backend, camera_write_attr)


@requires_gpu
def test_step_with_texture_feeds_the_fresh_post_upload_monitor():
    """The policy input reflects the uploaded monitor (fresh render, not a stale obs)."""
    from monitor_hijack_backend import MonitorHijackBackend
    from monitor_upload_probe import setup_monitor_env

    from rendering.monitor import _policy_input_frame, dilate_mask

    backend, env, geom, handle = setup_monitor_env()
    try:
        h, w = handle.dims

        # Monitor region in policy-input (pre-crop 224) space, via a black<->white contrast.
        handle.upload(np.zeros((h, w, 3), dtype=np.uint8))
        env.sim.forward()
        pin_black = _policy_input_frame(env)
        handle.upload(np.full((h, w, 3), 255, dtype=np.uint8))
        env.sim.forward()
        pin_white = _policy_input_frame(env)
        mask = dilate_mask(
            np.abs(pin_white.astype(int) - pin_black.astype(int)).max(axis=2) > 12, 2
        )
        assert mask.any()

        mb = MonitorHijackBackend()
        mb._policy = mb._load_policy()
        user_task = "pick up the alphabet soup and place it in the basket"
        white = np.full((h, w, 3), 255, dtype=np.uint8)
        result = mb.step_with_texture(env, handle, white, user_task)

        # Canonical-stage evidence: three distinct, non-empty hashes; sane shapes.
        assert result.policy_image.shape == (224, 224, 3)
        assert result.action.shape == (7,)
        assert len({result.s1_hash, result.s2_hash, result.s3_hash}) == 3

        # The fed image shows the WHITE monitor (reflects the upload through a fresh render),
        # clearly brighter in the monitor region than the black-monitor frame -> not stale.
        assert (
            result.policy_image[mask].mean() > pin_black[mask].mean() + 50
        )
    finally:
        env.close()
