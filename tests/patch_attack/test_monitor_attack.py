"""Task 5 — neutral-teacher render + S0 sanity gate.

The pure logic (the canonical neutral monitor content and the S0 seed-exclusion policy)
is unit-tested here on CPU. The GPU seams -- ``teacher_tokens`` (OpenVLA's real-path
target tokens on the neutral frame) and ``s0_sanity`` (per-seed target success with the
neutral monitor present) -- are exercised behind ``PPIP_GPU_TESTS`` in the GPU rollout env.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

requires_gpu = pytest.mark.skipif(
    not os.environ.get("PPIP_GPU_TESTS"),
    reason="set PPIP_GPU_TESTS=1 in the GPU rollout env (GPU 1) to run the OpenVLA seam",
)


def test_neutral_texture_is_uniform_mid_gray_at_the_monitor_dims():
    from monitor_attack import neutral_texture

    tex = neutral_texture((256, 256))

    assert tex.shape == (256, 256, 3)
    assert tex.dtype == np.uint8
    # Uniform: neutral content carries no injected pattern/text (unlike an attack texture).
    assert len(np.unique(tex)) == 1
    # Mid-range: unambiguously NOT the black(0)/white(255) calibration probes.
    assert 64 < int(tex[0, 0, 0]) < 192


def test_summarize_s0_flags_failing_seeds_as_excluded_not_usable():
    from monitor_attack import summarize_s0

    report = summarize_s0({0: True, 1: True, 2: False, 3: True, 4: True})

    # A seed whose TARGET fails with the neutral monitor is flagged out, never silently used.
    assert report.usable == (0, 1, 3, 4)
    assert report.excluded == (2,)
    assert report.all_pass is False
    assert report.results == {0: True, 1: True, 2: False, 3: True, 4: True}


def test_summarize_s0_all_pass_only_when_every_seed_succeeds():
    from monitor_attack import summarize_s0

    report = summarize_s0({0: True, 1: True, 2: True})

    # The gate passes: no seed is contaminated, so every seed is a usable teacher reference.
    assert report.all_pass is True
    assert report.excluded == ()
    assert report.usable == (0, 1, 2)


@requires_gpu
def test_neutral_and_attack_frames_differ_only_inside_the_monitor():
    """Neutral render matches deployment geometry: occlusion identical to the attack frame."""
    from monitor_attack import neutral_texture
    from monitor_upload_probe import setup_monitor_env

    from rendering.monitor import _policy_input_frame, dilate_mask, outside_mask_delta

    backend, env, geom, handle = setup_monitor_env()
    try:
        h, w = handle.dims
        # Pre-crop monitor mask in policy-input space, via a black<->white contrast.
        handle.upload(np.zeros((h, w, 3), dtype=np.uint8))
        env.sim.forward()
        black = _policy_input_frame(env)
        handle.upload(np.full((h, w, 3), 255, dtype=np.uint8))
        env.sim.forward()
        white = _policy_input_frame(env)
        mask = dilate_mask(np.abs(white.astype(int) - black.astype(int)).max(axis=2) > 12, 2)

        handle.upload(neutral_texture((h, w)))
        env.sim.forward()
        neutral = _policy_input_frame(env)
        handle.upload(np.full((h, w, 3), 255, dtype=np.uint8))
        env.sim.forward()
        attack = _policy_input_frame(env)

        # Same sim state -> the scene/robot occlusion is identical: the two frames differ
        # ONLY inside the monitor region (neutral gray vs the attack's white content).
        assert outside_mask_delta(neutral, attack, mask) == 0.0
        assert np.abs(neutral[mask].astype(int) - attack[mask].astype(int)).max() > 50
    finally:
        env.close()


@requires_gpu
def test_teacher_tokens_are_deterministic_target_tokens_on_the_neutral_frame():
    from monitor_attack import neutral_texture, teacher_tokens
    from monitor_hijack_backend import MonitorHijackBackend
    from monitor_upload_probe import setup_monitor_env

    _base, env, _geom, handle = setup_monitor_env()
    try:
        backend = MonitorHijackBackend()
        backend._policy = backend._load_policy()
        neutral = neutral_texture(handle.dims)

        t1 = teacher_tokens(backend, env, handle, neutral)
        t2 = teacher_tokens(backend, env, handle, neutral)

        assert tuple(t1.shape) == (1, 7)
        assert not t1.dtype.is_floating_point  # action *token ids*, not decoded floats
        # Greedy real path -> the same neutral frame yields the same teacher tokens.
        assert bool((t1 == t2).all())
        # Valid action-vocabulary tokens.
        vocab = backend._policy[0].vocab_size
        assert int(t1.min()) >= 0 and int(t1.max()) < vocab
    finally:
        env.close()


@requires_gpu
def test_s0_sanity_reports_per_seed_target_success_structure():
    from monitor_attack import S0Report, s0_sanity
    from monitor_hijack_backend import MonitorHijackBackend

    backend = MonitorHijackBackend()
    # A short-horizon smoke: assert the report SHAPE (per-seed booleans), not that the
    # target succeeds in 8 steps. The real gate runs seeds 0-4 at full horizon.
    report = s0_sanity(backend, [0], max_steps=8)

    assert isinstance(report, S0Report)
    assert set(report.results) == {0}
    assert isinstance(report.results[0], bool)
    assert isinstance(report.all_pass, bool)
