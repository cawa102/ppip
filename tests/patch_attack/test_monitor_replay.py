"""Task 8 — open-loop replay (Stage 2) + controls + margin (GATE B).

The deployment threat is a *recorded* video played strictly by time index -- no
re-optimisation, no state-conditioned frame selection. The time-indexing, the scramble
control, and the margin panel are CPU-pure and unit-tested here; the actual replay rollout
(attack + blank/scrambled controls) is a GPU seam behind ``PPIP_GPU_TESTS``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

requires_gpu = pytest.mark.skipif(
    not os.environ.get("PPIP_GPU_TESTS"),
    reason="set PPIP_GPU_TESTS=1 in the GPU rollout env (GPU 1) to run the OpenVLA seam",
)


def _video(n):
    """A distinguishable n-frame video: frame t is a uniform texture of value t."""
    return [np.full((4, 4, 3), t, dtype=np.uint8) for t in range(n)]


def test_time_indexed_texture_depends_only_on_the_step_index():
    from monitor_replay import time_indexed_texture

    video = _video(3)

    # texture_t = video[t] -- a function of t ALONE (no env/state argument exists).
    assert int(time_indexed_texture(video, 0)[0, 0, 0]) == 0
    assert int(time_indexed_texture(video, 1)[0, 0, 0]) == 1
    assert int(time_indexed_texture(video, 2)[0, 0, 0]) == 2
    # Past the end it holds the last frame (never re-optimises or picks by state).
    assert int(time_indexed_texture(video, 9)[0, 0, 0]) == 2


def test_scramble_video_is_a_deterministic_permutation_that_reorders():
    from monitor_replay import scramble_video

    video = _video(6)

    scrambled = scramble_video(video, seed=0)

    # Same frames, so the control shows the SAME content -- only the timing is destroyed.
    assert sorted(int(f[0, 0, 0]) for f in scrambled) == [0, 1, 2, 3, 4, 5]
    # Deterministic given the seed (reproducible control).
    again = scramble_video(video, seed=0)
    assert [int(f[0, 0, 0]) for f in scrambled] == [int(f[0, 0, 0]) for f in again]
    # It actually breaks the time order (this is the point of the control).
    assert [int(f[0, 0, 0]) for f in scrambled] != [0, 1, 2, 3, 4, 5]


def _result(kind, *, targeted, phase, dist):
    from monitor_replay import ReplayResult

    return ReplayResult(
        seed=0, kind=kind, targeted_success=targeted, commanded_success=False,
        max_phase=phase, min_target_dist=dist, steps=10,
    )


def test_margin_report_flags_a_hijack_that_beats_both_controls():
    from monitor_replay import margin_report

    attack = _result("attack", targeted=True, phase=3, dist=0.05)
    controls = [
        _result("blank", targeted=False, phase=1, dist=0.30),
        _result("scrambled", targeted=False, phase=2, dist=0.20),
    ]

    report = margin_report(attack, controls)

    # Margin is over the BEST control, not the average.
    assert report["phase_margin"] == 3 - 2
    # A real hijack: the attack succeeds and neither control does.
    assert report["hijack_beats_controls"] is True
    assert report["attack"]["targeted_success"] is True
    assert {c["kind"] for c in report["controls"]} == {"blank", "scrambled"}


def test_margin_report_denies_the_hijack_when_a_control_also_succeeds():
    from monitor_replay import margin_report

    attack = _result("attack", targeted=True, phase=3, dist=0.05)
    controls = [_result("scrambled", targeted=True, phase=3, dist=0.05)]

    report = margin_report(attack, controls)

    # If a content-identical but time-scrambled video also hijacks, it is not the trajectory.
    assert report["hijack_beats_controls"] is False
    assert report["phase_margin"] == 0


@requires_gpu
def test_run_replay_and_controls_produce_the_full_margin_panel():
    from monitor_attack import TEX_HW, neutral_texture
    from monitor_hijack_backend import MonitorHijackBackend
    from monitor_replay import margin_report, run_control, run_replay

    backend = MonitorHijackBackend()
    # A 2-frame smoke video (structure only -- not a hijack claim): validate the replay +
    # control + margin wiring runs open-loop through the monitor.
    video = [neutral_texture(TEX_HW), np.full((*TEX_HW, 3), 255, dtype=np.uint8)]

    attack = run_replay(backend, 0, video)
    blank = run_control(backend, 0, "blank", video)
    scrambled = run_control(backend, 0, "scrambled", video)

    for result in (attack, blank, scrambled):
        assert result.steps == 2
        assert isinstance(result.targeted_success, bool)
        assert result.max_phase >= 0
    assert (attack.kind, blank.kind, scrambled.kind) == ("attack", "blank", "scrambled")

    report = margin_report(attack, [blank, scrambled])
    assert "phase_margin" in report
    assert "hijack_beats_controls" in report
