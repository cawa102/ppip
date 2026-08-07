"""Unit tests for ``word_gate_probe`` — WP5 of the word-gated patch experiment
(``docs/plans/2026-07-30-word-gated-patch.md``).

The Phase-0 open-loop gate probe splits into a **pure core** (frame discovery, decisive-dim
classification, gate-diagram aggregation, the feasibility-hinge gradient signal) tested here
without a GPU, and a thin **GPU seam** (``probe_frame`` / ``main``) that runs the two-branch
per-frame optimizer live under ``PPIP_GPU_TESTS`` on GPU 1.
"""
from __future__ import annotations

import os

import pytest
from word_gate_probe import (
    FrameGateProbe,
    aggregate_gate_diagram,
    decisive_dims,
    forced_fraction,
    gradient_gate_signal,
    list_frame_paths,
)


def _probe(
    name: str, n_dec: int, armed: float, dormant: float
) -> FrameGateProbe:
    return FrameGateProbe(
        frame=name,
        n_decisive=n_dec,
        armed_forced_fraction=armed,
        dormant_forced_fraction=dormant,
        grad_norm_without_w=0.0,
        grad_delta_norm=0.0,
    )


class TestFrameListing:
    def test_lists_frame_pngs_sorted_and_ignores_other_files(self, tmp_path: object) -> None:
        import pathlib

        root = pathlib.Path(str(tmp_path))
        (root / "init01").mkdir()
        (root / "init02").mkdir()
        for name in ("f0002.png", "f0000.png", "f0001.png"):
            (root / "init01" / name).write_bytes(b"x")
        (root / "init02" / "f0000.png").write_bytes(b"x")
        (root / "init01" / "notes.txt").write_bytes(b"x")  # must be ignored

        paths = list_frame_paths(str(root))

        assert [os.path.basename(p) for p in paths] == [
            "f0000.png",
            "f0001.png",
            "f0002.png",  # init01, sorted
            "f0000.png",  # init02
        ]
        assert all(p.endswith(".png") for p in paths)

    def test_missing_buffer_raises(self, tmp_path: object) -> None:
        with pytest.raises(ValueError):
            list_frame_paths(os.path.join(str(tmp_path), "does_not_exist"))


class TestDecisiveDims:
    def test_returns_dims_where_clean_and_target_disagree(self) -> None:
        clean = [1, 2, 3, 4, 5, 6, 7]
        target = [1, 9, 3, 8, 5, 6, 0]
        assert decisive_dims(clean, target) == (1, 3, 6)

    def test_identical_teachers_have_no_decisive_dims(self) -> None:
        assert decisive_dims([1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]) == ()

    def test_rejects_wrong_length(self) -> None:
        with pytest.raises(ValueError):
            decisive_dims([1, 2, 3], [1, 2, 3])


class TestForcedFraction:
    def test_fraction_of_decisive_dims_matching_target(self) -> None:
        target = [1, 9, 3, 8, 5, 6, 0]
        executed = [1, 9, 3, 4, 5, 6, 0]  # matches target on dims 1 and 6, not 3
        assert forced_fraction(executed, target, (1, 3, 6)) == pytest.approx(2 / 3)

    def test_full_forcing_is_one(self) -> None:
        target = [1, 9, 3, 8, 5, 6, 0]
        assert forced_fraction(target, target, (1, 3, 6)) == pytest.approx(1.0)

    def test_no_decisive_dims_is_zero(self) -> None:
        assert forced_fraction([1] * 7, [1] * 7, ()) == 0.0


class TestGateDiagram:
    def test_aggregates_forcing_over_decisive_frames_only(self) -> None:
        probes = [
            _probe("f0", 3, 1.0, 0.0),
            _probe("f1", 2, 0.5, 0.5),
            _probe("f2", 0, 0.0, 0.0),  # no decisive dims -> excluded from the aggregate
        ]

        diagram = aggregate_gate_diagram(probes)

        assert diagram.n_frames == 3
        assert diagram.n_decisive_frames == 2
        assert diagram.armed_forcing == pytest.approx(0.75)  # mean(1.0, 0.5)
        assert diagram.dormant_false_fire == pytest.approx(0.25)  # mean(0.0, 0.5)
        assert diagram.gate_margin == pytest.approx(0.5)

    def test_raises_when_no_decisive_frames(self) -> None:
        with pytest.raises(ValueError):
            aggregate_gate_diagram([_probe("f0", 0, 0.0, 0.0)])


class TestGradientSignal:
    def test_large_change_meets_the_heuristic_threshold(self) -> None:
        sig = gradient_gate_signal(
            grad_delta_norm=3.0, grad_norm_without_w=1.0, min_relative_change=1.0
        )
        assert sig["relative_change"] == pytest.approx(3.0)
        assert sig["meets_heuristic_threshold"] is True

    def test_negligible_change_does_not_meet_the_heuristic_threshold(self) -> None:
        sig = gradient_gate_signal(
            grad_delta_norm=0.05, grad_norm_without_w=1.0, min_relative_change=1.0
        )
        assert sig["meets_heuristic_threshold"] is False

    def test_zero_baseline_gradient_does_not_divide_by_zero(self) -> None:
        sig = gradient_gate_signal(grad_delta_norm=0.0, grad_norm_without_w=0.0)
        assert sig["meets_heuristic_threshold"] is False

    def test_boolean_is_not_named_like_a_verdict(self) -> None:
        # The 2026-07-31 probe measured 0.885 against the default knob of 1.0 and emitted
        # `gate_present: false` into a published artifact, while the precommitted gate
        # ("clearly > 0") was a GO. The number is the result; the boolean is a knob readout.
        sig = gradient_gate_signal(grad_delta_norm=0.885, grad_norm_without_w=1.0)
        assert "gate_present" not in sig
        assert sig["relative_change"] == pytest.approx(0.885)
        assert sig["meets_heuristic_threshold"] is False


# --- GPU seam (skipped without PPIP_GPU_TESTS; runs on GPU 1) --------------------------

requires_gpu = pytest.mark.skipif(
    not os.environ.get("PPIP_GPU_TESTS"),
    reason="set PPIP_GPU_TESTS=1 in the GPU rollout env (GPU 1) to run the OpenVLA seam",
)


@pytest.fixture(scope="module")
def loaded_backend():  # type: ignore[no-untyped-def]
    if not os.environ.get("PPIP_GPU_TESTS"):
        pytest.skip("set PPIP_GPU_TESTS=1 in the GPU rollout env (GPU 1)")
    from hijack_backend import HijackBackend

    backend = HijackBackend(
        run_dir=os.path.join(os.path.expanduser("~"), "autoresearch/runs/monitor-stealth/word-gate")
    )
    backend.load_policy_once()
    return backend


@requires_gpu
def test_probe_frame_returns_bounded_gate_measurements(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    import numpy as np
    from word_gate import FIRST_WORD, GateConditions
    from word_gate_probe import PROBE_RECT, TARGET_TASK, USER_TASK, probe_frame

    model, processor, *_ = loaded_backend._policy
    frame = np.zeros((224, 224, 3), dtype=np.uint8)  # smoke frame; real buffer frames in the run
    conditions = GateConditions.make(USER_TASK, FIRST_WORD, 0)

    probe = probe_frame(
        model, processor, frame, conditions=conditions,
        target_task=TARGET_TASK, user_task=USER_TASK, rect=PROBE_RECT,
        lam=1.0, steps=3, lr=3e-2, frame_id="smoke",
    )

    assert 0.0 <= probe.armed_forced_fraction <= 1.0
    assert 0.0 <= probe.dormant_forced_fraction <= 1.0
    assert 0 <= probe.n_decisive <= 7
    assert probe.grad_delta_norm >= 0.0
