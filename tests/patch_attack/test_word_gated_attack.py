"""Unit tests for ``word_gated_attack`` — WP6 of the word-gated patch experiment
(``docs/plans/2026-07-30-word-gated-patch.md``).

The closed-loop driver's **pure core** (result assembly, the held-out reportability gate, the
E2.1 deferral) is tested here without a GPU. The GPU seam (``run_static_dos_gate``) composes the
existing fixed pieces — ``carrier_candidate`` + ``set_instruction_override`` + ``frozen_evaluation``
+ ``run_rollouts_at_inits`` — and runs under ``PPIP_GPU_TESTS`` on GPU 1.
"""
from __future__ import annotations

import os
from collections.abc import Sequence

import pytest
import shared_inits
from gate_metrics import Effect
from word_gate import GateConditions
from word_gated_attack import (
    TARGET_TASK,
    USER_TASK,
    WordGateResult,
    assemble_word_gate_result,
    reportable_inits,
)

from evaluator.metrics import RolloutOutcome


def _outcomes(
    *, targeted: list[bool], commanded: list[bool], inits: Sequence[int]
) -> list[RolloutOutcome]:
    return [
        RolloutOutcome(seed=inits[i], episode_index=i, commanded_success=c, targeted_success=t)
        for i, (t, c) in enumerate(zip(targeted, commanded, strict=True))
    ]


HELDOUT3 = list(shared_inits.HELDOUT_INITS[:3])


class TestAssembleResult:
    def test_dos_result_carries_the_gate_report_and_conditions(self) -> None:
        conditions = GateConditions.make(USER_TASK, "please", 0)
        armed = _outcomes(
            targeted=[False, False, False], commanded=[False, False, True], inits=HELDOUT3
        )  # 2/3 denied with the word
        dormant = _outcomes(
            targeted=[False, False, False], commanded=[True, True, True], inits=HELDOUT3
        )  # 0 denied without it

        result = assemble_word_gate_result(
            conditions=conditions,
            effect=Effect.DOS,
            target_task=TARGET_TASK,
            init_indices=HELDOUT3,
            armed_outcomes=armed,
            dormant_outcomes=dormant,
        )

        assert isinstance(result, WordGateResult)
        assert result.effect is Effect.DOS
        assert result.conditions.armed == conditions.armed
        assert result.report.effect is Effect.DOS
        assert result.report.gate_margin == pytest.approx(2 / 3)  # DoS: dormant−armed commanded
        assert result.reportable is True  # all inits are held-out


class TestReportability:
    def test_all_heldout_inits_are_reportable(self) -> None:
        assert reportable_inits(HELDOUT3) is True

    def test_a_train_init_makes_the_run_diagnostic_only(self) -> None:
        assert reportable_inits([*HELDOUT3, shared_inits.TRAIN_INITS[0]]) is False

    def test_the_contaminated_legacy_gate_init_is_not_reportable(self) -> None:
        assert reportable_inits([shared_inits.LEGACY_GATE_INIT]) is False


class TestPerframeDeferral:
    def test_perframe_targeted_gate_refuses_until_wp7(self) -> None:
        from word_gated_attack import run_perframe_targeted_gate

        with pytest.raises(NotImplementedError, match="WP7"):
            run_perframe_targeted_gate()


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
def test_run_static_dos_gate_scores_both_conditions(loaded_backend) -> None:  # type: ignore[no-untyped-def]
    import numpy as np
    from word_gated_attack import run_static_dos_gate

    patch = np.full((64, 64, 3), 0.5, dtype=np.float32)  # grey BL 64×64 smoke patch
    init = [shared_inits.HELDOUT_INITS[0]]

    result = run_static_dos_gate(
        loaded_backend, patch=patch, init_indices=init, top_left=(160, 0)
    )

    assert result.effect.value == "dos"
    assert result.report.n_armed >= 1
    assert result.report.n_dormant >= 1
    assert result.reportable is True
