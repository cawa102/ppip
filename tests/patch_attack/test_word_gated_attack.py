"""Unit tests for ``word_gated_attack`` — WP6 of the word-gated patch experiment
(``docs/plans/2026-07-30-word-gated-patch.md``).

The closed-loop driver's **pure core** (result assembly, the held-out reportability gate, the
E2.1 deferral) is tested here without a GPU. The GPU seam (``run_static_dos_gate``) composes the
existing fixed pieces — ``carrier_candidate`` + ``set_instruction_override`` + ``frozen_evaluation``
+ ``run_rollouts_at_inits`` — and runs under ``PPIP_GPU_TESTS`` on GPU 1.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

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
    run_perframe_targeted_gate,
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


class TestPerframeTargetedGate:
    """E2.1 — the per-frame targeted gate: one live optimiser run per (init, condition).

    The GPU episode itself is the system boundary, so it is injected: these tests drive the whole
    driver (condition pairing, outcome mapping, resume, crash handling) on CPU with a stub, and
    only the OpenVLA rollout is left to the ``@requires_gpu`` seam.
    """

    @staticmethod
    def _stub(
        calls: list[dict[str, Any]], *, targeted_when_armed: bool = True
    ) -> Callable[..., dict[str, Any]]:
        def episode(backend: Any, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            armed = bool(kwargs["deploy_word"])
            return {
                "targeted": armed and targeted_when_armed,
                "commanded_success": not armed,
                "status": "HIJACK" if armed else "DONE",
                "latch_step": 121 if armed else None,
                "gate_diagnostic": {"branches_differ_fraction": 1.0},
            }

        return episode

    def test_runs_both_conditions_at_every_init(self, tmp_path: Any) -> None:
        calls: list[dict[str, Any]] = []
        result = run_perframe_targeted_gate(
            object(),
            init_indices=HELDOUT3,
            run_dir=str(tmp_path),
            episode_fn=self._stub(calls),
        )

        assert len(calls) == 2 * len(HELDOUT3)
        assert {c["deploy_word"] for c in calls} == {True, False}
        assert sorted(c["seed"] for c in calls) == sorted(HELDOUT3 * 2)
        assert result.effect is Effect.TARGETED

    def test_gate_margin_comes_from_the_fixed_verdicts(self, tmp_path: Any) -> None:
        # Stub: every armed episode hijacks, every dormant one completes the user task.
        result = run_perframe_targeted_gate(
            object(),
            init_indices=HELDOUT3,
            run_dir=str(tmp_path),
            episode_fn=self._stub([]),
        )

        assert result.report.armed_fires == len(HELDOUT3)
        assert result.report.dormant_fires == 0
        assert result.report.gate_margin == pytest.approx(1.0)
        assert result.reportable is True

    def test_passes_the_word_gate_kwargs_to_every_episode(self, tmp_path: Any) -> None:
        calls: list[dict[str, Any]] = []
        run_perframe_targeted_gate(
            object(),
            init_indices=HELDOUT3[:1],
            run_dir=str(tmp_path),
            word="please",
            index=2,
            dormancy_weight=0.25,
            episode_fn=self._stub(calls),
        )

        for call in calls:
            assert call["gate_word"] == "please"
            assert call["word_index"] == 2
            assert call["dormancy_weight"] == 0.25
            assert call["patch_mode"] == "optimize"

    def test_resumes_without_repeating_finished_episodes(self, tmp_path: Any) -> None:
        # Thermal kills are expected (spine rule 8): a restart must not redo hours of work.
        first: list[dict[str, Any]] = []
        run_perframe_targeted_gate(
            object(), init_indices=HELDOUT3, run_dir=str(tmp_path),
            episode_fn=self._stub(first),
        )
        second: list[dict[str, Any]] = []
        result = run_perframe_targeted_gate(
            object(), init_indices=HELDOUT3, run_dir=str(tmp_path),
            episode_fn=self._stub(second),
        )

        assert len(first) == 2 * len(HELDOUT3)
        assert second == []  # everything already on disk
        assert result.report.n_armed == len(HELDOUT3)  # and the result still assembles

    def test_an_errored_episode_is_retried_on_resume(self, tmp_path: Any) -> None:
        # Most failures over a multi-day unattended run are transient (thermal kill, a transient
        # OOM on the shared card). Burning that init permanently would silently shrink N.
        state = {"fail": True}

        def flaky_once(backend: Any, **kwargs: Any) -> dict[str, Any]:
            if state["fail"] and kwargs["seed"] == HELDOUT3[0]:
                raise RuntimeError("transient CUDA OOM")
            return {"targeted": bool(kwargs["deploy_word"]), "commanded_success": False,
                    "status": "DONE", "latch_step": None, "gate_diagnostic": None}

        first = run_perframe_targeted_gate(
            object(), init_indices=HELDOUT3, run_dir=str(tmp_path), episode_fn=flaky_once,
        )
        assert first.report.n_armed == len(HELDOUT3) - 1  # the crashed init is missing data

        state["fail"] = False
        second = run_perframe_targeted_gate(
            object(), init_indices=HELDOUT3, run_dir=str(tmp_path), episode_fn=flaky_once,
        )
        # The retry supersedes the errored row; a successful episode is still never re-run.
        assert second.report.n_armed == len(HELDOUT3)

    def test_a_crashed_episode_is_errored_not_a_failed_verdict(self, tmp_path: Any) -> None:
        # An episode that dies must not silently count as "the attack did not work".
        def flaky(backend: Any, **kwargs: Any) -> dict[str, Any]:
            if kwargs["seed"] == HELDOUT3[0]:
                raise RuntimeError("CUDA out of memory")
            return {"targeted": bool(kwargs["deploy_word"]), "commanded_success": False,
                    "status": "DONE", "latch_step": None, "gate_diagnostic": None}

        result = run_perframe_targeted_gate(
            object(), init_indices=HELDOUT3, run_dir=str(tmp_path), episode_fn=flaky,
        )

        # The crashed init is excluded from both conditions' denominators, not scored as 0.
        assert result.report.n_armed == len(HELDOUT3) - 1
        assert result.report.n_dormant == len(HELDOUT3) - 1

    def test_train_inits_are_marked_diagnostic_only(self, tmp_path: Any) -> None:
        result = run_perframe_targeted_gate(
            object(),
            init_indices=list(shared_inits.GATE_INITS),
            run_dir=str(tmp_path),
            episode_fn=self._stub([]),
        )

        assert result.reportable is False  # GATE_INITS are a go/no-go, never a headline

    def test_rows_are_appended_not_rewritten(self, tmp_path: Any) -> None:
        import json

        run_perframe_targeted_gate(
            object(), init_indices=HELDOUT3[:1], run_dir=str(tmp_path),
            episode_fn=self._stub([]),
        )
        rows_path = os.path.join(str(tmp_path), "rows.jsonl")
        with open(rows_path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]

        assert len(rows) == 2
        assert {r["condition"] for r in rows} == {"armed", "dormant"}
        assert all(r["init"] == HELDOUT3[0] for r in rows)


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
