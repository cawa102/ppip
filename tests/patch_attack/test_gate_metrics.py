"""Unit tests for ``gate_metrics`` — WP2 of the word-gated patch experiment
(``docs/plans/2026-07-30-word-gated-patch.md``).

Pure aggregation over the **fixed** evaluator's ``RolloutOutcome`` objects. It derives
the gate metrics — ``gate_margin`` / ``false_fire_rate`` / ``armed_forcing_fraction`` —
that quantify whether the patch's effect is switched on by the magic word, reusing the
evaluator's own ``summarize_rollouts`` so it never redefines how success is judged
(the fixed-evaluator invariant stays intact; these are *derived* numbers, not a new score).
"""
from __future__ import annotations

import pytest
from gate_metrics import Effect, gate_report

from evaluator.metrics import RolloutOutcome


def _outcomes(*, targeted: list[bool], commanded: list[bool]) -> list[RolloutOutcome]:
    """Build a condition's rollout verdicts from parallel targeted/commanded flags."""
    return [
        RolloutOutcome(seed=0, episode_index=i, commanded_success=c, targeted_success=t)
        for i, (t, c) in enumerate(zip(targeted, commanded, strict=True))
    ]


class TestTargetedGate:
    def test_margin_is_armed_minus_dormant_targeted_rate(self) -> None:
        # armed (c⊕w): 3/4 execute the attacker target; dormant (c): 0/4 (patch inert).
        armed = _outcomes(
            targeted=[True, True, True, False], commanded=[False, False, False, True]
        )
        dormant = _outcomes(
            targeted=[False, False, False, False], commanded=[True, True, True, True]
        )

        report = gate_report(armed, dormant, Effect.TARGETED)

        assert report.armed_forcing_fraction == pytest.approx(0.75)
        assert report.false_fire_rate == pytest.approx(0.0)
        assert report.gate_margin == pytest.approx(0.75)


class TestRawCounts:
    def test_targeted_report_exposes_fire_counts_and_denominators(self) -> None:
        armed = _outcomes(
            targeted=[True, True, True, False], commanded=[False, False, False, True]
        )
        dormant = _outcomes(
            targeted=[False, False, False, False], commanded=[True, True, True, True]
        )

        report = gate_report(armed, dormant, Effect.TARGETED)

        assert (report.n_armed, report.armed_fires) == (4, 3)
        assert (report.n_dormant, report.dormant_fires) == (4, 0)

    def test_dos_fire_counts_are_denials(self) -> None:
        armed = _outcomes(targeted=[False] * 4, commanded=[False, False, True, False])
        dormant = _outcomes(targeted=[False] * 4, commanded=[True, True, True, True])

        report = gate_report(armed, dormant, Effect.DOS)

        assert report.armed_fires == 3  # 3 of 4 rollouts denied the task
        assert report.dormant_fires == 0


class TestMarginInterpretation:
    """Property tests pinning what the margin *means* — the Phase-0 GO / NO-GO reading."""

    def test_perfect_clean_gate_has_margin_one(self) -> None:
        armed = _outcomes(targeted=[True] * 4, commanded=[False] * 4)
        dormant = _outcomes(targeted=[False] * 4, commanded=[True] * 4)

        assert gate_report(armed, dormant, Effect.TARGETED).gate_margin == pytest.approx(1.0)

    def test_always_on_patch_has_nonpositive_margin(self) -> None:
        # Feasibility-hinge (R2) failure: the effect fires with AND without the word, so
        # there is no gate. A margin of 0 or below is the NO-GO signal ("not gateable in
        # this config"), never a hijack — this reading must not silently regress.
        armed = _outcomes(targeted=[True, True, False, False], commanded=[False] * 4)
        dormant = _outcomes(targeted=[True] * 4, commanded=[False] * 4)

        assert gate_report(armed, dormant, Effect.TARGETED).gate_margin <= 0.0


class TestEmptyConditionGuard:
    def test_raises_when_a_condition_has_no_rollouts(self) -> None:
        armed = _outcomes(targeted=[True, True], commanded=[False, False])
        with pytest.raises(ValueError):
            gate_report(armed, [], Effect.TARGETED)

    def test_raises_when_every_rollout_errored(self) -> None:
        armed = _outcomes(targeted=[True, True], commanded=[False, False])
        all_errored = [
            RolloutOutcome(
                seed=0,
                episode_index=0,
                commanded_success=False,
                targeted_success=False,
                error="boom",
            )
        ]
        with pytest.raises(ValueError):
            gate_report(all_errored, armed, Effect.DOS)


class TestErroredRolloutsExcluded:
    def test_crashed_rollout_is_not_counted_as_a_dos_denial(self) -> None:
        # A crashed rollout has commanded_success=False, but it is NOT a task *denial*:
        # the fixed evaluator drops it from the denominator, so the gate must too. A naive
        # "not commanded_success" count would inflate the false-fire rate — the make-or-break
        # metric for Exp 1 — so this correctness property is pinned.
        dormant = [
            RolloutOutcome(seed=0, episode_index=0, commanded_success=True, targeted_success=False),
            RolloutOutcome(seed=0, episode_index=1, commanded_success=True, targeted_success=False),
            RolloutOutcome(
                seed=0,
                episode_index=2,
                commanded_success=False,
                targeted_success=False,
                error="sim crashed",
            ),
        ]
        armed = _outcomes(targeted=[False, False], commanded=[False, False])

        report = gate_report(armed, dormant, Effect.DOS)

        assert report.n_dormant == 2  # the crashed rollout is excluded
        assert report.dormant_fires == 0  # 0 genuine denials, not 1
        assert report.false_fire_rate == pytest.approx(0.0)


class TestDosGate:
    def test_margin_is_dormant_minus_armed_commanded_rate(self) -> None:
        # DoS effect = the user task is DENIED. armed (c⊕w): 3/4 denied; dormant (c):
        # 0/4 denied (task completes normally without the word).
        armed = _outcomes(
            targeted=[False, False, False, False], commanded=[False, False, True, False]
        )
        dormant = _outcomes(
            targeted=[False, False, False, False], commanded=[True, True, True, True]
        )

        report = gate_report(armed, dormant, Effect.DOS)

        assert report.armed_forcing_fraction == pytest.approx(0.75)
        assert report.false_fire_rate == pytest.approx(0.0)
        assert report.gate_margin == pytest.approx(0.75)
