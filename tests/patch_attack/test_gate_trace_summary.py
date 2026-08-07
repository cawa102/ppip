"""Tests for the same-frame gate-trace aggregator (`gate_trace_summary`).

The aggregator turns the per-step ``gate`` slots that `ce_monitor_patch_attack` already writes
into the paper's mechanism number: how often ONE patch satisfies BOTH word conditions at once.
It is a derived read of immutable artifacts -- it must never re-judge a rollout, so every test
here works on hand-built step records rather than on a live model.
"""
from __future__ import annotations

import json

import pytest
from gate_trace_summary import (
    ACTION_DIM,
    gate_records,
    summarize_gate_steps,
    summarize_trace_files,
)


def step(
    *,
    armed_match: int = ACTION_DIM,
    dormant_match: int = ACTION_DIM,
    armed_forced: float = 1.0,
    dormant_forced: float = 0.0,
    branches_differ: bool = True,
    n_decisive_dims: int = 3,
) -> dict[str, object]:
    """One trace row shaped like `ce_monitor_patch_attack`'s per-step record."""
    return {
        "step": 0,
        "n_decisive_dims": n_decisive_dims,
        "gate": {
            "armed_tokens": [0] * ACTION_DIM,
            "dormant_tokens": [0] * ACTION_DIM,
            "armed_match": armed_match,
            "dormant_match": dormant_match,
            "armed_forced": armed_forced,
            "dormant_forced": dormant_forced,
            "branches_differ": branches_differ,
        },
    }


class TestGateRecords:
    def test_extracts_only_rows_carrying_a_gate_slot(self) -> None:
        # Arrange -- the ungated path writes `gate: null`, which must not become a data point.
        trace = [step(), {"step": 1, "n_decisive_dims": 2, "gate": None}, step()]

        # Act
        records = gate_records(trace)

        # Assert
        assert len(records) == 2

    def test_carries_the_decisive_dim_count_from_the_enclosing_row(self) -> None:
        # Arrange -- `n_decisive_dims` lives on the step, not inside the gate slot.
        trace = [step(n_decisive_dims=5)]

        # Act
        records = gate_records(trace)

        # Assert
        assert records[0]["n_decisive_dims"] == 5

    def test_empty_trace_yields_no_records(self) -> None:
        assert gate_records([]) == []


class TestSummarizeGateSteps:
    def test_all_perfect_steps_report_full_simultaneous_satisfaction(self) -> None:
        # Arrange -- the ideal gate: target under c+w, clean under c, on the same pixels.
        records = gate_records([step(), step(), step()])

        # Act
        summary = summarize_gate_steps(records)

        # Assert
        assert summary.n_steps == 3
        assert summary.both_perfect_fraction == 1.0
        assert summary.mean_armed_match == float(ACTION_DIM)
        assert summary.mean_dormant_match == float(ACTION_DIM)

    def test_both_perfect_requires_both_branches_not_just_one(self) -> None:
        # Arrange -- a step where the armed branch is perfect but the dormant one is not.
        records = gate_records([step(), step(dormant_match=6)])

        # Act
        summary = summarize_gate_steps(records)

        # Assert -- this is the whole point: a one-sided win is not a gate.
        assert summary.armed_perfect_fraction == 1.0
        assert summary.dormant_perfect_fraction == 0.5
        assert summary.both_perfect_fraction == 0.5

    def test_means_are_over_steps_not_over_episodes(self) -> None:
        # Arrange
        records = gate_records([step(armed_match=7), step(armed_match=5)])

        # Act
        summary = summarize_gate_steps(records)

        # Assert
        assert summary.mean_armed_match == pytest.approx(6.0)

    def test_forced_fractions_and_branch_divergence_are_averaged(self) -> None:
        # Arrange
        records = gate_records([
            step(armed_forced=1.0, dormant_forced=0.0, branches_differ=True),
            step(armed_forced=0.5, dormant_forced=0.5, branches_differ=False),
        ])

        # Act
        summary = summarize_gate_steps(records)

        # Assert
        assert summary.mean_armed_forced == pytest.approx(0.75)
        assert summary.mean_dormant_forced == pytest.approx(0.25)
        assert summary.branches_differ_fraction == pytest.approx(0.5)

    def test_decisive_dim_histogram_counts_every_step(self) -> None:
        # Arrange -- steps with zero decisive dims are real and must stay visible in the
        # histogram (they are excluded from *forcing*, not from the record).
        records = gate_records([
            step(n_decisive_dims=0), step(n_decisive_dims=3), step(n_decisive_dims=3)
        ])

        # Act
        summary = summarize_gate_steps(records)

        # Assert
        assert summary.decisive_dim_histogram == {0: 1, 3: 2}

    def test_empty_input_raises_rather_than_reporting_a_vacuous_zero(self) -> None:
        # A summary over no steps would print 0.0 everywhere and read as a null result.
        with pytest.raises(ValueError, match="no gate-bearing steps"):
            summarize_gate_steps([])

    @pytest.mark.parametrize("bad", [-1, ACTION_DIM + 1])
    def test_out_of_range_match_counts_raise(self, bad: int) -> None:
        records = gate_records([step(armed_match=bad)])
        with pytest.raises(ValueError, match="armed_match"):
            summarize_gate_steps(records)

    def test_out_of_range_dormant_match_raises(self) -> None:
        records = gate_records([step(dormant_match=9)])
        with pytest.raises(ValueError, match="dormant_match"):
            summarize_gate_steps(records)


class TestSummarizeTraceFiles:
    def test_aggregates_across_files_and_reports_provenance(self, tmp_path) -> None:
        # Arrange -- two "episodes", one perfect and one with a single imperfect armed step.
        a = tmp_path / "trace_armed_init01_trial0.json"
        b = tmp_path / "trace_dormant_init01_trial0.json"
        a.write_text(json.dumps([step(), step()]), encoding="utf-8")
        b.write_text(json.dumps([step(), step(armed_match=6)]), encoding="utf-8")

        # Act
        report = summarize_trace_files([str(a), str(b)])

        # Assert
        assert report["n_traces"] == 2
        assert report["summary"]["n_steps"] == 4
        assert report["summary"]["both_perfect_fraction"] == pytest.approx(0.75)
        assert sorted(report["traces"]) == sorted([a.name, b.name])

    def test_missing_gate_slots_across_all_files_raises(self, tmp_path) -> None:
        # An ungated run must fail loudly here, not summarise into an empty table.
        p = tmp_path / "trace_ungated.json"
        p.write_text(
            json.dumps([{"step": 0, "n_decisive_dims": 2, "gate": None}]), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="no gate-bearing steps"):
            summarize_trace_files([str(p)])

    def test_no_trace_files_raises(self) -> None:
        with pytest.raises(ValueError, match="no trace files"):
            summarize_trace_files([])
