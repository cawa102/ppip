"""Behavior of loop memory: read-only summaries of prior candidate history."""

from __future__ import annotations

import pytest

from autoresearch_loop.ledger import append_ledger_row
from autoresearch_loop.memory import summarize_history


def _seed_ledger(path):
    append_ledger_row(path, {"candidate_id": "c1", "attack_score": 0.3, "valid": True,
                             "targeted_success_rate": 0.4})
    append_ledger_row(path, {"candidate_id": "c2", "attack_score": 0.5, "valid": True,
                             "targeted_success_rate": 0.6})
    append_ledger_row(path, {"candidate_id": "c3", "attack_score": -0.05, "valid": False,
                             "targeted_success_rate": 0.0})


def test_summary_reports_counts_and_incumbent(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    _seed_ledger(ledger)

    summary = summarize_history(ledger)

    assert summary["n_candidates"] == 3
    assert summary["n_valid"] == 2
    assert summary["n_invalid"] == 1
    assert summary["best_score"] == pytest.approx(0.5)
    assert summary["best_candidate_id"] == "c2"
    assert summary["invalid_candidate_rate"] == pytest.approx(1 / 3)


def test_summary_does_not_modify_the_ledger(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    _seed_ledger(ledger)
    before = ledger.read_text(encoding="utf-8")

    summarize_history(ledger)

    assert ledger.read_text(encoding="utf-8") == before


def test_summary_of_empty_history_is_safe(tmp_path):
    summary = summarize_history(tmp_path / "ledger.jsonl")
    assert summary["n_candidates"] == 0
    assert summary["best_score"] is None
    assert summary["best_candidate_id"] is None
