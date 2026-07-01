"""Behavior of the append-only run ledger."""

from __future__ import annotations

import pytest

from autoresearch_loop.ledger import (
    LedgerError,
    append_ledger_row,
    read_ledger,
    select_incumbent,
)


def _row(candidate_id: str, score: float, **extra):
    return {"candidate_id": candidate_id, "attack_score": score, **extra}


def test_rows_are_read_back_in_append_order(tmp_path):
    ledger = tmp_path / "ledger.jsonl"

    append_ledger_row(ledger, _row("c1", 0.1))
    append_ledger_row(ledger, _row("c2", 0.4))

    rows = read_ledger(ledger)
    assert [r["candidate_id"] for r in rows] == ["c1", "c2"]
    assert rows[1]["attack_score"] == pytest.approx(0.4)


def test_select_incumbent_returns_highest_scoring_row(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    append_ledger_row(ledger, _row("c1", 0.1))
    append_ledger_row(ledger, _row("c2", 0.4))
    append_ledger_row(ledger, _row("c3", 0.2))

    incumbent = select_incumbent(ledger)

    assert incumbent["candidate_id"] == "c2"


def test_select_incumbent_on_empty_ledger_raises(tmp_path):
    with pytest.raises(LedgerError):
        select_incumbent(tmp_path / "ledger.jsonl")


def test_duplicate_candidate_id_cannot_overwrite_history(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    append_ledger_row(ledger, _row("c1", 0.1))

    with pytest.raises(LedgerError):
        append_ledger_row(ledger, _row("c1", 0.9))  # attempt to re-record c1

    # Original record is untouched.
    rows = read_ledger(ledger)
    assert len(rows) == 1
    assert rows[0]["attack_score"] == pytest.approx(0.1)
