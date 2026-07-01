"""Behavior of record_result: turning a candidate + metrics into a ledger row."""

from __future__ import annotations

import json

import pytest

from autoresearch_loop.ledger import read_ledger
from autoresearch_loop.run_loop import record_result
from tests.support import make_candidate, write_candidate


def _write_metrics(path, **fields):
    path.write_text(json.dumps(fields), encoding="utf-8")
    return path


def test_record_result_appends_row_linking_candidate_and_metrics(tmp_path):
    candidate_path = write_candidate(tmp_path / "candidate_1.json", make_candidate())
    metrics_path = _write_metrics(
        tmp_path / "metrics_human_ppia_001.json",
        candidate_id="human_ppia_001",
        condition="human_ppia",
        valid=True,
        error=None,
        attack_score=0.42,
        targeted_success_rate=0.5,
        commanded_success_rate=0.1,
    )
    ledger = tmp_path / "ledger.jsonl"

    record_result(str(candidate_path), str(metrics_path), str(ledger))

    rows = read_ledger(ledger)
    assert len(rows) == 1
    row = rows[0]
    assert row["candidate_id"] == "human_ppia_001"
    assert row["condition"] == "human_ppia"
    assert row["attack_score"] == pytest.approx(0.42)
    assert row["metrics_path"] == str(metrics_path)
    assert row["candidate_path"] == str(candidate_path)
    assert row["error"] is None
