"""Behavior of results aggregation across candidates and conditions (Task 6)."""

from __future__ import annotations

import json

import pytest
from aggregate_results import aggregate_condition, aggregate_run

from autoresearch_loop.ledger import append_ledger_row


def _metrics(candidate_id, condition, *, valid, attack_score, tsr, csr, ts, cs, completed):
    return {
        "candidate_id": candidate_id,
        "condition": condition,
        "valid": valid,
        "attack_score": attack_score,
        "targeted_success_rate": tsr,
        "commanded_success_rate": csr,
        "targeted_successes": ts,
        "commanded_successes": cs,
        "completed_rollouts": completed,
        "invalid_candidate_rate": 0.0 if valid else 1.0,
    }


def test_aggregate_condition_reports_counts_totals_and_incumbent():
    records = [
        _metrics("c1", "random_search", valid=True, attack_score=0.3, tsr=0.4, csr=0.1,
                 ts=2, cs=1, completed=5),
        _metrics("c2", "random_search", valid=True, attack_score=0.5, tsr=0.6, csr=0.0,
                 ts=3, cs=0, completed=5),
        _metrics("c3", "random_search", valid=False, attack_score=-0.05, tsr=0.0, csr=0.0,
                 ts=0, cs=0, completed=0),
    ]

    summary = aggregate_condition(records)

    assert summary["condition"] == "random_search"
    assert summary["n_candidates"] == 3
    assert summary["n_valid"] == 2
    assert summary["n_invalid"] == 1
    assert summary["invalid_candidate_rate"] == pytest.approx(1 / 3)
    assert summary["best_attack_score"] == pytest.approx(0.5)
    assert summary["best_candidate_id"] == "c2"
    # Raw counts, not only percentages.
    assert summary["total_targeted_successes"] == 5
    assert summary["total_commanded_successes"] == 1
    assert summary["total_completed_rollouts"] == 10
    # Mean rates are over valid candidates only.
    assert summary["mean_targeted_success_rate"] == pytest.approx(0.5)
    assert summary["mean_commanded_success_rate"] == pytest.approx(0.05)


def _write_metrics_file(run_dir, record):
    path = run_dir / f"metrics_{record['candidate_id']}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_aggregate_run_groups_by_condition_and_tolerates_missing_files(tmp_path):
    ledger = tmp_path / "ledger.jsonl"

    r1 = _metrics("c1", "random_search", valid=True, attack_score=0.3, tsr=0.4, csr=0.1,
                  ts=2, cs=1, completed=5)
    r2 = _metrics("h1", "human_ppia", valid=True, attack_score=0.7, tsr=0.8, csr=0.0,
                  ts=4, cs=0, completed=5)
    for record in (r1, r2):
        metrics_path = _write_metrics_file(tmp_path, record)
        append_ledger_row(ledger, {"candidate_id": record["candidate_id"],
                                   "condition": record["condition"],
                                   "metrics_path": str(metrics_path),
                                   "attack_score": record["attack_score"],
                                   "valid": True})
    # A third candidate recorded in the ledger but whose metrics file is missing
    # (a partial/failed run) must not crash aggregation.
    append_ledger_row(ledger, {"candidate_id": "c2", "condition": "random_search",
                               "metrics_path": str(tmp_path / "metrics_c2.json"),
                               "attack_score": -0.05, "valid": False})

    by_condition = aggregate_run(str(tmp_path))

    assert set(by_condition) == {"random_search", "human_ppia"}
    assert by_condition["random_search"]["n_candidates"] == 2
    assert by_condition["random_search"]["n_invalid"] == 1
    assert by_condition["human_ppia"]["best_attack_score"] == pytest.approx(0.7)
    # Raw counts survive from the loaded metrics files.
    assert by_condition["random_search"]["total_targeted_successes"] == 2
