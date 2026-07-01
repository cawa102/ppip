"""Aggregate candidate/condition results for reporting and metric-gaming checks.

Reads per-candidate metrics records (as emitted by the evaluator) and rolls them
up per condition. It reports raw counts alongside rates so the official score can
always be recomputed from saved metrics, and it tolerates partial/failed runs
(invalid or errored candidates are counted, not dropped).

Only summaries are meant to be tracked in git; heavy per-rollout artifacts stay
ignored (see .gitignore).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from autoresearch_loop.ledger import read_ledger


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _load_record(row: dict[str, Any]) -> dict[str, Any]:
    """Prefer the full metrics file; fall back to the ledger row if it's missing."""
    metrics_path = row.get("metrics_path")
    if metrics_path and Path(metrics_path).exists():
        loaded: dict[str, Any] = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        return loaded
    return row


def aggregate_run(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Aggregate one run's ledger into per-condition summaries."""
    rows = read_ledger(Path(run_dir) / "ledger.jsonl")
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        record = _load_record(row)
        by_condition[record["condition"]].append(record)
    return {condition: aggregate_condition(records) for condition, records in by_condition.items()}


def aggregate_condition(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll up a list of per-candidate metrics into one condition summary."""
    n_candidates = len(records)
    valid = [r for r in records if r.get("valid")]
    n_valid = len(valid)
    n_invalid = n_candidates - n_valid

    best = max(records, key=lambda r: r.get("attack_score", float("-inf")), default=None)
    condition = records[0]["condition"] if records else None

    return {
        "condition": condition,
        "n_candidates": n_candidates,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "invalid_candidate_rate": n_invalid / n_candidates if n_candidates else 0.0,
        "best_attack_score": best["attack_score"] if best else None,
        "best_candidate_id": best["candidate_id"] if best else None,
        "mean_attack_score": _mean([r.get("attack_score", 0.0) for r in records]),
        "total_targeted_successes": sum(r.get("targeted_successes", 0) for r in records),
        "total_commanded_successes": sum(r.get("commanded_successes", 0) for r in records),
        "total_completed_rollouts": sum(r.get("completed_rollouts", 0) for r in records),
        "mean_targeted_success_rate": _mean(
            [r.get("targeted_success_rate", 0.0) for r in valid]
        ),
        "mean_commanded_success_rate": _mean(
            [r.get("commanded_success_rate", 0.0) for r in valid]
        ),
    }
