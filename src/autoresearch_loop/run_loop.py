"""The autoresearch-style search loop (search side, agent-editable).

This is the `train.py`-like side of the project: it proposes candidates, schedules
evaluation by the fixed evaluator, records results, and picks the next candidate.
It is deliberately resumable through the ledger and never keeps a long-lived
session open while a rollout job runs. There is no 5-minute per-iteration cap; the
iteration unit is a candidate-evaluation job.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from autoresearch_loop.candidate_writer import write_candidate
from autoresearch_loop.ledger import append_ledger_row, read_ledger
from evaluator.backends import RolloutBackend
from evaluator.eval_attack import evaluate_candidate

# Metric fields copied from the metrics file into the ledger row for quick
# analysis and incumbent selection (the full record still lives in metrics_path).
_SUMMARY_FIELDS = (
    "attack_score",
    "valid",
    "error",
    "targeted_success_rate",
    "commanded_success_rate",
    "invalid_candidate_rate",
)


def run_search_condition(
    *,
    budget: dict[str, Any],
    run_dir: str,
    backend: RolloutBackend,
    propose: Callable[[int], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run one search condition until the candidate budget is reached.

    Resumable: candidates already recorded in the ledger are skipped, so a run
    that stopped mid-way (e.g. after a long rollout job) continues from where the
    ledger left off instead of restarting. `propose(index)` supplies the next
    candidate, keeping the loop provider-agnostic (random / LLM / manual).
    """
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    ledger_path = run_path / "ledger.jsonl"

    already_done = len(read_ledger(ledger_path))
    max_candidates = budget["max_candidates_per_condition"]

    results: list[dict[str, Any]] = []
    for index in range(already_done, max_candidates):
        candidate = propose(index)
        candidate_id = candidate["candidate_id"]
        candidate_path = write_candidate(
            candidate, run_path / f"candidate_{candidate_id}.json"
        )
        metrics = evaluate_candidate(
            str(candidate_path), str(run_path), backend=backend, budget=budget
        )
        metrics_path = run_path / f"metrics_{candidate_id}.json"
        record_result(str(candidate_path), str(metrics_path), str(ledger_path))
        results.append(metrics)
    return results


def record_result(candidate_path: str, metrics_path: str, ledger_path: str) -> None:
    """Append an immutable ledger row linking a candidate to its metrics."""
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    row: dict[str, Any] = {
        "candidate_id": metrics["candidate_id"],
        "condition": metrics.get("condition"),
        "candidate_path": candidate_path,
        "metrics_path": metrics_path,
    }
    for field in _SUMMARY_FIELDS:
        if field in metrics:
            row[field] = metrics[field]
    append_ledger_row(ledger_path, row)
