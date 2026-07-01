"""The fixed attack evaluator — the integrity boundary the search side may not edit.

`evaluate_candidate` is the whole candidate lifecycle for one candidate:
validate -> run rollouts (via an injected backend) -> summarize -> score ->
write an immutable metrics file. The OpenVLA rollout is injected as a backend so
the orchestration is testable on CPU; the real backend runs on the GPU host.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluator.backends import RolloutBackend
from evaluator.metrics import compute_attack_score, summarize_rollouts
from evaluator.validation import CandidateValidationError, validate_candidate

__all__ = ["evaluate_candidate", "validate_candidate", "CandidateValidationError"]


def evaluate_candidate(
    candidate_path: str,
    run_dir: str,
    *,
    backend: RolloutBackend,
    budget: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one candidate and write its metrics record; return the metrics dict."""
    candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))

    try:
        validate_candidate(candidate)
    except CandidateValidationError as exc:
        return _emit_invalid(run_dir, candidate, str(exc))

    seeds = budget["seeds"]
    rollouts_per_candidate = budget["rollouts_per_candidate"]
    error: str | None = None
    try:
        outcomes = backend.run_rollouts(
            candidate=candidate,
            seeds=seeds,
            rollouts_per_candidate=rollouts_per_candidate,
        )
        summary = summarize_rollouts(outcomes)
    except Exception as exc:  # noqa: BLE001 - one bad rollout must not kill a batch
        error = f"rollout failed: {exc}"
        summary = _errored_summary(len(seeds) * rollouts_per_candidate)

    metrics: dict[str, Any] = {
        "candidate_id": candidate["candidate_id"],
        "condition": candidate["condition"],
        "valid": True,
        "error": error,
        **summary,
        "invalid_candidate_rate": 0.0,
    }
    metrics["attack_score"] = compute_attack_score(metrics)

    _write_metrics(run_dir, candidate["candidate_id"], metrics)
    return metrics


def _errored_summary(expected_rollouts: int) -> dict[str, Any]:
    """A rollout summary for the case where the whole rollout call crashed."""
    return {
        "rollout_count": expected_rollouts,
        "completed_rollouts": 0,
        "errored_rollouts": expected_rollouts,
        "commanded_successes": 0,
        "targeted_successes": 0,
        "commanded_success_rate": 0.0,
        "targeted_success_rate": 0.0,
    }


def _emit_invalid(run_dir: str, candidate: dict[str, Any], error: str) -> dict[str, Any]:
    """Emit a zeroed, penalized metrics record for a rejected candidate."""
    candidate_id = candidate.get("candidate_id", "unknown")
    metrics: dict[str, Any] = {
        "candidate_id": candidate_id,
        "condition": candidate.get("condition"),
        "valid": False,
        "error": error,
        "rollout_count": 0,
        "completed_rollouts": 0,
        "errored_rollouts": 0,
        "commanded_successes": 0,
        "targeted_successes": 0,
        "commanded_success_rate": 0.0,
        "targeted_success_rate": 0.0,
        "invalid_candidate_rate": 1.0,
    }
    metrics["attack_score"] = compute_attack_score(metrics)
    _write_metrics(run_dir, candidate_id, metrics)
    return metrics


def _write_metrics(run_dir: str, candidate_id: str, metrics: dict[str, Any]) -> None:
    out_path = Path(run_dir) / f"metrics_{candidate_id}.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
