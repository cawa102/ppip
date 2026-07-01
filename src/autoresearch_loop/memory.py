"""Loop memory — read-only summaries of prior candidate history.

`loop_with_memory` uses this to condition the next proposal on what already
failed. It is strictly read-only over the ledger: it never edits evaluator
outputs, metrics, or ledger rows, so using memory can't influence a score.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autoresearch_loop.ledger import read_ledger


def summarize_history(ledger_path: str | Path) -> dict[str, Any]:
    """Summarize prior candidates: counts, incumbent, and invalid rate."""
    rows = read_ledger(ledger_path)
    n_candidates = len(rows)
    valid_rows = [r for r in rows if r.get("valid")]
    n_valid = len(valid_rows)
    n_invalid = n_candidates - n_valid

    best_row = max(rows, key=lambda r: r.get("attack_score", float("-inf")), default=None)
    return {
        "n_candidates": n_candidates,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "best_score": best_row["attack_score"] if best_row else None,
        "best_candidate_id": best_row["candidate_id"] if best_row else None,
        "invalid_candidate_rate": n_invalid / n_candidates if n_candidates else 0.0,
    }
