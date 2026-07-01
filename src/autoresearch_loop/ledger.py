"""The run ledger — an append-only JSONL log of candidate results.

The ledger is the resumable backbone of the search loop: a later session reads
completed rows from `runs/<run_id>/ledger.jsonl` and continues, without any
long-lived process. Rows are immutable once written; the search side appends and
reads but never rewrites history.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LedgerError(ValueError):
    """Raised on an illegal ledger operation (e.g. rewriting an existing row)."""


def append_ledger_row(ledger_path: str | Path, row: dict[str, Any]) -> None:
    """Append one row as a JSON line.

    A candidate_id may be recorded at most once: re-recording is rejected so a
    result can never be silently overwritten.
    """
    path = Path(ledger_path)
    candidate_id = row.get("candidate_id")
    if candidate_id is not None:
        existing = {r.get("candidate_id") for r in read_ledger(path)}
        if candidate_id in existing:
            raise LedgerError(f"candidate_id {candidate_id!r} is already recorded in the ledger")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def read_ledger(ledger_path: str | Path) -> list[dict[str, Any]]:
    """Read all ledger rows in append order (empty list if the ledger is absent)."""
    path = Path(ledger_path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def select_incumbent(ledger_path: str | Path) -> dict[str, Any]:
    """Return the current best row under the official attack_score.

    Ties keep the earliest-recorded candidate (max is stable over append order).
    """
    rows = read_ledger(ledger_path)
    if not rows:
        raise LedgerError("cannot select an incumbent from an empty ledger")
    return max(rows, key=lambda row: row["attack_score"])
