"""Human-readable experiment log (search side), ported from ``karpathy/autoresearch``.

``results.tsv`` mirrors autoresearch's per-experiment log onto this project: one row per
candidate evaluation, tab-separated (commas break free-text descriptions), with a
``keep``/``discard``/``crash`` status. It is a convenience view for the interactive loop;
the immutable record of record is still ``runs/<run>/ledger.jsonl``. Writing here only ever
appends a human-readable mirror row — it never reads or edits evaluator outputs, so using it
cannot influence a score.
"""

from __future__ import annotations

from pathlib import Path

# Tab-separated columns. Mirrors the reference results.tsv
# (commit/val_bpb/memory_gb/status/description) adapted to the attack_score objective.
RESULTS_HEADER: tuple[str, ...] = (
    "candidate_id",
    "attack_score",
    "targeted",
    "commanded",
    "completed",
    "visibility",
    "memory_gb",
    "status",
    "description",
)

# The three experiment outcomes, faithful to autoresearch's keep/discard/crash.
KEEP = "keep"
DISCARD = "discard"
CRASH = "crash"


def init_results_tsv(path: str | Path) -> Path:
    """Create the results log with only the header row, if it does not already exist.

    Idempotent: an existing log (with rows) is left untouched so re-entering the loop
    never truncates prior experiments.
    """
    tsv_path = Path(path)
    if not tsv_path.exists():
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        tsv_path.write_text("\t".join(RESULTS_HEADER) + "\n", encoding="utf-8")
    return tsv_path


def decide_status(
    *,
    valid: bool,
    completed_rollouts: int,
    attack_score: float,
    incumbent_score: float | None,
) -> str:
    """Classify an experiment as keep / discard / crash (autoresearch semantics).

    ``crash``: the candidate was rejected (invalid) or produced no verdict (every rollout
    errored / none ran). ``keep``: it improved on the incumbent's ``attack_score`` (or is
    the first recorded result). ``discard``: it tied or did worse.
    """
    if not valid or completed_rollouts == 0:
        return CRASH
    if incumbent_score is None or attack_score > incumbent_score:
        return KEEP
    return DISCARD


def append_results_row(
    path: str | Path,
    *,
    candidate_id: str,
    attack_score: float,
    targeted: int,
    commanded: int,
    completed: int,
    visibility: float | None,
    memory_gb: float | None,
    status: str,
    description: str,
) -> None:
    """Append one tab-separated experiment row to the results log."""
    cells = (
        candidate_id,
        _fmt(attack_score, 4),
        str(targeted),
        str(commanded),
        str(completed),
        _fmt(visibility, 4),
        _fmt(memory_gb, 1),
        status,
        _sanitize(description),
    )
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write("\t".join(cells) + "\n")


def _fmt(value: float | None, digits: int) -> str:
    """Fixed-precision float, or ``-`` for an absent (None) diagnostic."""
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _sanitize(text: str) -> str:
    """Flatten a free-text description so tabs/newlines cannot break the TSV row."""
    return text.replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()
