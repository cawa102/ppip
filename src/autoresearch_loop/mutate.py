"""A programmatic feedback proposer (search side, agent-editable).

``propose_mutation`` is a deterministic, ledger-driven candidate proposer: it reads
the run ledger, selects the current incumbent (best ``attack_score`` so far), and
perturbs it to propose the next candidate. Because it conditions each proposal on the
recorded history of prior results, it exercises the ``loop_with_memory`` feedback path
end to end without a live LLM session.

Pilot stand-in, stated plainly: in pilot-001 this replaces the LLM-in-the-loop
proposer so the four-condition comparison can run unattended. It validates the
feedback *machinery* and equal-budget plumbing, not LLM search quality; the real
LLM-driven ``loop_with_memory`` / ``loop_with_skill`` comparison is a follow-up
interactive run. It never edits evaluator outputs or ledger rows — it only reads them.

The task pair (``user_task``/``target_task``) is treated as fixed and is never mutated,
preserving the controlled comparison. Perturbations stay inside the evaluator's own
placement/style bounds so every proposal remains schema-valid.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

from autoresearch_loop.ledger import LedgerError, read_ledger, select_incumbent
from evaluator.validation import PLACEMENT_BOUNDS

# Keep mutated labels inside a legible size range (the schema allows 0.01-10.0, but a
# readable billboard lives in a much narrower band). Font size is clamped to the schema.
_SCALE_RANGE = (0.4, 1.0)
_FONT_SIZE_RANGE = (12, 60)
_POSITION_JITTER = 0.03
_SCALE_JITTER = 0.1
_FONT_JITTER = 6


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _load_incumbent_candidate(ledger_path: str | Path) -> dict[str, Any] | None:
    """Return the incumbent's candidate dict, or None if it cannot be read.

    Reads the best-scoring row via ``select_incumbent`` and loads the candidate JSON
    it points at. Any gap (empty ledger, missing path, unreadable file) yields None so
    the caller falls back to the seed candidate.
    """
    if not read_ledger(ledger_path):
        return None
    try:
        incumbent = select_incumbent(ledger_path)
    except LedgerError:
        return None
    candidate_path = incumbent.get("candidate_path")
    if not candidate_path or not Path(candidate_path).exists():
        return None
    loaded: dict[str, Any] = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    return loaded


def propose_mutation(
    *,
    ledger_path: str | Path,
    seed_candidate: dict[str, Any],
    index: int,
    condition: str = "loop_with_memory",
    id_prefix: str = "loop_with_memory",
    base_seed: int = 0,
) -> dict[str, Any]:
    """Propose candidate ``index`` by perturbing the ledger incumbent (or the seed).

    Deterministic in ``index`` (the RNG is seeded from ``base_seed`` + ``index``), so a
    resumed run reproduces the same proposal it would have made in one pass. The first
    candidate (or any time no incumbent can be read) is the seed candidate.
    """
    rng = random.Random(base_seed * 100_000 + index)
    parent = None if index == 0 else _load_incumbent_candidate(ledger_path)

    candidate = copy.deepcopy(parent if parent is not None else seed_candidate)
    candidate["candidate_id"] = f"{id_prefix}_{index:02d}"
    candidate["condition"] = condition
    # The task pair is the controlled variable's anchor: never mutate it.
    candidate["user_task"] = seed_candidate["user_task"]
    candidate["target_task"] = seed_candidate["target_task"]

    if parent is None:
        candidate["metadata"] = {
            **candidate.get("metadata", {}),
            "created_by": "programmatic_mutation_proposer",
            "notes": (
                "pilot-001 loop_with_memory stand-in (index 0): seed candidate, no "
                "incumbent yet. Programmatic feedback proposer, not an LLM."
            ),
        }
        return candidate

    _perturb_in_place(candidate, rng)
    parent_id = parent.get("candidate_id", "<unknown>")
    candidate["metadata"] = {
        **candidate.get("metadata", {}),
        "created_by": "programmatic_mutation_proposer",
        "notes": (
            f"pilot-001 loop_with_memory stand-in (index {index}): perturbed the ledger "
            f"incumbent {parent_id!r}. Programmatic feedback proposer, not an LLM."
        ),
    }
    return candidate


def _perturb_in_place(candidate: dict[str, Any], rng: random.Random) -> None:
    """Jitter placement/style around the incumbent, staying inside evaluator bounds."""
    placement = candidate["placement"]
    position = list(placement["position"])
    for axis_index, (_axis, (low, high)) in enumerate(PLACEMENT_BOUNDS.items()):
        jittered = position[axis_index] + rng.uniform(-_POSITION_JITTER, _POSITION_JITTER)
        position[axis_index] = round(_clamp(jittered, low, high), 4)
    placement["position"] = position

    scale = placement.get("scale", 0.6) + rng.uniform(-_SCALE_JITTER, _SCALE_JITTER)
    placement["scale"] = round(_clamp(scale, *_SCALE_RANGE), 3)

    style = candidate["style"]
    font = int(round(style["font_size"] + rng.randint(-_FONT_JITTER, _FONT_JITTER)))
    style["font_size"] = int(_clamp(font, *_FONT_SIZE_RANGE))
