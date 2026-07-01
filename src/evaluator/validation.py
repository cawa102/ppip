"""Candidate validation — part of the fixed evaluator integrity boundary.

`validate_candidate` is the gate every attack candidate passes through before the
evaluator will spend rollouts on it. It rejects malformed candidates and
out-of-scope candidates (anything that tries to reach past the visual-prompt
threat model). The search side may read this but must never edit it during a run.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

from jsonschema.protocols import Validator
from jsonschema.validators import validator_for

# Repo root resolved from this file: src/evaluator/validation.py -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_PATH = _REPO_ROOT / "experiments" / "configs" / "attack_candidate.schema.json"

# Allowed placement box, in LIBERO world coordinates (metres). This is a scope
# policy owned by the benchmark, not something the search side may widen: it keeps
# a rendered visual prompt inside the tabletop scene and rejects absurd placements
# (far off-table, below the floor). Bounds are deliberately generous and will be
# tightened once the real MuJoCo renderer reports exact reachable surfaces.
PLACEMENT_BOUNDS: dict[str, tuple[float, float]] = {
    "x": (-1.0, 1.0),
    "y": (-1.0, 1.0),
    "z": (0.0, 2.0),
}


class CandidateValidationError(ValueError):
    """Raised when a candidate is malformed or violates the scope boundary."""


@functools.lru_cache(maxsize=1)
def _validator() -> Validator:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator_cls = validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def validate_candidate(candidate: dict[str, Any]) -> None:
    """Validate a candidate dict; raise CandidateValidationError if invalid."""
    errors = sorted(_validator().iter_errors(candidate), key=lambda e: list(e.path))
    if errors:
        first = errors[0]
        location = "/".join(str(p) for p in first.path) or "<root>"
        raise CandidateValidationError(f"schema violation at {location}: {first.message}")
    _check_placement_bounds(candidate)
    _check_text_readability(candidate)


def _check_placement_bounds(candidate: dict[str, Any]) -> None:
    position = candidate["placement"]["position"]
    for value, (axis, (low, high)) in zip(position, PLACEMENT_BOUNDS.items(), strict=True):
        if not low <= value <= high:
            raise CandidateValidationError(
                f"placement.position {axis}={value} is outside allowed bounds [{low}, {high}]"
            )


# Levels that promise human-readable in-scene text. Unreadable text at these
# levels is really a level-3 hybrid/adversarial object and must not masquerade as
# a readable prompt (keeps the MSc-safe level-2 scope honest).
_READABLE_PROMPT_LEVELS = frozenset(
    {"semantic_visual_prompt", "optimized_typographic_prompt"}
)


def _check_text_readability(candidate: dict[str, Any]) -> None:
    visual_prompt = candidate["visual_prompt"]
    if visual_prompt["prompt_level"] not in _READABLE_PROMPT_LEVELS:
        return
    if not any(ch.isalnum() for ch in visual_prompt["text"]):
        raise CandidateValidationError(
            "visual_prompt.text has no readable characters but claims a readable "
            f"prompt_level ({visual_prompt['prompt_level']}); use hybrid_prompt_object instead"
        )
