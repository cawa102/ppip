"""Candidate generation and writing (search side, agent-editable).

`generate_random_candidate` is the `random_search` baseline: it samples a valid
candidate inside the evaluator's own placement/style bounds, so every candidate
it produces passes the fixed validator. It is deterministic given its RNG, which
keeps the baseline reproducible under a fixed budget.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from evaluator.validation import PLACEMENT_BOUNDS

# Small instruction vocabulary; random candidates draw readable, target-flavoured
# text from the target task plus generic manipulation verbs.
_GENERIC_WORDS = ("put", "place", "move", "grab", "take", "the", "into", "on", "now")

# Deterministic default timestamp; the loop may overwrite with a real one. Kept
# fixed so generation stays reproducible from the RNG alone.
_DEFAULT_CREATED_AT = "1970-01-01T00:00:00Z"


def _random_hex_color(rng: random.Random) -> str:
    return f"#{rng.randint(0, 0xFFFFFF):06x}"


def _random_text(rng: random.Random, target_task: str) -> str:
    pool = list(_GENERIC_WORDS) + target_task.replace("_", " ").split()
    n_words = rng.randint(3, 8)
    words = [rng.choice(pool) for _ in range(n_words)]
    return " ".join(words)[:160]


def generate_random_candidate(
    *,
    rng: random.Random,
    user_task: str,
    target_task: str,
    candidate_id: str,
    condition: str = "random_search",
    created_at: str = _DEFAULT_CREATED_AT,
) -> dict[str, Any]:
    """Sample a schema-valid candidate within the evaluator's allowed bounds."""
    position = [round(rng.uniform(low, high), 4) for low, high in PLACEMENT_BOUNDS.values()]
    rotation = [round(rng.uniform(-3.14159, 3.14159), 4) for _ in range(3)]
    return {
        "candidate_id": candidate_id,
        "condition": condition,
        "user_task": user_task,
        "target_task": target_task,
        "visual_prompt": {
            "text": _random_text(rng, target_task),
            "prompt_level": "semantic_visual_prompt",
            "rationale": "Random-search baseline candidate.",
        },
        "placement": {
            "scene_anchor": "table_surface",
            "position": position,
            "rotation": rotation,
            "scale": round(rng.uniform(0.5, 2.0), 3),
        },
        "style": {
            "foreground_color": _random_hex_color(rng),
            "background_color": _random_hex_color(rng),
            "font_size": rng.randint(4, 96),
            "font_family": "sans-serif",
        },
        "metadata": {
            "created_by": condition,
            "created_at": created_at,
            "notes": "Auto-generated random-search candidate.",
        },
    }


def write_candidate(candidate: dict[str, Any], output_path: str | Path) -> Path:
    """Write a candidate dict to disk as pretty JSON and return the path."""
    path = Path(output_path)
    path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
    return path
