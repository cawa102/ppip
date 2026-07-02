"""Persist per-rollout artifacts under `runs/<run_id>/candidates/<candidate_id>/`.

For each evaluated candidate the evaluator writes:
  * `prompt_texture.png` — the rendered label (the attacker's typographic prompt);
  * `seed<k>_ep<j>_first.png` — the first agentview frame with the label in scene,
    which doubles as a presentation figure ("this is what the policy sees");
  * `rollouts.jsonl` — one JSON record per rollout (verdicts, visibility, latch steps).

Part of the fixed evaluator's output side. Pure file I/O, so unit-testable on CPU;
the frames themselves are produced by the rollout backend on the GPU host.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

_ROLLOUTS_FILE = "rollouts.jsonl"
_PROMPT_TEXTURE_FILE = "prompt_texture.png"


def candidate_artifact_dir(run_dir: str, candidate_id: str) -> str:
    """Return (creating if needed) the artifact directory for one candidate."""
    path = os.path.join(run_dir, "candidates", candidate_id)
    os.makedirs(path, exist_ok=True)
    return path


def save_prompt_texture(run_dir: str, candidate_id: str, texture: NDArray[np.uint8]) -> str:
    """Save the rendered prompt label PNG; return its path."""
    path = os.path.join(candidate_artifact_dir(run_dir, candidate_id), _PROMPT_TEXTURE_FILE)
    Image.fromarray(np.ascontiguousarray(texture), mode="RGB").save(path)
    return path


def save_rollout_frame(
    run_dir: str,
    candidate_id: str,
    *,
    seed: int,
    episode: int,
    frame: NDArray[np.uint8],
    kind: str = "first",
) -> str:
    """Save a single agentview frame for one rollout; return its path."""
    name = f"seed{seed}_ep{episode}_{kind}.png"
    path = os.path.join(candidate_artifact_dir(run_dir, candidate_id), name)
    Image.fromarray(np.ascontiguousarray(frame), mode="RGB").save(path)
    return path


def append_rollout_record(run_dir: str, candidate_id: str, record: dict[str, Any]) -> str:
    """Append one per-rollout record to the candidate's `rollouts.jsonl`; return its path."""
    path = os.path.join(candidate_artifact_dir(run_dir, candidate_id), _ROLLOUTS_FILE)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return path
