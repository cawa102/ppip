"""Persist per-rollout artifacts under `runs/<run_id>/candidates/<candidate_id>/`.

For each evaluated candidate the evaluator writes:
  * `prompt_texture.png` — the rendered label (the attacker's typographic prompt);
  * sampled agentview frames such as `seed<k>_ep<j>_first.png`,
    `seed<k>_ep<j>_step20.png`, and `seed<k>_ep<j>_last.png`;
  * `rollouts.jsonl` — one JSON record per rollout (verdicts, visibility, latch
    steps, diagnostic target miss-distance, and sampled frame paths).

Part of the fixed evaluator's output side. Pure file I/O, so unit-testable on CPU;
the frames themselves are produced by the rollout backend in the GPU rollout
environment.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from evaluator.metrics import RolloutOutcome

_ROLLOUTS_FILE = "rollouts.jsonl"
_PROMPT_TEXTURE_FILE = "prompt_texture.png"
DEFAULT_FRAME_SAMPLE_STEPS = (0, 20)


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


def rollout_frame_kinds(
    step_index: int,
    *,
    is_last: bool,
    sample_steps: Sequence[int] = DEFAULT_FRAME_SAMPLE_STEPS,
) -> tuple[str, ...]:
    """Return stable screenshot labels to capture for a rollout step.

    The default dissertation/presentation sample is first policy frame, step 20
    when reached, and the last policy frame. This avoids per-step screenshots while
    still making a specific failure case reproducible.
    """
    if step_index < 0:
        raise ValueError("step_index must be non-negative")

    kinds: list[str] = []
    for sample_step in sample_steps:
        if sample_step < 0:
            raise ValueError("sample_steps must be non-negative")
        if step_index == sample_step:
            kinds.append("first" if sample_step == 0 else f"step{sample_step}")
    if is_last:
        kinds.append("last")
    return tuple(dict.fromkeys(kinds))


def rollout_record_from_outcome(
    outcome: RolloutOutcome,
    *,
    latch_step: int | None,
    geom_name: str | None,
    frame_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the canonical JSONL record for one rollout outcome."""
    record: dict[str, Any] = {
        "seed": outcome.seed,
        "episode_index": outcome.episode_index,
        "commanded_success": outcome.commanded_success,
        "targeted_success": outcome.targeted_success,
        "prompt_visibility": outcome.prompt_visibility,
        "latch_step": latch_step,
        "geom_name": geom_name,
        "error": outcome.error,
    }
    if outcome.target_diagnostics is not None:
        record["target_diagnostics"] = outcome.target_diagnostics.as_record()
    if frame_paths:
        record["frame_paths"] = dict(frame_paths)
    return record


def append_rollout_record(run_dir: str, candidate_id: str, record: dict[str, Any]) -> str:
    """Append one per-rollout record to the candidate's `rollouts.jsonl`; return its path."""
    path = os.path.join(candidate_artifact_dir(run_dir, candidate_id), _ROLLOUTS_FILE)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return path
