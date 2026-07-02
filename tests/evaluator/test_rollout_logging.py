"""Contract of the per-rollout artifact logger (CPU-pure).

Under `runs/<run_id>/candidates/<candidate_id>/` the evaluator persists the rendered
prompt texture, first-frame agentview renders (which double as presentation figures),
and a `rollouts.jsonl` of per-rollout records. This module only writes files, so it is
fully unit-testable; the backend supplies the frames on the GPU host.
"""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from evaluator.rollout_logging import (
    append_rollout_record,
    candidate_artifact_dir,
    save_prompt_texture,
    save_rollout_frame,
)


def test_artifact_dir_is_scoped_by_candidate(tmp_path):
    a = candidate_artifact_dir(str(tmp_path), "cand_a")
    b = candidate_artifact_dir(str(tmp_path), "cand_b")

    assert a != b
    assert "cand_a" in a and "cand_b" in b


def test_save_prompt_texture_round_trips(tmp_path):
    texture = np.random.default_rng(0).integers(0, 255, (12, 20, 3), dtype=np.uint8)

    path = save_prompt_texture(str(tmp_path), "cand_a", texture)

    loaded = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    assert np.array_equal(loaded, texture)


def test_save_rollout_frame_writes_named_png(tmp_path):
    frame = np.zeros((16, 16, 3), dtype=np.uint8)

    path = save_rollout_frame(str(tmp_path), "cand_a", seed=1, episode=2, frame=frame)

    assert path.endswith(".png")
    assert "seed1" in path and "ep2" in path
    assert Image.open(path).size == (16, 16)


def test_append_rollout_record_writes_one_jsonl_line_per_call(tmp_path):
    r1 = {"seed": 0, "episode_index": 0, "commanded_success": False,
          "targeted_success": True, "prompt_visibility": 0.03, "error": None}
    r2 = {"seed": 0, "episode_index": 1, "commanded_success": True,
          "targeted_success": False, "prompt_visibility": 0.01, "error": None}

    path = append_rollout_record(str(tmp_path), "cand_a", r1)
    append_rollout_record(str(tmp_path), "cand_a", r2)

    from pathlib import Path

    lines = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert lines[0]["targeted_success"] is True
    assert lines[1]["seed"] == 0 and lines[1]["commanded_success"] is True
