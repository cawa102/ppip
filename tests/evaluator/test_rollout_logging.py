"""Contract of the per-rollout artifact logger (CPU-pure).

Under `runs/<run_id>/candidates/<candidate_id>/` the evaluator persists the rendered
prompt texture, sampled agentview renders (which double as presentation figures), and
a `rollouts.jsonl` of per-rollout records. This module only writes files, so it is fully
unit-testable; the backend supplies the frames in the GPU rollout environment.
"""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from evaluator.metrics import RolloutOutcome, TargetDiagnostics
from evaluator.rollout_logging import (
    append_rollout_record,
    candidate_artifact_dir,
    rollout_frame_kinds,
    rollout_record_from_outcome,
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


def test_rollout_frame_kinds_samples_first_step20_and_last():
    assert rollout_frame_kinds(0, is_last=False) == ("first",)
    assert rollout_frame_kinds(20, is_last=False) == ("step20",)
    assert rollout_frame_kinds(37, is_last=True) == ("last",)
    assert rollout_frame_kinds(20, is_last=True) == ("step20", "last")


def test_rollout_record_from_outcome_includes_diagnostics_and_frame_paths():
    outcome = RolloutOutcome(
        seed=0,
        episode_index=1,
        commanded_success=False,
        targeted_success=False,
        prompt_visibility=0.02,
        target_diagnostics=TargetDiagnostics(
            target_object="cream_cheese_1",
            target_region="basket_1_contain_region",
            final_target_distance_m=0.42,
            min_target_distance_m=0.18,
            target_object_moved_m=0.20,
            failure_mode="moved_target_but_not_to_region",
        ),
    )

    record = rollout_record_from_outcome(
        outcome,
        latch_step=None,
        geom_name="ppia_prompt__c1",
        frame_paths={"first": "seed0_ep1_first.png", "last": "seed0_ep1_last.png"},
    )

    assert record["target_diagnostics"]["final_target_distance_m"] == 0.42
    assert record["target_diagnostics"]["failure_mode"] == "moved_target_but_not_to_region"
    assert record["frame_paths"]["last"] == "seed0_ep1_last.png"


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
