"""Behavior of the fixed evaluator's candidate validator (integrity boundary)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluator.validation import CandidateValidationError, validate_candidate
from tests.support import make_candidate

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CANDIDATE_FILES = sorted((_REPO_ROOT / "experiments" / "candidates").glob("*.json"))


@pytest.mark.parametrize("candidate_file", _CANDIDATE_FILES, ids=lambda p: p.name)
def test_shipped_example_candidates_validate(candidate_file):
    candidate = json.loads(candidate_file.read_text(encoding="utf-8"))
    validate_candidate(candidate)


def test_valid_candidate_passes():
    validate_candidate(make_candidate())


def test_missing_required_field_is_rejected():
    candidate = make_candidate()
    del candidate["target_task"]
    with pytest.raises(CandidateValidationError):
        validate_candidate(candidate)


def test_position_outside_scene_bounds_is_rejected():
    candidate = make_candidate()
    candidate["placement"]["position"] = [5.0, 0.0, 0.02]  # x far outside workspace
    with pytest.raises(CandidateValidationError):
        validate_candidate(candidate)


def test_unreadable_text_rejected_for_readable_prompt_levels():
    candidate = make_candidate()
    candidate["visual_prompt"]["prompt_level"] = "optimized_typographic_prompt"
    candidate["visual_prompt"]["text"] = "░▒▓█"  # no readable characters
    with pytest.raises(CandidateValidationError):
        validate_candidate(candidate)


def test_hybrid_prompt_object_may_use_non_readable_text():
    # Level-3 (hybrid_prompt_object) is the stretch escape hatch: it is allowed to
    # carry non-readable object/texture cues, unlike the readable level-1/2 prompts.
    candidate = make_candidate()
    candidate["visual_prompt"]["prompt_level"] = "hybrid_prompt_object"
    candidate["visual_prompt"]["text"] = "░▒▓█"
    validate_candidate(candidate)  # must not raise


@pytest.mark.parametrize(
    "override_key",
    ["seeds", "rollouts_per_candidate", "attack_score", "evaluator_config"],
)
def test_evaluator_override_injection_is_rejected(override_key):
    # Integrity boundary: no candidate field may reach the evaluator's budget,
    # seeds, or score. Any injected control key must be rejected.
    candidate = make_candidate()
    candidate[override_key] = "malicious"
    with pytest.raises(CandidateValidationError):
        validate_candidate(candidate)
