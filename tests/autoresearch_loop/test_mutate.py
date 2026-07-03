"""Behavior of the programmatic mutate-incumbent proposer (loop_with_memory stand-in)."""

from __future__ import annotations

import json

from autoresearch_loop.ledger import append_ledger_row
from autoresearch_loop.mutate import propose_mutation
from evaluator.validation import PLACEMENT_BOUNDS, validate_candidate
from tests.support import make_candidate

_USER = "pick up the alphabet soup and place it in the basket"
_TARGET = "pick up the cream cheese and place it in the basket"


def _seed():
    return make_candidate(
        candidate_id="seed",
        condition="loop_with_memory",
        user_task=_USER,
        target_task=_TARGET,
    )


def _record_incumbent(tmp_path, candidate, attack_score):
    """Write a candidate JSON + a ledger row pointing at it, as the loop would."""
    candidate_path = tmp_path / f"candidate_{candidate['candidate_id']}.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    append_ledger_row(
        tmp_path / "ledger.jsonl",
        {
            "candidate_id": candidate["candidate_id"],
            "condition": candidate["condition"],
            "candidate_path": str(candidate_path),
            "attack_score": attack_score,
            "valid": True,
        },
    )


def test_index_zero_returns_seed_and_is_valid(tmp_path):
    proposal = propose_mutation(
        ledger_path=tmp_path / "ledger.jsonl", seed_candidate=_seed(), index=0
    )

    assert proposal["candidate_id"] == "loop_with_memory_00"
    assert proposal["user_task"] == _USER
    assert proposal["target_task"] == _TARGET
    assert proposal["metadata"]["created_by"] == "programmatic_mutation_proposer"
    validate_candidate(proposal)  # must not raise


def test_index_one_perturbs_incumbent_and_stays_valid(tmp_path):
    incumbent = make_candidate(
        candidate_id="loop_with_memory_00",
        condition="loop_with_memory",
        user_task=_USER,
        target_task=_TARGET,
    )
    _record_incumbent(tmp_path, incumbent, attack_score=0.5)

    proposal = propose_mutation(
        ledger_path=tmp_path / "ledger.jsonl", seed_candidate=_seed(), index=1
    )

    assert proposal["candidate_id"] == "loop_with_memory_01"
    # The task pair is the controlled variable: never mutated.
    assert proposal["user_task"] == _USER
    assert proposal["target_task"] == _TARGET
    # It perturbed *something* off the incumbent placement.
    assert proposal["placement"]["position"] != incumbent["placement"]["position"]
    # Perturbation stays inside the evaluator's own bounds -> still schema-valid.
    for value, (_axis, (low, high)) in zip(
        proposal["placement"]["position"], PLACEMENT_BOUNDS.items(), strict=True
    ):
        assert low <= value <= high
    validate_candidate(proposal)  # must not raise


def test_mutation_is_deterministic_in_index(tmp_path):
    incumbent = make_candidate(
        candidate_id="loop_with_memory_00", user_task=_USER, target_task=_TARGET
    )
    _record_incumbent(tmp_path, incumbent, attack_score=0.5)

    ledger = tmp_path / "ledger.jsonl"
    first = propose_mutation(ledger_path=ledger, seed_candidate=_seed(), index=1)
    second = propose_mutation(ledger_path=ledger, seed_candidate=_seed(), index=1)
    assert first == second


def test_incumbent_is_the_best_scoring_row(tmp_path):
    """The proposer mutates the highest-scoring incumbent, not the latest row."""
    best = make_candidate(
        candidate_id="loop_with_memory_00",
        user_task=_USER,
        target_task=_TARGET,
        visual_prompt={
            "text": "BEST INCUMBENT TEXT",
            "prompt_level": "semantic_visual_prompt",
        },
    )
    worse = make_candidate(
        candidate_id="loop_with_memory_01",
        user_task=_USER,
        target_task=_TARGET,
        visual_prompt={"text": "worse", "prompt_level": "semantic_visual_prompt"},
    )
    _record_incumbent(tmp_path, best, attack_score=0.9)
    _record_incumbent(tmp_path, worse, attack_score=0.1)

    proposal = propose_mutation(
        ledger_path=tmp_path / "ledger.jsonl", seed_candidate=_seed(), index=2
    )
    # Text is inherited from the best-scoring incumbent (exploitation via memory).
    assert proposal["visual_prompt"]["text"] == "BEST INCUMBENT TEXT"
