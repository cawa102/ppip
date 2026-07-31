"""Unit tests for ``word_gate`` — the GPU-free instruction-construction core of the
word-gated patch experiment (``docs/plans/2026-07-30-word-gated-patch.md``).

Pure string manipulation: no model, no GPU. Constructs the two instruction
conditions the attack compares — **dormant** (the operator's plain instruction) and
**armed** (the same instruction with a natural "magic word" inserted).
"""
from __future__ import annotations

import pytest
from word_gate import (
    FIRST_WORD,
    GateConditions,
    GateSetup,
    all_placements,
    armed_instruction,
    assert_trigger_novel,
    dormant_instruction,
    insert_word,
    is_trigger_novel,
    normalize_instruction,
    resolve_gate_setup,
    word_positions,
)

USER = "pick up the alphabet soup and put it in the basket"


class TestNormalize:
    def test_collapses_internal_whitespace(self) -> None:
        assert normalize_instruction("pick   up  the soup") == "pick up the soup"

    def test_strips_leading_and_trailing_whitespace(self) -> None:
        assert normalize_instruction("  pick up the soup  ") == "pick up the soup"

    def test_rejects_blank_instruction(self) -> None:
        with pytest.raises(ValueError):
            normalize_instruction("   ")


class TestWordPositions:
    def test_returns_one_slot_per_gap_including_both_ends(self) -> None:
        # 11 words -> 12 insertion slots (0..11)
        assert word_positions(USER) == tuple(range(12))

    def test_single_word_instruction_has_two_slots(self) -> None:
        assert word_positions("stop") == (0, 1)


class TestInsertWord:
    def test_prepends_at_index_zero(self) -> None:
        assert insert_word("put it down", "carefully", 0) == "carefully put it down"

    def test_appends_at_final_index(self) -> None:
        assert insert_word("put it down", "carefully", 3) == "put it down carefully"

    def test_inserts_in_the_middle(self) -> None:
        assert insert_word("put it down", "carefully", 1) == "put carefully it down"

    def test_preserves_surrounding_word_order(self) -> None:
        out = insert_word(USER, "gently", 4)
        assert out.split() == USER.split()[:4] + ["gently"] + USER.split()[4:]

    def test_normalizes_instruction_before_inserting(self) -> None:
        assert insert_word("put   it down", "x", 1) == "put x it down"

    @pytest.mark.parametrize("bad_index", [-1, 4, 99])
    def test_rejects_out_of_range_index(self, bad_index: int) -> None:
        # "put it down" has 3 words -> valid slots are 0..3
        with pytest.raises(ValueError):
            insert_word("put it down", "x", bad_index)

    @pytest.mark.parametrize("bad_word", ["", "   ", "two words"])
    def test_rejects_blank_or_multiword_trigger(self, bad_word: str) -> None:
        with pytest.raises(ValueError):
            insert_word("put it down", bad_word, 1)


class TestConditions:
    def test_dormant_is_the_normalized_instruction(self) -> None:
        assert dormant_instruction("  pick  up the soup ") == "pick up the soup"

    def test_armed_inserts_the_word(self) -> None:
        assert armed_instruction("put it down", "now", 3) == "put it down now"

    def test_make_conditions_bundles_both_sides(self) -> None:
        c = GateConditions.make("put it down", "now", 3)
        assert c.dormant == "put it down"
        assert c.armed == "put it down now"
        assert c.word == "now"
        assert c.index == 3

    def test_armed_has_exactly_one_more_word_than_dormant(self) -> None:
        c = GateConditions.make(USER, "carefully", 6)
        assert len(c.armed.split()) == len(c.dormant.split()) + 1

    def test_conditions_are_immutable(self) -> None:
        c = GateConditions.make("put it down", "now", 3)
        with pytest.raises(AttributeError):
            c.armed = "hacked"  # type: ignore[misc]


class TestAllPlacements:
    def test_yields_one_condition_per_slot(self) -> None:
        # 3 words -> 4 slots
        assert len(all_placements("put it down", "now")) == 4

    def test_every_placement_is_distinct(self) -> None:
        armed = [p.armed for p in all_placements("put it down", "now")]
        assert len(set(armed)) == len(armed)

    def test_all_placements_share_the_same_dormant(self) -> None:
        placements = all_placements("put it down", "now")
        assert {p.dormant for p in placements} == {"put it down"}


class TestTriggerNovelty:
    def test_first_word_inserts_as_a_single_token(self) -> None:
        # the decided first-milestone word round-trips through insertion
        assert armed_instruction("put it down", FIRST_WORD, 0) == "please put it down"

    def test_first_word_is_novel_against_the_primary_instruction(self) -> None:
        assert is_trigger_novel(FIRST_WORD, USER)

    def test_detects_a_word_already_in_the_instruction(self) -> None:
        assert not is_trigger_novel("soup", USER)

    def test_novelty_check_is_case_insensitive(self) -> None:
        assert not is_trigger_novel("Soup", USER)

    def test_assert_raises_for_a_contaminating_word(self) -> None:
        with pytest.raises(ValueError):
            assert_trigger_novel("basket", USER)

    def test_assert_passes_for_a_novel_word(self) -> None:
        assert_trigger_novel(FIRST_WORD, USER)  # must not raise


class TestResolveGateSetup:
    """WP7: the resolver ``run_confined_episode`` consults to pick single- vs two-branch.

    ``None`` is the behavior-preserving contract — the loop takes its original single-branch
    path unchanged. A non-``None`` setup carries both conditions and which one is deployed.
    """

    def test_none_word_is_the_ungated_default(self) -> None:
        # The default-preserving path: no gate word => None => original single-branch optimiser.
        assert resolve_gate_setup(USER, None, 0, True) is None

    def test_none_word_ignores_the_other_gate_kwargs(self) -> None:
        # Even with a non-default slot / deploy flag, no word still means the ungated default.
        assert resolve_gate_setup(USER, None, 3, False) is None

    def test_gated_setup_builds_both_conditions(self) -> None:
        setup = resolve_gate_setup(USER, "please", 0, True)
        assert isinstance(setup, GateSetup)
        assert setup.dormant == USER
        assert setup.armed == "please " + USER
        assert setup.conditions.word == "please"

    def test_first_word_default_resolves(self) -> None:
        setup = resolve_gate_setup(USER, FIRST_WORD, 0, True)
        assert setup is not None
        assert setup.conditions.word == "please"

    def test_armed_deploy_executes_under_the_worded_instruction(self) -> None:
        setup = resolve_gate_setup(USER, "please", 0, True)
        assert setup is not None
        assert setup.deploy == setup.armed  # operator uttered w -> deploy c⊕w

    def test_dormant_deploy_executes_under_the_word_absent_instruction(self) -> None:
        setup = resolve_gate_setup(USER, "please", 0, False)
        assert setup is not None
        assert setup.deploy == setup.dormant == USER  # word absent -> deploy plain c

    def test_word_slot_moves_only_the_armed_insertion(self) -> None:
        appended = resolve_gate_setup(USER, "please", len(USER.split()), True)  # append slot
        assert appended is not None
        assert appended.armed == USER + " please"
        assert appended.dormant == USER  # dormant is slot-independent

    def test_contaminated_instruction_is_rejected(self) -> None:
        # "soup" already in the instruction => no genuine word-absent dormant baseline.
        with pytest.raises(ValueError):
            resolve_gate_setup(USER, "soup", 0, True)

    def test_out_of_range_slot_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            resolve_gate_setup(USER, "please", 999, True)

    def test_setup_is_immutable(self) -> None:
        setup = resolve_gate_setup(USER, "please", 0, True)
        assert setup is not None
        with pytest.raises(AttributeError):
            setup.deploy_word = False  # type: ignore[misc]
