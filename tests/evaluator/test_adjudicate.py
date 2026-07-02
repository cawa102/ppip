"""Tests for target-predicate adjudication (CPU-pure).

`eval_goal_state` decides `targeted_success` by evaluating a target task's
goal-state predicate tuples against a live env's `object_states_dict`, using
LIBERO's own predicate functions. The tests inject fake object-state objects
implementing the minimal predicate interface, so the scientific-integrity-critical
decision logic is verified without building a MuJoCo env.

They still call LIBERO's `eval_predicate_fn`, so are skipped where LIBERO is absent.
"""

from __future__ import annotations

from typing import Any

import pytest

from evaluator.adjudicate import UnevaluableGoalError, eval_goal_state

pytest.importorskip("libero.libero.envs.predicates")


class _FakeRegion:
    """Minimal stand-in for a LIBERO region object-state used by the `in` predicate.

    `In(item, region)` returns `region.check_contact(item) and region.check_contain(item)`,
    so contact and containment are independent inputs; a faithful fake must be able to
    disagree between them. `contain` defaults to `contact` for the common satisfied case.
    """

    def __init__(self, contact: list[Any], contain: list[Any] | None = None) -> None:
        self._contact = contact
        self._contain = contact if contain is None else contain

    def check_contact(self, other: Any) -> bool:
        return other in self._contact

    def check_contain(self, other: Any) -> bool:
        return other in self._contain


def test_single_satisfied_in_predicate_is_true() -> None:
    item = object()
    object_states = {
        "ketchup_1": item,
        "basket_1_contain_region": _FakeRegion([item]),
    }
    goal_state = [["in", "ketchup_1", "basket_1_contain_region"]]

    assert eval_goal_state(goal_state, object_states) is True


def test_single_unsatisfied_predicate_is_false() -> None:
    item = object()
    object_states = {
        "ketchup_1": item,
        "basket_1_contain_region": _FakeRegion([]),  # basket does not contain item
    }
    goal_state = [["in", "ketchup_1", "basket_1_contain_region"]]

    assert eval_goal_state(goal_state, object_states) is False


def test_in_predicate_needs_both_contact_and_containment() -> None:
    item = object()
    goal_state = [["in", "ketchup_1", "basket_1_contain_region"]]

    # Touching the rim but not settled inside -> not "in".
    contact_only = {
        "ketchup_1": item,
        "basket_1_contain_region": _FakeRegion(contact=[item], contain=[]),
    }
    assert eval_goal_state(goal_state, contact_only) is False

    # Inside the containment volume but not registered as contact -> not "in".
    contain_only = {
        "ketchup_1": item,
        "basket_1_contain_region": _FakeRegion(contact=[], contain=[item]),
    }
    assert eval_goal_state(goal_state, contain_only) is False


def test_conjunction_true_only_when_both_predicates_hold() -> None:
    soup, ketchup = object(), object()
    basket_with_both = _FakeRegion([soup, ketchup])
    basket_with_one = _FakeRegion([soup])
    goal_state = [
        ["in", "soup_1", "basket_1_contain_region"],
        ["in", "ketchup_1", "basket_1_contain_region"],
    ]

    both = {
        "soup_1": soup,
        "ketchup_1": ketchup,
        "basket_1_contain_region": basket_with_both,
    }
    one = {
        "soup_1": soup,
        "ketchup_1": ketchup,
        "basket_1_contain_region": basket_with_one,
    }

    assert eval_goal_state(goal_state, both) is True
    assert eval_goal_state(goal_state, one) is False


def test_missing_object_raises_unevaluable() -> None:
    object_states = {"basket_1_contain_region": _FakeRegion([])}
    goal_state = [["in", "ketchup_1", "basket_1_contain_region"]]

    with pytest.raises(UnevaluableGoalError, match="ketchup_1"):
        eval_goal_state(goal_state, object_states)


def test_missing_object_in_later_predicate_still_raises_despite_short_circuit() -> None:
    # First predicate is False (would short-circuit a naive conjunction to False),
    # but the second references an object absent from the scene: the honest verdict
    # is "unevaluable", not a fabricated False. All objects must be validated first.
    soup = object()
    object_states = {
        "soup_1": soup,
        "basket_1_contain_region": _FakeRegion([]),  # soup not in basket -> False
    }
    goal_state = [
        ["in", "soup_1", "basket_1_contain_region"],
        ["in", "ketchup_1", "basket_1_contain_region"],  # ketchup_1 is missing
    ]

    with pytest.raises(UnevaluableGoalError, match="ketchup_1"):
        eval_goal_state(goal_state, object_states)


def test_empty_goal_state_raises_unevaluable() -> None:
    with pytest.raises(UnevaluableGoalError, match="empty"):
        eval_goal_state([], {"basket_1_contain_region": _FakeRegion([])})
