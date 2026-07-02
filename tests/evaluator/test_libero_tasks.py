"""Tests for candidate-task-string -> libero_object task resolution.

These exercise the real LIBERO benchmark registry (a system boundary we wrap),
so they are skipped on hosts where LIBERO is not importable. On the GPU host run
with `PYTHONPATH=$HOME/LIBERO`.
"""

from __future__ import annotations

import pytest

from evaluator.libero_tasks import (
    ResolvedTask,
    TaskResolutionError,
    parse_goal_state,
    resolve_task,
)

# Resolution genuinely loads the LIBERO suite; skip where the benchmark is absent.
pytest.importorskip("libero.libero.benchmark")


def test_resolves_language_string_to_task_id_and_goal_state() -> None:
    resolved = resolve_task("pick up the bbq sauce and place it in the basket")

    assert isinstance(resolved, ResolvedTask)
    assert resolved.task_id == 3
    assert resolved.name == "pick_up_the_bbq_sauce_and_place_it_in_the_basket"
    assert resolved.language == "pick up the bbq sauce and place it in the basket"
    assert resolved.goal_state == [["in", "bbq_sauce_1", "basket_1_contain_region"]]
    assert resolved.bddl_path.endswith(
        "pick_up_the_bbq_sauce_and_place_it_in_the_basket.bddl"
    )


def test_parse_goal_state_returns_predicate_tuples() -> None:
    resolved = resolve_task("pick up the ketchup and place it in the basket")

    goal_state = parse_goal_state(resolved.bddl_path)

    assert goal_state == [["in", "ketchup_1", "basket_1_contain_region"]]


def test_resolves_despite_whitespace_and_case() -> None:
    resolved = resolve_task("  Pick Up  the BBQ Sauce  and place it in the basket ")

    assert resolved.task_id == 3


def test_unknown_task_string_raises() -> None:
    with pytest.raises(TaskResolutionError):
        resolve_task("open the microwave and heat the burrito")


class _FakeTask:
    def __init__(self, name: str, language: str) -> None:
        self.name = name
        self.language = language


class _FakeSuite:
    """A suite with two tasks sharing a language, to drive the ambiguity guard."""

    n_tasks = 2

    def get_task(self, i: int) -> _FakeTask:
        return _FakeTask(f"dup_{i}", "pick up the widget and place it in the basket")


def test_ambiguous_task_string_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from libero.libero import benchmark

    monkeypatch.setattr(
        benchmark, "get_benchmark_dict", lambda: {"libero_object": _FakeSuite}
    )

    with pytest.raises(TaskResolutionError):
        resolve_task("pick up the widget and place it in the basket")
