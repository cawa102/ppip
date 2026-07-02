"""Adjudicate a target task's success against a live env's object states.

`eval_goal_state` is the single place the *targeted-success* verdict is computed.
It mirrors what LIBERO's own `_check_success` does internally: evaluate each
goal-state predicate tuple with `eval_predicate_fn` over the referenced object
states, and take the conjunction. Keeping it pure (no sim access) lets the
scientific-integrity-critical decision be unit-tested off-GPU; the GPU rollout
body hands it `env.<...>.object_states_dict`.

LIBERO is imported lazily so the module stays importable on a CPU host.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class UnevaluableGoalError(RuntimeError):
    """A goal-state could not be evaluated against the given object states.

    Raised when a predicate references an object absent from `object_states`, or
    when the goal-state is empty (a real goal always has >=1 predicate). The caller
    records this as an unevaluable rollout rather than inventing a verdict.
    """


def eval_goal_state(
    goal_state: Sequence[Sequence[str]],
    object_states: Mapping[str, Any],
) -> bool:
    """Return the conjunction of `goal_state`'s predicate tuples over `object_states`.

    Each tuple is ``(predicate_name, *object_names)``; it is evaluated via LIBERO
    ``eval_predicate_fn(name, object_states[obj1], ...)``. Raises
    ``UnevaluableGoalError`` if the goal is empty or references a missing object.

    Objects are resolved for *all* predicates before any predicate is evaluated, so an
    unevaluable goal is reported honestly regardless of predicate order or conjunction
    short-circuiting -- a missing object never hides behind an earlier `False` term.
    """
    from libero.libero.envs.predicates import eval_predicate_fn

    if not goal_state:
        raise UnevaluableGoalError("goal_state is empty; nothing to adjudicate")

    resolved: list[tuple[str, list[Any]]] = []
    for predicate in goal_state:
        name, *object_names = predicate
        try:
            states = [object_states[obj] for obj in object_names]
        except KeyError as exc:
            raise UnevaluableGoalError(
                f"goal predicate {list(predicate)!r} references object "
                f"{exc.args[0]!r} not present in the scene"
            ) from exc
        resolved.append((name, states))

    return all(eval_predicate_fn(name, *states) for name, states in resolved)
