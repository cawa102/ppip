"""Resolve a candidate's free-text task string to a concrete LIBERO task.

The attack-candidate schema carries ``user_task``/``target_task`` as human-readable
strings. The rollout backend needs the benchmark-owned task behind each: its suite
index, the language description fed to OpenVLA, and the goal-state predicate tuples
used to adjudicate success. This module is that bridge.

LIBERO is imported lazily inside the functions so the module stays importable on a
CPU host without the benchmark installed (mirroring ``openvla_backend``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResolvedTask:
    """A candidate task string resolved to its LIBERO benchmark task."""

    task_id: int
    name: str
    language: str
    bddl_path: str
    goal_state: list[list[str]]


class TaskResolutionError(ValueError):
    """A task string matched no task, or matched more than one, in the suite."""


def _normalize(task_str: str) -> str:
    """Normalise a task string for matching: strip, lowercase, collapse spaces."""
    return re.sub(r"\s+", " ", task_str.strip().lower())


def parse_goal_state(bddl_path: str) -> list[list[str]]:
    """Return the goal-state predicate tuples for a BDDL file.

    Each tuple is ``[predicate_name, *object_names]`` (lowercased by the parser),
    e.g. ``["in", "ketchup_1", "basket_1_contain_region"]``.
    """
    from libero.libero.envs import bddl_utils

    parsed = bddl_utils.robosuite_parse_problem(bddl_path)
    return [list(predicate) for predicate in parsed["goal_state"]]


def resolve_task(task_str: str, *, suite: str = "libero_object") -> ResolvedTask:
    """Resolve a free-text task string to its LIBERO task in ``suite``.

    Raises ``TaskResolutionError`` if the string matches no task or is ambiguous.
    """
    from libero.libero import benchmark

    suite_obj: Any = benchmark.get_benchmark_dict()[suite]()
    wanted = _normalize(task_str)

    matches = [
        i for i in range(suite_obj.n_tasks)
        if _normalize(suite_obj.get_task(i).language) == wanted
    ]
    if len(matches) != 1:
        detail = "no task" if not matches else f"{len(matches)} tasks"
        raise TaskResolutionError(
            f"task string {task_str!r} matched {detail} in suite {suite!r}"
        )

    task_id = matches[0]
    task = suite_obj.get_task(task_id)
    bddl_path = suite_obj.get_task_bddl_file_path(task_id)
    return ResolvedTask(
        task_id=task_id,
        name=task.name,
        language=task.language,
        bddl_path=bddl_path,
        goal_state=parse_goal_state(bddl_path),
    )
