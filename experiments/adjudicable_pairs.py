"""Enumerate adjudicable (user_task, target_task) pairs for libero_object.

Search-side setup tool (does NOT touch the evaluator). A target task is only
*adjudicable* inside a user task's rollout if the target object is instantiated in
the user task's scene — each libero_object task instantiates only 7 objects
(target + basket + 5 task-specific distractors), so an arbitrary (user, target)
pair is often unevaluable (see docs/research/targeted-success-design.md,
"Adjudicability constraint"). This parses each task's BDDL ``:objects`` roster and
emits every pair where the target's object is present in the user's scene.

CPU-only, no LIBERO import — it just reads the shipped BDDL text.

    PYTHONPATH=src python experiments/adjudicable_pairs.py            # human table
    PYTHONPATH=src python experiments/adjudicable_pairs.py --json     # machine list
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BDDL_DIR = Path.home() / "LIBERO" / "libero" / "libero" / "bddl_files" / "libero_object"

# object type name (from the BDDL :objects lines, e.g. "cream_cheese_1 - cream_cheese")
# -> the English noun used in the "pick up the ... and place it in the basket" task string.
# For libero_object the object type == the task noun with underscores as spaces.
_OBJECTS_BLOCK = re.compile(r"\(:objects\s*(.*?)\)", re.DOTALL)
_TASK_NAME = re.compile(r"pick_up_the_(.+?)_and_place_it_in_the_basket")


def _roster(bddl_path: Path) -> set[str]:
    """The set of manipulable object *types* in a task's scene (excludes basket/floor)."""
    text = bddl_path.read_text(encoding="utf-8")
    match = _OBJECTS_BLOCK.search(text)
    if match is None:
        raise ValueError(f"no :objects block in {bddl_path}")
    types: set[str] = set()
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or " - " not in line:
            continue
        obj_type = line.split(" - ", 1)[1].strip()
        if obj_type in {"basket", "floor"}:
            continue
        types.add(obj_type)
    return types


def _task_noun(bddl_path: Path) -> str:
    m = _TASK_NAME.search(bddl_path.stem)
    if m is None:
        raise ValueError(f"unexpected task filename {bddl_path.stem}")
    return m.group(1)


def _task_string(noun: str) -> str:
    return f"pick up the {noun.replace('_', ' ')} and place it in the basket"


def enumerate_pairs() -> list[dict[str, object]]:
    """All adjudicable (user, target) pairs, user != target, target-in-user-scene."""
    bddls = sorted(BDDL_DIR.glob("pick_up_the_*_and_place_it_in_the_basket.bddl"))
    if not bddls:
        raise FileNotFoundError(f"no libero_object BDDLs under {BDDL_DIR}")
    rosters = {_task_noun(p): _roster(p) for p in bddls}

    pairs: list[dict[str, object]] = []
    for user_noun, roster in sorted(rosters.items()):
        for target_type in sorted(roster):
            if target_type == user_noun:
                continue
            if target_type not in rosters:
                # object present in a scene but has no task of its own -> no user pairing
                continue
            pairs.append(
                {
                    "user_noun": user_noun,
                    "target_noun": target_type,
                    "user_task": _task_string(user_noun),
                    "target_task": _task_string(target_type),
                }
            )
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit JSON list of pairs")
    args = ap.parse_args()

    bddls = sorted(BDDL_DIR.glob("pick_up_the_*_and_place_it_in_the_basket.bddl"))
    rosters = {_task_noun(p): _roster(p) for p in bddls}
    pairs = enumerate_pairs()

    if args.json:
        print(json.dumps(pairs, indent=2))
        return

    print(f"libero_object tasks: {len(rosters)}")
    for noun in sorted(rosters):
        print(f"  {noun:18s} scene = {{{', '.join(sorted(rosters[noun]))}}}")
    print(f"\nadjudicable (user -> target) pairs: {len(pairs)}")
    by_user: dict[str, list[str]] = {}
    for p in pairs:
        by_user.setdefault(str(p["user_noun"]), []).append(str(p["target_noun"]))
    for user in sorted(by_user):
        print(f"  user={user:18s} targets: {', '.join(by_user[user])}")


if __name__ == "__main__":
    main()
