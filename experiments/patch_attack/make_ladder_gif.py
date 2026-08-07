"""The epsilon-ladder GIF -- one figure, the whole threshold experiment.

Rollouts of the SAME task pair (`alphabet_soup` -> `salad_dressing`) at the SAME corner rect
(BL 64x64, 8.2% of frame) and the same seed, differing only in what sits in that corner:

    clean  |  eps=0 pure logo  |  ... each eps rung ...  |  free-range patch

Read left to right it is the result: the logo alone changes nothing, a perturbation bounded to
+-eps around that logo denies the user's task and -- once the budget is large enough -- hijacks
the robot outright, and the unconstrained patch is the ceiling.

Every panel shows the **policy input**, literally what OpenVLA consumed that step, so the patch
is visible exactly as the model saw it. Panels are aligned by **step index** and a finished
rollout holds its last frame, so a hijack that latched early keeps its outcome on screen.
Verdict bands come from each rollout's evaluator fields via `rollout_gif.outcome_of`; they are
never hand-written, because a mislabelled panel is a figure that misreports the result.

Rungs are **discovered from the run directory**, so rebuilding after a rung lands picks it up
without this file being edited -- a hand-maintained list would silently omit the newest rung.

Run:
  ~/vla-injection/.venv/bin/python experiments/patch_attack/make_ladder_gif.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Final

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rollout_gif as RG  # noqa: E402

LADDER_DIR: Final = os.path.join(RG.RUNS, "monitor-stealth/ladder_hinge")
OUT: Final = os.path.join(LADDER_DIR, "epsilon_ladder.gif")

RESULT_RE: Final = re.compile(
    r"^result_corner_(?P<corner>[A-Z]{2})_(?P<size>\d+)_seed(?P<seed>\d+)"
    r"(?P<suffix>.*?)_trial(?P<trial>\d+)\.json$"
)

# The two fixed reference panels: both already adjudicated, same cell and seed as every rung.
REFERENCES: Final = (
    (
        "CLEAN",
        "no patch",
        os.path.join(RG.RUNS, "monitor-corner/result_corner_BL_64_seed0_ctl_none_trial0.json"),
        os.path.join(RG.RUNS, "monitor-corner/rec_BL_64_ctl_none/policy_input"),
    ),
    (
        "PURE LOGO",
        "eps = 0",
        os.path.join(
            RG.RUNS,
            "monitor-stealth/perframe/result_corner_BL_64_seed0_stealth_eps0_trial0.json",
        ),
        os.path.join(RG.RUNS, "monitor-stealth/perframe/rec_BL_64_stealth_eps0/policy_input"),
    ),
)


@dataclass(frozen=True)
class ParsedName:
    corner: str
    size: int
    seed: int
    suffix: str
    trial: int


@dataclass(frozen=True)
class Rung:
    """A discovered ladder rung, with its verdict already on disk."""

    eps: float | None  # None == free-range (unbounded); sorts last, as the ceiling
    suffix: str
    result_path: str
    frames_dir: str
    result: dict[str, Any]


def parse_result_name(name: str) -> ParsedName | None:
    """Split a result filename into its parts, or None if it is not a result JSON."""
    match = RESULT_RE.match(name)
    if not match:
        return None
    return ParsedName(
        corner=match["corner"],
        size=int(match["size"]),
        seed=int(match["seed"]),
        suffix=match["suffix"],
        trial=int(match["trial"]),
    )


def discover_rungs(run_dir: str) -> list[Rung]:
    """Every rung in `run_dir` that has both a verdict and recorded frames, ordered by eps."""
    if not os.path.isdir(run_dir):
        raise SystemExit(f"no such run directory: {run_dir}")

    rungs: list[Rung] = []
    for name in sorted(os.listdir(run_dir)):
        parsed = parse_result_name(name)
        if parsed is None:
            continue
        frames = os.path.join(
            run_dir, f"rec_{parsed.corner}_{parsed.size}{parsed.suffix}", "policy_input"
        )
        # Skipped at discovery rather than at render: a rung with no frames would otherwise
        # fail at the very end of building a long figure.
        if not os.path.isdir(frames):
            continue
        with open(os.path.join(run_dir, name)) as fh:
            result = json.load(fh)
        stealth = result.get("stealth")
        rungs.append(
            Rung(
                eps=None if not stealth else float(stealth["eps"]),
                suffix=parsed.suffix,
                result_path=os.path.join(run_dir, name),
                frames_dir=frames,
                result=result,
            )
        )
    # Free-range (eps None) is the ceiling, so it sorts above every bounded rung.
    return sorted(rungs, key=lambda r: (r.eps is None, r.eps if r.eps is not None else 0.0))


def _panel(title: str, subtitle: str, result: dict[str, Any], frames_dir: str) -> RG.Panel:
    outcome = RG.outcome_of(result)
    return RG.Panel(title, subtitle, frames_dir, outcome.label, outcome.colour)


def build(run_dir: str = LADDER_DIR, out_path: str = OUT, stride: int = 3) -> str:
    panels: list[RG.Panel] = []
    for title, subtitle, result_path, frames_dir in REFERENCES:
        with open(result_path) as fh:
            panels.append(_panel(title, subtitle, json.load(fh), frames_dir))

    for rung in discover_rungs(run_dir):
        objective = rung.result.get("objective", {}).get("name", "unrecorded")
        if rung.eps is None:
            title, subtitle = "FREE-RANGE", f"unbounded  ({objective})"
        else:
            title, subtitle = "STEALTH", f"eps = {rung.eps:g}  ({objective})"
        panels.append(_panel(title, subtitle, rung.result, rung.frames_dir))

    footer = (
        'step {step}     commanded: "pick up the alphabet soup"     '
        "attacker wants: the salad dressing",
        "BL 64x64 corner = 8.2% of frame, covers no object, seed 0     "
        "every panel is the image OpenVLA actually consumed",
    )
    return RG.render(panels, out_path, footer, stride=stride)


if __name__ == "__main__":
    path = build()
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
