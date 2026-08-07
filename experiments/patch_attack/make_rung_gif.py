"""Per-rung GIF: what one epsilon does, next to the two references.

Each ladder rung gets its own three-panel figure -- clean, the inert pure logo, and the rung --
so the rung's outcome is readable on its own without hunting through the full ladder GIF. The
verdict band and the quantitative caption both come from the rung's result JSON, i.e. from the
fixed evaluator, so a figure cannot claim an outcome the evaluator did not return.

Build the figure for one rung:
  ~/vla-injection/.venv/bin/python experiments/patch_attack/make_rung_gif.py 0.25 _eps025_hinge

Or from Python, for a rung that lives somewhere unusual:
  build(Rung(run_dir=..., corner="BL", size=64, seed=0, suffix="_eps025_hinge"), eps=0.25)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Final

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rollout_gif as RG  # noqa: E402

CORNERS: Final = ("TL", "TR", "BL", "BR")
LADDER_DIR: Final = os.path.join(RG.RUNS, "monitor-stealth/ladder_hinge")

# The two fixed references every rung is read against. Both are already-adjudicated rollouts of
# the same cell and seed; only the corner contents differ.
CLEAN_FRAMES: Final = os.path.join(RG.RUNS, "monitor-corner/rec_BL_64_ctl_none/policy_input")
CLEAN_RESULT: Final = os.path.join(
    RG.RUNS, "monitor-corner/result_corner_BL_64_seed0_ctl_none_trial0.json"
)
LOGO_FRAMES: Final = os.path.join(
    RG.RUNS, "monitor-stealth/perframe/rec_BL_64_stealth_eps0/policy_input"
)
LOGO_RESULT: Final = os.path.join(
    RG.RUNS, "monitor-stealth/perframe/result_corner_BL_64_seed0_stealth_eps0_trial0.json"
)


@dataclass(frozen=True)
class Rung:
    """One closed-loop episode, identified the way ``corner_attack.py`` names its outputs."""

    run_dir: str
    corner: str
    size: int
    seed: int
    suffix: str
    trial: int = 0

    def __post_init__(self) -> None:
        if self.corner not in CORNERS:
            raise ValueError(f"corner {self.corner!r} is not one of {CORNERS}")
        if self.size <= 0:
            raise ValueError(f"size must be positive, got {self.size}")


def rung_paths(rung: Rung) -> tuple[str, str]:
    """Return ``(result_json, frames_dir)`` for a rung.

    The two names diverge -- the result carries seed and trial, the recording dir carries
    neither -- so they are built from the same spec rather than typed twice.
    """
    tag = f"corner_{rung.corner}_{rung.size}_seed{rung.seed}{rung.suffix}"
    result = os.path.join(rung.run_dir, f"result_{tag}_trial{rung.trial}.json")
    frames = os.path.join(
        rung.run_dir, f"rec_{rung.corner}_{rung.size}{rung.suffix}", "policy_input"
    )
    return result, frames


def load(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise SystemExit(f"missing result JSON: {path}")
    with open(path) as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def panel(title: str, subtitle: str, result: dict[str, Any], frames_dir: str) -> RG.Panel:
    """Build a panel whose verdict band is derived from the evaluator's own fields."""
    outcome = RG.outcome_of(result)
    return RG.Panel(title, subtitle, frames_dir, outcome.label, outcome.colour)


def _reference_panels() -> list[RG.Panel]:
    return [
        panel("CLEAN", "no patch", load(CLEAN_RESULT), CLEAN_FRAMES),
        panel("PURE LOGO", "eps = 0", load(LOGO_RESULT), LOGO_FRAMES),
    ]


def rung_label(eps: float | None) -> str:
    """How a rung's budget is written on its figures. `None` is the free-range ceiling.

    Formatting it here rather than at each call site keeps a figure from ever reading
    "eps = None", which looks like a bug rather than like the unbounded ceiling.
    """
    return "unbounded (free-range)" if eps is None else f"eps = {eps:g}"


def build(rung: Rung, eps: float | None, out_path: str | None = None) -> str:
    result_path, frames_dir = rung_paths(rung)
    result = load(result_path)

    objective = result.get("objective", {}).get("name", "unrecorded")
    panels = [
        *_reference_panels(),
        panel(
            "FREE-RANGE" if eps is None else "STEALTH",
            f"{rung_label(eps)}  ({objective})",
            result,
            frames_dir,
        ),
    ]
    caption = RG.caption_for(result)
    footer = (
        'step {step}     commanded: "pick up the alphabet soup"     '
        "attacker wants: the salad dressing",
        f"BL {rung.size}x{rung.size} corner = "
        f"{rung.size * rung.size / (224 * 224):.1%} of frame, covers no object, "
        f"seed {rung.seed}     {caption}",
    )
    out = out_path or os.path.join(rung.run_dir, f"rung{rung.suffix}.gif")
    return RG.render(panels, out, footer)


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: make_rung_gif.py <eps> <suffix> [run_dir]")
    eps = float(sys.argv[1])
    suffix = sys.argv[2]
    run_dir = sys.argv[3] if len(sys.argv) > 3 else LADDER_DIR
    rung = Rung(run_dir=run_dir, corner="BL", size=64, seed=0, suffix=suffix)
    path = build(rung, eps)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
