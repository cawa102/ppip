"""Post-process one ladder rung: verify it, measure it, draw it, record it.

A finished rung leaves a result JSON and a directory of recorded frames. Turning that into a
ladder row takes four steps that are easy to do inconsistently by hand, so they happen together
here:

1. **Verify** the rung actually ran under the objective the ladder claims. `corner_attack.py`
   takes the objective from an env var and reaches the closed-loop core directly, so a typo
   yields a rung that is wrong in a way nothing else would reveal.
2. **Measure** spatial stealth (LPIPS vs the carrier) and temporal churn from the executed
   patch crops.
3. **Draw** the three-panel GIF for that rung.
4. **Append** a summary row to the ladder table.

Run:
  ~/vla-injection/.venv/bin/python experiments/patch_attack/finalize_rung.py \
      _eps025_hinge 0.25 [run_dir]
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Final

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import make_patch_gif as PGIF  # noqa: E402
import make_rung_gif as RGIF  # noqa: E402
import rollout_gif as RG  # noqa: E402
import stealth_metrics as SM  # noqa: E402

CARRIER: Final = os.path.join(RG.RUNS, "monitor-stealth/bases/aurora_64.png")
LADDER_TABLE: Final = "ladder_table.json"


#: Budget label, owned by the GIF module so figures and tables cannot disagree.
rung_label = RGIF.rung_label


def rung_gain(eps: float | None) -> int:
    """Difference-panel amplification. The ceiling has no budget to normalise against, and
    amplifying an unbounded perturbation would saturate the panel everywhere."""
    return 1 if eps is None else PGIF.gain_for_eps(eps)


def verify_objective(result: dict[str, Any], expected: str) -> None:
    """Reject a rung that did not run under the ladder's objective."""
    objective = result.get("objective")
    if not objective or not objective.get("name"):
        raise ValueError(
            "rung has no recorded objective -- it predates the objective dispatch and cannot "
            "be admitted to a ladder that claims one objective throughout"
        )
    name = objective["name"]
    if name != expected:
        raise ValueError(
            f"rung ran under objective {name!r}, but the ladder is {expected!r}; "
            "check MC_OBJECTIVE on the launch"
        )


def summary_row(result: dict[str, Any], stealth: dict[str, Any]) -> dict[str, Any]:
    """One ladder row: the evaluator's verdict plus the numbers the rung is judged on."""
    budget = result.get("stealth") or {}
    return {
        "outcome": RG.outcome_of(result).key,
        "eps": budget.get("eps"),
        "targeted": result["targeted"],
        "commanded_success": result["commanded_success"],
        "latch_step": result.get("latch_step"),
        "min_target_dist_m": result.get("min_target_dist_m"),
        "mean_decisive_forcing": result.get("mean_decisive_forcing"),
        "objective": (result.get("objective") or {}).get("name"),
        "linf_measured_max": budget.get("linf_measured_max"),
        "bound_holds": budget.get("bound_holds"),
        "linf_vs_carrier": stealth["linf_vs_carrier"],
        "lpips_patch_vs_carrier": stealth["patch_vs_carrier"]["lpips_mean"],
        "churn_mean_abs_delta": stealth["churn"]["mean_abs_delta"],
        "churn_frac_steps_visible": stealth["churn"]["frac_steps_above_threshold"],
        # Granted vs spent (design section 5): eps is the budget the optimiser was GIVEN, and the
        # typical pixel spends only 39-58% of it across the ladder even though L-inf reaches the cap
        # at every rung. A row carrying eps alone describes the worst pixel, not the patch, so
        # occupancy travels beside it or not at all.
        "ball_mean_occupancy": (stealth.get("ball_occupancy") or {}).get("mean_ratio"),
        "ball_frac_at_boundary": (stealth.get("ball_occupancy") or {}).get("frac_above_90pct"),
    }


def append_row(run_dir: str, suffix: str, row: dict[str, Any]) -> str:
    """Add (or replace) this rung's row in the ladder table, keeping it sorted by eps."""
    path = os.path.join(run_dir, LADDER_TABLE)
    table: dict[str, Any] = {}
    if os.path.exists(path):
        with open(path) as fh:
            table = json.load(fh)
    table[suffix] = row
    ordered = dict(
        sorted(table.items(), key=lambda kv: (kv[1]["eps"] is None, kv[1]["eps"] or 0.0))
    )
    with open(path, "w") as fh:
        json.dump(ordered, fh, indent=2)
    return path


def finalize(
    run_dir: str,
    suffix: str,
    eps: float | None,
    *,
    objective: str = "hinge",
    carrier: str = CARRIER,
) -> dict[str, Any]:
    """Finalize one rung. `eps=None` is the free-range ceiling: no budget, so no occupancy."""
    rung = RGIF.Rung(run_dir=run_dir, corner="BL", size=64, seed=0, suffix=suffix)
    result_path, _ = RGIF.rung_paths(rung)
    result = RGIF.load(result_path)

    verify_objective(result, objective)
    print(f"[finalize] objective verified: {result['objective']}", flush=True)

    rec_dir = os.path.join(run_dir, f"rec_BL_64{suffix}")
    stealth = SM.measure(rec_dir, carrier, eps=eps)
    stealth_path = os.path.join(run_dir, f"stealth_metrics{suffix}.json")
    with open(stealth_path, "w") as fh:
        json.dump(stealth, fh, indent=2)
    print(f"[finalize] wrote {stealth_path}", flush=True)

    gif = RGIF.build(rung, eps)
    print(f"[finalize] wrote {gif} ({os.path.getsize(gif) / 1e6:.1f} MB)", flush=True)

    # Gain is chosen from eps: a fixed one saturates the difference panel at the top of the
    # ladder and leaves it flat at the bottom.
    gain = rung_gain(eps)
    patch_gif = PGIF.build(
        rec_dir, carrier, os.path.join(run_dir, f"patch_evolution{suffix}.gif"), gain
    )
    print(
        f"[finalize] wrote {patch_gif} (x{gain} difference, "
        f"{os.path.getsize(patch_gif) / 1e6:.1f} MB)",
        flush=True,
    )

    row = summary_row(result, stealth)
    table = append_row(run_dir, suffix, row)
    print(f"[finalize] wrote {table}", flush=True)
    print(json.dumps(row, indent=2), flush=True)
    return row


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: finalize_rung.py <suffix> <eps> [run_dir] [objective]")
    # "free" names the unbounded ceiling: no epsilon, so no ball and no occupancy.
    suffix = sys.argv[1]
    eps = None if sys.argv[2].lower() in ("free", "none") else float(sys.argv[2])
    run_dir = sys.argv[3] if len(sys.argv) > 3 else RGIF.LADDER_DIR
    objective = sys.argv[4] if len(sys.argv) > 4 else "hinge"
    finalize(run_dir, suffix, eps, objective=objective)


if __name__ == "__main__":
    main()
