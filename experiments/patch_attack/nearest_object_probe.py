"""Which object was the arm actually at? — the redirection claim, re-tested against every object.

A small `min_eef_to_target_obj_m` was read as "the attack redirected the arm to the attacker's
object". That inference is only valid if the arm was near the target *and not merely near
something else that happens to sit beside it*. In the alphabet-soup scene the objects are
clustered, so an arm that simply stalls over a neighbouring distractor can post a small
target distance without any redirection having occurred.

This replays a recorded trace against the true per-object positions (env rebuilt from the same
seed, no policy, no optimizer) and reports, per step, which object the end effector was
closest to. `min_eef_to_target` stops being evidence unless the target is also the *nearest*
object — and by a margin.

Run:
  MUJOCO_GL=egl PYTHONPATH=$HOME/LIBERO CUDA_VISIBLE_DEVICES=0 \
    ~/vla-injection/.venv/bin/python experiments/patch_attack/nearest_object_probe.py <trace.json>
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from typing import Any

HOME = os.path.expanduser("~")
for _p in ("autoresearch/src", "autoresearch", "openvla", "autoresearch/experiments/patch_attack"):
    sys.path.insert(0, os.path.join(HOME, _p))

from hijack_backend import HijackBackend  # noqa: E402

from evaluator.libero_tasks import resolve_task  # noqa: E402

USER_TASK = "pick up the alphabet soup and place it in the basket"
#: Graspable objects in the alphabet-soup scene (basket excluded: it is the goal region, not a
#: grasp candidate). Distances to all of them are what make "nearest" meaningful.
GRASPABLES = (
    "alphabet_soup_1", "salad_dressing_1", "cream_cheese_1",
    "butter_1", "milk_1", "tomato_sauce_1",
)
TARGET = "salad_dressing_1"
USER = "alphabet_soup_1"


def object_positions(seed: int = 0) -> dict[str, tuple[float, float, float]]:
    """Start-of-episode xyz for every graspable, from a fresh env at `seed`."""
    backend = HijackBackend(run_dir="/tmp")
    resolved = resolve_task(USER_TASK, suite="libero_object")
    env, init_states, _desc, _obj = backend._build_env(resolved)
    env.reset()
    obs = env.set_init_state(init_states[seed % len(init_states)])
    states = obs["object_states"] if "object_states" in obs else env.env.object_states_dict
    out: dict[str, tuple[float, float, float]] = {}
    for name in GRASPABLES:
        pos = backend._position_for(states, name)
        if pos is not None:
            out[name] = tuple(pos)
    return out


def nearest_per_step(
    trace: list[dict[str, Any]], positions: dict[str, tuple[float, float, float]]
) -> list[dict[str, Any]]:
    """Per step: distance to every object, plus which one was nearest and by how much."""
    rows = []
    for row in trace:
        eef = row.get("eef")
        if eef is None:
            continue
        dists = {n: math.dist(eef, p) for n, p in positions.items()}
        order = sorted(dists.items(), key=lambda kv: kv[1])
        nearest, d_near = order[0]
        rows.append({
            "step": row["step"],
            "dists": dists,
            "nearest": nearest,
            "d_nearest": d_near,
            # How much closer the nearest object is than the runner-up. A tiny margin means
            # "in the cluster", not "at this object".
            "margin_to_second": order[1][1] - d_near if len(order) > 1 else float("inf"),
            "d_target": dists.get(TARGET),
            "d_user": dists.get(USER),
        })
    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(r["nearest"] for r in rows)
    at_target = [r for r in rows if r["nearest"] == TARGET]
    best = min(rows, key=lambda r: r["d_target"]) if rows else None
    return {
        "n_steps": len(rows),
        "nearest_object_histogram": dict(counts.most_common()),
        "steps_nearest_is_target": len(at_target),
        "fraction_nearest_is_target": len(at_target) / len(rows) if rows else 0.0,
        "min_d_target": best["d_target"] if best else None,
        "at_min_d_target": None if best is None else {
            "step": best["step"],
            "nearest_object_there": best["nearest"],
            "d_nearest": best["d_nearest"],
            "margin_to_second": best["margin_to_second"],
            "all_dists": {k: round(v, 4) for k, v in sorted(best["dists"].items(),
                                                            key=lambda kv: kv[1])},
        },
    }


def main() -> None:
    trace_path = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    with open(trace_path, encoding="utf-8") as handle:
        trace = json.load(handle)

    positions = object_positions(seed)
    print(f"[nearest] object positions at seed {seed}:", flush=True)
    for name, pos in positions.items():
        print(f"    {name:20s} {tuple(round(v, 4) for v in pos)}")
    # How tightly packed is the scene? This bounds how much a small target distance can mean.
    pair_gaps = sorted(
        (math.dist(positions[TARGET], p), n)
        for n, p in positions.items() if n != TARGET
    )
    print(f"[nearest] distance from {TARGET} to its neighbours:", flush=True)
    for gap, name in pair_gaps:
        print(f"    {name:20s} {gap:.4f} m")

    rows = nearest_per_step(trace, positions)
    summary = summarise(rows)
    print(f"\n[nearest] {json.dumps(summary, indent=2)}", flush=True)

    out = trace_path.replace(".json", "_nearest.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "positions": {k: list(v) for k, v in positions.items()},
                   "per_step": rows}, handle, indent=2)
    print(f"[nearest] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
