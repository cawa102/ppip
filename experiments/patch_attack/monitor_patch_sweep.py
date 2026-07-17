"""Shrink-sweep driver for the confined monitor-patch hijack: find the SMALLEST rectangle
that still hijacks (alphabet_soup -> salad_dressing), loading OpenVLA-7B ONCE and running one
fresh episode per rectangle size.

    MP_RECTS="70,100,100,100;90,120,60,60;100,130,40,40;105,135,30,30" \
    MP_SEED=0 MP_MAX_STEPS=200 CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYTHONPATH=$HOME/LIBERO \
      ~/vla-injection/.venv/bin/python experiments/patch_attack/monitor_patch_sweep.py
"""

from __future__ import annotations

import json
import os
import sys

HOME = os.path.expanduser("~")
for _p in ("autoresearch/src", "autoresearch", "openvla", "autoresearch/experiments/patch_attack"):
    sys.path.insert(0, os.path.join(HOME, _p))

from hijack_backend import HijackBackend  # noqa: E402
from monitor_patch_attack import run_confined_episode  # noqa: E402

SEED = int(os.environ.get("MP_SEED", "0"))
MAX_STEPS = int(os.environ.get("MP_MAX_STEPS", "200"))
TRIAL = os.environ.get("MP_TRIAL", "0")
RUN_DIR = os.environ.get("MP_RUN_DIR", os.path.join(HOME, "autoresearch/runs/monitor-patch"))
# Descending squares centred on the decision region (~row 120, col 150), unless overridden.
DEFAULT_RECTS = "70,100,100,100;90,120,60,60;100,130,40,40;105,135,30,30"


def _parse(spec: str) -> list[tuple[int, int, int, int]]:
    out = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        r0, c0, ph, pw = (int(x) for x in chunk.split(","))
        out.append((r0, c0, ph, pw))
    return out


def main() -> None:
    rects = _parse(os.environ.get("MP_RECTS", DEFAULT_RECTS))
    os.makedirs(RUN_DIR, exist_ok=True)
    backend = HijackBackend(run_dir=RUN_DIR, max_steps=MAX_STEPS)
    backend._policy = backend._load_policy()  # load ONCE, reuse across the sweep

    summary = []
    for rect in rects:
        r0, c0, ph, pw = rect
        tag = f"sweep_seed{SEED}_{ph}x{pw}"
        rec = os.path.join(RUN_DIR, f"rec_{ph}x{pw}") if os.environ.get("MP_RECORD") == "1" else ""
        print(f"\n===== SWEEP rect {rect} ({ph}x{pw}, {ph * pw}px) =====", flush=True)
        result = run_confined_episode(
            backend, rect=rect, seed=SEED, max_steps=MAX_STEPS, chunk=MAX_STEPS + 10,
            trial=TRIAL, run_dir=RUN_DIR, tag=tag, record_dir=rec,
        )
        summary.append(result)
        with open(os.path.join(RUN_DIR, f"sweep_summary_seed{SEED}.json"), "w") as fh:
            json.dump(summary, fh, indent=2)

    print("\n===== SWEEP SUMMARY =====", flush=True)
    for r in summary:
        print(f"  {r['rect']}  area={r['area_frac']:.1%}  targeted={r['targeted']}  "
              f"latch={r['latch_step']}  min_dist={r['min_target_dist_m']}  "
              f"mean_match={r['mean_token_match']:.2f}", flush=True)


if __name__ == "__main__":
    main()
