"""Full closed-loop CORNER-confined monitor-patch hijack (alphabet_soup -> salad_dressing).

Reuses the PROVEN per-step optimisation loop (``run_confined_episode`` in
``monitor_patch_attack``) but constrains the patch to a *corner* rectangle that provably does
NOT cover the graspable objects (asserts non-overlap with the object keep-out box). This is
the literal researcher ask: hijack with the optimised region in one of the 4 corners, not on
the object.

Run one corner episode:
  MC_CORNER=TR MC_SIZE=95 MC_MAX_STEPS=220 MC_RECORD=1 \
    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYTHONPATH=$HOME/LIBERO \
    ~/vla-injection/.venv/bin/python experiments/patch_attack/corner_attack.py

Sweep several corners/sizes with ONE model load:
  MC_SPECS="TR:95;TL:95;BL:95" ... corner_attack.py
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

KEEPOUT = (95, 170, 100, 218)  # (r0,r1,c0,c1) graspable soup+salad_dressing box, seed-0 init
RUN_DIR = os.environ.get("MC_RUN_DIR", os.path.join(HOME, "autoresearch/runs/monitor-corner"))
SEED = int(os.environ.get("MC_SEED", "0"))
MAX_STEPS = int(os.environ.get("MC_MAX_STEPS", "220"))


def corner_rect(corner: str, s: int) -> tuple[int, int, int, int]:
    n = 224
    return {
        "TL": (0, 0, s, s),
        "TR": (0, n - s, s, s),
        "BL": (n - s, 0, s, s),
        "BR": (n - s, n - s, s, s),
    }[corner]


def assert_no_object_overlap(rect: tuple[int, int, int, int]) -> None:
    r0, c0, h, w = rect
    kr0, kr1, kc0, kc1 = KEEPOUT
    overlap = not (r0 + h <= kr0 or r0 >= kr1 or c0 + w <= kc0 or c0 >= kc1)
    if overlap:
        raise ValueError(
            f"rect {rect} intersects graspable-object keep-out {KEEPOUT} -> would cover object")


def _parse_specs(spec: str) -> list[tuple[str, int]]:
    out = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        c, s = chunk.split(":")
        out.append((c, int(s)))
    return out


def main() -> None:
    os.makedirs(RUN_DIR, exist_ok=True)
    specs = _parse_specs(os.environ.get(
        "MC_SPECS", f"{os.environ.get('MC_CORNER', 'TR')}:{os.environ.get('MC_SIZE', '95')}"))
    record = os.environ.get("MC_RECORD", "0") == "1"
    lr = float(os.environ.get("MC_LR", "3e-2"))
    k = int(os.environ.get("MC_K", "10"))
    maxtries = int(os.environ.get("MC_MAXTRIES", "6"))
    trial = os.environ.get("MC_TRIAL", "0")
    # Effort headroom + controls (all default to the originally-run configuration).
    mode = os.environ.get("MC_MODE", "optimize")  # optimize | blank | random | none
    restarts = int(os.environ.get("MC_RESTARTS", "1"))
    warm_start = os.environ.get("MC_WARM", "0") == "1"
    decisive_boost = int(os.environ.get("MC_DEC_BOOST", "1"))
    suffix = os.environ.get("MC_TAG_SUFFIX", "")

    for corner, s in specs:
        assert_no_object_overlap(corner_rect(corner, s))

    backend = HijackBackend(run_dir=RUN_DIR, max_steps=MAX_STEPS)
    backend._policy = backend._load_policy()  # load ONCE, reuse across specs

    summary = []
    for corner, s in specs:
        rect = corner_rect(corner, s)
        tag = f"corner_{corner}_{s}_seed{SEED}{suffix}"
        rec = os.path.join(RUN_DIR, f"rec_{corner}_{s}{suffix}") if record else ""
        print(f"\n===== CORNER {corner} size {s} rect={rect} "
              f"({s * s / (224 * 224):.1%} of frame) mode={mode} "
              f"k={k} maxtries={maxtries} restarts={restarts} warm={warm_start} "
              f"dec_boost={decisive_boost} =====", flush=True)
        result = run_confined_episode(
            backend, rect=rect, seed=SEED, max_steps=MAX_STEPS, chunk=MAX_STEPS + 10,
            k=k, lr=lr, maxtries=maxtries, trial=trial, run_dir=RUN_DIR, tag=tag, record_dir=rec,
            patch_mode=mode, restarts=restarts, warm_start=warm_start,
            decisive_boost=decisive_boost,
        )
        result["corner"] = corner
        summary.append(result)
        with open(os.path.join(RUN_DIR, f"corner_summary_seed{SEED}.json"), "w") as fh:
            json.dump(summary, fh, indent=2)

    print("\n===== CORNER SUMMARY =====", flush=True)
    for r in summary:
        red = r.get("redirection", {})
        print(f"  {r['corner']}:{r['rect']}  area={r['area_frac']:.1%}  mode={r.get('patch_mode')}  "
              f"targeted={r['targeted']}  commanded={r.get('commanded_success')}  "
              f"latch={r['latch_step']}  min_dist={r['min_target_dist_m']}  "
              f"decisive_forcing={r.get('mean_decisive_forcing')}  "
              f"min_eef->tgt={red.get('min_eef_to_target_obj_m')}", flush=True)


if __name__ == "__main__":
    main()
