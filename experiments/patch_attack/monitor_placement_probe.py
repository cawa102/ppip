"""Placement diagnostic for Experiment 2 (through-render monitor hijack).

Cheap geometry probe (NO rollout / no per-step optimisation): for each candidate monitor
placement (scale / position / rotation), build the deployment scene, make the monitor
emissive + neutral, measure its projected footprint in the 224 policy input (area % + bbox)
and save the scene frame + neutral policy input so occlusion of the salad dressing can be
inspected by eye. The goal (H1) is a BIG (>=25-40% of frame) NON-occluding placement -- more
projected pixels restore the post-render DOF the low-pass reality-gap destroys, while an
un-occluded object keeps the target action easy to force.

Search/rendering side only; evaluator untouched. GPU 1 only.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import numpy as np

HOME = os.path.expanduser("~")
for _p in ("autoresearch/src", "autoresearch", "openvla", "autoresearch/experiments/patch_attack"):
    sys.path.insert(0, os.path.join(HOME, _p))

# Candidate placements: (tag, scale, pos[x,y,z], rot[rx,ry,rz]).
# rot [90,90,0] stands the panel upright facing the +x agentview camera (fronto-parallel-ish,
# minimising perspective warp -> the BPDA straight-through identity assumption holds best).
# x = depth (camera on +x looks toward -x), y = left/right, z = height above table.
# Raise z to float the panel ABOVE the low table objects (top of frame = empty receding floor).
# WIDE (non-square) placements: a square monitor clips its top corners off-frame above ~44%
# (calibrate_uv then fails), and decisive-5 (forcing yaw) still collapses at that size. A wide,
# SHORT panel (tex H<W) can cover more frame area (more render DOF -> can sustain the yaw token)
# without the top-corner clip. Tuple: (tag, scale, pos[x,y,z], rot, tex_hw=(H,W)).
# FRONTO-PARALLEL rotation sweep at a fixed square panel: (90,90,0) faces +x but the agentview
# camera is ELEVATED looking down, so the panel is foreshortened (bbox wider than tall). Tilting
# the panel to face the camera squarely (bbox h~=w, larger area) reduces foreshortening -> more
# effective pixels/DOF for the same size AND makes the BPDA identity-gradient more accurate.
# Report bbox aspect (h/w) and area per rotation; pick the most-square, largest, calibrate-ok one.
# CENTERED big placements: earlier big monitors were positioned HIGH (z~0.26-0.34) so their TOP
# corners clipped off-frame (calibrate_uv fails) -- capping usable size at ~44.6%. Centering the
# panel (lower z) brings the top corner into frame, so a much larger monitor (rows ~20-200) can
# keep all 4 corners in-frame -> more render DOF. Sweep scale x z; report area + calibrate_ok.
PLACEMENTS: list[tuple[str, float, list[float], list[float], tuple[int, int]]] = [
    ("ctr_s65_z18", 6.5, [-0.05, 0.0, 0.18], [90.0, 90.0, 0.0], (256, 256)),
    ("ctr_s70_z16", 7.0, [-0.05, 0.0, 0.16], [90.0, 90.0, 0.0], (256, 256)),
    ("ctr_s75_z14", 7.5, [-0.05, 0.0, 0.14], [90.0, 90.0, 0.0], (256, 256)),
    ("ctr_s80_z12", 8.0, [-0.05, 0.0, 0.12], [90.0, 90.0, 0.0], (256, 256)),
    ("ctr_s90_z10", 9.0, [-0.05, 0.0, 0.10], [90.0, 90.0, 0.0], (256, 256)),
    ("ctr_s75_z10", 7.5, [-0.05, 0.0, 0.10], [90.0, 90.0, 0.0], (256, 256)),
]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """(r0, c0, h, w) bounding box of the True region of a 2D bool mask (0,0,0,0 if empty)."""
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if ys.size == 0:
        return (0, 0, 0, 0)
    r0, c0 = int(ys.min()), int(xs.min())
    return (r0, c0, int(ys.max()) - r0 + 1, int(xs.max()) - c0 + 1)


def main() -> None:
    import imageio.v2 as imageio
    import mujoco
    from experiments.robot.libero.libero_utils import get_libero_image
    from monitor_attack import neutral_texture
    from monitor_hijack_backend import MonitorHijackBackend
    from monitor_render_attack import _setup_monitor

    from rendering.geometry import _mujoco_safe_name
    from rendering.monitor import (
        _fresh_obs,
        _policy_input_frame,
        _raw_model,
        _sim_of,
        calibrate_uv,
    )

    seed = int(os.environ.get("MPP_SEED", "0"))
    out_dir = os.path.join(HOME, "autoresearch/runs/monitor-render", "placement_probe")
    os.makedirs(out_dir, exist_ok=True)

    backend = MonitorHijackBackend(run_dir=os.path.join(HOME, "autoresearch/runs/monitor-render"))
    backend._policy = backend._load_policy()
    _model = backend._policy[0]
    resize_size = backend._policy[3]

    summary: list[dict[str, Any]] = []
    for tag, scale, pos, rot, tex_hw in PLACEMENTS:
        env, handle, _ru, _rt = _setup_monitor(
            backend, seed, scale=scale, pos=pos, rot=rot, tex_hw=tex_hw
        )
        try:
            # emissive (a real screen glows) -- runtime material mutation only
            _sim = _sim_of(env)
            _m = _raw_model(_sim)
            emat = int(mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_MATERIAL,
                                         f"{_mujoco_safe_name('monitor_render')}__mat"))
            if emat >= 0:
                _m.mat_emission[emat] = 1.0
                _m.mat_specular[emat] = 0.0
                _m.mat_shininess[emat] = 0.0
                _sim.forward()

            # projected footprint via black<->white contrast (pre-crop 224)
            h, w = handle.dims
            handle.upload(np.zeros((h, w, 3), dtype=np.uint8))
            env.sim.forward()
            black = _policy_input_frame(env, resize_size)
            handle.upload(np.full((h, w, 3), 255, dtype=np.uint8))
            env.sim.forward()
            white = _policy_input_frame(env, resize_size)
            mask = np.abs(white.astype(np.int64) - black.astype(np.int64)).max(axis=2) > 12.0
            area = int(mask.sum())
            frac = area / (224 * 224)
            bbox = _bbox(mask)

            # calibrate_uv is what the ATTACK uses; it fails if any of the 4 corners projects
            # off-frame. Test it here so we only run the attack on usable placements.
            try:
                calibrate_uv(env, handle, resize_size=resize_size)
                calib_ok = True
            except Exception as exc:  # noqa: BLE001 - probe reports the failure, not raises
                calib_ok = False
                print(f"[place] {tag}: calibrate_uv FAILED ({exc})", flush=True)

            # save neutral scene + policy input for occlusion inspection
            handle.upload(np.ascontiguousarray(neutral_texture((h, w))))
            env.sim.forward()
            neutral_input = _policy_input_frame(env, resize_size)
            scene = get_libero_image(_fresh_obs(env), 384)
            imageio.imwrite(os.path.join(out_dir, f"{tag}_scene.png"), scene)
            imageio.imwrite(os.path.join(out_dir, f"{tag}_input.png"), neutral_input)

            row = {"tag": tag, "scale": scale, "pos": pos, "rot": rot, "tex_hw": list(tex_hw),
                   "area_px": area, "area_frac": round(frac, 4), "bbox_r0c0hw": list(bbox),
                   "calibrate_ok": calib_ok}
            summary.append(row)
            print(f"[place] {tag}: area={frac:.1%} ({area}px) bbox(r0,c0,h,w)={bbox} "
                  f"calib_ok={calib_ok}", flush=True)
        finally:
            if hasattr(env, "close"):
                env.close()

    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[place] wrote {out_dir}/summary.json ({len(summary)} placements)", flush=True)


if __name__ == "__main__":
    main()
