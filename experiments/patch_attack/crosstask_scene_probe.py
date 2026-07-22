"""Render a libero_object scene's seed-0 init frame and check a corner rect is empty floor.

Needed before running the corner attack in a NEW scene: the keep-out box asserted by
``corner_attack.py`` (rows 95..170, cols 100..218) was measured on the alphabet-soup layout,
so a different task's layout must be re-checked rather than assumed.

Every ``libero_object`` BDDL places its objects on the SAME six floor regions
(``target_object_region`` + ``other_object_region_0..4``, identical coordinates in all ten
files) -- only *which* object sits on each region changes. So the occupied pixel envelope
should be scene-independent, but this script measures it instead of trusting that: it renders
the init frame with objects, then re-renders with every graspable object teleported far below
the floor, and takes the per-pixel difference as the object mask.  The reported bounding box
is that mask's extent, and the requested corner rect is checked against it.

No OpenVLA load -- env + EGL render only.

Run:
  SP_TASK="pick up the milk and place it in the basket" SP_RECT=BL:64 \
    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYTHONPATH=$HOME/LIBERO \
    ~/vla-injection/.venv/bin/python experiments/patch_attack/crosstask_scene_probe.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HOME = os.path.expanduser("~")
for _p in ("autoresearch/src", "autoresearch", "openvla", "autoresearch/experiments/patch_attack"):
    sys.path.insert(0, os.path.join(HOME, _p))

from hijack_backend import HijackBackend  # noqa: E402

from evaluator.libero_tasks import resolve_task  # noqa: E402

RUN_DIR = os.environ.get("SP_RUN_DIR", os.path.join(HOME, "autoresearch/runs/monitor-crosstask"))
TASK = os.environ.get("SP_TASK", "pick up the milk and place it in the basket")
SEED = int(os.environ.get("SP_SEED", "0"))
DIFF_THRESH = int(os.environ.get("SP_DIFF_THRESH", "12"))  # per-pixel uint8 delta = "object here"


def corner_rect(corner: str, s: int) -> tuple[int, int, int, int]:
    n = 224
    return {"TL": (0, 0, s, s), "TR": (0, n - s, s, s),
            "BL": (n - s, 0, s, s), "BR": (n - s, n - s, s, s)}[corner]


def main() -> None:
    os.makedirs(RUN_DIR, exist_ok=True)
    corner, size = os.environ.get("SP_RECT", "BL:64").split(":")
    rect = corner_rect(corner, int(size))
    resolved = resolve_task(TASK, suite="libero_object")

    from experiments.robot.libero.libero_utils import get_libero_image, resize_image

    def render224(env) -> np.ndarray:
        """Render the policy's agentview straight from the sim.

        ``env._get_observations()`` does NOT re-render after a raw ``sim.set_state``, so
        diffing observations silently yields an all-zero mask (a vacuous "no overlap").
        Rendering from the sim is the only path that reflects the mutated state; the same
        180-degree rotation as ``get_libero_image`` is applied so pixel coordinates match
        the frame the attack's rect is defined in.
        """
        im = env.sim.render(height=256, width=256, camera_name="agentview")
        return resize_image(im[::-1, ::-1], (224, 224))

    backend = HijackBackend(run_dir=RUN_DIR, max_steps=240)
    cfg = backend._build_cfg()
    env, init_states, _desc, _obj = backend._build_env(resolved)
    obs = env.set_init_state(init_states[SEED % len(init_states)])
    for _ in range(backend.num_steps_wait):
        obs, _r, _d, _i = env.step(backend._dummy_action(cfg))
    with_objects = render224(env)
    # Guard the coordinate convention: the sim render must match the frame the policy sees,
    # otherwise the rect check would be done in the wrong pixel space.
    obs_frame = get_libero_image(obs, 224)
    conv_err = float(np.abs(with_objects.astype(int) - obs_frame.astype(int)).mean())
    if conv_err > 8.0:
        raise RuntimeError(
            f"sim render disagrees with the policy frame (mean |delta| = {conv_err:.1f}/255) -- "
            "orientation/convention mismatch, rect coordinates would be meaningless")

    # Sink every graspable object (everything but the basket/fixtures) far below the floor and
    # re-render: whatever changes is object pixels.
    sim = env.sim
    names = [n for n in sim.model.body_names
             if n.endswith("_main") and "basket" not in n and "robot" not in n
             and "gripper" not in n and "table" not in n and "floor" not in n]
    state = sim.get_state()
    moved = []
    for n in names:
        bid = sim.model.body_name2id(n)
        jadr = sim.model.body_jntadr[bid]
        if jadr < 0:
            continue
        qadr = sim.model.jnt_qposadr[jadr]
        state.qpos[qadr + 2] -= 5.0  # drop 5 m through the floor
        moved.append(n)
    sim.set_state(state)
    sim.forward()
    without_objects = render224(env)
    if np.abs(with_objects.astype(int) - without_objects.astype(int)).max() == 0:
        raise RuntimeError(
            "sinking the objects changed nothing in the render -- the mask would be vacuously "
            f"empty. Bodies attempted: {moved}")

    diff = np.abs(with_objects.astype(int) - without_objects.astype(int)).max(axis=2)
    mask = diff > DIFF_THRESH
    rows, cols = np.where(mask)
    bbox = ([int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())]
            if rows.size else None)

    r0, c0, h, w = rect
    overlap_px = int(mask[r0:r0 + h, c0:c0 + w].sum())

    os.makedirs(os.path.join(RUN_DIR, "overlays"), exist_ok=True)
    slug = resolved.name
    Image.fromarray(with_objects).save(os.path.join(RUN_DIR, "overlays", f"scene_{slug}.png"))
    Image.fromarray((mask * 255).astype(np.uint8)).save(
        os.path.join(RUN_DIR, "overlays", f"objmask_{slug}.png"))
    ov = Image.fromarray(with_objects).convert("RGB").resize((448, 448), Image.NEAREST)
    dr = ImageDraw.Draw(ov)
    dr.rectangle([c0 * 2, r0 * 2, (c0 + w) * 2 - 1, (r0 + h) * 2 - 1], outline=(255, 0, 0), width=3)
    if bbox:
        dr.rectangle([bbox[2] * 2, bbox[0] * 2, bbox[3] * 2, bbox[1] * 2],
                     outline=(0, 255, 0), width=3)
    ov.save(os.path.join(RUN_DIR, "overlays", f"overlay_{slug}_{corner}{size}.png"))

    out = {"task": TASK, "seed": SEED, "rect": list(rect), "corner": f"{corner}:{size}",
           "objects_sunk": moved, "object_bbox_r0r1c0c1": bbox,
           "object_pixels_in_rect": overlap_px, "clear": overlap_px == 0,
           "diff_threshold": DIFF_THRESH}
    with open(os.path.join(RUN_DIR, f"scene_probe_{slug}_{corner}{size}.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2), flush=True)
    print(f"\n{'CLEAR' if overlap_px == 0 else 'OVERLAP'}: rect {rect} contains {overlap_px} "
          f"object pixels; object bbox (r0,r1,c0,c1) = {bbox}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
