"""Measured non-occlusion: does the corner patch rect cover any task object, per init?

Replaces the static seed-0 keep-out box hardcoded at `corner_probe.py:44`
(`KEEPOUT = (95, 170, 100, 218)`), which was eyeballed from one init frame and then reused
for every result. Codex F8 asks for the measured version: per-init segmentation overlap for
the target object, the user object, the basket and the gripper, reported as evidence across
inits rather than asserted once.

Method — MuJoCo geom-id segmentation, mapped into the policy's 224x224 frame:

  1. Build the user task's scene, jump to init `i`, settle with the evaluator's own
     `num_steps_wait` dummy steps, so the state matches a scored episode's step 0.
  2. Render a geom-id segmentation via the evaluator's `_segmentation` seam.
  3. Map it into policy space. `get_libero_image` does `obs["agentview_image"][::-1, ::-1]`
     then resizes 256 -> 224, so the segmentation needs the same chain. The orientation of
     `sim.render` relative to `obs` is **calibrated empirically** per run (see
     `calibrate_orientation`) rather than assumed: getting it wrong would silently swap the
     corner being measured, turning "provably off the objects" into its opposite.
  4. Dilate each entity mask by one pixel before intersecting with the rect, so nearest-
     neighbour resampling at a boundary can only ever *over*-report occlusion, never hide it.

Cross-check: at init 0 the measured object bounding box is compared against the legacy
hardcoded keep-out. Independent agreement there validates the whole orientation chain.

No policy is loaded (rendering only), so this is cheap and does not contend for VRAM with a
running rollout job.

Run:

    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYTHONPATH=$HOME/LIBERO \
      ~/vla-injection/.venv/bin/python experiments/patch_attack/occlusion_probe.py

Scope: this measures the **episode-start** state for every precommitted init, which is the
per-init variation Codex F8 asks about (object layout differs by init). Occlusion *during*
the trajectory -- an object being carried through the rect -- is not covered here and needs
a per-step segmentation render inside a rollout; recorded as a follow-up.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray

HOME = os.path.expanduser("~")
for _p in ("autoresearch/src", "autoresearch", "openvla", "autoresearch/experiments/patch_attack"):
    sys.path.insert(0, os.path.join(HOME, _p))

import shared_inits  # noqa: E402
from hijack_backend import HijackBackend  # noqa: E402

from evaluator.libero_tasks import resolve_task  # noqa: E402

ImageArray = NDArray[np.uint8]
IntArray = NDArray[np.integer[Any]]
BoolArray = NDArray[np.bool_]

USER_TASK = "pick up the alphabet soup and place it in the basket"
POLICY_SIDE = 224
RENDER_SIDE = 256

#: Corner sizes every previously-claimed corner result used, so this re-validates them all.
CORNER_SIZES: tuple[int, ...] = (80, 64, 48, 40, 32)
#: BR is excluded by construction -- the graspable objects live there.
CORNERS: tuple[str, ...] = ("BL", "TL", "TR")

#: Geom-name substrings identifying each entity the patch must not cover.
ENTITY_PATTERNS: dict[str, tuple[str, ...]] = {
    "user_object": ("alphabet_soup",),
    "target_object": ("salad_dressing",),
    "basket": ("basket", "wooden_tray"),
    "gripper": ("gripper", "robot0_finger", "finger"),
    "other_objects": ("cream_cheese", "tomato_sauce", "butter", "milk", "ketchup",
                      "orange_juice", "chocolate_pudding", "bbq_sauce"),
}

#: The eyeballed seed-0 box this probe replaces, kept only as a cross-check target.
LEGACY_KEEPOUT = (95, 170, 100, 218)  # (r0, r1, c0, c1) in policy space

_FLIPS: dict[str, Any] = {
    "identity": lambda a: a,
    "vflip": lambda a: a[::-1],
    "hflip": lambda a: a[:, ::-1],
    "rot180": lambda a: a[::-1, ::-1],
}


def split_of(init: int) -> str:
    """Which precommit split an init belongs to.

    Init 0 belongs to neither: it is the selection-contaminated legacy gate. Labelling it
    "heldout" (the naive `train if ... else heldout`) would quietly let a contaminated
    episode be read as held-out evidence.
    """
    if init in shared_inits.TRAIN_INITS:
        return "train"
    if init in shared_inits.HELDOUT_INITS:
        return "heldout"
    return "legacy_gate"


def corner_rect(corner: str, size: int) -> tuple[int, int, int, int]:
    """(r0, c0, h, w) of a corner rect in the policy's 224x224 frame."""
    n = POLICY_SIDE
    return {
        "TL": (0, 0, size, size),
        "TR": (0, n - size, size, size),
        "BL": (n - size, 0, size, size),
        "BR": (n - size, n - size, size, size),
    }[corner]


def calibrate_orientation(rendered: ImageArray, observed: ImageArray) -> str:
    """Return the flip mapping `sim.render` output onto `obs["agentview_image"]`.

    MuJoCo renders bottom-up while robosuite's observable pipeline may already have flipped
    the frame, and the answer decides which corner of the image the rect refers to. Rather
    than assume, score all four flips and require the winner to be decisively better.
    """
    reference = observed.astype(np.float64)
    scores = {
        name: float(np.mean(np.abs(flip(rendered).astype(np.float64) - reference)))
        for name, flip in _FLIPS.items()
    }
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    best, runner_up = ranked[0], ranked[1]
    if best[1] > 1.0 and best[1] > 0.5 * runner_up[1]:
        raise RuntimeError(
            f"could not calibrate render orientation; mean-abs-diff per flip = {scores}. "
            "Refusing to guess: the corner identity of every non-occlusion claim depends on it."
        )
    return best[0]


def to_policy_space(seg_raw: IntArray, flip: str) -> IntArray:
    """Map a raw segmentation frame into the policy's 224x224 orientation and scale."""
    as_obs = _FLIPS[flip](seg_raw)
    rotated = as_obs[::-1, ::-1]  # get_libero_image's documented 180-degree rotation
    index = (np.arange(POLICY_SIDE) * (rotated.shape[0] / POLICY_SIDE)).astype(int)
    resampled: IntArray = np.asarray(rotated[index][:, index])
    return resampled


def dilate(mask: BoolArray) -> BoolArray:
    """1-pixel 4-neighbour dilation, so resampling slop can only over-report overlap."""
    out: BoolArray = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def geom_names(env: Any) -> dict[int, str]:
    """geom id -> name for every named geom in the model."""
    model = env.sim.model
    names: dict[int, str] = {}
    for geom_id in range(int(model.ngeom)):
        try:
            name = model.geom_id2name(geom_id)
        except Exception:  # noqa: BLE001 - unnamed geoms are expected and simply skipped
            name = None
        if name:
            names[geom_id] = str(name)
    return names


def entity_masks(
    seg: IntArray, names: dict[int, str]
) -> tuple[dict[str, BoolArray], dict[str, list[str]]]:
    """Boolean policy-space masks per entity, plus the geom names each one matched."""
    masks: dict[str, BoolArray] = {}
    matched: dict[str, list[str]] = {}
    for entity, patterns in ENTITY_PATTERNS.items():
        ids = [gid for gid, name in names.items() if any(p in name.lower() for p in patterns)]
        masks[entity] = np.isin(seg, ids) if ids else np.zeros_like(seg, dtype=bool)
        matched[entity] = sorted({names[gid] for gid in ids})
    objects = ("user_object", "target_object", "other_objects", "basket")
    masks["any_task_object"] = np.logical_or.reduce([masks[k] for k in objects])
    matched["any_task_object"] = sorted({n for k in objects for n in matched[k]})
    return masks, matched


def bbox(mask: BoolArray) -> tuple[int, int, int, int] | None:
    """(r0, r1, c0, c1) inclusive bounds of a boolean mask, or None if empty."""
    rows, cols = np.where(mask)
    if rows.size == 0:
        return None
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def measure(seg: IntArray, names: dict[int, str]) -> dict[str, Any]:
    """Overlap of every corner rect with every entity, for one init's segmentation."""
    masks, matched = entity_masks(seg, names)
    dilated = {entity: dilate(mask) for entity, mask in masks.items()}

    overlaps: list[dict[str, Any]] = []
    for corner in CORNERS:
        for size in CORNER_SIZES:
            r0, c0, height, width = corner_rect(corner, size)
            window = (slice(r0, r0 + height), slice(c0, c0 + width))
            per_entity = {
                entity: int(mask[window].sum()) for entity, mask in dilated.items()
            }
            overlaps.append({
                "corner": corner,
                "size": size,
                "rect": [r0, c0, height, width],
                "area_frac": (height * width) / (POLICY_SIDE * POLICY_SIDE),
                "overlap_px": per_entity,
                "occludes_any_object": per_entity["any_task_object"] > 0,
            })
    return {
        "matched_geoms": matched,
        "entity_pixels": {entity: int(mask.sum()) for entity, mask in masks.items()},
        "entity_bbox": {entity: bbox(mask) for entity, mask in masks.items()},
        "overlaps": overlaps,
    }


def probe_init(backend: HijackBackend, env: Any, init_states: Any, cfg: Any, init: int,
               flip: str | None) -> tuple[dict[str, Any], str]:
    """Settle init `init` exactly as a scored episode does, then measure it."""
    obs = env.set_init_state(init_states[init % len(init_states)])
    for _ in range(backend.num_steps_wait):
        obs, _r, _d, _i = env.step(backend._dummy_action(cfg))

    if flip is None:
        rendered = np.asarray(
            env.sim.render(width=RENDER_SIDE, height=RENDER_SIDE, camera_name="agentview")
        )
        flip = calibrate_orientation(rendered, np.asarray(obs["agentview_image"]))
        print(f"[occlusion] render orientation calibrated: {flip}", flush=True)

    seg = to_policy_space(np.asarray(backend._segmentation(env)), flip)
    result = measure(seg, geom_names(env))
    result["init"] = init
    result["split"] = split_of(init)
    result["orientation"] = flip
    return result, flip


def legacy_crosscheck(result: dict[str, Any]) -> dict[str, Any]:
    """Compare the measured graspable-object box against the legacy hardcoded keep-out."""
    boxes = [result["entity_bbox"][k] for k in ("user_object", "target_object")]
    present = [b for b in boxes if b is not None]
    if not present:
        return {"status": "no objects segmented"}
    measured = (
        min(b[0] for b in present), max(b[1] for b in present),
        min(b[2] for b in present), max(b[3] for b in present),
    )
    return {
        "measured_object_box": list(measured),
        "legacy_keepout": list(LEGACY_KEEPOUT),
        "max_edge_difference_px": max(
            abs(m - lg) for m, lg in zip(measured, LEGACY_KEEPOUT, strict=True)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measured per-init non-occlusion probe.")
    parser.add_argument("--user-task", default=USER_TASK)
    parser.add_argument("--inits", default=None, help="comma-separated override")
    parser.add_argument("--crosscheck-init", type=int, default=0,
                        help="init to compare against the legacy keep-out (-1 to skip)")
    parser.add_argument("--out", default=os.path.join(HOME, "autoresearch/runs/monitor-stealth"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shared_inits.verify_precommit()
    os.makedirs(args.out, exist_ok=True)

    inits = ([int(x) for x in args.inits.split(",")] if args.inits
             else list(shared_inits.SHARED_INITS))
    if args.crosscheck_init >= 0 and args.crosscheck_init not in inits:
        inits = [args.crosscheck_init, *inits]

    backend = HijackBackend(run_dir=args.out)
    resolved = resolve_task(args.user_task, suite=backend.task_suite)
    env, init_states, _desc, _obj = backend._build_env(resolved)
    cfg = backend._build_cfg()

    print(f"[occlusion] {shared_inits.summary()}", flush=True)
    results: list[dict[str, Any]] = []
    flip: str | None = None
    for init in inits:
        result, flip = probe_init(backend, env, init_states, cfg, init, flip)
        results.append(result)
        worst = [o for o in result["overlaps"] if o["occludes_any_object"]]
        print(
            f"[occlusion] init={init:02d} ({result['split']:>7}) "
            f"object_box={result['entity_bbox']['any_task_object']} "
            f"rects_occluding={len(worst)}/{len(result['overlaps'])}",
            flush=True,
        )
    env.close()

    crosscheck = (
        legacy_crosscheck(next(r for r in results if r["init"] == args.crosscheck_init))
        if args.crosscheck_init >= 0 else None
    )
    # A rect is CLEAR only if it is clear on every init measured -- the point of the upgrade.
    clear: dict[str, bool] = {}
    for corner in CORNERS:
        for size in CORNER_SIZES:
            key = f"{corner}:{size}"
            clear[key] = all(
                not o["occludes_any_object"]
                for r in results for o in r["overlaps"]
                if o["corner"] == corner and o["size"] == size
            )

    summary = {
        "precommit": shared_inits.summary(),
        "user_task": args.user_task,
        "n_inits": len(results),
        "inits": [r["init"] for r in results],
        "orientation": flip,
        "clear_across_all_inits": clear,
        "legacy_crosscheck": crosscheck,
        "per_init": results,
    }
    out_path = os.path.join(args.out, "occlusion.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n===== NON-OCCLUSION (clear on EVERY measured init) =====", flush=True)
    for key, is_clear in clear.items():
        print(f"  {key:>7}  {'CLEAR' if is_clear else 'OCCLUDES'}", flush=True)
    if crosscheck:
        print(f"\n[occlusion] legacy cross-check: {crosscheck}", flush=True)
    print(f"[occlusion] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
