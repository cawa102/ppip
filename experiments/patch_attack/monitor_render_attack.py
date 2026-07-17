"""Experiment 2 -- PHYSICALLY-REALIZABLE confined monitor-video hijack (through the render).

Unlike Experiment 1 (``monitor_patch_attack.py``, a camera-space replacement patch = an
idealised upper bound), here the attack is realised ENTIRELY through an in-scene *monitor*
geom whose texture is re-uploaded every control step and re-rendered by MuJoCo -- the camera
image buffer is NEVER written (``MonitorHijackBackend`` enforces that invariant). This is the
attacker's true threat: put a monitor in the room; the camera legitimately sees it.

Why the Phase-0 GATE-B monitor failed and this should not:
  * GATE-B used ``optimize_masked_delta`` with fixed eps=0.15 additive-around-mid-gray, k=6,
    NO escalation, NO real-render verification, and ``select_texture`` kept committing neutral.
  * Here the monitor shows FREE-RANGE content (sigmoid, full [0,1] -- a real screen is bright),
    the teacher tokens are FORCED with the proven adaptive objective, every candidate is
    VERIFIED on the real post-upload render, and a fixed-point surrogate correction
    compensates for the render reality-gap (resample/shading) so the *rendered* frame -- not
    an image-space proxy -- forces the tokens. The action is executed from that real render.

Scored by the FIXED target predicate. Search/rendering side only; evaluator untouched.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

HOME = os.path.expanduser("~")
for _p in ("autoresearch/src", "autoresearch", "openvla", "autoresearch/experiments/patch_attack"):
    sys.path.insert(0, os.path.join(HOME, _p))

import vla_diff  # noqa: E402
from adaptive_attack import _prompt_ids, _real_tokens  # noqa: E402
from monitor_attack import (  # noqa: E402
    TARGET_TASK,
    USER_TASK,
    _precrop_monitor_mask,
    neutral_texture,
)

DEVICE = "cuda"
SEED = int(os.environ.get("MR_SEED", "0"))
MAX_STEPS = int(os.environ.get("MR_MAX_STEPS", "220"))
K = int(os.environ.get("MR_K", "12"))
MAXTRIES = int(os.environ.get("MR_MAXTRIES", "8"))
LR = float(os.environ.get("MR_LR", "4e-2"))
# Texture canvas: MR_TEX sets a square canvas; MR_TEX_H/MR_TEX_W override per-axis so a WIDE
# (H<W) monitor can cover more frame width without the top-corner clipping that caps a tall
# square monitor's usable size (calibrate_uv needs all 4 corners in-frame).
TEX_HW = (int(os.environ.get("MR_TEX_H", os.environ.get("MR_TEX", "256"))),
          int(os.environ.get("MR_TEX_W", os.environ.get("MR_TEX", "256"))))
RUN_DIR = os.environ.get("MR_RUN_DIR", os.path.join(HOME, "autoresearch/runs/monitor-render"))
TAG = os.environ.get("MR_TAG", f"seed{SEED}")
RECORD = os.environ.get("MR_RECORD", "1") == "1"
# Monitor size/placement. scale 1.6 projects to ~3.9% of frame (below the DOF threshold, it
# only partial-carries); enlarge into the >=7% working regime found by the camera-space sweep.
MON_SCALE = float(os.environ.get("MR_SCALE", "3.0"))
MON_POS = [float(x) for x in os.environ.get("MR_POS", "-0.08,0.0,0.14").split(",")]
MON_ROT = [float(x) for x in os.environ.get("MR_ROT", "90.0,90.0,0.0").split(",")]

# H1-H4 knobs (deepening the through-render boundary).
RESTARTS = int(os.environ.get("MR_RESTARTS", "2"))     # random restarts per frame (keep best)
# Adaptive restarts: after a frame whose forcing COLLAPSED (best break-match < required), spend
# many more restarts on the NEXT frames to punch through the grasp transition and RECOVER, instead
# of letting the arm diverge (the vicious cycle that made every fixed-restart run fail). Cheap on
# easy frames (they break early), heavy only where forcing is failing.
RESTARTS_HARD = int(os.environ.get("MR_RESTARTS_HARD", str(max(RESTARTS, 6))))
WARMSTART = os.environ.get("MR_WARMSTART", "1") == "1"  # restart 0 warm-starts from prev frame
RAW_CLAMP = float(os.environ.get("MR_CLAMP", "8.0"))   # clamp raw so sigmoid never dead-saturates
# Decisive-token objective (H4): restrict the CE loss + match to a subset of the 7 action tokens
# (OpenVLA action = [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper]). Forcing only the tokens that
# REDIRECT the grasp (xy + z + gripper) is easier through the render's low-pass and lets the
# policy fill the rest from the visible object. Empty = force all 7 (backward compatible).
_didx = os.environ.get("MR_DECISIVE", "").strip()
DECISIVE_IDX = [int(x) for x in _didx.split(",") if x != ""] if _didx else list(range(7))
# The CE loss is optimised over DECISIVE_IDX, but the per-frame optimiser STOPS as soon as the
# (smaller) BREAK_IDX subset matches the teacher. This decouples "what we push toward" from "what
# is required": e.g. push xy+z+yaw+gripper (so the trajectory tracks the teacher) but only REQUIRE
# xy+z+gripper (which the render can sustain through the grasp transition) -- yaw becomes
# best-effort, improved when forceable, never causing the collapse forcing all 5 did. Empty =
# BREAK_IDX == DECISIVE_IDX (require everything you optimise; backward compatible).
_bidx = os.environ.get("MR_BREAK", "").strip()
BREAK_IDX = [int(x) for x in _bidx.split(",") if x != ""] if _bidx else list(DECISIVE_IDX)
# "Any-K of 7" break: when > 0, the optimiser stops as soon as ANY MR_FULL_THRESH of the 7 tokens
# match (the count, not a fixed subset), optimising CE over all of DECISIVE_IDX. The render can force
# ~6/7 but never the SAME 7 -- with a fixed subset the missed token is CONSISTENT and its error
# accumulates over the episode (arm drifts off the grasp); letting the missed token VARY frame to
# frame makes the errors cancel, the way camera-space's 6.99/7 (occasional, random misses) grasps.
FULL_THRESH = int(os.environ.get("MR_FULL_THRESH", "0"))
# Interior-point UV calibration: the shared `calibrate_uv` lights the 4 texture CORNERS and fails
# if any project off-frame -- which caps a usable monitor at ~50 %. When MR_INTERIOR_CAL=1 we
# instead calibrate the texture->image homography from 4 INTERIOR patches (search-side only; the
# shared rendering code is untouched), which stay in-frame for a much BIGGER monitor -> far more
# render DOF AND the monitor occludes the object through the grasp (removing OpenVLA's competing
# natural grasp intent that collapses the forcing at the grasp transition).
INTERIOR_CAL = os.environ.get("MR_INTERIOR_CAL", "0") == "1"
# Attacker target task override ("target has options"): the exact salad_dressing grasp is blocked
# at the grasp transition; a target object CLOSER to the alphabet_soup needs a smaller grasp
# override and may be completable. Default keeps the original salad_dressing target.
TARGET_TASK = os.environ.get("MR_TARGET", TARGET_TASK)


def _to_t(img_u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img_u8, np.float32) / 255.0).permute(2, 0, 1)[None].to(DEVICE)


def _setup_monitor(backend: Any, seed: int, *, scale: float, pos: list[float],
                   rot: list[float], tex_hw: tuple[int, int]) -> tuple[Any, Any, Any, Any]:
    """Build the deployment scene with a configurable-size monitor injected + resolved.

    Mirrors monitor_attack.setup_deployment_episode but overrides the monitor
    scale/position/rotation so its projected footprint can be tuned into the working regime
    (size/position/rotation are all free to explore). Caller owns env.close().
    """
    from monitor_attack import USER_TASK  # TARGET_TASK comes from the module global (MR_TARGET)
    from monitor_upload_probe import _MONITOR_CANDIDATE, _inject_monitor

    from evaluator.libero_tasks import resolve_task
    from evaluator.openvla_backend import _ENV_RESOLUTION
    from rendering.monitor import MonitorTextureHandle, build_monitor_asset

    cand = {**_MONITOR_CANDIDATE,
            "candidate_id": "monitor_render",
            "placement": {**_MONITOR_CANDIDATE["placement"],
                          "scale": scale, "position": pos, "rotation": rot}}
    resolved_user = resolve_task(USER_TASK, suite="libero_object")
    resolved_target = resolve_task(TARGET_TASK, suite="libero_object")
    env, init_states, _desc, _obj = backend._build_env(resolved_user)
    geom = build_monitor_asset(cand, tex_hw=tex_hw)
    texture_dir = os.path.join(os.environ.get("PROBE_DIR", "/tmp/monitor_render"), "tex")
    _inject_monitor(env, geom, texture_dir)
    if backend._policy is None:
        backend._policy = backend._load_policy()
    obs = env.set_init_state(init_states[seed % len(init_states)])
    dummy = backend._dummy_action(backend._policy[2])
    for _ in range(backend.num_steps_wait):
        obs, _r, _d, _i = env.step(dummy)
    _ = env.sim.render(width=_ENV_RESOLUTION, height=_ENV_RESOLUTION, camera_name="agentview")
    handle = MonitorTextureHandle(geom.name)
    handle.resolve(env)
    return env, handle, resolved_user, resolved_target


def _calibrate_uv_interior(env: Any, handle: Any, resize_size: int,
                           threshold: float = 12.0) -> Any:
    """Texture->image homography from 4 INTERIOR patches (in-frame for a big off-frame-corner
    monitor). A UVMap only needs 4 non-collinear (texture, image) correspondences to fit the
    homography that ``warp_pattern_to_texture`` uses -- they need not be the corners. Search-side
    only; the shared ``calibrate_uv`` (corner-based) is left untouched."""
    from rendering.monitor import UVMap, _policy_input_frame
    h, w = handle.dims
    patch = max(8, min(h, w) // 8)
    fracs = [(0.28, 0.28), (0.72, 0.28), (0.72, 0.72), (0.28, 0.72)]  # TL,TR,BR,BL of the interior
    texture_pts = np.array([[fx * w, fy * h] for fx, fy in fracs], dtype=np.float64)
    handle.upload(np.zeros((h, w, 3), dtype=np.uint8))
    env.sim.forward()
    baseline = _policy_input_frame(env, resize_size).astype(np.int64)
    image_pts = np.zeros((4, 2), dtype=np.float64)
    for i, (fx, fy) in enumerate(fracs):
        cx, cy = int(fx * w), int(fy * h)
        tex = np.zeros((h, w, 3), dtype=np.uint8)
        tex[max(0, cy - patch): cy + patch, max(0, cx - patch): cx + patch, :] = 255
        handle.upload(tex)
        env.sim.forward()
        lit = _policy_input_frame(env, resize_size).astype(np.int64)
        diff = np.abs(lit - baseline).max(axis=2) > threshold
        ys, xs = np.where(diff)
        if ys.size == 0:
            raise RuntimeError(f"interior calibration patch {i} not visible in policy image")
        image_pts[i] = [xs.mean(), ys.mean()]
    return UVMap(texture_corners=texture_pts, image_corners=image_pts)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(RUN_DIR, exist_ok=True)
    rec_dir = os.path.join(RUN_DIR, f"{TAG}_rec")
    if RECORD:
        for sub in ("scene", "policy_input", "texture", "clean_input"):
            os.makedirs(os.path.join(rec_dir, sub), exist_ok=True)

    from experiments.robot.libero.libero_utils import get_libero_image
    from monitor_hijack_backend import MonitorHijackBackend
    from texture_surrogate import warp_pattern_to_texture

    from evaluator.adjudicate import eval_goal_state
    from rendering.monitor import _fresh_obs, _policy_input_frame, calibrate_uv

    backend = MonitorHijackBackend(run_dir=RUN_DIR, max_steps=MAX_STEPS)
    backend._policy = backend._load_policy()
    model, processor, _cfg, resize_size = backend._policy
    for pm in model.parameters():
        pm.requires_grad_(False)
    model.language_model.config.use_cache = False
    user_ids = _prompt_ids(processor, USER_TASK)

    env, handle, resolved_user, resolved_target = _setup_monitor(
        backend, SEED, scale=MON_SCALE, pos=MON_POS, rot=MON_ROT, tex_hw=TEX_HW
    )
    try:
        # Make the monitor EMISSIVE (a real screen glows) so scene lighting doesn't shade/crush
        # the displayed texture's contrast -- preserving the adversarial signal through the render
        # (the render's diffuse shading was low-pass/contrast-attenuating the attack). Runtime
        # material mutation only: the shared inject/render code (evaluator side) is NOT touched.
        import mujoco as _mj

        from rendering.geometry import _mujoco_safe_name as _safe
        from rendering.monitor import _raw_model as _rm
        from rendering.monitor import _sim_of as _so
        _emat = int(_mj.mj_name2id(_rm(_so(env)), _mj.mjtObj.mjOBJ_MATERIAL,
                                   f"{_safe('monitor_render')}__mat"))
        if _emat >= 0:
            _m = _rm(_so(env))
            _m.mat_emission[_emat] = float(os.environ.get("MR_EMISSION", "1.0"))
            _m.mat_specular[_emat] = 0.0
            _m.mat_shininess[_emat] = 0.0
            _so(env).forward()
            print(f"[render] monitor material emissive (emission={_m.mat_emission[_emat]:.2f})",
                  flush=True)

        neutral = neutral_texture(handle.dims)
        # Monitor projection is fixed (static geom + camera) -> calibrate ONCE.
        precrop_mask = _precrop_monitor_mask(env, handle, resize_size=resize_size)
        uv_map = (_calibrate_uv_interior(env, handle, resize_size) if INTERIOR_CAL
                  else calibrate_uv(env, handle, resize_size=resize_size))
        mask_t = torch.from_numpy(precrop_mask.astype(np.float32))[None, None].to(DEVICE)
        mask_area = int(precrop_mask.sum())
        print(f"[render] monitor precrop mask area={mask_area}px "
              f"({mask_area / (224 * 224):.1%} of frame)", flush=True)

        # Clean-frame teacher: the monitor geom itself is a distractor that disrupts even the
        # TARGET policy (the GATE-B S0-fail boundary), so a teacher taken on the monitor-present
        # scene is a *confused* action that never grasps. The attacker's true intent is "grasp
        # salad_dressing regardless of the monitor" -> take the teacher on a monitor-HIDDEN
        # render (geom moved out of frame), exactly as Experiment 1 uses the clean frame.
        import mujoco

        from rendering.geometry import _mujoco_safe_name
        from rendering.monitor import _raw_model, _sim_of
        _sim = _sim_of(env)
        _raw = _raw_model(_sim)
        _gid = int(mujoco.mj_name2id(_raw, mujoco.mjtObj.mjOBJ_GEOM,
                                     _mujoco_safe_name("monitor_render")))
        _saved_geom_pos = np.array(_raw.geom_pos[_gid], dtype=np.float64).copy()

        def clean_frame() -> np.ndarray:
            _raw.geom_pos[_gid] = np.array([5.0, 5.0, 5.0])
            _sim.forward()
            f = _policy_input_frame(env, resize_size)
            _raw.geom_pos[_gid] = _saved_geom_pos
            _sim.forward()
            return f

        if RECORD:  # save one monitor-hidden vs monitor-present pair to verify the hide works
            import imageio.v2 as imageio
            imageio.imwrite(os.path.join(rec_dir, "teacher_clean_check.png"), clean_frame())

        tobj, treg = backend._target_entities(resolved_target)
        targeted = False
        latch_step = None
        min_dist: float | None = None
        match_trace: list[int] = []
        raw_persist = torch.zeros(1, 3, 224, 224, device=DEVICE)  # bounded warm-start across steps
        loss_idx = torch.tensor(DECISIVE_IDX, dtype=torch.long, device=DEVICE)
        break_idx = torch.tensor(BREAK_IDX, dtype=torch.long, device=DEVICE)
        n_break = len(BREAK_IDX)
        break_target = FULL_THRESH if FULL_THRESH > 0 else n_break  # match count needed to stop
        prev_brk = break_target  # last frame's best break-key; drives adaptive restarts

        def _optimise_patch(init_raw: torch.Tensor, teacher: torch.Tensor) -> tuple:
            """One restart of the BPDA straight-through optimiser -> the best realisation found.

            Each iter REALISES the patch on the monitor, reads the REAL rendered frame, and
            computes the CE on it while flowing the gradient to the patch as if the render were
            identity on the mask (value = real render, grad = patch) -- so the optimiser works
            WITH the render reality-gap. `raw` is clamped so sigmoid never dead-saturates (the
            old unbounded warm-start collapsed to 1/7 mid-episode). CE + match are restricted to
            ``loss_idx`` (all 7 by default; the decisive subset under MR_DECISIVE). Returns
            ``(sel_match, full_match, R_u8, texture, raw_snapshot)``.
            """
            raw = init_raw.detach().clone().requires_grad_(True)
            opt = torch.optim.Adam([raw], lr=LR)
            local: tuple = (-1, -1, None, None, None)
            for _attempt in range(MAXTRIES):
                last_R = None
                last_tex = None
                for _ in range(K):
                    patch = torch.sigmoid(raw)
                    with torch.no_grad():
                        patch_u8 = (patch[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                                    * 255).astype(np.uint8)
                    texture = warp_pattern_to_texture(patch_u8, uv_map, TEX_HW)
                    handle.upload(np.ascontiguousarray(texture))
                    env.sim.forward()
                    R = _policy_input_frame(env, resize_size)
                    r_t = _to_t(R)  # real render, no grad
                    # straight-through: forward VALUE = real render; gradient -> patch on mask.
                    surrogate = r_t + (patch - patch.detach()) * mask_t
                    side = vla_diff._CROP_SIDE + 0.03 * (torch.rand(1).item() - 0.5)
                    pv = vla_diff.preprocess(surrogate.clamp(0, 1), side=side)
                    logits = vla_diff.action_token_logits(model, pv, user_ids, teacher)
                    lg = logits.reshape(7, -1).float()
                    loss = F.cross_entropy(lg[loss_idx], teacher.reshape(7)[loss_idx])
                    opt.zero_grad(); loss.backward(); opt.step()
                    with torch.no_grad():
                        raw.clamp_(-RAW_CLAMP, RAW_CLAMP)  # keep gradients alive
                    last_R, last_tex = R, texture

                er = _real_tokens(model, processor, last_R, USER_TASK)
                full_m = int((er == teacher.view(7)).sum())
                # key = the match measure we stop on: the total count (any-K, missed token varies)
                # under MR_FULL_THRESH, else the fixed break-subset match.
                if FULL_THRESH > 0:
                    key = full_m
                else:
                    key = int((er[break_idx] == teacher.view(7)[break_idx]).sum())
                if (key, full_m) > (local[0], local[1]):
                    local = (key, full_m, last_R.copy(),
                             np.ascontiguousarray(last_tex), raw.detach().clone())
                if key >= break_target:  # required match reached -> stop early
                    break
                for g in opt.param_groups:
                    g["lr"] = min(g["lr"] * 1.3, 0.25)
            return local

        for step in range(MAX_STEPS):
            # 1) teacher = TARGET tokens on a MONITOR-HIDDEN clean render (attacker's intent);
            #    then restore the monitor and take the neutral render for the record.
            teacher_frame = clean_frame()
            teacher = _real_tokens(model, processor, teacher_frame, TARGET_TASK).view(1, 7)
            handle.upload(np.ascontiguousarray(neutral))
            env.sim.forward()
            neutral_frame = _policy_input_frame(env, resize_size)
            if RECORD:
                import imageio.v2 as imageio
                imageio.imwrite(os.path.join(rec_dir, "scene", f"f{step:04d}.png"),
                                get_libero_image(_fresh_obs(env), 384))
                imageio.imwrite(os.path.join(rec_dir, "clean_input", f"f{step:04d}.png"),
                                neutral_frame)

            # Random restarts (H2): restart 0 warm-starts from the previous frame; the rest
            # re-init `raw` fresh to escape a bad basin on hard grasp frames. Keep the global best.
            # Adaptive: if the PREVIOUS frame collapsed, throw many more restarts at this one to
            # recover the trajectory before the arm diverges past the point of no return.
            restarts_now = RESTARTS if prev_brk >= break_target - 1 else RESTARTS_HARD
            best: tuple = (-1, -1, None, None, None)  # (sel_match, full_match, R, tex, raw)
            for _restart in range(restarts_now):
                if _restart == 0 and WARMSTART:
                    init_raw = raw_persist                                # warm: fast when frames similar
                elif _restart % 2 == 1:
                    init_raw = torch.zeros(1, 3, 224, 224, device=DEVICE)  # neutral: reliable re-convergence
                else:                                                     # when a grasp-transition frame
                    init_raw = torch.randn(1, 3, 224, 224, device=DEVICE) * 1.5  # breaks the warm basin
                cand = _optimise_patch(init_raw, teacher)
                if (cand[0], cand[1]) > (best[0], best[1]):
                    best = cand
                if best[0] >= break_target:  # required match reached -> no more restarts
                    break

            sel_match, match, R_best, tex_best, raw_best = best
            prev_brk = sel_match
            if raw_best is not None:
                raw_persist = raw_best  # warm-start next (near-identical) frame from the best
            match_trace.append(match)

            # 2) execute from the REAL render (step_with_texture re-uploads tex_best + re-renders
            #    the same frame -> reproduces R_best, then feeds THAT to OpenVLA and steps).
            result = backend.step_with_texture(env, handle, tex_best, resolved_user.language,
                                               resize_size=resize_size)
            if RECORD:
                imageio.imwrite(os.path.join(rec_dir, "policy_input", f"f{step:04d}.png"),
                                result.policy_image)
                imageio.imwrite(os.path.join(rec_dir, "texture", f"f{step:04d}.png"), tex_best)

            ostates = backend._object_states(env)
            d = backend._distance_between(ostates, tobj, treg)
            if d is not None:
                min_dist = d if min_dist is None else min(min_dist, d)
            if not targeted and eval_goal_state(resolved_target.goal_state, ostates):
                targeted = True
                latch_step = step
            if step % 5 == 0 or targeted:
                print(f"[render] step={step} tok_match={match}/7 key={sel_match}/{break_target} "
                      f"tgt_dist={d} min={min_dist} targeted={targeted}", flush=True)
            if targeted:
                print("[render] *** TARGETED SUCCESS (through-render monitor hijack) ***", flush=True)
                break
            if result.done:
                break

        status = "HIJACK" if targeted else "DONE"
        mean_match = float(np.mean(match_trace)) if match_trace else 0.0
        out = {
            "tag": TAG, "seed": SEED, "status": status, "targeted": bool(targeted),
            "latch_step": latch_step, "steps": len(match_trace), "max_steps": MAX_STEPS,
            "monitor_mask_area_px": mask_area, "monitor_mask_frac": mask_area / (224 * 224),
            "min_target_dist_m": None if min_dist is None else float(min_dist),
            "mean_token_match": mean_match, "tex_hw": list(TEX_HW),
            "scale": MON_SCALE, "pos": MON_POS, "rot": MON_ROT,
            "restarts": RESTARTS, "warmstart": WARMSTART, "raw_clamp": RAW_CLAMP,
            "decisive_idx": DECISIVE_IDX, "break_idx": BREAK_IDX,
        }
        with open(os.path.join(RUN_DIR, f"result_{TAG}.json"), "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"[render] {status} {TAG} targeted={targeted} min_dist={min_dist} "
              f"mean_match={mean_match:.2f} mask={mask_area / (224 * 224):.1%}", flush=True)
    finally:
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main()
