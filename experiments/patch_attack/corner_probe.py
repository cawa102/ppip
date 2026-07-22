"""Cheap leverage probe for the CORNER-confined monitor-patch hijack.

Question: can a per-step optimised replacement patch placed in a *corner* (NOT covering the
graspable objects) force OpenVLA's action tokens toward the attacker target
(salad_dressing) on realistic rollout frames? This ranks corners/sizes WITHOUT paying for a
full closed-loop rollout: it reuses saved clean agentview frames and, per (corner, size,
frame), runs the SAME proven per-step optimisation as ``monitor_patch_attack`` and reports
the REAL-inference-path token match (out of 7).

Object keep-out box (graspable soup + salad_dressing, seed-0 init frame): rows 95..170,
cols 100..218. Any corner rect that does not intersect this box provably does not cover the
objects; the probe asserts non-overlap and refuses object-covering rects.

Env-configurable:
  CP_CORNERS="TL,TR,BL"   CP_SIZES="95,64"   CP_FRAMES="0,30,60,90"
  CP_K=10 CP_MAXTRIES=6 CP_LR=3e-2
Run:
  CP_CORNERS=TL,TR,BL CP_SIZES=95 CP_FRAMES=0,30,60,90 CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl \
    PYTHONPATH=$HOME/LIBERO ~/vla-injection/.venv/bin/python \
    experiments/patch_attack/corner_probe.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HOME = os.path.expanduser("~")
for _p in ("autoresearch/src", "autoresearch", "openvla", "autoresearch/experiments/patch_attack"):
    sys.path.insert(0, os.path.join(HOME, _p))

import patch_config as C  # noqa: E402
import vla_diff  # noqa: E402
from adaptive_attack import _prompt_ids, _real_tokens  # noqa: E402
from hijack_backend import HijackBackend  # noqa: E402

DEVICE = "cuda"
KEEPOUT = (95, 170, 100, 218)  # (r0,r1,c0,c1) graspable-object box; corner rects must avoid it
FRAME_DIR = os.environ.get(
    "CP_FRAME_DIR", os.path.join(HOME, "autoresearch/runs/monitor-patch/run2_rec/clean_input")
)
RUN_DIR = os.environ.get("CP_RUN_DIR", os.path.join(HOME, "autoresearch/runs/monitor-corner"))


def corner_rect(corner: str, s: int) -> tuple[int, int, int, int]:
    n = 224
    return {
        "TL": (0, 0, s, s),
        "TR": (0, n - s, s, s),
        "BL": (n - s, 0, s, s),
        "BR": (n - s, n - s, s, s),
    }[corner]


def _intersects_keepout(rect: tuple[int, int, int, int]) -> bool:
    r0, c0, h, w = rect
    kr0, kr1, kc0, kc1 = KEEPOUT
    return not (r0 + h <= kr0 or r0 >= kr1 or c0 + w <= kc0 or c0 >= kc1)


def _rect_mask(rect: tuple[int, int, int, int]) -> torch.Tensor:
    r0, c0, ph, pw = rect
    m = torch.zeros(1, 1, 224, 224, device=DEVICE)
    m[:, :, r0 : min(r0 + ph, 224), c0 : min(c0 + pw, 224)] = 1.0
    return m


def probe_cell(
    model, processor, user_ids, image_u8: np.ndarray, rect: tuple[int, int, int, int],
    *, k: int, maxtries: int, lr: float,
    user_task: str = C.USER_TASK, restarts: int = 1,
) -> dict:
    """Optimise a corner replacement patch on one frame; return real-path token match.

    ``user_task`` is the instruction the patch must fool (defaults to the original
    alphabet-soup user task, so existing callers are unchanged); ``user_ids`` must be
    the tokenised prompt for that same instruction. ``restarts`` re-runs the inner
    optimisation from a different basin and keeps the best real-path match, matching
    the escalated effort used for the 64x64 corner hijack.

    Also reports the *decisive* dims (where the un-attacked user-instructed policy and
    the target teacher disagree on this frame) and how many of them were forced --
    ``mean_token_match`` is inflated by agreement and must not be read as progress.
    """
    img224 = torch.from_numpy(image_u8.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(DEVICE)
    mask = _rect_mask(rect)
    teacher = _real_tokens(model, processor, image_u8, C.TARGET_TASK).view(1, 7)
    clean_user = _real_tokens(model, processor, image_u8, user_task).view(7)
    dec_dims = [i for i in range(7) if int(clean_user[i]) != int(teacher.view(7)[i])]
    best: tuple[int, torch.Tensor | None] = (-1, None)
    for _restart in range(max(1, restarts)):
        raw = ((torch.zeros(1, 3, 224, 224, device=DEVICE) if _restart == 0
                else torch.randn(1, 3, 224, 224, device=DEVICE) * 1.5).requires_grad_(True))
        opt = torch.optim.Adam([raw], lr=lr)
        for _attempt in range(maxtries):
            for _ in range(k):
                patch01 = torch.sigmoid(raw)
                composite = (img224 * (1 - mask) + patch01 * mask).clamp(0, 1)
                side = vla_diff._CROP_SIDE + 0.03 * (torch.rand(1).item() - 0.5)
                pv = vla_diff.preprocess(composite, side=side)
                logits = vla_diff.action_token_logits(model, pv, user_ids, teacher)
                loss = F.cross_entropy(logits.reshape(7, -1).float(), teacher.reshape(7))
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                patch01 = torch.sigmoid(raw)
                composite = (img224 * (1 - mask) + patch01 * mask).clamp(0, 1)
                pu8 = (composite[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            er = _real_tokens(model, processor, pu8, user_task)
            m = int((er == teacher.view(7)).sum())
            if m > best[0]:
                best = (m, er)
            if m == 7:
                break
            for g in opt.param_groups:
                g["lr"] = min(g["lr"] * 1.5, 0.3)
        if best[0] == 7:
            break
    er_best = best[1]
    dec_hits = (0 if er_best is None
                else sum(1 for i in dec_dims if int(er_best[i]) == int(teacher.view(7)[i])))
    return {"match": best[0], "n_decisive": len(dec_dims), "decisive_hits": dec_hits}


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    os.makedirs(RUN_DIR, exist_ok=True)
    corners = os.environ.get("CP_CORNERS", "TL,TR,BL").split(",")
    sizes = [int(x) for x in os.environ.get("CP_SIZES", "95").split(",")]
    frames = [int(x) for x in os.environ.get("CP_FRAMES", "0,30,60,90").split(",")]
    k = int(os.environ.get("CP_K", "10"))
    maxtries = int(os.environ.get("CP_MAXTRIES", "6"))
    lr = float(os.environ.get("CP_LR", "3e-2"))

    backend = HijackBackend(run_dir=RUN_DIR, max_steps=200)
    model, processor, cfg, resize_size = backend._load_policy()
    for pm in model.parameters():
        pm.requires_grad_(False)
    model.language_model.config.use_cache = False
    user_ids = _prompt_ids(processor, C.USER_TASK)

    imgs = {f: np.array(Image.open(os.path.join(FRAME_DIR, f"f{f:04d}.png")).convert("RGB"))
            for f in frames}

    results = []
    for corner in corners:
        for s in sizes:
            rect = corner_rect(corner, s)
            if _intersects_keepout(rect):
                print(f"[probe] SKIP {corner}:{s} rect={rect} hits object keep-out", flush=True)
                continue
            matches = []
            for f in frames:
                cell = probe_cell(model, processor, user_ids, imgs[f], rect,
                                  k=k, maxtries=maxtries, lr=lr)
                matches.append(cell["match"])
                print(f"[probe] {corner}:{s} rect={rect} frame f{f:04d} "
                      f"real_match={cell['match']}/7", flush=True)
            rec = {"corner": corner, "size": s, "rect": list(rect),
                   "frames": frames, "matches": matches,
                   "mean_match": float(np.mean(matches)), "min_match": int(np.min(matches))}
            results.append(rec)
            print(f"[probe] === {corner}:{s} mean={rec['mean_match']:.2f} "
                  f"min={rec['min_match']} matches={matches} ===", flush=True)
            with open(os.path.join(RUN_DIR, "probe_results.json"), "w") as fh:
                json.dump(results, fh, indent=2)

    print("\n===== PROBE SUMMARY (higher mean = more corner leverage) =====", flush=True)
    for r in sorted(results, key=lambda x: -x["mean_match"]):
        print(f"  {r['corner']}:{r['size']:>3}  mean={r['mean_match']:.2f}  "
              f"min={r['min_match']}  matches={r['matches']}", flush=True)


if __name__ == "__main__":
    main()
