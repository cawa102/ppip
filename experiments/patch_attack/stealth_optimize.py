"""EoT optimization of ONE static stealth patch — the `train.py` analog of the loop.

Every hijack this project has produced re-optimizes the patch **every frame**, which makes it
a per-episode digital artifact and (fatally for a stealth claim) a *flickering* logo. This
optimizes a single frame-independent patch over a distribution of frames instead:

    minimise  E_frames [ L( OpenVLA(composite(frame, patch), USER_TASK), teacher(frame) ) ]
              + tv_weight * TV(patch - base)
    subject to |patch - base|_inf <= eps                      (by parameterization)

where `teacher(frame) = OpenVLA(frame, TARGET_TASK)` — the same teacher the proven per-frame
corner attack uses, so stealth and staticness are the only variables changed.

`L` is selectable (`--objective`, see `forcing_loss`) and defaults to the **margin hinge**
rather than the cross-entropy this file originally used. Three reasons, all measured here:
the two instructions already agree on ~3 of 7 dims, so all-7 CE spends 43% of every step on
settled dims; CE never saturates, so it keeps buying margin on frames already won while the
binding constraint is capacity across frames; and the 256 action bins are ordered, so CE's
"every wrong bin is equally wrong" charges for exactness the attack does not need. Pass
`--objective ce` to reproduce runs from before 2026-07-30.

What makes it a *distribution* and not one frame:
  * frames come from several clean rollouts (`OPTIMIZE_INITS`), so object layouts vary;
  * only **decisive** frames are optimized on — those where the user-instructed and
    target-instructed policies actually disagree. Forcing a token both instructions already
    agree on proves nothing, and the mean token match is dominated by that free agreement
    (measured at ~6.88/7 in GATE-B), so it is a misleading objective *and* a waste of budget;
  * the existing crop-side jitter is retained as the EoT nuisance parameter.

Scope: this writes a candidate artifact and search-side diagnostics. It never produces a
score — `eval_static_patch.py` does that, on inits this file never sees.

Run (GPU 1 only):

    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYTHONPATH=$HOME/LIBERO \
      ~/vla-injection/.venv/bin/python experiments/patch_attack/stealth_optimize.py \
        --base aurora --eps 1.0 --rect BL:64 --steps 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

HOME = os.path.expanduser("~")
for _p in ("autoresearch/src", "autoresearch", "openvla", "autoresearch/experiments/patch_attack"):
    sys.path.insert(0, os.path.join(HOME, _p))

import crop_geometry as CG  # noqa: E402
import forcing_loss as FL  # noqa: E402
import shared_inits  # noqa: E402
import stealth_patch as SP  # noqa: E402
import vla_diff  # noqa: E402
from adaptive_attack import _prompt_ids, _real_tokens  # noqa: E402
from hijack_backend import HijackBackend  # noqa: E402
from make_logo import build_base, save_png  # noqa: E402
from occlusion_probe import corner_rect  # noqa: E402

DEVICE = "cuda"
USER_TASK = "pick up the alphabet soup and place it in the basket"
TARGET_TASK = "pick up the salad dressing and place it in the basket"
DEFAULT_FRAMES = os.path.join(HOME, "autoresearch/runs/monitor-stealth/ceiling/frames/train")
DEFAULT_OUT = os.path.join(HOME, "autoresearch/runs/monitor-stealth/patches")

#: A frame counts as decisive when the two instructions disagree on at least this many dims.
#: Matches `corner_decisive_probe`'s CD_MIN_DIFF default, so gate and optimizer agree on terms.
MIN_DECISIVE_DIMS = 2

#: Selectable objectives (see `forcing_loss` for why each exists).
#:   `ce`            - the original: cross-entropy over all 7 dims. Kept to reproduce prior runs.
#:   `ce_decisive`   - the same, restricted to the dims the two instructions disagree on.
#:   `ce_saturating` - `ce_decisive` plus the hinge's won-dim release, and nothing else. The cell
#:                     that makes `ce`-vs-`hinge` decomposable into shape and saturation.
#:   `hinge`         - Carlini-Wagner margin; a won dim stops consuming capacity.
#:   `directional`   - signed progress from the user's action toward the teacher's.
OBJECTIVES = ("ce", "ce_decisive", "ce_saturating", "hinge", "directional")
DEFAULT_OBJECTIVE = "hinge"


@dataclass(frozen=True)
class Frame:
    """One clean agentview frame plus what the two instructions want it to do."""

    init: int
    step: int
    image: NDArray[np.uint8]  # [224,224,3]
    teacher: tuple[int, ...]  # target-instructed action tokens
    clean_user: tuple[int, ...]  # user-instructed action tokens
    decisive_dims: tuple[int, ...]

    @property
    def is_decisive(self) -> bool:
        return len(self.decisive_dims) >= MIN_DECISIVE_DIMS


def frame_paths(
    frames_root: str, inits: tuple[int, ...], stride: int
) -> list[tuple[int, int, str]]:
    """(init, step, path) for every `stride`-th recorded frame of the given inits."""
    found: list[tuple[int, int, str]] = []
    for init in inits:
        init_dir = os.path.join(frames_root, f"init{init:02d}")
        if not os.path.isdir(init_dir):
            raise SystemExit(
                f"no frames for init {init} at {init_dir}. Run ceiling_screen.py --phase a first."
            )
        names = sorted(n for n in os.listdir(init_dir) if n.startswith("f") and n.endswith(".png"))
        found.extend((init, int(n[1:5]), os.path.join(init_dir, n)) for n in names[::stride])
    return found


def build_frames(
    model: Any, processor: Any, paths: list[tuple[int, int, str]], *,
    user_task: str, target_task: str, cache_path: str,
) -> list[Frame]:
    """Load frames and label each with teacher / clean-user tokens (cached across runs).

    Two real-inference forwards per frame, so this is the expensive part of a cold start;
    the cache makes every later candidate on the same buffer effectively free.
    """
    cache: dict[str, list[int]] = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as handle:
            cache = json.load(handle)

    frames: list[Frame] = []
    misses = 0
    for init, step, path in paths:
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        key = f"{init}:{step}"
        if key not in cache:
            teacher = _real_tokens(model, processor, image, target_task).view(7)
            clean = _real_tokens(model, processor, image, user_task).view(7)
            cache[key] = [int(t) for t in teacher] + [int(c) for c in clean]
            misses += 1
            if misses % 25 == 0:
                print(f"[stealth-opt] labelled {misses} new frames...", flush=True)
        values = cache[key]
        teacher_ids, clean_ids = tuple(values[:7]), tuple(values[7:])
        frames.append(
            Frame(
                init=init, step=step, image=image,
                teacher=teacher_ids, clean_user=clean_ids,
                decisive_dims=tuple(i for i in range(7) if teacher_ids[i] != clean_ids[i]),
            )
        )

    if misses:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(cache, handle)
    print(f"[stealth-opt] {len(frames)} frames ({misses} newly labelled), "
          f"{sum(f.is_decisive for f in frames)} decisive", flush=True)
    return frames


def save_checkpoint(
    path: str, raw: torch.Tensor, optimizer: torch.optim.Optimizer, step: int
) -> None:
    """Persist the patch parameter, Adam moments and step count.

    Adam's moments matter: restoring `raw` alone restarts the optimizer cold and throws away
    the adaptive scaling built up over hundreds of steps.
    """
    torch.save(
        {"raw": raw.detach().cpu(), "optimizer": optimizer.state_dict(), "step": step}, path
    )


def load_checkpoint(
    path: str, raw: torch.Tensor, optimizer: torch.optim.Optimizer
) -> int:
    """Restore a checkpoint in place; return the step to resume from."""
    state = torch.load(path, map_location=raw.device)
    with torch.no_grad():
        raw.copy_(state["raw"].to(raw.device))
    optimizer.load_state_dict(state["optimizer"])
    return int(state["step"])


def _as_tensor(image: NDArray[np.uint8]) -> torch.Tensor:
    return torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(DEVICE)


def real_path_forcing(
    model: Any, processor: Any, frames: list[Frame], patch_hwc: NDArray[np.float32],
    rect: tuple[int, int, int, int], user_task: str,
) -> dict[str, float]:
    """Decisive-dim forcing of a FROZEN patch on `frames`, via the real inference path.

    The optimizer's cross-entropy is a surrogate computed on a differentiable
    reimplementation of preprocessing; this is the actual deployed path, and it is what any
    progress claim must rest on.
    """
    r0, c0, height, width = rect
    forced, total, exact = 0, 0, 0
    with torch.no_grad():
        for frame in frames:
            attacked = frame.image.copy()
            attacked[r0 : r0 + height, c0 : c0 + width] = np.clip(
                patch_hwc * 255.0, 0, 255
            ).astype(np.uint8)
            tokens = _real_tokens(model, processor, attacked, user_task).view(7)
            hits = sum(1 for i in frame.decisive_dims if int(tokens[i]) == frame.teacher[i])
            forced += hits
            total += len(frame.decisive_dims)
            exact += int(hits == len(frame.decisive_dims))
    return {
        "decisive_forcing": forced / total if total else 0.0,
        "frames_fully_forced": exact / len(frames) if frames else 0.0,
        "n_frames": float(len(frames)),
    }


def real_path_forcing_masked(
    model: Any, processor: Any, frames: list[Frame], patch_full: NDArray[np.float32],
    mask_hw: NDArray[np.float32], user_task: str,
) -> dict[str, float]:
    """`real_path_forcing` for an arbitrary masked region rather than a single rect."""
    forced, total, exact = 0, 0, 0
    where = mask_hw > 0.5
    with torch.no_grad():
        for frame in frames:
            attacked = frame.image.copy()
            patch_u8 = np.clip(patch_full * 255.0, 0, 255).astype(np.uint8)
            attacked[where] = patch_u8[where]
            tokens = _real_tokens(model, processor, attacked, user_task).view(7)
            hits = sum(1 for i in frame.decisive_dims if int(tokens[i]) == frame.teacher[i])
            forced += hits
            total += len(frame.decisive_dims)
            exact += int(hits == len(frame.decisive_dims))
    return {
        "decisive_forcing": forced / total if total else 0.0,
        "frames_fully_forced": exact / len(frames) if frames else 0.0,
        "n_frames": float(len(frames)),
    }


def frame_loss(
    logits: torch.Tensor, frame: Frame, *,
    objective: str, kappa: float, temperature: float, anchor: float,
) -> torch.Tensor:
    """Per-frame scalar loss under `objective`. `logits` is `[7, vocab]`.

    `anchor` mixes a small decisive-dim cross-entropy into the saturating objectives. Both of
    those go flat once satisfied — which is the point — but `directional` also reads the head
    through a softmax whose gradient thins out when the distribution is peaked, so a light CE
    term keeps a usable signal. Zero by default; ~0.1 is a reasonable starting weight.
    """
    teacher = torch.tensor(frame.teacher, dtype=torch.long, device=logits.device)
    user = torch.tensor(frame.clean_user, dtype=torch.long, device=logits.device)
    return FL.action_loss(
        logits, teacher, user, frame.decisive_dims,
        objective=objective, kappa=kappa, temperature=temperature, anchor=anchor,
    )


def _frame_forward(
    model: Any, patch: torch.Tensor, frame: Frame, *,
    rect: tuple[int, int, int, int], mask: torch.Tensor | None, user_ids: torch.Tensor,
    jitter: float, objective: str, kappa: float, temperature: float, anchor: float,
) -> torch.Tensor:
    """Composite `patch` into `frame`, run the differentiable path, return the frame's loss."""
    image = _as_tensor(frame.image)
    teacher = torch.tensor(frame.teacher, device=DEVICE).view(1, 7)
    composited = (
        SP.composite_masked(image, patch, mask) if mask is not None
        else SP.composite(image, patch, rect)
    )
    logits = vla_diff.action_token_logits(
        model, vla_diff.preprocess(composited, side=vla_diff._CROP_SIDE + jitter),
        user_ids, teacher,
    )
    return frame_loss(
        logits.reshape(FL.ACTION_DIM, -1).float(), frame,
        objective=objective, kappa=kappa, temperature=temperature, anchor=anchor,
    )


def optimize(
    model: Any, processor: Any, frames: list[Frame], *,
    base: torch.Tensor, eps: float, rect: tuple[int, int, int, int], user_ids: torch.Tensor,
    steps: int, batch_size: int, lr: float, tv_weight: float, seed: int,
    log_every: int, max_frames: int | None = None,
    checkpoint_path: str | None = None, checkpoint_every: int = 100, resume: bool = False,
    mask: torch.Tensor | None = None,
    objective: str = DEFAULT_OBJECTIVE, kappa: float = FL.DEFAULT_KAPPA,
    temperature: float = FL.DEFAULT_TEMPERATURE, anchor: float = 0.0, pool_size: int = 0,
    distortion_weight: float = 0.0,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    """Adam on a single shared `raw` over the decisive-frame distribution.

    `max_frames` caps how many decisive frames the patch must serve. Sweeping it is the
    capacity ladder: one static patch trivially fits ONE frame (that is the proven per-frame
    attack), and the question is how far the distribution can widen before forcing collapses.
    Holding `steps` fixed while raising `max_frames` deliberately *lowers* the per-frame
    budget, so a ladder must scale `steps` with `max_frames` to separate "too little search"
    from "too little capacity".

    `pool_size > batch_size` turns each step into worst-case (CVaR) selection: score a pool of
    that many frames under `no_grad`, then spend the gradient on the hardest `batch_size` of
    them. Optimizing the plain batch mean leaves a tail of unforced frames alive, and those are
    the ones that derail a rollout — a patch measured at 0.910 *mean* forcing still had ~10% of
    decisive dims wrong and changed no behaviour at all. Selecting rather than reweighting
    keeps memory flat: only the chosen frames ever build a graph. `pool_size <= batch_size`
    disables the scoring pass entirely, so the default costs nothing.
    """
    decisive = [f for f in frames if f.is_decisive]
    if max_frames is not None:
        decisive = decisive[:max_frames]
    if not decisive:
        raise SystemExit("no decisive frames in the buffer — nothing to optimize toward")

    generator = np.random.default_rng(seed)
    torch.manual_seed(seed)
    raw = torch.zeros_like(base, requires_grad=True)  # raw=0 -> patch == base (the pure logo)
    optimizer = torch.optim.Adam([raw], lr=lr)

    start_step = 0
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        start_step = load_checkpoint(checkpoint_path, raw, optimizer)
        print(f"[stealth-opt] resumed from {checkpoint_path} at step {start_step}", flush=True)

    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; choose from {OBJECTIVES}")
    terms = {"objective": objective, "kappa": kappa, "temperature": temperature, "anchor": anchor}
    batch_size = min(batch_size, len(decisive))
    pool_size = min(max(pool_size, batch_size), len(decisive))

    history: list[dict[str, float]] = []
    started = time.time()
    for step in range(start_step, steps):
        jitter = 0.03 * (torch.rand(1).item() - 0.5)  # EoT nuisance, shared across the step
        pool = [decisive[i] for i in generator.choice(len(decisive), size=pool_size, replace=False)]

        pool_losses: list[float] = []
        if pool_size > batch_size:  # CVaR: spend the gradient on the hardest frames in the pool
            with torch.no_grad():
                frozen = SP.stealth_patch(raw, base, eps)
                pool_losses = [
                    float(_frame_forward(
                        model, frozen, f, rect=rect, mask=mask, user_ids=user_ids,
                        jitter=jitter, **terms,  # type: ignore[arg-type]
                    ).item())
                    for f in pool
                ]
            batch = [pool[i] for i in np.argsort(pool_losses)[::-1][:batch_size]]
        else:
            batch = pool

        optimizer.zero_grad()
        batch_loss = 0.0
        batch_losses: list[float] = []
        for frame in batch:
            patch = SP.stealth_patch(raw, base, eps)
            loss = _frame_forward(
                model, patch, frame, rect=rect, mask=mask, user_ids=user_ids,
                jitter=jitter, **terms,  # type: ignore[arg-type]
            )
            batch_losses.append(float(loss.item()))
            if tv_weight:
                loss = loss + tv_weight * SP.total_variation(patch - base)
            if distortion_weight and eps > 0:
                # The soft half of the stealth constraint (design 4.5). Distinct from TV: TV
                # smooths delta, this SHRINKS it. eps-normalised so one lambda travels across
                # the ladder. Recorded in `diagnostics` so an artifact names the loss that made
                # it — the same reason `objective` is recorded.
                loss = loss + distortion_weight * SP.distortion(
                    patch, base, mask, eps=eps
                )
            (loss / len(batch)).backward()
            batch_loss += float(loss.item()) / len(batch)
        optimizer.step()

        # The tail, not just the mean: a low mean with a live tail is the signature of the
        # patch that forced 91% of dims and still changed no behaviour. `frames_satisfied` is
        # only meaningful for the saturating objectives, where zero means "nothing left to do".
        tail = torch.tensor(pool_losses or batch_losses)
        history.append({
            "step": step,
            "loss": batch_loss,
            "loss_worst_quartile": float(FL.cvar(tail, 0.25).item()),
            "frames_satisfied": float((tail <= 0.0).float().mean().item()),
        })
        if checkpoint_path and (step % checkpoint_every == 0 or step == steps - 1):
            save_checkpoint(checkpoint_path, raw, optimizer, step + 1)
        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                measured = SP.l_inf(SP.stealth_patch(raw, base, eps), base)
            print(f"[stealth-opt] step {step:>4}/{steps} loss={batch_loss:.4f} "
                  f"worst25={history[-1]['loss_worst_quartile']:.4f} "
                  f"satisfied={history[-1]['frames_satisfied']:.2f} "
                  f"|delta|_inf={measured:.4f} ({time.time() - started:.0f}s)", flush=True)

    with torch.no_grad():
        patch = SP.stealth_patch(raw, base, eps)
        delta = patch - base
        region: CG.Rect | torch.Tensor = mask if mask is not None else rect
        diagnostics = {
            "objective": objective,
            "kappa": kappa, "temperature": temperature, "anchor": anchor,
            "distortion_weight": distortion_weight,
            "batch_size": batch_size, "pool_size": pool_size,
            "final_loss": history[-1]["loss"] if history else float("nan"),
            "final_loss_worst_quartile": (
                history[-1]["loss_worst_quartile"] if history else float("nan")
            ),
            "final_frames_satisfied": history[-1]["frames_satisfied"] if history else float("nan"),
            "resumed_from_step": start_step,
            "mean_loss_last_10pct": float(
                np.mean([h["loss"] for h in history[-max(1, steps // 10) :]])
            ),
            "linf_measured": SP.l_inf(patch, base),
            "delta_tv": float(SP.total_variation(delta).item()),
            "delta_mean_abs": float(delta.abs().mean().item()),
            # The epsilon ball is not fully available where the base is near 0 or 1, so two
            # carriers at one eps do not get the same budget. Recorded so a base sweep can
            # separate structure from headroom.
            "headroom_fraction": SP.headroom_fraction(base, eps),
            # Nominal area is what a defender sees; input area is what the ViT consumes after
            # the 0.9-area centre crop. A flush corner patch loses a third of itself at 32x32.
            "nominal_area_fraction": CG.nominal_area_fraction(region),
            "input_area_fraction": CG.input_area_fraction(region),
            "retained_fraction": CG.retained_fraction(region),
            "n_decisive_frames": len(decisive),
            "grad_steps_per_frame": steps * batch_size / len(decisive),
            "seconds": round(time.time() - started, 1),
        }
    return SP.to_hwc(patch).numpy().astype(np.float32), diagnostics


def parse_rect(spec: str) -> tuple[int, int, int, int]:
    """Rect spec -> (r0, c0, h, w) in the 224 frame.

    Two forms: `"BL:64"` for a named corner square, or `"r0,c0,h,w"` for an explicit rect.
    The explicit form exists for capacity probes that need a non-square region (e.g. the
    full-width band below the objects) to bound what patch *area* can buy.
    """
    if "," in spec:
        r0, c0, height, width = (int(x) for x in spec.split(","))
        return r0, c0, height, width
    corner, _, size = spec.partition(":")
    return corner_rect(corner.upper(), int(size))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EoT-optimize one static stealth patch.")
    parser.add_argument("--base", default="aurora", help="make_logo base name")
    parser.add_argument(
        "--eps", type=float, default=1.0, help="L-inf stealth budget; 1.0 = free-range"
    )
    parser.add_argument("--rect", default="BL:64", help="corner rect, e.g. BL:64")
    parser.add_argument(
        "--regions", default=None,
        help="semicolon-separated r0,c0,h,w rects forming ONE masked region; overrides --rect. "
             "Needed to push area past what a single non-occluding rect allows.",
    )
    parser.add_argument(
        "--inset", action="store_true",
        help="shift the rect inside the 0.9-area centre crop. A flush corner loses 17-33%% of "
             "itself before the ViT; this recovers it at no cost in nominal area. Re-check "
             "occlusion_probe before using it on a bottom-flush rect.",
    )
    parser.add_argument("--tv", type=float, default=0.0, help="weight on TV(delta)")
    parser.add_argument(
        "--lam", type=float, default=0.0,
        help="weight on the soft distortion penalty lambda*MSE(patch, base), eps-normalised. "
             "Distinct from --tv: TV smooths delta, this shrinks it. Added INSIDE the eps-ball, "
             "never instead of it (design 2026-08-04 eps-threshold, section 4.5).",
    )
    parser.add_argument(
        "--objective", choices=OBJECTIVES, default=DEFAULT_OBJECTIVE,
        help="forcing objective; 'ce' reproduces pre-2026-07-30 runs",
    )
    parser.add_argument(
        "--kappa", type=float, default=FL.DEFAULT_KAPPA,
        help="hinge margin at which a decisive dim stops receiving gradient",
    )
    parser.add_argument(
        "--temperature", type=float, default=FL.DEFAULT_TEMPERATURE,
        help="softmax temperature for the directional objective's soft action decode",
    )
    parser.add_argument(
        "--anchor", type=float, default=0.0,
        help="weight of a decisive-dim CE term mixed into a saturating objective (try 0.1)",
    )
    parser.add_argument(
        "--pool", type=int, default=0,
        help="score this many frames per step and spend the gradient on the hardest --batch "
             "of them (CVaR). 0 or <= --batch disables the extra scoring pass.",
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=4, help="frames per optimizer step")
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--stride", type=int, default=5, help="frame subsampling stride")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="cap the decisive-frame pool; sweep it for the static-capacity ladder",
    )
    parser.add_argument("--user-task", default=USER_TASK)
    parser.add_argument("--target-task", default=TARGET_TASK)
    parser.add_argument("--frames", default=DEFAULT_FRAMES)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument(
        "--resume", action="store_true",
        help="continue from this candidate's checkpoint if one exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shared_inits.verify_precommit()
    rect = parse_rect(args.rect)
    if args.inset:
        shifted = CG.shift_rect_into_crop(rect)
        print(f"[stealth-opt] --inset: rect {rect} -> {shifted}; the model's share of it goes "
              f"{CG.input_area_fraction(rect):.2%} -> {CG.input_area_fraction(shifted):.2%} "
              f"at the same nominal area", flush=True)
        rect = shifted
    size = rect[2]
    area_frac = rect[2] * rect[3] / (224 * 224)
    candidate_id = args.candidate_id or (
        f"{args.base}_eps{args.eps:g}_tv{args.tv:g}_{args.rect.replace(':', '')}_s{args.seed}"
    )
    os.makedirs(args.out, exist_ok=True)

    backend = HijackBackend(run_dir=args.out)
    model, processor, _cfg, _resize = backend.load_policy_once()
    for parameter in model.parameters():
        parameter.requires_grad_(False)  # weights frozen: only the patch is optimized
    model.language_model.config.use_cache = False

    paths = frame_paths(args.frames, shared_inits.OPTIMIZE_INITS, args.stride)
    frames = build_frames(
        model, processor, paths,
        user_task=args.user_task, target_task=args.target_task,
        cache_path=os.path.join(args.out, "token_cache.json"),
    )

    regions = (
        [tuple(int(x) for x in part.split(",")) for part in args.regions.split(";")]
        if args.regions else None
    )
    mask = None
    if regions is not None:
        mask = SP.rects_to_mask(regions, device=DEVICE)  # type: ignore[arg-type]
        area_frac = float(mask.mean().item())
        base_hwc = np.full((224, 224, 3), 0.5, dtype=np.float32)
        base = SP.from_hwc(torch.from_numpy(base_hwc), DEVICE)
        print(f"[stealth-opt] MASKED region {regions} -> {area_frac:.1%} of frame", flush=True)
    base_hwc = base_hwc if regions is not None else (
        build_base(args.base, size) if rect[2] == rect[3] else np.broadcast_to(
        build_base(args.base, min(rect[2], rect[3])).mean(axis=(0, 1)),
        (rect[2], rect[3], 3),
    ).astype(np.float32).copy())
    base = base if regions is not None else SP.from_hwc(torch.from_numpy(base_hwc), DEVICE)
    print(f"[stealth-opt] candidate={candidate_id} rect={rect} "
          f"({area_frac:.1%} of frame) eps={args.eps} tv={args.tv} "
          f"optimize_inits={list(shared_inits.OPTIMIZE_INITS)}", flush=True)

    patch_hwc, diagnostics = optimize(
        model, processor, frames,
        base=base, eps=args.eps, rect=rect,
        user_ids=_prompt_ids(processor, args.user_task),
        steps=args.steps, batch_size=args.batch, lr=args.lr, tv_weight=args.tv,
        seed=args.seed, log_every=args.log_every, max_frames=args.max_frames,
        checkpoint_path=os.path.join(args.out, f"{candidate_id}.ckpt.pt"),
        checkpoint_every=args.checkpoint_every, resume=args.resume, mask=mask,
        objective=args.objective, kappa=args.kappa, temperature=args.temperature,
        anchor=args.anchor, pool_size=args.pool, distortion_weight=args.lam,
    )

    n_eval = args.max_frames if args.max_frames else 24
    eval_frames = [f for f in frames if f.is_decisive][:n_eval]
    forcing = (
        real_path_forcing_masked(
            model, processor, eval_frames, patch_hwc,
            mask[0, 0].cpu().numpy(), args.user_task,
        )
        if mask is not None
        else real_path_forcing(model, processor, eval_frames, patch_hwc, rect, args.user_task)
    )
    patch_path = os.path.join(args.out, f"{candidate_id}.npy")
    np.save(patch_path, patch_hwc)
    save_png(patch_hwc, os.path.join(args.out, f"{candidate_id}.png"))
    save_png(base_hwc, os.path.join(args.out, f"base_{args.base}_{size}.png"))

    meta = {
        "candidate_id": candidate_id,
        "patch_path": patch_path,
        "base": args.base, "eps": args.eps, "tv_weight": args.tv,
        "rect": list(rect), "regions": regions, "area_frac": area_frac,
        "user_task": args.user_task, "target_task": args.target_task,
        "optimize_inits": list(shared_inits.OPTIMIZE_INITS),
        "effort": {"steps": args.steps, "batch": args.batch, "lr": args.lr,
                   "stride": args.stride, "seed": args.seed},
        "n_frames": len(frames),
        "diagnostics": diagnostics,
        "train_real_path_forcing": forcing,
    }
    with open(os.path.join(args.out, f"{candidate_id}.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print(f"\n[stealth-opt] DONE {candidate_id} (objective={args.objective})", flush=True)
    print(f"  |delta|_inf = {diagnostics['linf_measured']:.4f} (eps={args.eps}), "
          f"clamp headroom = {diagnostics['headroom_fraction']:.3f}", flush=True)
    print(f"  area: {diagnostics['nominal_area_fraction']:.2%} of frame -> "
          f"{diagnostics['input_area_fraction']:.2%} of the model's input "
          f"({diagnostics['retained_fraction']:.1%} of the patch survives the crop)", flush=True)
    print(f"  train decisive forcing = {forcing['decisive_forcing']:.3f}, "
          f"fully-forced frames = {forcing['frames_fully_forced']:.3f}", flush=True)
    print(f"  artifact: {patch_path}", flush=True)


if __name__ == "__main__":
    main()
