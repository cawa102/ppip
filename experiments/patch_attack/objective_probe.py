"""Per-frame objective probe — the go/no-go before the ε ladder (design §4.2).

**The question.** The ε-threshold experiment runs the *per-frame* attack, which re-optimises a
patch on every observed frame. Its objective was chosen on principle (a minimum-perturbation
threshold measured with a non-saturating loss is an upper bound, not a threshold), but the margin
objective has never been validated in that regime: every comparison so far used
`stealth_optimize --max-frames N`, which fits ONE patch to N frames — the *static* problem.

**Why it is cheap and worth it.** A ladder rung is a ~9.5 h rollout. If the hinge is pathological
at tight ε — it measured 0.667 forcing for every κ from 1 to 100 on a single frame — this finds
out in half an hour instead of after a multi-day sweep.

**Fair-comparison rule.** Effort is *identical* across objectives and deliberately below the
ladder's, because the probe measures objectives relative to one another, not absolute forcing.
Reading an absolute capability number off this probe would be a mistake; the closed-loop ladder is
what produces those.

This mirrors the proven per-frame optimise loop in `ce_monitor_patch_attack.run_confined_episode`
(restarts → attempts → escalating lr → best-by-real-path), as `word_gate_probe.probe_frame`
already does for the gate work. It never adjudicates anything: forcing is a search-side
diagnostic, and every verdict in this project still comes from the fixed evaluator.

Run (GPU 0 or 1):

    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYTHONPATH=$HOME/LIBERO \
      ~/vla-injection/.venv/bin/python experiments/patch_attack/objective_probe.py \
        --frames 8 --steps 240 --eps 0.06
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

HOME = os.path.expanduser("~")
for _p in ("autoresearch/src", "autoresearch", "openvla", "autoresearch/experiments/patch_attack"):
    sys.path.insert(0, os.path.join(HOME, _p))

import forcing_loss as FL  # noqa: E402

DEVICE = "cuda"
USER_TASK = "pick up the alphabet soup and place it in the basket"
TARGET_TASK = "pick up the salad dressing and place it in the basket"
PROBE_RECT = (160, 0, 64, 64)  # the measured-clear BL 64x64 corner the ladder uses
DEFAULT_FRAMES = os.path.join(HOME, "autoresearch/runs/monitor-stealth/ceiling/frames/train")
DEFAULT_OUT = os.path.join(HOME, "autoresearch/runs/monitor-stealth/objective_probe")


@dataclass(frozen=True)
class ObjectiveSpec:
    """One objective configuration to compare. Frozen so it can key a result dict."""

    objective: str
    kappa: float = FL.DEFAULT_KAPPA
    temperature: float = FL.DEFAULT_TEMPERATURE
    anchor: float = 0.0


#: What the probe compares. `ce` is the reference — the loss every published closed-loop result
#: was produced with — so the others are read as deltas against a known quantity.
DEFAULT_SPECS: tuple[ObjectiveSpec, ...] = (
    ObjectiveSpec(objective="ce"),
    ObjectiveSpec(objective="ce_decisive"),
    ObjectiveSpec(objective="hinge", kappa=6.0),
    ObjectiveSpec(objective="hinge", kappa=12.0),
    ObjectiveSpec(objective="directional"),
)

#: Objectives for which `kappa` is meaningless; labelling it would imply it had been varied.
_KAPPA_FREE = ("ce", "ce_decisive", "directional")


def spec_label(spec: ObjectiveSpec) -> str:
    """Short stable name — the key in every output table."""
    label = spec.objective
    if spec.objective not in _KAPPA_FREE:
        label += f"@k{spec.kappa:g}"
    if spec.anchor:
        label += f"+a{spec.anchor:g}"
    return label


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per-spec summary over frames.

    Frames with no decisive dims are dropped, not scored 0.0: there is nothing to force on them,
    so counting them would dilute every objective identically and hide the difference being
    measured (same convention as `word_gate_probe.forced_fraction`).

    Both a per-frame mean and a dim-weighted mean are reported. They differ whenever frames have
    unequal numbers of contested dims, and the capacity tables elsewhere in this project use the
    dim-weighted one — so emitting a single number would invite a silent convention mismatch.
    """
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["n_decisive"]) > 0:
            buckets[str(row["spec"])].append(row)

    out: dict[str, dict[str, float]] = {}
    for label, group in buckets.items():
        forced_dims = sum(float(r["forced_fraction"]) * int(r["n_decisive"]) for r in group)
        total_dims = sum(int(r["n_decisive"]) for r in group)
        out[label] = {
            "mean_forcing": sum(float(r["forced_fraction"]) for r in group) / len(group),
            "dim_weighted_forcing": forced_dims / total_dims if total_dims else 0.0,
            "fully_forced_fraction": sum(1.0 for r in group if r["fully_forced"]) / len(group),
            "n_frames": float(len(group)),
            "n_decisive_dims": float(total_dims),
        }
    return out


# --- GPU seam ------------------------------------------------------------------------


@dataclass
class ProbeConfig:
    """Effort, held identical across specs so the comparison is fair."""

    steps: int = 240
    k: int = 30
    lr: float = 3e-2
    eps: float = 0.06
    base: str = "aurora"
    rect: tuple[int, int, int, int] = PROBE_RECT
    seed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def probe_frame(
    model: Any, processor: Any, frame_u8: Any, spec: ObjectiveSpec, config: ProbeConfig,
) -> dict[str, Any]:
    """Fit a stealth patch on ONE frame under `spec`; score it on the real inference path.

    Mirrors `run_confined_episode`'s inner loop: attempts of `k` gradient steps, escalating lr
    on a stubborn frame, keeping the best patch by *real-path* token match rather than by the
    surrogate loss. Stops early once the frame is fully forced, exactly as the rollout does.
    """
    import numpy as np
    import stealth_patch as SP
    import torch
    import vla_diff
    from adaptive_attack import _prompt_ids, _real_tokens
    from make_logo import build_base
    from ce_monitor_patch_attack import _rect_mask
    from word_gate_probe import decisive_dims, forced_fraction

    for parameter in model.parameters():
        parameter.requires_grad_(False)  # frozen weights: only the patch is optimised

    frame_u8 = np.asarray(frame_u8, dtype=np.uint8)
    img224 = (
        torch.from_numpy(frame_u8.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(DEVICE)
    )
    r0, c0, height, width = config.rect
    mask = _rect_mask(config.rect)

    teacher = _real_tokens(model, processor, frame_u8, TARGET_TASK).view(1, 7)
    clean_user = _real_tokens(model, processor, frame_u8, USER_TASK).view(7)
    target_list = teacher.view(7).tolist()
    dec = decisive_dims(clean_user.tolist(), target_list)
    if not dec:
        return {"spec": spec_label(spec), "forced_fraction": 0.0,
                "n_decisive": 0, "fully_forced": False}

    carrier = np.full((224, 224, 3), 0.5, dtype=np.float32)
    carrier[r0 : r0 + height, c0 : c0 + width] = build_base(config.base, height)
    base = SP.from_hwc(torch.from_numpy(carrier), DEVICE)
    user_ids = _prompt_ids(processor, USER_TASK)

    torch.manual_seed(config.seed)
    raw = torch.zeros(1, 3, 224, 224, device=DEVICE, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=config.lr)
    best_forced, best_full = 0.0, False

    attempts = max(1, config.steps // config.k)
    for _attempt in range(attempts):
        for _ in range(config.k):
            patch01 = SP.stealth_patch(raw, base, config.eps)
            composite = (img224 * (1 - mask) + patch01 * mask).clamp(0, 1)
            side = vla_diff._CROP_SIDE + 0.03 * (torch.rand(1).item() - 0.5)  # EoT nuisance
            logits = vla_diff.action_token_logits(
                model, vla_diff.preprocess(composite, side=side), user_ids, teacher
            )
            loss = FL.action_loss(
                logits.reshape(7, -1).float(), teacher.reshape(7), clean_user, dec,
                objective=spec.objective, kappa=spec.kappa,
                temperature=spec.temperature, anchor=spec.anchor,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Score on the REAL path, not the differentiable surrogate: the surrogate is a
        # reimplementation of preprocessing and any progress claim has to survive the deployed one.
        with torch.no_grad():
            patch01 = SP.stealth_patch(raw, base, config.eps)
            composite = (img224 * (1 - mask) + patch01 * mask).clamp(0, 1)
            attacked = (composite[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        tokens = _real_tokens(model, processor, attacked, USER_TASK).view(7).tolist()
        forced = forced_fraction(tokens, target_list, dec)
        if forced > best_forced:
            best_forced, best_full = forced, forced == 1.0
        if best_full:
            break
        for group in optimizer.param_groups:  # escalate pressure on a stubborn frame
            group["lr"] = min(group["lr"] * 1.5, 0.3)

    return {"spec": spec_label(spec), "forced_fraction": best_forced,
            "n_decisive": len(dec), "fully_forced": best_full}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare per-frame objectives at fixed effort.")
    parser.add_argument("--frames", type=int, default=8, help="decisive frames to probe")
    parser.add_argument("--steps", type=int, default=240, help="gradient steps per frame per spec")
    parser.add_argument("--k", type=int, default=30, help="steps between real-path checks")
    parser.add_argument("--eps", type=float, default=0.06)
    parser.add_argument("--base", default="aurora")
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--frames-dir", default=DEFAULT_FRAMES)
    parser.add_argument("--out", default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import shared_inits
    from hijack_backend import HijackBackend
    from stealth_optimize import build_frames, frame_paths

    shared_inits.verify_precommit()
    os.makedirs(args.out, exist_ok=True)

    backend = HijackBackend(run_dir=args.out)
    model, processor, _cfg, _resize = backend.load_policy_once()
    model.language_model.config.use_cache = False

    paths = frame_paths(args.frames_dir, shared_inits.OPTIMIZE_INITS, args.stride)
    frames = build_frames(
        model, processor, paths, user_task=USER_TASK, target_task=TARGET_TASK,
        cache_path=os.path.join(args.out, "token_cache.json"),
    )
    decisive = [f for f in frames if f.is_decisive][: args.frames]
    config = ProbeConfig(steps=args.steps, k=args.k, lr=args.lr, eps=args.eps, base=args.base)

    print(f"[objprobe] {len(decisive)} frames x {len(DEFAULT_SPECS)} specs, "
          f"{args.steps} steps each, eps={args.eps}, base={args.base}, "
          f"rect={PROBE_RECT} — effort identical across specs", flush=True)

    rows: list[dict[str, Any]] = []
    for spec in DEFAULT_SPECS:
        label = spec_label(spec)
        for index, frame in enumerate(decisive):
            row = probe_frame(model, processor, frame.image, spec, config)
            row["init"], row["step"] = frame.init, frame.step
            rows.append(row)
            print(f"[objprobe] {label:<16} frame {index + 1}/{len(decisive)} "
                  f"(init {frame.init} step {frame.step}) forced={row['forced_fraction']:.3f} "
                  f"dims={row['n_decisive']}", flush=True)
        summary = aggregate([r for r in rows if r["spec"] == label])[label]
        print(f"[objprobe] == {label}: mean {summary['mean_forcing']:.3f}, "
              f"dim-weighted {summary['dim_weighted_forcing']:.3f}, "
              f"fully {summary['fully_forced_fraction']:.3f}", flush=True)

    summary = aggregate(rows)
    out_path = os.path.join(args.out, f"objective_probe_eps{args.eps:g}.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({
            "config": {"eps": args.eps, "base": args.base, "rect": list(PROBE_RECT),
                       "steps": args.steps, "k": args.k, "lr": args.lr,
                       "n_frames": len(decisive), "inits": list(shared_inits.OPTIMIZE_INITS)},
            "summary": summary, "rows": rows,
        }, handle, indent=2)

    print(f"\n{'spec':<18}{'mean':>8}{'dim-wt':>9}{'fully':>8}{'frames':>8}", flush=True)
    for label, stats in sorted(summary.items(), key=lambda kv: -kv[1]["dim_weighted_forcing"]):
        print(f"{label:<18}{stats['mean_forcing']:>8.3f}{stats['dim_weighted_forcing']:>9.3f}"
              f"{stats['fully_forced_fraction']:>8.3f}{stats['n_frames']:>8.0f}", flush=True)
    print(f"\n[objprobe] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
