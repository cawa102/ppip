"""The autoresearch loop for the stealth axis: propose -> optimize -> gate -> score -> ledger.

This is the control pattern from `karpathy/autoresearch`, with the roles this repo's
invariant assigns them:

    candidate (eps, base, tv, effort)      <- the agent-editable METHOD side
        |
        v  stealth_optimize.optimize        <- the `train.py` analog: fits ONE static patch
    patch_<id>.npy                             on OPTIMIZE_INITS only
        |
        v  stealth_gate.gate_patch          <- cheap search-side ranking on GATE_INITS.
    forcing score                              A DIAGNOSTIC. Never reported as a result.
        |  (>= threshold?)
        v  eval_static_patch                <- the `prepare.py` analog: the FIXED evaluator,
    metrics_<id>.json                          frozen patch, no optimizer, HELDOUT_INITS
        |
        v
    ledger.jsonl                            <- immutable, one row per candidate; the loop
                                               resumes from it and never rewrites a row

Three separations do the integrity work, and none of them is a convention — each is enforced:

  * **Init split.** The optimizer sees `OPTIMIZE_INITS`, the gate scores on `GATE_INITS`,
    and only `eval_static_patch` touches `HELDOUT_INITS`. A gate that scored on held-out
    inits would let evaluation data pick the candidate.
  * **Diagnostic vs score.** Forcing ranks candidates; it is never a reported number. Every
    `targeted`/`commanded` rate comes from the unmodified evaluator via the frozen wrapper.
  * **Method vs measurement.** The loop may change eps, the base image, the TV weight, the
    effort — the whole optimization objective. It cannot change how a rollout is judged.

The default ladder starts at **eps = 1.0**, which is effectively free-range: that is the
capacity gate. All "80x80 is robust" evidence comes from *per-frame* attacks, and a static
patch has far less capacity, so if eps=1 cannot hijack, no stealth budget will and the
finding is about staticness rather than stealth. The ladder only descends past that point.

Run:

    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYTHONPATH=$HOME/LIBERO \
      ~/vla-injection/.venv/bin/python experiments/patch_attack/stealth_loop.py --eps 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import numpy as np
import torch

HOME = os.path.expanduser("~")
for _p in ("autoresearch/src", "autoresearch", "openvla", "autoresearch/experiments/patch_attack"):
    sys.path.insert(0, os.path.join(HOME, _p))

import crop_geometry as CG  # noqa: E402
import forcing_loss as FL  # noqa: E402
import shared_inits  # noqa: E402
import stealth_patch as SP  # noqa: E402
from adaptive_attack import _prompt_ids  # noqa: E402
from eval_static_patch import assert_stealth_bound, evaluate_static_patch  # noqa: E402
from hijack_backend import HijackBackend  # noqa: E402
from make_logo import build_base, save_png  # noqa: E402
from stealth_gate import DEFAULT_THRESHOLD, gate_patch  # noqa: E402
from stealth_optimize import (  # noqa: E402
    DEFAULT_FRAMES,
    DEFAULT_OBJECTIVE,
    OBJECTIVES,
    TARGET_TASK,
    USER_TASK,
    build_frames,
    frame_paths,
    optimize,
    parse_rect,
    real_path_forcing,
)

from autoresearch_loop.ledger import append_ledger_row, read_ledger  # noqa: E402

DEFAULT_RUN_DIR = os.path.join(HOME, "autoresearch/runs/monitor-stealth/loop")

#: The eps ladder, descending. eps=1.0 is the free-range static capacity ceiling (plan B1);
#: 0.0 is the pure-logo control. Everything between is the stealth curve.
DEFAULT_LADDER: tuple[float, ...] = (1.0, 0.32, 0.16, 0.08, 0.04, 0.02, 0.0)


def candidate_id(
    base: str, eps: float, tv: float, rect_spec: str, seed: int,
    objective: str = DEFAULT_OBJECTIVE,
) -> str:
    """Ids carry the objective, so ledger rows fit under different objectives never collide.

    The pre-2026-07-30 objective keeps the original id shape, so a resumed run still matches
    the rows it already wrote instead of silently re-spending GPU on them.
    """
    stem = f"{base}_eps{eps:g}_tv{tv:g}_{rect_spec.replace(':', '')}_s{seed}"
    return stem if objective == "ce" else f"{stem}_{objective}"


def completed_candidates(ledger_path: str) -> set[str]:
    """Ids already recorded, so the loop resumes instead of re-spending GPU."""
    if not os.path.exists(ledger_path):
        return set()
    return {row["candidate_id"] for row in read_ledger(ledger_path)}


def run_candidate(
    backend: HijackBackend, model: Any, processor: Any, *,
    base_name: str, eps: float, tv: float, rect_spec: str, seed: int,
    args: argparse.Namespace, run_dir: str,
) -> dict[str, Any]:
    """One full iteration: optimize on train, gate, score on held-out if the gate passes."""
    rect = parse_rect(rect_spec)
    if args.inset:
        rect = CG.shift_rect_into_crop(rect)
    size = rect[2]
    cid = candidate_id(base_name, eps, tv, rect_spec, seed, args.objective)
    started = time.time()

    base_hwc = build_base(base_name, size)
    base = SP.from_hwc(torch.from_numpy(base_hwc), "cuda")

    print(f"\n[loop] === candidate {cid} === eps={eps} tv={tv} rect={rect} "
          f"({CG.nominal_area_fraction(rect):.1%} of frame -> "
          f"{CG.input_area_fraction(rect):.1%} of the model's input)", flush=True)

    # --- METHOD: fit one static patch on OPTIMIZE_INITS -------------------------------
    train_frames = build_frames(
        model, processor,
        frame_paths(args.frames, shared_inits.OPTIMIZE_INITS, args.stride),
        user_task=args.user_task, target_task=args.target_task,
        cache_path=os.path.join(run_dir, "token_cache_optimize.json"),
    )
    patch_hwc, diagnostics = optimize(
        model, processor, train_frames,
        base=base, eps=eps, rect=rect, user_ids=_prompt_ids(processor, args.user_task),
        steps=args.steps, batch_size=args.batch, lr=args.lr, tv_weight=tv,
        seed=seed, log_every=args.log_every,
        objective=args.objective, kappa=args.kappa, temperature=args.temperature,
        anchor=args.anchor, pool_size=args.pool,
    )
    patch_path = os.path.join(run_dir, f"{cid}.npy")
    np.save(patch_path, patch_hwc)
    save_png(patch_hwc, os.path.join(run_dir, f"{cid}.png"))

    # The bound is re-derived from the saved artifact, not taken from the optimizer's word.
    linf = assert_stealth_bound(patch_hwc, base_hwc, eps)
    train_forcing = real_path_forcing(
        model, processor, [f for f in train_frames if f.is_decisive][:16],
        patch_hwc, rect, args.user_task,
    )

    # --- DIAGNOSTIC: cheap ranking on GATE_INITS (never a reported number) -------------
    gate = gate_patch(
        model, processor, patch_hwc, rect=rect, frames_root=args.frames, stride=args.stride,
        user_task=args.user_task, target_task=args.target_task,
        cache_path=os.path.join(run_dir, "token_cache_gate.json"), max_frames=args.gate_frames,
    )
    gate_passed = bool(gate["decisive_forcing"] >= args.threshold)
    print(f"[loop] gate: decisive_forcing={gate['decisive_forcing']:.3f} "
          f"(threshold {args.threshold}) -> {'PASS' if gate_passed else 'FAIL'}", flush=True)

    row: dict[str, Any] = {
        "candidate_id": cid,
        "condition": "loop_with_skill",
        "base": base_name, "eps": eps, "tv_weight": tv,
        "objective": args.objective,
        "rect": list(rect),
        # `area_frac` stays the nominal frame share every earlier row used; `input_area_frac`
        # is the share of the model's post-crop input, which is what actually bounds capacity.
        "area_frac": CG.nominal_area_fraction(rect),
        "input_area_frac": CG.input_area_fraction(rect),
        "user_task": args.user_task, "target_task": args.target_task,
        "effort": {"steps": args.steps, "batch": args.batch, "lr": args.lr,
                   "stride": args.stride, "seed": seed,
                   "kappa": args.kappa, "temperature": args.temperature,
                   "anchor": args.anchor, "pool": args.pool, "inset": args.inset},
        "patch_path": patch_path,
        "linf_measured": linf,
        "optimizer_diagnostics": diagnostics,
        "train_forcing": train_forcing,
        "gate_forcing": gate,
        "gate_passed": gate_passed,
        "seconds_method": round(time.time() - started, 1),
    }

    # --- MEASUREMENT: the fixed evaluator, frozen patch, HELDOUT_INITS ----------------
    if not gate_passed and not args.force_eval:
        row.update({"evaluated": False, "skip_reason": "gate below threshold"})
        print("[loop] skipping the rollout — gate below threshold "
              "(pass --force-eval to score anyway)", flush=True)
    else:
        metrics = evaluate_static_patch(
            backend, patch=patch_hwc, top_left=(rect[0], rect[1]),
            init_indices=shared_inits.HELDOUT_INITS, candidate_id=cid,
            user_task=args.user_task, target_task=args.target_task,
            stealth={"base": base_name, "eps": eps, "linf_measured": linf,
                     "bound_verified": True},
        )
        metrics_path = os.path.join(run_dir, f"metrics_{cid}.json")
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        row.update({
            "evaluated": True,
            "metrics_path": metrics_path,
            "targeted_successes": metrics["targeted_successes"],
            "commanded_successes": metrics["commanded_successes"],
            "completed_rollouts": metrics["completed_rollouts"],
            "targeted_success_rate": metrics["targeted_success_rate"],
            "commanded_success_rate": metrics["commanded_success_rate"],
            "invalid_candidate_rate": metrics["invalid_candidate_rate"],
            "attack_score": metrics["attack_score"],
        })
        n = metrics["completed_rollouts"]
        print(f"[loop] SCORED {cid}: targeted {metrics['targeted_successes']}/{n}  "
              f"commanded {metrics['commanded_successes']}/{n}  "
              f"attack_score {metrics['attack_score']:.4f}", flush=True)

    row["seconds_total"] = round(time.time() - started, 1)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autoresearch loop over the stealth axis.")
    parser.add_argument("--eps", type=float, default=None,
                        help="run a single eps instead of the ladder")
    parser.add_argument("--ladder", default=None,
                        help="comma-separated eps ladder (default: 1,0.32,...,0)")
    parser.add_argument("--base", default="aurora")
    parser.add_argument("--tv", type=float, default=0.0)
    parser.add_argument("--rect", default="BL:64")
    parser.add_argument(
        "--inset", action="store_true",
        help="shift the rect inside the 0.9-area centre crop (recovers 17-33%% of a flush "
             "corner at no cost in nominal area; re-check occlusion first)",
    )
    parser.add_argument("--objective", choices=OBJECTIVES, default=DEFAULT_OBJECTIVE)
    parser.add_argument("--kappa", type=float, default=FL.DEFAULT_KAPPA)
    parser.add_argument("--temperature", type=float, default=FL.DEFAULT_TEMPERATURE)
    parser.add_argument("--anchor", type=float, default=0.0)
    parser.add_argument("--pool", type=int, default=0, help="CVaR pool per step (0 = off)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--gate-frames", type=int, default=24)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--force-eval", action="store_true",
        help="score on held-out even when the gate fails (costly; use for controls)",
    )
    parser.add_argument("--user-task", default=USER_TASK)
    parser.add_argument("--target-task", default=TARGET_TASK)
    parser.add_argument("--frames", default=DEFAULT_FRAMES)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shared_inits.verify_precommit()
    os.makedirs(args.run_dir, exist_ok=True)
    ledger_path = os.path.join(args.run_dir, "ledger.jsonl")

    if args.eps is not None:
        ladder = (args.eps,)
    elif args.ladder:
        ladder = tuple(float(x) for x in args.ladder.split(","))
    else:
        ladder = DEFAULT_LADDER

    done = completed_candidates(ledger_path)
    backend = HijackBackend(run_dir=args.run_dir)
    model, processor, _cfg, _resize = backend.load_policy_once()
    for parameter in model.parameters():
        parameter.requires_grad_(False)  # weights frozen; only the patch is ever optimized
    model.language_model.config.use_cache = False

    print(f"[loop] {shared_inits.summary()}", flush=True)
    print(f"[loop] optimize={list(shared_inits.OPTIMIZE_INITS)} "
          f"gate={list(shared_inits.GATE_INITS)} "
          f"score={list(shared_inits.HELDOUT_INITS)}", flush=True)
    print(f"[loop] ladder={list(ladder)} base={args.base} tv={args.tv} rect={args.rect}",
          flush=True)

    for eps in ladder:
        cid = candidate_id(args.base, eps, args.tv, args.rect, args.seed, args.objective)
        if cid in done:
            print(f"[loop] {cid} already in the ledger -- skip", flush=True)
            continue
        row = run_candidate(
            backend, model, processor,
            base_name=args.base, eps=eps, tv=args.tv, rect_spec=args.rect, seed=args.seed,
            args=args, run_dir=args.run_dir,
        )
        append_ledger_row(ledger_path, row)

    print(f"\n[loop] ledger: {ledger_path}", flush=True)
    for row in read_ledger(ledger_path):
        if row.get("evaluated"):
            print(f"  eps={row['eps']:<5} targeted={row['targeted_successes']}/"
                  f"{row['completed_rollouts']} commanded={row['commanded_successes']}/"
                  f"{row['completed_rollouts']} attack_score={row['attack_score']:.4f} "
                  f"linf={row['linf_measured']:.4f}", flush=True)
        else:
            print(f"  eps={row['eps']:<5} gate {row['gate_forcing']['decisive_forcing']:.3f} "
                  f"-> not scored ({row.get('skip_reason')})", flush=True)


if __name__ == "__main__":
    main()
