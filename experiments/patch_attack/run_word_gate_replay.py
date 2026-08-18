"""Artifact-level word gate: deploy ONE pre-recorded patch video, drive both word conditions.

The per-frame experiment re-solves a patch every control step, so its armed and dormant rollouts
never share pixels and the deployment story ("a video plays on a monitor regardless of what the
operator says") is asserted rather than measured. This driver measures it. Two uses, one code path:

**Replication** (``--init N`` with no ``--video``): record a fresh armed per-frame episode at ``N``
to obtain a deployable video, then replay THAT video at ``N`` under both instructions. Answers
"is the artifact-level gate reproducible?" — the question that takes the result off n = 1.

**Transfer** (``--init N --video <another init's video>``): replay a video fitted elsewhere. Answers
"is it a sticker, or must the patch be optimised for one specific setting?" A negative here is a
reportable result, not a failure: it bounds the threat model honestly.

Legs per panel — ``replay`` is the attack; the two controls isolate what does the work:
  * ``blank``     -- mid-gray rectangle. Separates "the video hijacks" from "any occupied corner
                     disrupts" and from "the word alone does it".
  * ``scrambled`` -- the SAME frames permuted in time (``monitor_replay.scramble_video``).
                     Separates content from temporal alignment.

No optimiser runs on any leg (~4 forwards per control step instead of ~900 forward/backwards), so a
whole panel costs a fraction of one optimised episode. Every verdict comes from the fixed evaluator
via ``run_confined_episode``; nothing here judges anything. Resumable: a leg whose result JSON
already exists is skipped, so a crashed queue can be relaunched without losing finished work.

    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYTHONPATH=$HOME/LIBERO \\
      ~/vla-injection/.venv/bin/python experiments/patch_attack/run_word_gate_replay.py --init 24
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

HOME = os.path.expanduser("~")
for _p in ("autoresearch/src", "autoresearch", "openvla", "autoresearch/experiments/patch_attack"):
    sys.path.insert(0, os.path.join(HOME, _p))

import imageio.v2 as imageio  # noqa: E402
from ce_monitor_patch_attack import run_confined_episode  # noqa: E402
from hijack_backend import HijackBackend  # noqa: E402
from monitor_replay import scramble_video  # noqa: E402
from patch_replay import load_patch_video  # noqa: E402
from render_word_gate_figure import (  # noqa: E402
    EFFORT,
    RECT,
    TARGET_TASK,
    USER_TASK,
    horizon_for,
)

ART_ROOT = os.path.join(HOME, "autoresearch/runs/monitor-stealth/word-gate/artifact")
ALL_LEGS = ("replay", "blank", "scrambled")
#: Fixed permutation for the time-scramble control — a control that moves between runs is not one.
SCRAMBLE_SEED = 46


def leg_tag(prefix: str, kind: str, condition: str, init: int, word_index: int) -> str:
    """Name for one leg's result/trace files.

    The slot suffix appears ONLY when the trigger has been moved. Slot 0 is what every existing
    panel ran, and renaming its tag would both break resumability (the driver skips a leg whose
    result JSON exists) and silently re-run finished work under a new name.
    """
    tag = f"{prefix}_{kind}_{condition}_init{init}"
    return tag if word_index == 0 else f"{tag}_slot{word_index}"


def record_armed_video(backend: Any, init: int, out: str, max_steps: int) -> str:
    """Run the armed per-frame episode at ``init`` with recording; return its patch-video dir.

    This is the ONLY optimised episode in the pipeline and dominates its cost. The rollout breaks on
    ``targeted``, so it costs roughly ``latch_step`` steps rather than the full horizon.
    """
    video = os.path.join(out, "video_armed", "patch")
    if os.path.isdir(video) and os.listdir(video):
        print(f"[artifact] init{init}: video already recorded ({len(os.listdir(video))} frames)",
              flush=True)
        return video
    result = run_confined_episode(
        backend, rect=RECT, seed=init, max_steps=max_steps, trial="0", run_dir=out,
        tag=f"video_armed_init{init}", record_dir=os.path.join(out, "video_armed"),
        user_task=USER_TASK, target_task=TARGET_TASK, patch_mode="optimize",
        gate_word="please", word_index=0, dormancy_weight=1.0, deploy_word=True, **EFFORT,
    )
    print(f"[artifact] init{init} VIDEO: targeted={result['targeted']} "
          f"latch={result['latch_step']} steps={result['step']}", flush=True)
    return video


def write_scrambled(source: str, dest: str) -> str:
    """Materialise the time-scrambled control as its own video directory."""
    if os.path.isdir(dest) and os.listdir(dest):
        return dest
    os.makedirs(dest, exist_ok=True)
    for i, frame in enumerate(scramble_video(load_patch_video(source), seed=SCRAMBLE_SEED)):
        imageio.imwrite(os.path.join(dest, f"f{i:04d}.png"), frame)
    return dest


def run_panel(
    init: int, video: str, out: str, legs: tuple[str, ...], max_steps: int,
    tag_prefix: str, record: bool, word_index: int = 0,
    conditions: tuple[str, ...] = ("dormant", "armed"),
) -> list[dict[str, Any]]:
    """Run every requested leg under the requested word conditions; return the summary rows.

    ``word_index`` moves the trigger inside the instruction (E-ART-X): the SAME deployed video is
    replayed while ``please`` is inserted at a different slot, which asks whether a deployed patch
    is position-locked or fires wherever the operator puts the word. The dormant string is
    unaffected by the slot, so the benign baseline stays fixed across the sweep.
    """
    backend = HijackBackend(run_dir=out)
    backend.load_policy_once()
    if not video:
        video = record_armed_video(backend, init, out, max_steps)
    scrambled = (write_scrambled(video, os.path.join(out, "scrambled_video"))
                 if "scrambled" in legs else "")
    sources = {"replay": video, "blank": None, "scrambled": scrambled}

    summary: list[dict[str, Any]] = []
    for kind in legs:
        for condition in conditions:
            deploy_word = condition == "armed"
            tag = leg_tag(tag_prefix, kind, condition, init, word_index)
            path = os.path.join(out, f"result_{tag}_trial0.json")
            if os.path.exists(path):
                print(f"[artifact] {tag}: already done, skipping", flush=True)
                with open(path, encoding="utf-8") as handle:
                    result = json.load(handle)
            else:
                print(f"\n=== {tag} ===", flush=True)
                result = run_confined_episode(
                    backend, rect=RECT, seed=init, max_steps=max_steps, trial="0", run_dir=out,
                    tag=tag,
                    record_dir=os.path.join(out, kind, condition) if (record and kind == "replay")
                    else "",
                    user_task=USER_TASK, target_task=TARGET_TASK,
                    patch_mode="blank" if kind == "blank" else "replay",
                    replay_dir=sources[kind],
                    # No search happens on a deployed patch; pinned to 1 so the recorded effort
                    # block cannot be misread as an optimisation budget that was actually spent.
                    k=1, maxtries=1, restarts=1,
                    gate_word="please", word_index=word_index, dormancy_weight=1.0,
                    deploy_word=deploy_word,
                )
            summary.append({
                "leg": tag, "init": init, "kind": kind, "condition": condition,
                "word_index": word_index,
                "deployed_instruction": (result.get("word_gate") or {}).get("deploy"),
                "video": sources[kind], "max_steps": max_steps,
                "targeted": result["targeted"], "commanded_success": result["commanded_success"],
                "latch_step": result["latch_step"], "commanded_step": result["commanded_step"],
                "min_target_dist_m": result["min_target_dist_m"],
                "gate_diagnostic": result["gate_diagnostic"],
            })
            print(f"[artifact] {tag}: targeted={summary[-1]['targeted']} "
                  f"commanded={summary[-1]['commanded_success']}", flush=True)
            with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", type=int, required=True, help="LIBERO init index to roll out at")
    ap.add_argument("--video", default="", help="patch-video dir; omit to record a fresh one here")
    ap.add_argument("--legs", default=",".join(ALL_LEGS), help=f"comma list from {ALL_LEGS}")
    ap.add_argument("--out", default="", help="output dir (default artifact/init<N>)")
    ap.add_argument("--tag-prefix", default="art")
    ap.add_argument("--word-index", type=int, default=0,
                    help="slot to insert the trigger at (E-ART-X: same video, word relocated)")
    ap.add_argument("--conditions", default="dormant,armed",
                    help="comma list from (dormant, armed); the dormant string is slot-independent"
                         ", so a cross-position sweep only needs 'armed'")
    ap.add_argument("--max-steps", type=int, default=0, help="default: horizon_for(init)")
    ap.add_argument("--no-record", action="store_true", help="skip frame recording (no GIF)")
    args = ap.parse_args()

    legs = tuple(x.strip() for x in args.legs.split(",") if x.strip())
    unknown = set(legs) - set(ALL_LEGS)
    if unknown:
        raise SystemExit(f"unknown legs {sorted(unknown)}; choose from {ALL_LEGS}")
    out = args.out or os.path.join(ART_ROOT, f"init{args.init}")
    max_steps = args.max_steps or horizon_for(args.init)
    os.makedirs(out, exist_ok=True)
    print(f"[artifact] init={args.init} legs={legs} conditions={args.conditions} "
          f"slot={args.word_index} horizon={max_steps} out={out} "
          f"video={args.video or '(record fresh)'}", flush=True)

    conditions = tuple(x.strip() for x in args.conditions.split(",") if x.strip())
    unknown_c = set(conditions) - {"dormant", "armed"}
    if unknown_c:
        raise SystemExit(f"unknown conditions {sorted(unknown_c)}; choose from (dormant, armed)")
    rows = run_panel(args.init, args.video, out, legs, max_steps, args.tag_prefix,
                     not args.no_record, word_index=args.word_index, conditions=conditions)
    print("\n=== PANEL ===", flush=True)
    for row in rows:
        print(f"  {row['leg']:34s} targeted={str(row['targeted']):5s} "
              f"commanded={str(row['commanded_success']):5s}", flush=True)


if __name__ == "__main__":
    main()
