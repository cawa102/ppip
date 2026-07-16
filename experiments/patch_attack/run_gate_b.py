"""GATE-B experiment driver: Stage-1 oracle -> Stage-2 open-loop replay + controls.

Runs the real physically-realizable monitor-video hijack end to end on one seed:
  1. S0 sanity (does the TARGET still succeed with a neutral monitor present?).
  2. Stage-1 oracle (per-step re-optimised monitor texture; the upper bound), recording
     texture_0..T through the real renderer.
  3. Stage-2 open-loop replay of that recorded video, plus the blank + time-scrambled
     controls, and the GATE-B margin report.

This only ORCHESTRATES the tested/GPU-verified seams (run_oracle, run_replay, run_control,
margin_report, s0_sanity) -- it writes candidate/experiment artefacts only, never touches
the evaluator. Results land under runs/monitor-hijack/seed<seed>/ (git-ignored).

    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYTHONPATH=$HOME/LIBERO \
      ~/vla-injection/.venv/bin/python experiments/patch_attack/run_gate_b.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys

HOME = os.path.expanduser("~")
for _p in ("autoresearch/src", "autoresearch", "openvla", "autoresearch/experiments/patch_attack"):
    sys.path.insert(0, os.path.join(HOME, _p))

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
from monitor_attack import run_oracle, s0_sanity  # noqa: E402
from monitor_hijack_backend import MonitorHijackBackend  # noqa: E402
from monitor_replay import margin_report, run_control, run_replay  # noqa: E402

SEED = int(os.environ.get("GATEB_SEED", "0"))
ORACLE_STEPS = int(os.environ.get("GATEB_ORACLE_STEPS", "130"))
K = int(os.environ.get("GATEB_K", "6"))
EPS = float(os.environ.get("GATEB_EPS", "0.15"))
RUN_DIR = os.path.join(HOME, "autoresearch/runs/monitor-hijack", f"seed{SEED}")


def _load_video(record_dir: str) -> list[np.ndarray]:
    files = sorted(
        f for f in os.listdir(record_dir) if f.startswith("texture_") and f.endswith(".png")
    )
    return [np.asarray(imageio.imread(os.path.join(record_dir, f)), dtype=np.uint8) for f in files]


def main() -> None:
    os.makedirs(RUN_DIR, exist_ok=True)
    backend = MonitorHijackBackend()

    # --- S0 sanity (this seed) -------------------------------------------------------------
    s0 = s0_sanity(backend, [SEED], max_steps=ORACLE_STEPS)
    print(f"[gate-b] S0 seed={SEED}: results={s0.results} all_pass={s0.all_pass}", flush=True)

    # --- Stage 1: oracle (records texture_0..T through the real renderer) -------------------
    tex_dir = os.path.join(RUN_DIR, "oracle_textures")
    oracle = run_oracle(backend, SEED, max_steps=ORACLE_STEPS, k=K, eps=EPS, record_dir=tex_dir)
    print(
        f"[gate-b] ORACLE seed={SEED}: targeted={oracle.targeted_success} "
        f"commanded={oracle.commanded_success} steps={oracle.steps} "
        f"max_phase={oracle.max_phase} latch_step={oracle.latch_step}",
        flush=True,
    )
    video = _load_video(tex_dir)
    print(f"[gate-b] recorded video frames: {len(video)}", flush=True)

    # --- Stage 2: open-loop replay + controls ----------------------------------------------
    attack = run_replay(backend, SEED, video, kind="attack")
    blank = run_control(backend, SEED, "blank", video)
    scrambled = run_control(backend, SEED, "scrambled", video)
    report = margin_report(attack, [blank, scrambled])
    print(f"[gate-b] REPLAY attack: {attack}", flush=True)
    print(f"[gate-b] CONTROL blank: {blank}", flush=True)
    print(f"[gate-b] CONTROL scrambled: {scrambled}", flush=True)
    print(f"[gate-b] MARGIN: {json.dumps(report, default=str)}", flush=True)

    result = {
        "seed": SEED,
        "oracle_steps": ORACLE_STEPS,
        "k": K,
        "eps": EPS,
        "s0": {"results": s0.results, "all_pass": s0.all_pass},
        "oracle": dataclasses.asdict(
            dataclasses.replace(oracle, step_logs=())
        ),
        "oracle_token_match_trace": [log.token_match for log in oracle.step_logs],
        "oracle_phase_trace": [log.progress_phase for log in oracle.step_logs],
        "replay": {
            "attack": dataclasses.asdict(attack),
            "blank": dataclasses.asdict(blank),
            "scrambled": dataclasses.asdict(scrambled),
        },
        "margin": report,
        "gate_b_pass": bool(report["hijack_beats_controls"]),
    }
    out = os.path.join(RUN_DIR, "gate_b_result.json")
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"[gate-b] GATE-B pass={result['gate_b_pass']} -> {out}", flush=True)


if __name__ == "__main__":
    main()
