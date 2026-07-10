"""DAgger-style state collection: gather the states the arm ACTUALLY visits under the USER
task (optionally with the current delta), and label each with the TARGET (salad_dressing)
teacher action tokens. Training the delta on THESE states (vs the target trajectory the arm
never reaches) directly addresses closed-loop distribution shift.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

HOME = os.path.expanduser("~")
for p in ("autoresearch/src", "autoresearch", "openvla", "autoresearch/experiments/patch_attack"):
    sys.path.insert(0, os.path.join(HOME, p))

import patch_config as C  # noqa: E402
import vla_diff  # noqa: E402
from collect_states import _carrier, teacher_tokens  # noqa: E402
from hijack_backend import HijackBackend  # noqa: E402

SEEDS = [int(s) for s in os.environ.get("DAGGER_SEEDS", "0").split(",") if s.strip()]
MAX_STEPS = int(os.environ.get("DAGGER_MAX_STEPS", "120"))
SUBSAMPLE = int(os.environ.get("DAGGER_SUBSAMPLE", "2"))
USE_DELTA = os.environ.get("DAGGER_USE_DELTA", "0") == "1"
OUT = os.environ.get("DAGGER_OUT", os.path.join(C.RUN_DIR, "states_dagger.npz"))
DELTA_NPY = os.path.join(C.RUN_DIR, "delta.npy")


def main() -> None:
    backend = HijackBackend(run_dir=C.RUN_DIR, max_steps=MAX_STEPS)
    backend.set_instruction_override(None)  # roll out under the USER task (alphabet_soup)
    if USE_DELTA and os.path.exists(DELTA_NPY):
        backend.set_delta(np.load(DELTA_NPY))
        print(f"[dagger] rolling out WITH current delta", flush=True)
    all_frames: list[np.ndarray] = []
    for s in SEEDS:
        backend._collect = []
        backend.run_rollouts(candidate=_carrier(f"dagger_s{s}"), seeds=[s], rollouts_per_candidate=1)
        frames = backend._collect[:MAX_STEPS][::SUBSAMPLE]
        all_frames.extend(frames)
        print(f"[dagger] seed {s}: {len(backend._collect)} frames -> kept {len(frames)}", flush=True)
    frames = np.stack(all_frames).astype(np.uint8)
    model, processor, _, _ = backend._policy
    print(f"[dagger] teacher(salad_dressing) tokens for {len(frames)} user-trajectory states ...",
          flush=True)
    tokens = teacher_tokens(model, processor, frames, C.TARGET_TASK)
    # Merge with any existing target-trajectory states for coverage of both distributions.
    if os.path.exists(C.STATES_NPZ):
        base = np.load(C.STATES_NPZ)
        frames = np.concatenate([base["images"], frames], axis=0)
        tokens = np.concatenate([base["tokens"], tokens], axis=0)
    np.savez_compressed(OUT, images=frames, tokens=tokens)
    print(f"[dagger] saved {len(frames)} combined states -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
