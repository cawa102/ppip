"""Collect patched USER-command states and pair them with TARGET trajectory tokens."""

from __future__ import annotations

import os
import sys

import numpy as np

HOME = os.path.expanduser("~")
for p in ("autoresearch/src", "autoresearch", "openvla", "autoresearch/experiments/patch_attack"):
    sys.path.insert(0, os.path.join(HOME, p))

import patch_config as C  # noqa: E402
from eval_patch import _candidate  # noqa: E402
from hijack_backend import HijackBackend  # noqa: E402

SEED = int(os.environ.get("PATCH_ATTACK_STUDENT_SEED", "0"))
MAX_STEPS = int(os.environ.get("PATCH_ATTACK_MAX_STEPS", "180"))
TARGET_NPZ = os.environ.get("PATCH_ATTACK_TARGET_NPZ", C.STATES_NPZ)
OUT_NPZ = os.environ.get(
    "PATCH_ATTACK_STUDENT_OUT",
    os.path.join(C.RUN_DIR, "states_student_to_target.npz"),
)


def main() -> None:
    target = np.load(TARGET_NPZ)
    target_tokens = target["tokens"]
    target_actions = target["actions"]
    patch = np.load(os.environ.get("PATCH_ATTACK_NPY", C.PATCH_NPY))
    row0 = int(os.environ.get("PATCH_ATTACK_ROW0", str(C.PATCH_ROW0)))
    col0 = int(os.environ.get("PATCH_ATTACK_COL0", str(C.PATCH_COL0)))

    backend = HijackBackend(run_dir=C.RUN_DIR, max_steps=MAX_STEPS)
    backend.set_instruction_override(None)
    backend.set_patch(patch, (row0, col0))
    backend._collect = []
    outcomes = backend.run_rollouts(
        candidate=_candidate(), seeds=[SEED], rollouts_per_candidate=1
    )

    frames = np.stack(backend._collect).astype(np.uint8)
    n = min(len(frames), len(target_tokens), MAX_STEPS)
    frames = frames[:n]
    tokens = target_tokens[:n]
    actions = target_actions[:n]
    np.savez_compressed(OUT_NPZ, images=frames, tokens=tokens, actions=actions)
    print(f"[student] saved {n} frames -> {OUT_NPZ}", flush=True)
    for outcome in outcomes:
        td = outcome.target_diagnostics
        print(
            f"[student] seed={outcome.seed} targeted={outcome.targeted_success} "
            f"commanded={outcome.commanded_success} "
            f"target_moved={getattr(td, 'target_object_moved_m', None)} "
            f"min_target_dist={getattr(td, 'min_target_distance_m', None)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
