"""Run OpenVLA-7B on a LIBERO task — CLEAN, no attack. Start here.

This is the "hello world" of this project: it loads the OpenVLA policy, builds a LIBERO
scene, and closes the perception->action loop until the task succeeds or the step cap is
hit. There is **no injected prompt, no patch, no perturbation** — it is the baseline every
attack result is measured against.

What one control step looks like (this is the whole loop):

    obs  --get_libero_image-->  224x224 RGB  ┐
    obs  --eef pos/quat/gripper-->  8-d state ┤--> OpenVLA(image, instruction) --> 7-d action
                                              ┘
    action --normalize/invert gripper--> env.step(action) --> next obs, done

`done` is LIBERO's own goal predicate for the task the *scene* was built from, so
`done == True` means the commanded task genuinely succeeded.

--------------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------------

You need a Python env with the OpenVLA + LIBERO stack. On this project's GPU host that
is a reused venv, `~/vla-injection/.venv/bin/python` (third_party/README.md has the
pinned commits and package versions). Substitute your own interpreter if it differs.

`MUJOCO_GL=egl` selects headless off-screen rendering — without it MuJoCo tries to open
a window and dies on a server. `CUDA_VISIBLE_DEVICES` picks the GPU: **on this host use
GPU 1 only, GPU 0 is reserved for other people's jobs.**

Run the user task on the first init state:

    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl \
      ~/vla-injection/.venv/bin/python experiments/clean_rollout.py \
        --task "pick up the alphabet soup and place it in the basket"

Run the "teacher"/attacker task instead — same suite, its own scene and predicate:

    ... experiments/clean_rollout.py \
        --task "pick up the salad dressing and place it in the basket"

Several init states + save a video of each episode:

    ... experiments/clean_rollout.py \
        --task "pick up the alphabet soup and place it in the basket" \
        --inits 0,1,2 --video-dir /tmp/clean_rollout

If the openvla / LIBERO checkouts are not siblings of this repo or in $HOME, point at
them explicitly (no PYTHONPATH needed — the script puts them on sys.path itself):

    OPENVLA_ROOT=/data/openvla LIBERO_ROOT=/data/LIBERO ... experiments/clean_rollout.py

Expected: the model loads (~30 s, ~16 GB VRAM), then ~1-3 steps/s. A good task/init
succeeds around step 120-200. `alphabet_soup` at init 0 is a known-good sanity check.

--------------------------------------------------------------------------------
Where this sits in the project
--------------------------------------------------------------------------------
Search side, not the evaluator. It reuses the *fixed* evaluator's GPU seams
(`OpenVLARolloutBackend._load_policy` / `._build_env`) so the physics, camera, and
preprocessing are bit-identical to a scored run — but it writes no metrics, no ledger
row, and no score. Nothing here can influence how an attack is graded, which is the one
invariant this project protects (see CLAUDE.md).

If you need an officially *scored* clean number, don't extend this file — run
`experiments/run_candidate.py` with an off-camera (visibility ~0) carrier candidate so
the result goes through the fixed evaluator and lands in the ledger.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# --- import bootstrap -------------------------------------------------------------
# Three separate source trees have to be importable, and none is pip-installed:
#   <repo>/src   -> this project's `evaluator` package
#   openvla      -> `experiments.robot.*` helpers (get_action, image utils)
#   LIBERO       -> the `libero` package (the simulator, tasks, and init states)
#
# Nothing below is hardcoded to a particular machine: the repo is found relative to this
# file, and the two external trees are found via $OPENVLA_ROOT / $LIBERO_ROOT, falling
# back to siblings of the repo and then to $HOME. Override the env vars if yours live
# somewhere else. `experiments` works as a namespace package across the repo and openvla
# because neither has an __init__.py, so `experiments.robot` resolves into openvla.
REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_tree(name: str, env_var: str, marker: str) -> Path:
    """Locate an external dependency tree, or exit with an actionable message.

    An explicitly-set $env_var must be correct: if it is set but wrong we fail loudly
    rather than quietly falling back, so a typo'd path can never send you off debugging
    the wrong checkout.
    """
    override = os.environ.get(env_var)
    if override:
        root = Path(override).expanduser()
        if not (root / marker).exists():
            raise SystemExit(
                f"{env_var}={override} does not look like a {name} checkout "
                f"(missing {marker}). Fix or unset it."
            )
        return root

    candidates = [REPO_ROOT.parent / name, Path.home() / name]
    for candidate in candidates:
        if (candidate / marker).exists():
            return candidate
    searched = "\n  ".join(str(c) for c in candidates)
    raise SystemExit(
        f"Could not find the {name} source tree (looked for {name}/{marker}).\n"
        f"Searched:\n  {searched}\n"
        f"Set {env_var}=/path/to/{name} and re-run. "
        f"See third_party/README.md for the pinned commit this project expects."
    )


OPENVLA_ROOT = _find_tree("openvla", "OPENVLA_ROOT", "experiments/robot/robot_utils.py")
LIBERO_ROOT = _find_tree("LIBERO", "LIBERO_ROOT", "libero/libero/__init__.py")
for _tree in (REPO_ROOT / "src", REPO_ROOT, OPENVLA_ROOT, LIBERO_ROOT):
    sys.path.insert(0, str(_tree))

from evaluator.libero_tasks import resolve_task  # noqa: E402
from evaluator.openvla_backend import OpenVLARolloutBackend  # noqa: E402

# LIBERO episode caps used by the fixed evaluator — keep these if you want your numbers
# comparable to the recorded runs.
DEFAULT_MAX_STEPS = {"libero_object": 280, "libero_spatial": 220, "libero_goal": 300}
DEMO_RES = 384  # video frames are rendered larger than the 224 the policy sees


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a clean (un-attacked) OpenVLA rollout on a LIBERO task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        default="pick up the alphabet soup and place it in the basket",
        help="Task language string. Builds the scene AND is the instruction given to OpenVLA. "
        "Must match a task in --suite exactly (up to case/punctuation).",
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="ADVANCED: command a different instruction inside --task's scene (the 's0 "
        "reachability' probe). Success is still judged by --task's predicate, so a "
        "success/failure here does NOT adjudicate the overridden instruction.",
    )
    parser.add_argument("--suite", default="libero_object", help="LIBERO task suite.")
    parser.add_argument(
        "--inits",
        default="0",
        help="Comma-separated init-state indices (LIBERO ships ~50 per task). "
        "The scene layout differs per index; results vary with it.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Control-step cap per episode (default: the suite's standard cap).",
    )
    parser.add_argument(
        "--video-dir",
        default=None,
        help="If set, write <suite>_<init>.mp4 per episode here.",
    )
    return parser.parse_args()


def run_episode(
    *,
    backend: OpenVLARolloutBackend,
    policy: tuple,
    resolved,
    init_index: int,
    instruction: str,
    max_steps: int,
    video_path: str | None,
) -> dict:
    """Roll one clean episode. Returns {success, steps, seconds}."""
    from experiments.robot.libero.libero_utils import get_libero_image, quat2axisangle
    from experiments.robot.robot_utils import (
        get_action,
        invert_gripper_action,
        normalize_gripper_action,
    )

    model, processor, cfg, resize_size = policy

    # Seed every RNG from the init index so the episode is reproducible.
    backend._seed_everything(init_index)

    # Build the scene, then jump to the chosen init state. LIBERO needs no explicit
    # reset() here — set_init_state does it.
    env, init_states, task_description, obj_of_interest = backend._build_env(resolved)
    obs = env.set_init_state(init_states[init_index % len(init_states)])

    # The scene drops objects in; a few no-op steps let physics settle before the policy
    # sees anything. Skipping this makes the first observations unrepresentative.
    dummy = backend._dummy_action(cfg)
    for _ in range(backend.num_steps_wait):
        obs, _reward, _done, _info = env.step(dummy)

    frames: list = []
    success = False
    started = time.time()
    step = 0
    try:
        for step in range(max_steps):
            if video_path is not None:
                frames.append(get_libero_image(obs, DEMO_RES))

            # --- what the policy actually consumes -------------------------------
            image = get_libero_image(obs, resize_size)  # uint8 [224, 224, 3]
            observation = {
                "full_image": image,
                "state": np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    )
                ),
            }
            action = get_action(cfg, model, observation, instruction, processor=processor)

            # OpenVLA's gripper convention differs from LIBERO's: binarize, then flip.
            # Get this wrong and the robot approaches correctly but never grasps.
            action = invert_gripper_action(normalize_gripper_action(action, binarize=True))

            obs, _reward, done, _info = env.step(
                action.tolist() if hasattr(action, "tolist") else action
            )
            if step % 20 == 0:
                print(f"    step {step:3d}/{max_steps}", flush=True)
            if done:  # LIBERO's own goal predicate for this task
                success = True
                break
    finally:
        env.close()

    if video_path is not None and frames:
        import imageio.v2 as imageio

        imageio.mimwrite(video_path, frames, fps=20)
        print(f"    video -> {video_path}", flush=True)

    return {"success": success, "steps": step + 1, "seconds": round(time.time() - started, 1)}


def main() -> None:
    args = parse_args()
    inits = [int(x) for x in args.inits.split(",") if x.strip()]
    max_steps = args.max_steps or DEFAULT_MAX_STEPS.get(args.suite, 280)

    # Fail fast on a typo'd task string, BEFORE spending 30 s loading a 7B model.
    resolved = resolve_task(args.task, suite=args.suite)
    instruction = args.instruction or resolved.language

    print(f"suite       : {args.suite}")
    print(f"scene task  : {resolved.language!r} (task_id={resolved.task_id})")
    print(f"instruction : {instruction!r}")
    if args.instruction:
        print("  ^ OVERRIDDEN: 'success' below still means the SCENE task's predicate fired.")
    print(f"inits       : {inits}   max_steps: {max_steps}\n")

    if args.video_dir:
        os.makedirs(args.video_dir, exist_ok=True)

    # One backend, one model load, reused across episodes. Re-loading per episode without
    # freeing the old model exhausts VRAM — that bug cost us a whole pilot run.
    backend = OpenVLARolloutBackend(task_suite=args.suite, max_steps=max_steps)
    print("loading OpenVLA-7B (~30 s, ~16 GB VRAM) ...", flush=True)
    policy = backend._load_policy()
    print("loaded.\n", flush=True)

    results = []
    for init_index in inits:
        print(f"[init {init_index}] rolling out ...", flush=True)
        video_path = (
            os.path.join(args.video_dir, f"{args.suite}_init{init_index}.mp4")
            if args.video_dir
            else None
        )
        result = run_episode(
            backend=backend,
            policy=policy,
            resolved=resolved,
            init_index=init_index,
            instruction=instruction,
            max_steps=max_steps,
            video_path=video_path,
        )
        results.append((init_index, result))
        verdict = "SUCCESS" if result["success"] else "FAIL   "
        print(
            f"[init {init_index}] {verdict}  steps={result['steps']}  "
            f"({result['seconds']}s)\n",
            flush=True,
        )

    n_success = sum(1 for _i, r in results if r["success"])
    print("=== SUMMARY ===")
    for init_index, result in results:
        print(f"  init {init_index}: success={result['success']} steps={result['steps']}")
    print(f"  {n_success}/{len(results)} succeeded")


if __name__ == "__main__":
    main()
