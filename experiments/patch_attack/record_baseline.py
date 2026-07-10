"""Record a CLEAN baseline rollout (no attack) for contrast in the demo video.

Commands the USER task (alphabet_soup) and runs OpenVLA normally -> the arm grasps the soup.
Dumps per-step high-res agentview frames to $ADAPT_RECORD_DIR/scene/.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HOME = os.path.expanduser("~")
for p in ("autoresearch/src", "autoresearch", "openvla", "autoresearch/experiments/patch_attack"):
    sys.path.insert(0, os.path.join(HOME, p))

import patch_config as C  # noqa: E402
from evaluator.libero_tasks import resolve_task  # noqa: E402
from hijack_backend import HijackBackend  # noqa: E402

SEED = int(os.environ.get("ADAPT_SEED", "0"))
MAX_STEPS = int(os.environ.get("BASE_MAX_STEPS", "130"))
RECORD_DIR = os.environ["ADAPT_RECORD_DIR"]
DEMO_RES = int(os.environ.get("ADAPT_DEMO_RES", "384"))


def main() -> None:
    import imageio.v2 as imageio
    from experiments.robot.libero.libero_utils import get_libero_image, quat2axisangle
    from experiments.robot.robot_utils import get_action, invert_gripper_action, normalize_gripper_action
    from rendering.inject import inject_prompt

    os.makedirs(os.path.join(RECORD_DIR, "scene"), exist_ok=True)
    resolved_user = resolve_task(C.USER_TASK, suite="libero_object")
    resolved_target = resolve_task(C.TARGET_TASK, suite="libero_object")
    backend = HijackBackend(run_dir=C.RUN_DIR, max_steps=MAX_STEPS)
    backend._policy = backend._load_policy()
    model, processor, cfg, resize_size = backend._policy
    env, init_states, _d, _o = backend._build_env(resolved_user)
    cand = {"candidate_id": f"baseline_seed{SEED}", "condition": "loop_with_skill",
            "user_task": C.USER_TASK, "target_task": C.TARGET_TASK,
            "visual_prompt": {"text": "(baseline)", "prompt_level": "optimized_typographic_prompt"},
            "placement": {"scene_anchor": "table_surface", "position": [0.9, 0.5, 1.9],
                          "rotation": [90.0, 90.0, 0.0], "scale": 0.6},
            "style": {"foreground_color": "#000000", "background_color": "#ffffff",
                      "font_size": 40, "font_family": "sans-serif"},
            "metadata": {"created_by": "baseline", "created_at": "2026-07-06T00:00:00Z", "notes": "x"}}
    inject_prompt(env, cand, texture_dir=backend._texture_dir_for(cand["candidate_id"]))
    obs = env.set_init_state(init_states[SEED % len(init_states)])
    for _ in range(backend.num_steps_wait):
        obs, _r, _d, _i = env.step(backend._dummy_action(cfg))

    tobj, treg = backend._target_entities(resolved_target)
    commanded = False
    for step in range(MAX_STEPS):
        imageio.imwrite(os.path.join(RECORD_DIR, "scene", f"f{step:04d}.png"),
                        get_libero_image(obs, DEMO_RES))
        image = get_libero_image(obs, resize_size)
        observation = {"full_image": image, "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))}
        action = get_action(cfg, model, observation, C.USER_TASK, processor=processor)
        action = invert_gripper_action(normalize_gripper_action(action, binarize=True))
        obs, _r, done, _i = env.step(action.tolist() if hasattr(action, "tolist") else action)
        if step % 10 == 0:
            print(f"[baseline] step={step} done={done}", flush=True)
        if done:
            commanded = True
            print(f"[baseline] commanded success at step {step}", flush=True)
            break
    env.close()
    print(f"[baseline] DONE commanded(alphabet_soup)={commanded}; frames in {RECORD_DIR}/scene", flush=True)


if __name__ == "__main__":
    main()
