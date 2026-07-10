"""Search-side subclass of the fixed evaluator for the vision-layer hijack attack.

The evaluator (rollout loop, `eval_goal_state` adjudication, metrics) is inherited
UNCHANGED. This subclass only adds two search-side capabilities:

  * a **digital camera-space patch** overlaid on the agentview image the policy sees
    (Tier A adversarial patch — the strongest vision-layer attacker), and
  * an **instruction override** (used by the S0 target-reachability probe).

Scoring is untouched: the fixed `targeted_success` predicate still decides the verdict,
so the optimizer cannot game its own score.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from evaluator.openvla_backend import OpenVLARolloutBackend


class HijackBackend(OpenVLARolloutBackend):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._patch: np.ndarray | None = None  # float [ph, pw, 3] in [0,1]
        self._patch_rc: tuple[int, int] = (0, 0)  # top-left (row, col) in the 224 image
        self._instruction_override: str | None = None
        self._collect: list | None = None  # if a list, every agentview frame is appended
        self._collect_actions: list | None = None  # if a list, raw OpenVLA actions are appended
        self._delta: np.ndarray | None = None  # additive full-image perturbation in [0,1] space

    # --- search-side setters ---
    def set_patch(self, patch: Any, top_left: tuple[int, int]) -> None:
        self._patch = None if patch is None else np.asarray(patch, dtype=np.float32)
        self._patch_rc = (int(top_left[0]), int(top_left[1]))

    def set_instruction_override(self, instruction: str | None) -> None:
        self._instruction_override = instruction

    def set_delta(self, delta: Any) -> None:
        """Full-image additive perturbation (in [0,1] pixel space), applied every step."""
        self._delta = None if delta is None else np.asarray(delta, dtype=np.float32)

    # --- overlay helper ---
    def _overlay(self, image: np.ndarray) -> np.ndarray:
        img = image.copy()
        r, c = self._patch_rc
        assert self._patch is not None
        ph, pw = self._patch.shape[:2]
        patch_u8 = np.clip(self._patch * 255.0, 0, 255).astype(np.uint8)
        img[r : r + ph, c : c + pw, :] = patch_u8
        return img

    # --- overridden action seam: overlay patch + optional instruction override ---
    def _policy_action(self, policy: Any, obs: Any, instruction: str) -> tuple[Any, Any]:
        from experiments.robot.libero.libero_utils import get_libero_image, quat2axisangle
        from experiments.robot.robot_utils import (
            get_action,
            invert_gripper_action,
            normalize_gripper_action,
        )

        model, processor, cfg, resize_size = policy
        image = get_libero_image(obs, resize_size)  # uint8 [224,224,3]
        if self._patch is not None:
            image = self._overlay(image)
        if self._delta is not None:
            pert = np.clip(image.astype(np.float32) / 255.0 + self._delta, 0.0, 1.0)
            image = (pert * 255.0).astype(np.uint8)
        if self._collect is not None:
            self._collect.append(image.copy())
        used_instruction = self._instruction_override or instruction
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
        action = get_action(cfg, model, observation, used_instruction, processor=processor)
        if self._collect_actions is not None:
            self._collect_actions.append(action.copy())
        env_action = normalize_gripper_action(action.copy(), binarize=True)
        env_action = invert_gripper_action(env_action)
        return env_action, image
