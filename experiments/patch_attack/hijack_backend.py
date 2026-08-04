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

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from evaluator.libero_tasks import resolve_task
from evaluator.metrics import RolloutOutcome
from evaluator.openvla_backend import OpenVLARolloutBackend, _require_openvla_stack


class HijackBackend(OpenVLARolloutBackend):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._patch: NDArray[np.float32] | None = None  # float [ph, pw, 3] in [0,1]
        self._patch_rc: tuple[int, int] = (0, 0)  # top-left (row, col) in the 224 image
        self._instruction_override: str | None = None
        # If a list, every agentview frame / raw OpenVLA action is appended to it.
        self._collect: list[NDArray[np.uint8]] | None = None
        self._collect_actions: list[Any] | None = None
        # Additive full-image perturbation in [0,1] space, applied every step.
        self._delta: NDArray[np.float32] | None = None
        # Masked full-frame replacement: the general region form. A single rect caps out near
        # 28% of this scene before covering task objects, so the area axis needs a mask.
        self._mask: NDArray[np.bool_] | None = None
        self._masked_patch: NDArray[np.float32] | None = None

    # --- search-side setters ---
    def set_patch(self, patch: Any, top_left: tuple[int, int]) -> None:
        self._patch = None if patch is None else np.asarray(patch, dtype=np.float32)
        self._patch_rc = (int(top_left[0]), int(top_left[1]))

    def set_instruction_override(self, instruction: str | None) -> None:
        self._instruction_override = instruction

    def set_delta(self, delta: Any) -> None:
        """Full-image additive perturbation (in [0,1] pixel space), applied every step."""
        self._delta = None if delta is None else np.asarray(delta, dtype=np.float32)

    def set_masked_patch(self, patch_full: Any, mask: Any) -> None:
        """Replace the agentview wherever `mask` is true, using `patch_full` [224,224,3].

        Generalises `set_patch` (one rect) to an arbitrary region. Passing `None` for either
        clears it, so a backend can be reused across conditions without leaking state.
        """
        if patch_full is None or mask is None:
            self._masked_patch, self._mask = None, None
            return
        self._masked_patch = np.asarray(patch_full, dtype=np.float32)
        self._mask = np.asarray(mask).astype(bool)
        if self._masked_patch.shape[:2] != self._mask.shape:
            raise ValueError(
                f"patch {self._masked_patch.shape[:2]} and mask {self._mask.shape} disagree"
            )

    # --- policy loading ---
    def load_policy_once(self) -> tuple[Any, Any, Any, Any]:
        """Return the cached policy, loading it into **this backend's cache** if absent.

        Calling `_load_policy()` directly and keeping the result in a local variable leaves
        `self._policy` as None, so the next `run_rollouts*` call loads a *second* copy of the
        7B model and OOMs a 24 GB card. Always go through here.
        """
        if self._policy is None:
            self._policy = self._load_policy()
        return self._policy

    # --- explicit init selection (the precommit needs non-contiguous indices) ---
    def run_rollouts_at_inits(
        self, *, candidate: dict[str, Any], init_indices: Sequence[int]
    ) -> list[RolloutOutcome]:
        """Run one episode per EXPLICIT init-state index, via the inherited episode path.

        `run_rollouts` derives `init_selector` from a seed's **list position**
        (`seed_index * rollouts_per_candidate + episode_index`), so it can only ever reach
        init states `0..len(seeds)-1`. Passing the precommitted indices as `seeds` would
        therefore roll inits 0,1,2,... while *labelling* them 4,7,22,... -- silently wrong
        numbers, not a crash. `shared_inits` is deliberately non-contiguous (init 0 is
        excluded as selection-contaminated), so selection has to be explicit.

        Scoring is untouched: this calls the inherited, unmodified `_run_one_episode`, so
        the fixed `eval_goal_state` predicates, latch-not-terminate semantics, and the
        run-to-`done`/`max_steps` episode length all apply exactly as in a scored run.
        `outcome.seed` carries the init index, so a row is traceable to its episode.
        """
        _require_openvla_stack()
        resolved_user = resolve_task(candidate["user_task"], suite=self.task_suite)
        resolved_target = resolve_task(candidate["target_task"], suite=self.task_suite)
        if self._policy is None:
            self._policy = self._load_policy()

        return [
            self._run_one_episode(
                candidate=candidate,
                policy=self._policy,
                resolved_user=resolved_user,
                resolved_target=resolved_target,
                seed=int(index),
                episode_index=0,
                init_selector=int(index),
            )
            for index in init_indices
        ]

    def _overlay_masked(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        img: NDArray[np.uint8] = image.copy()
        assert self._masked_patch is not None and self._mask is not None
        patch_u8 = np.clip(self._masked_patch * 255.0, 0, 255).astype(np.uint8)
        img[self._mask] = patch_u8[self._mask]
        return img

    # --- overlay helper ---
    def _overlay(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        img: NDArray[np.uint8] = image.copy()
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
        if self._mask is not None:
            image = self._overlay_masked(image)
        elif self._patch is not None:
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
