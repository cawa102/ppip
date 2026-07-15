"""Threat-faithful rollout backend for the in-scene monitor hijack (Task 4).

`MonitorHijackBackend` subclasses the fixed `OpenVLARolloutBackend` and mechanizes the
locked invariant: every scored policy input derives ONLY from a fresh MuJoCo render taken
AFTER the monitor texture upload; the camera image buffer is NEVER written. This is
deliberately NOT routed through `hijack_backend.py`/`adaptive_attack.py`, which execute
from a perturbed `pu8` (a camera-buffer write) -- that path is "hacking the camera" under
our threat model.

Three canonical image stages are hashed every step so the invariant is auditable:
  * S1 -- the raw MuJoCo agentview render (H x W x 3 uint8),
  * S2 -- the 224 policy-input image (get_libero_image: rot180 + resize) built FROM S1,
  * S3 -- the processor `pixel_values` tensor the model consumes.
`assert_policy_input_fresh` refuses any S2 that is not `transform(S1)` (e.g. a stale obs).

Scoring stays the fixed `eval_goal_state` object-state predicate -- untouched here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from evaluator.openvla_backend import OpenVLARolloutBackend


class StalePolicyInputError(RuntimeError):
    """Raised when the policy input is not the fresh post-upload render (invariant breach)."""


def _hash(array: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def canonical_stage_hashes(
    s1_raw: NDArray[Any], s2_policy_input: NDArray[Any], s3_pixel_values: NDArray[Any]
) -> tuple[str, str, str]:
    """SHA-256 of each canonical image stage (raw render, 224 policy input, pixel_values)."""
    return (_hash(s1_raw), _hash(s2_policy_input), _hash(s3_pixel_values))


def assert_policy_input_fresh(
    s1_raw: NDArray[Any],
    s2_policy_input: NDArray[Any],
    transform: Callable[[NDArray[Any]], NDArray[Any]],
) -> None:
    """Assert S2 is exactly ``transform(S1)`` -- i.e. built from the fresh post-upload render.

    Guards against feeding the policy a stale/cached camera image (which does not reflect the
    monitor upload) instead of the fresh render.
    """
    expected = transform(s1_raw)
    if _hash(expected) != _hash(s2_policy_input):
        raise StalePolicyInputError(
            "policy input is not the fresh post-upload render (stale or perturbed frame)"
        )


@dataclass(frozen=True)
class StepResult:
    """One monitor-hijack control step: the action taken plus the invariant evidence."""

    action: NDArray[np.float64]
    policy_image: NDArray[np.uint8]  # S2, the 224 image fed to the policy
    s1_hash: str
    s2_hash: str
    s3_hash: str
    done: bool


class MonitorHijackBackend(OpenVLARolloutBackend):
    """OpenVLA rollout backend whose policy input is always a fresh post-upload render."""

    def step_with_texture(
        self,
        env: Any,
        handle: Any,
        texture: NDArray[np.uint8],
        instruction: str,
        *,
        resize_size: int = 224,
        render_size: int = 256,
    ) -> StepResult:  # pragma: no cover - GPU seam, verified in GPU rollout env
        """Upload ``texture`` -> fresh render -> feed that exact frame to OpenVLA -> step env.

        Strict ordering enforces the invariant: the policy input is built ONLY from the
        post-upload MuJoCo render (never a stale obs, never a camera-buffer write). Returns
        the action taken plus the three canonical-stage hashes as invariant evidence.
        """
        import torch
        import vla_diff
        from experiments.robot.libero.libero_utils import get_libero_image, quat2axisangle
        from experiments.robot.robot_utils import (
            get_action,
            invert_gripper_action,
            normalize_gripper_action,
        )

        from rendering.monitor import _fresh_obs

        if self._policy is None:
            self._policy = self._load_policy()
        model, processor, cfg, _rs = self._policy

        # 1) upload the attacker's texture, 2) FRESH render (reflects the upload).
        handle.upload(np.ascontiguousarray(np.asarray(texture, dtype=np.uint8)))
        env.sim.forward()
        s1 = np.asarray(
            env.sim.render(width=render_size, height=render_size, camera_name="agentview")
        )

        def _transform(raw: NDArray[Any]) -> NDArray[np.uint8]:
            return np.asarray(
                get_libero_image({"agentview_image": np.asarray(raw)}, resize_size),
                dtype=np.uint8,
            )

        s2 = _transform(s1)
        # 3) invariant guard: the policy input IS the fresh post-upload render.
        assert_policy_input_fresh(s1, s2, _transform)

        # 4) build the policy observation from the fresh image + current proprioception.
        obs = _fresh_obs(env)
        observation = {
            "full_image": s2,
            "state": np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            ),
        }
        action = get_action(cfg, model, observation, instruction, processor=processor)

        # S3 audit tensor: the processor pixel_values equivalent, derived from S2.
        s2_t = torch.from_numpy(s2.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        s3 = vla_diff.preprocess(s2_t).detach().cpu().numpy()

        env_action = invert_gripper_action(
            normalize_gripper_action(np.asarray(action).copy(), binarize=True)
        )
        _obs, _r, done, _i = env.step(
            env_action.tolist() if hasattr(env_action, "tolist") else env_action
        )

        s1_hash, s2_hash, s3_hash = canonical_stage_hashes(s1, s2, s3)
        return StepResult(
            action=np.asarray(action, dtype=np.float64),
            policy_image=s2,
            s1_hash=s1_hash,
            s2_hash=s2_hash,
            s3_hash=s3_hash,
            done=bool(done),
        )
