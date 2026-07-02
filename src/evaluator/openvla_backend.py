"""The real OpenVLA+LIBERO rollout backend (GPU-only body).

This is the concrete `RolloutBackend` the harness uses on a GPU host. The
CPU-observable contract is exercised in tests here; the closed-loop rollout body
is to be implemented on this GPU host (Phase C), following the reference harness.

Defaults are grounded in the reference OpenVLA+LIBERO project. The task suite is
`libero_object`: its 10 tasks share one scene with 7 objects, and each goal places a
distinct object in the basket -- so a user/target task pair has independent,
benchmark-native success predicates (the targeted-vs-commanded distinction the study
needs). See docs/research/targeted-success-design.md.
  * model_id        openvla/openvla-7b-finetuned-libero-object
  * unnorm_key      libero_object    (action de-normalisation stats key)
  * max_steps       280              (libero_object episode cap; spatial=220, goal=300)
  * num_steps_wait  10               (no-op settle steps at episode start)

Intended rollout body (per candidate, for each seed x rollout):
  1. Load model/processor once (AutoModelForVision2Seq + AutoProcessor, bf16).
  2. Build the LIBERO env for the task suite; render the candidate's visual prompt
     into the scene (src/rendering) at its placement/style.
  3. Run the closed loop like the reference `run_episode`: get_libero_image ->
     get_action -> normalize/invert gripper -> env.step, until `done` or max_steps.
  4. commanded_success = env `done` on the user task.
  5. targeted_success = the attacker target task's fixed success predicate over
     the same rollout. Latch it once true, but do not terminate the episode only
     because the auxiliary target predicate fired.
  6. Return one RolloutOutcome per episode.
"""

from __future__ import annotations

from typing import Any

from evaluator.metrics import RolloutOutcome

_DEFAULT_MODEL_ID = "openvla/openvla-7b-finetuned-libero-object"
_DEFAULT_UNNORM_KEY = "libero_object"
_DEFAULT_MAX_STEPS = 280
_DEFAULT_NUM_STEPS_WAIT = 10


class OpenVLABackendUnavailable(RuntimeError):
    """Raised when the OpenVLA/LIBERO/torch stack is not importable on this host."""


def _require_openvla_stack() -> Any:
    """Import the GPU stack lazily; raise an actionable error if it is absent."""
    try:
        import torch  # noqa: F401  (presence check only)
        import transformers  # noqa: F401
    except ImportError as exc:
        raise OpenVLABackendUnavailable(
            "The OpenVLA rollout backend needs the GPU stack (torch, transformers, "
            "openvla, LIBERO). Install the 'gpu' extra on a CUDA host and run there; "
            f"import failed with: {exc}"
        ) from exc
    return torch


class OpenVLARolloutBackend:
    """Runs real OpenVLA rollouts for a candidate on a GPU host."""

    def __init__(
        self,
        *,
        model_id: str = _DEFAULT_MODEL_ID,
        unnorm_key: str = _DEFAULT_UNNORM_KEY,
        task_suite: str = "libero_object",
        max_steps: int = _DEFAULT_MAX_STEPS,
        num_steps_wait: int = _DEFAULT_NUM_STEPS_WAIT,
        device: str = "cuda",
    ) -> None:
        self.model_id = model_id
        self.unnorm_key = unnorm_key
        self.task_suite = task_suite
        self.max_steps = max_steps
        self.num_steps_wait = num_steps_wait
        self.device = device

    def run_rollouts(
        self,
        *,
        candidate: dict[str, Any],
        seeds: list[int],
        rollouts_per_candidate: int,
    ) -> list[RolloutOutcome]:
        """Run rollouts for a candidate (GPU-only; see module docstring)."""
        _require_openvla_stack()
        # The GPU host implements the closed loop described in the module docstring
        # and returns one RolloutOutcome per episode.
        raise NotImplementedError(  # pragma: no cover - implemented on the GPU host
            "OpenVLARolloutBackend.run_rollouts must be implemented on the GPU host; "
            "see the module docstring for the reference rollout procedure."
        )
