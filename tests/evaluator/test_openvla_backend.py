"""Contract of the real OpenVLA rollout backend (the GPU boundary).

On the GPU host the OpenVLA/torch stack is importable. We pin the CPU-observable
contract: the backend conforms to RolloutBackend, carries the reference-grounded
defaults, its dependency guard passes when the stack is present, and it still fails
loudly with an actionable error if the stack is ever missing (simulated here so the
check does not depend on the host) rather than failing obscurely deep in a rollout.
"""

from __future__ import annotations

import os
import sys

import pytest

from evaluator.backends import RolloutBackend
from evaluator.openvla_backend import (
    OpenVLABackendUnavailable,
    OpenVLARolloutBackend,
    _require_openvla_stack,
)
from tests.support import make_candidate

# The real-model tests load OpenVLA-7B and build a MuJoCo/EGL env, so they only run
# on the GPU host. Set PPIP_GPU_TESTS=1 (with CUDA_VISIBLE_DEVICES=1, MUJOCO_GL=egl,
# PYTHONPATH=$HOME/LIBERO) to enable them; the default CPU suite skips them.
requires_gpu = pytest.mark.skipif(
    not os.environ.get("PPIP_GPU_TESTS"),
    reason="set PPIP_GPU_TESTS=1 on the GPU host (GPU 1) to run real-model tests",
)


def test_backend_conforms_to_protocol_and_has_reference_defaults():
    backend = OpenVLARolloutBackend()

    assert isinstance(backend, RolloutBackend)
    # Defaults mirror the reference OpenVLA+LIBERO harness for the libero_object suite
    # (the suite whose task pairs give independent user/target predicates).
    assert backend.model_id == "openvla/openvla-7b-finetuned-libero-object"
    assert backend.unnorm_key == "libero_object"
    assert backend.task_suite == "libero_object"
    assert backend.max_steps == 280
    assert backend.num_steps_wait == 10


def test_build_cfg_has_openvla_helper_fields():
    # `_build_cfg` assembles the SimpleNamespace that OpenVLA's get_action /
    # get_image_resize_size read (a subset of run_libero_eval's GenerateConfig).
    # These fields must exactly mirror the reference; center_crop=True is essential
    # (the fine-tunes trained with random-crop aug).
    backend = OpenVLARolloutBackend()

    cfg = backend._build_cfg()

    assert cfg.model_family == "openvla"
    assert cfg.pretrained_checkpoint == "openvla/openvla-7b-finetuned-libero-object"
    assert cfg.unnorm_key == "libero_object"
    assert cfg.task_suite_name == "libero_object"
    assert cfg.center_crop is True
    assert cfg.load_in_8bit is False
    assert cfg.load_in_4bit is False


def test_require_openvla_stack_passes_on_gpu_host():
    # On the GPU host the OpenVLA/torch stack is importable, so the guard returns
    # torch instead of raising OpenVLABackendUnavailable.
    import torch

    assert _require_openvla_stack() is torch


def test_run_rollouts_raises_actionable_error_when_stack_missing(monkeypatch):
    # Defensive contract: if the GPU stack is ever unimportable, fail loudly and
    # actionably up front rather than obscurely deep inside a rollout. Simulate the
    # missing stack by making `import torch` raise, independent of the host.
    monkeypatch.setitem(sys.modules, "torch", None)
    backend = OpenVLARolloutBackend()
    with pytest.raises(OpenVLABackendUnavailable) as excinfo:
        backend.run_rollouts(candidate=make_candidate(), seeds=[0], rollouts_per_candidate=1)
    # The message must point the user at the missing dependency stack.
    assert "openvla" in str(excinfo.value).lower() or "torch" in str(excinfo.value).lower()


@requires_gpu
def test_build_env_matches_resolved_task_language():
    from evaluator.libero_tasks import resolve_task

    resolved = resolve_task("pick up the bbq sauce and place it in the basket")
    backend = OpenVLARolloutBackend()

    env, init_states, task_description, obj_of_interest = backend._build_env(resolved)
    try:
        assert str(task_description) == resolved.language
        assert len(init_states) > 0
        # Shared scene: the target object is present for adjudication.
        assert any("bbq_sauce" in name for name in obj_of_interest)
    finally:
        env.close()


@requires_gpu
def test_load_policy_fits_one_card():
    import torch

    backend = OpenVLARolloutBackend()
    model, processor, cfg, resize_size = backend._load_policy()

    device = torch.device(backend.device)
    props = torch.cuda.get_device_properties(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    assert peak_reserved < props.total_memory  # fits one A5000 card
    assert resize_size  # get_image_resize_size returned a usable size
    assert model is not None and processor is not None
