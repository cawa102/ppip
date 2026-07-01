"""Contract of the real OpenVLA rollout backend (the GPU boundary).

On the GPU host the OpenVLA/torch stack is importable. We pin the CPU-observable
contract: the backend conforms to RolloutBackend, carries the reference-grounded
defaults, its dependency guard passes when the stack is present, and it still fails
loudly with an actionable error if the stack is ever missing (simulated here so the
check does not depend on the host) rather than failing obscurely deep in a rollout.
"""

from __future__ import annotations

import sys

import pytest

from evaluator.backends import RolloutBackend
from evaluator.openvla_backend import (
    OpenVLABackendUnavailable,
    OpenVLARolloutBackend,
    _require_openvla_stack,
)
from tests.support import make_candidate


def test_backend_conforms_to_protocol_and_has_reference_defaults():
    backend = OpenVLARolloutBackend()

    assert isinstance(backend, RolloutBackend)
    # Defaults mirror the reference OpenVLA+LIBERO harness.
    assert backend.model_id == "openvla/openvla-7b-finetuned-libero-spatial"
    assert backend.unnorm_key == "libero_spatial"
    assert backend.max_steps == 220
    assert backend.num_steps_wait == 10


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
