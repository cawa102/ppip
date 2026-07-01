"""Contract of the real OpenVLA rollout backend (the GPU boundary).

The rollout body runs only on a GPU host, so here we pin the CPU-observable
contract: the backend conforms to RolloutBackend, carries the reference-grounded
defaults, and fails loudly with an actionable error when the OpenVLA/LIBERO stack
is not importable (as on this machine) instead of failing obscurely deep inside a
rollout.
"""

from __future__ import annotations

import pytest

from evaluator.backends import RolloutBackend
from evaluator.openvla_backend import OpenVLABackendUnavailable, OpenVLARolloutBackend
from tests.support import make_candidate


def test_backend_conforms_to_protocol_and_has_reference_defaults():
    backend = OpenVLARolloutBackend()

    assert isinstance(backend, RolloutBackend)
    # Defaults mirror the reference OpenVLA+LIBERO harness.
    assert backend.model_id == "openvla/openvla-7b-finetuned-libero-spatial"
    assert backend.unnorm_key == "libero_spatial"
    assert backend.max_steps == 220
    assert backend.num_steps_wait == 10


def test_run_rollouts_without_gpu_stack_raises_actionable_error():
    backend = OpenVLARolloutBackend()
    with pytest.raises(OpenVLABackendUnavailable) as excinfo:
        backend.run_rollouts(candidate=make_candidate(), seeds=[0], rollouts_per_candidate=1)
    # The message must point the user at the missing dependency stack.
    assert "openvla" in str(excinfo.value).lower() or "torch" in str(excinfo.value).lower()
