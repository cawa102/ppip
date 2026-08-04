"""`run_rollouts_at_inits` exists because `run_rollouts` cannot reach a non-contiguous
precommit, and the failure mode is silent (wrong episodes, right-looking labels). That makes
the index mapping a correctness contract, so it is pinned here on CPU."""

from __future__ import annotations

from typing import Any

import pytest
import shared_inits
from hijack_backend import HijackBackend


class _Recorder:
    """Stands in for the inherited episode path, capturing what it was asked to roll."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return kwargs


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> HijackBackend:
    """A backend with the GPU seams stubbed out -- only the index plumbing is under test."""
    import hijack_backend as module

    from evaluator import openvla_backend as parent

    # Both modules resolve tasks and gate on the GPU stack; the inherited `run_rollouts`
    # test below goes through the parent's copies, so stub each at its own import site.
    for target in (module, parent):
        monkeypatch.setattr(target, "_require_openvla_stack", lambda: None)
        monkeypatch.setattr(target, "resolve_task", lambda task, suite: f"resolved:{task}")
    instance = HijackBackend()
    instance._policy = ("model", "processor", "cfg", 224)  # already "loaded"
    return instance


CANDIDATE = {"user_task": "user task", "target_task": "target task"}


def test_each_requested_index_becomes_its_own_init_selector(
    backend: HijackBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(backend, "_run_one_episode", recorder)

    backend.run_rollouts_at_inits(candidate=CANDIDATE, init_indices=[4, 7, 22])

    # The whole point: index 22 must roll init state 22, not list position 2.
    assert [c["init_selector"] for c in recorder.calls] == [4, 7, 22]


def test_seed_label_matches_the_init_actually_rolled(
    backend: HijackBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(backend, "_run_one_episode", recorder)

    backend.run_rollouts_at_inits(candidate=CANDIDATE, init_indices=[4, 7, 22])

    # A mislabelled row would attribute a result to the wrong episode.
    assert [c["seed"] for c in recorder.calls] == [c["init_selector"] for c in recorder.calls]


def test_the_precommitted_heldout_set_maps_through_unchanged(
    backend: HijackBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(backend, "_run_one_episode", recorder)

    backend.run_rollouts_at_inits(
        candidate=CANDIDATE, init_indices=shared_inits.HELDOUT_INITS
    )

    assert [c["init_selector"] for c in recorder.calls] == list(shared_inits.HELDOUT_INITS)


def test_run_rollouts_positional_mapping_is_why_this_method_exists(
    backend: HijackBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Documents the trap: the inherited API derives init_selector from LIST POSITION, so
    # feeding it the precommit would roll inits 0,1,2 while reporting seeds 4,7,22.
    recorder = _Recorder()
    monkeypatch.setattr(backend, "_run_one_episode", recorder)

    backend.run_rollouts(candidate=CANDIDATE, seeds=[4, 7, 22], rollouts_per_candidate=1)

    assert [c["init_selector"] for c in recorder.calls] == [0, 1, 2]
    assert [c["seed"] for c in recorder.calls] == [4, 7, 22]


def test_empty_index_list_rolls_nothing(
    backend: HijackBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(backend, "_run_one_episode", recorder)

    assert backend.run_rollouts_at_inits(candidate=CANDIDATE, init_indices=[]) == []
    assert recorder.calls == []


def test_load_policy_once_populates_the_backends_own_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Keeping the policy only in a local (the `_load_policy()` shortcut) leaves the cache
    # empty, so the next run_rollouts* call loads a SECOND 7B model and OOMs the card.
    instance = HijackBackend()
    loads: list[int] = []

    def _fake_load() -> tuple[str, str, str, int]:
        loads.append(1)
        return ("model", "processor", "cfg", 224)

    monkeypatch.setattr(instance, "_load_policy", _fake_load)

    first = instance.load_policy_once()
    second = instance.load_policy_once()

    assert first is second
    assert instance._policy is first
    assert len(loads) == 1, "the policy must be loaded exactly once per backend"
