"""WP7 — the word-gate kwargs on ``run_confined_episode`` (two-branch optimiser).

The surgical two-branch change lives deep inside the GPU optimise loop, so it is split like the
other patch-track seams: the CPU-testable *decision* logic is the pure ``resolve_gate_setup``
(tested in ``test_word_gate.py``), and the live two-branch rollout is a ``@requires_gpu`` smoke
seam here. One guard — ``gate_word`` requires ``patch_mode='optimize'`` — is validated before any
GPU work, so it is exercised on CPU with a dummy backend. See
``docs/plans/2026-07-30-word-gated-patch.md`` (WP7).
"""
from __future__ import annotations

import os
from typing import Any

import pytest

requires_gpu = pytest.mark.skipif(
    not os.environ.get("PPIP_GPU_TESTS"),
    reason="set PPIP_GPU_TESTS=1 in the GPU rollout env (GPU 1) to run the OpenVLA seam",
)

USER_TASK = "pick up the alphabet soup and place it in the basket"
TARGET_TASK = "pick up the salad dressing and place it in the basket"
PROBE_RECT = (160, 0, 64, 64)  # the measured-clear BL 64x64 corner


def _trace_rows(run_dir: object) -> list[dict[str, Any]]:
    """The per-step trace ``run_confined_episode`` writes beside its result JSON."""
    import glob
    import json

    paths = glob.glob(os.path.join(str(run_dir), "trace_*.json"))
    assert len(paths) == 1, f"expected exactly one trace file, got {paths}"
    with open(paths[0], encoding="utf-8") as handle:
        rows: list[dict[str, Any]] = json.load(handle)
    return rows


def test_gate_word_requires_optimize_patch_mode_fails_before_gpu() -> None:
    """The mode guard resolves the gate then rejects a non-optimise mode before touching the GPU.

    A dummy backend proves the raise happens before any policy load / env build — the guard is the
    first thing ``run_confined_episode`` does.
    """
    from ce_monitor_patch_attack import run_confined_episode

    with pytest.raises(ValueError, match="optimis"):
        run_confined_episode(
            object(),  # never used: the guard raises first
            rect=PROBE_RECT,
            seed=0,
            run_dir="/tmp/wp7-unused",
            tag="wg_guard",
            patch_mode="blank",
            gate_word="please",
        )


def test_contaminated_trigger_rejected_before_gpu() -> None:
    """A trigger already in the instruction has no word-absent baseline — reject before GPU work."""
    from ce_monitor_patch_attack import run_confined_episode

    with pytest.raises(ValueError):
        run_confined_episode(
            object(),
            rect=PROBE_RECT,
            seed=0,
            run_dir="/tmp/wp7-unused",
            tag="wg_contam",
            user_task=USER_TASK,
            gate_word="soup",  # "alphabet soup" already present
        )


# --- GPU seam (skipped without PPIP_GPU_TESTS; runs on GPU 1) --------------------------


@requires_gpu
def test_armed_word_gated_episode_runs_two_branch_optimiser_end_to_end(tmp_path: object) -> None:
    """Smoke: an armed word-gated episode runs the two-branch loop and reaches a fixed verdict.

    Two steps only — this asserts the wiring (deploy under c⊕w, target/clean teachers, the
    ``two_branch_loss`` path) executes without raising and self-describes the gate; it is NOT a
    hijack claim (the closed-loop existence run is the GPU experiment, not this test).
    """
    from hijack_backend import HijackBackend
    from ce_monitor_patch_attack import run_confined_episode

    backend = HijackBackend(run_dir=str(tmp_path), max_steps=2)
    result = run_confined_episode(
        backend,
        rect=PROBE_RECT,
        seed=0,
        max_steps=2,
        chunk=2,
        k=2,
        maxtries=1,
        trial="0",
        run_dir=str(tmp_path),
        tag="wg_armed_smoke",
        user_task=USER_TASK,
        target_task=TARGET_TASK,
        gate_word="please",
        word_index=0,
        dormancy_weight=1.0,
        deploy_word=True,
    )

    assert result["status"] in {"HIJACK", "DONE", "PAUSED"}
    assert result["word_gate"] is not None
    assert result["word_gate"]["word"] == "please"
    assert result["word_gate"]["deploy"] == "please " + USER_TASK  # armed deploy
    assert result["word_gate"]["dormant"] == USER_TASK
    # The verdict keys still come from the fixed predicates, unchanged by the gate.
    assert isinstance(result["targeted"], bool)
    assert isinstance(result["commanded_success"], bool)

    # Per-step same-frame gate record: both instructions evaluated on the SAME composite, which
    # the condition-blind selection already computes (so it costs no extra forward pass).
    rows = _trace_rows(tmp_path)
    gated = [r["gate"] for r in rows if r["gate"] is not None]
    assert gated, "a gated episode must record the per-step armed/dormant comparison"
    for g in gated:
        assert len(g["armed_tokens"]) == 7 and len(g["dormant_tokens"]) == 7
        assert 0 <= g["armed_match"] <= 7 and 0 <= g["dormant_match"] <= 7
        assert 0.0 <= g["armed_forced"] <= 1.0 and 0.0 <= g["dormant_forced"] <= 1.0
        assert g["branches_differ"] == (g["armed_tokens"] != g["dormant_tokens"])


@requires_gpu
def test_dormant_word_gated_episode_deploys_the_plain_instruction(tmp_path: object) -> None:
    """The dormant rollout runs the SAME two-branch optimiser but deploys the word-absent ``c``."""
    from hijack_backend import HijackBackend
    from ce_monitor_patch_attack import run_confined_episode

    backend = HijackBackend(run_dir=str(tmp_path), max_steps=2)
    result = run_confined_episode(
        backend,
        rect=PROBE_RECT,
        seed=0,
        max_steps=2,
        chunk=2,
        k=2,
        maxtries=1,
        trial="0",
        run_dir=str(tmp_path),
        tag="wg_dormant_smoke",
        user_task=USER_TASK,
        target_task=TARGET_TASK,
        gate_word="please",
        deploy_word=False,
    )

    assert result["word_gate"]["deploy_word"] is False
    assert result["word_gate"]["deploy"] == USER_TASK  # deployed the plain instruction
    # The dormant rollout records the same two-branch comparison as the armed one: the optimiser
    # and the selection rule are identical in both conditions, only the deployment differs.
    assert [r["gate"] for r in _trace_rows(tmp_path) if r["gate"] is not None]


@requires_gpu
def test_ungated_default_records_no_word_gate(tmp_path: object) -> None:
    """The default (no ``gate_word``) leaves ``word_gate`` null — the behavior-preserving path."""
    from hijack_backend import HijackBackend
    from ce_monitor_patch_attack import run_confined_episode

    backend = HijackBackend(run_dir=str(tmp_path), max_steps=2)
    result = run_confined_episode(
        backend,
        rect=PROBE_RECT,
        seed=0,
        max_steps=2,
        chunk=2,
        k=2,
        maxtries=1,
        trial="0",
        run_dir=str(tmp_path),
        tag="wg_ungated_smoke",
        user_task=USER_TASK,
        target_task=TARGET_TASK,
    )

    assert result["word_gate"] is None
    assert result["gate_diagnostic"] is None
    # The trace schema is unchanged for every pre-existing run: the gate slot is simply null.
    assert all(row["gate"] is None for row in _trace_rows(tmp_path))
