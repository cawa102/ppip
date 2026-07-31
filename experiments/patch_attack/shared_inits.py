"""Precommitted LIBERO init-state indices shared by every controllability experiment.

This is the *precommit* required by the program's standing methodology rule 2
(`docs/plans/2026-07-22-controllability-program.md`): a single fixed set of init indices,
**identical across every condition and every experiment (C / A / B)**, declared before any
optimisation runs so no result can be produced by choosing favourable episodes after the fact.

Three properties make it auditable:

* **Literal, not computed at import.** The indices below are hard-coded. `verify_precommit()`
  re-derives them from the recorded rule and asserts equality, so a future change to numpy's
  RNG cannot silently move the goalposts; the literals are what the paper reports.
* **Seed 0 is excluded from both splits.** Every prior corner result (64/48/40/32, the effort
  ladder, the escalation schedule) was tuned on init 0, so it is contaminated by selection.
  It stays available as `LEGACY_GATE_INIT` -- a cheap go/no-go probe, **never** a claim.
* **Train / held-out are disjoint by construction.** Exp B optimises the static patch on the
  frames of `TRAIN_INITS` only; every reported targeted/commanded rate comes from
  `HELDOUT_INITS`, which the optimiser never sees (Codex F9 leakage guard).

Derivation rule (reproducible):

    rng = numpy.random.default_rng(20260722)
    perm = rng.permutation(numpy.arange(1, 50))   # 1..49; 0 excluded as contaminated
    TRAIN_INITS   = sorted(perm[:8])
    HELDOUT_INITS = sorted(perm[8:20])

`init_selector` semantics match the fixed evaluator: LIBERO indexes its own 50 saved init
states per task, and both the evaluator (`openvla_backend.py`) and the search-side rollouts
select with `init_states[i % len(init_states)]`, so an index here means the same episode
everywhere.
"""

from __future__ import annotations

from typing import Final

import numpy as np

#: Number of init states LIBERO ships per `libero_object` task (verified: shape (50, 110)).
N_LIBERO_INIT_STATES: Final[int] = 50

#: Seed of the recorded derivation rule; changing it invalidates the precommit.
PRECOMMIT_SEED: Final[int] = 20260722

#: Init 0 -- used by every pre-2026-07-28 corner result, therefore selection-contaminated.
#: Legitimate as a cheap existence gate; must never appear in a reported rate.
LEGACY_GATE_INIT: Final[int] = 0

#: Inits whose frames may enter an optimiser (EoT training buffer, DAgger rounds).
TRAIN_INITS: Final[tuple[int, ...]] = (1, 13, 14, 18, 20, 34, 41, 44)

#: Inits reserved for evaluation. No optimiser may ever see these frames.
HELDOUT_INITS: Final[tuple[int, ...]] = (4, 7, 22, 24, 26, 33, 36, 38, 39, 45, 46, 49)

#: `TRAIN_INITS` split again, because the search side needs its *own* generalisation signal.
#: The cheap candidate-ranking gate must not score a patch on the frames it was fitted to
#: (that measures fit, not transfer) and must not touch `HELDOUT_INITS` (that leaks the
#: evaluation set into candidate selection). So the optimiser sees `OPTIMIZE_INITS` and the
#: gate scores on `GATE_INITS` -- disjoint, both inside the training half.
OPTIMIZE_INITS: Final[tuple[int, ...]] = (1, 13, 14, 18, 20)
GATE_INITS: Final[tuple[int, ...]] = (34, 41, 44)

#: Every init this program touches, in a stable order.
SHARED_INITS: Final[tuple[int, ...]] = TRAIN_INITS + HELDOUT_INITS


def verify_precommit() -> None:
    """Re-derive the splits from the recorded rule and assert the literals still match.

    Raises `AssertionError` if the constants above have drifted from their derivation, if the
    splits overlap, if the contaminated legacy init leaked in, or if an index is out of range.
    """
    rng = np.random.default_rng(PRECOMMIT_SEED)
    perm = rng.permutation(np.arange(1, N_LIBERO_INIT_STATES))
    expected_train = tuple(sorted(int(i) for i in perm[: len(TRAIN_INITS)]))
    expected_held = tuple(
        sorted(int(i) for i in perm[len(TRAIN_INITS) : len(TRAIN_INITS) + len(HELDOUT_INITS)])
    )

    assert expected_train == TRAIN_INITS, (
        f"TRAIN_INITS drifted: {TRAIN_INITS} != {expected_train}"
    )
    assert expected_held == HELDOUT_INITS, (
        f"HELDOUT_INITS drifted: {HELDOUT_INITS} != {expected_held}"
    )
    assert not set(TRAIN_INITS) & set(HELDOUT_INITS), "train/held-out overlap => leakage"
    assert not set(OPTIMIZE_INITS) & set(GATE_INITS), "optimize/gate overlap => gate measures fit"
    assert set(OPTIMIZE_INITS) | set(GATE_INITS) == set(TRAIN_INITS), (
        "optimize+gate must partition TRAIN_INITS exactly"
    )
    assert not set(GATE_INITS) & set(HELDOUT_INITS), "the ranking gate must not touch held-out"
    assert LEGACY_GATE_INIT not in SHARED_INITS, "contaminated legacy init leaked into the splits"
    assert all(0 <= i < N_LIBERO_INIT_STATES for i in SHARED_INITS), "init index out of range"
    assert len(HELDOUT_INITS) >= 10, "standing methodology requires N >= 10 held-out inits"


def summary() -> str:
    """One-line human-readable description of the precommit, for run logs and result files."""
    return (
        f"shared_inits(seed={PRECOMMIT_SEED}): "
        f"train={list(TRAIN_INITS)} heldout={list(HELDOUT_INITS)} "
        f"legacy_gate={LEGACY_GATE_INIT} (excluded from both)"
    )


if __name__ == "__main__":
    verify_precommit()
    print(summary())
