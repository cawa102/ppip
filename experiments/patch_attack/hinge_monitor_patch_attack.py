"""Confined "monitor-video" hijack, **hinge-objective** entry point (Experiment 1, dynamic).

Same per-frame mechanism as ``ce_monitor_patch_attack`` -- the patch is re-optimised at every
step, which is the **dynamic** regime this project targets -- with the saturating margin loss
as the default instead of all-7 cross-entropy.

Why a separate module rather than an ``objective='hinge'`` kwarg at the call site:
  A run's objective is the difference between a result that is comparable to
  ``runs/monitor-corner/`` and one that is not. A kwarg default is easy to forget, easy to
  shadow in a sweep script, and invisible in a shell history six weeks later; a module name is
  none of those. So the two objectives get two entry points, and this one **refuses** a
  non-saturating objective (see ``SATURATING``) -- the filename is a guarantee, not a hint.

  Nothing is duplicated: ``run_confined_episode`` below delegates to the shared core in
  ``ce_monitor_patch_attack``, so the two paths can never drift mechanically, only in
  objective. Every fix to the episode loop lands in one place and applies to both.

Why the hinge is the right default *here* (dynamic-regime reasons only):
  * **Only some dims are contested.** The user- and target-instructed policies already agree
    on ~3 of 7 dims (measured 3.96/7), so all-7 CE spends ~43% of every gradient step on
    settled dims -- and *anchors* them against the same perturbation budget the decisive dims
    need. ``hinge`` is restricted to the decisive dims.
  * **This is a threshold measurement.** The stealth ladder asks for the *minimum* epsilon that
    still forces the policy. A non-saturating loss keeps buying logit margin on a dim that has
    already won, so the epsilon it reports is an **upper bound on the threshold, not the
    threshold**. Saturation is a measurement-validity requirement here, not an efficiency
    preference. (This is the argument that survives the static->dynamic move; the multi-frame
    "one artifact must serve many frames" capacity argument does **not** -- each solve in this
    regime sees exactly one frame.)
  * **The 256 action bins are ordered** (median user/target disagreement ~23 bins, ~9% of
    range), which ``directional`` exploits and CE cannot see.

``kappa`` defaults to ``forcing_loss.DEFAULT_KAPPA``, which is **6.0** as of 2026-08-04. The
previous 3.0 silently capped every hinge run this project had done at 2-of-3 decisive dims;
see ``forcing_loss.DEFAULT_KAPPA`` for the sweep. Do not hardcode a literal here -- that is
exactly how the cap went unnoticed.

Scoring is untouched: every verdict still comes from the fixed ``eval_goal_state`` predicate
via the shared core. This module only chooses what gradient the patch receives.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Final

HOME = os.path.expanduser("~")
for p in ("autoresearch/src", "autoresearch", "openvla", "autoresearch/experiments/patch_attack"):
    sys.path.insert(0, os.path.join(HOME, p))

import forcing_loss as FL  # noqa: E402
from ce_monitor_patch_attack import run_confined_episode as _run_confined_episode  # noqa: E402
from hijack_backend import HijackBackend  # noqa: E402

#: Objectives this entry point accepts -- the margin family. The cross-entropy objectives are
#: rejected rather than silently honoured, so a run launched from this module is always
#: attributable to its loss family from its filename alone.
#:
#: ``ce_saturating`` is excluded too, and NOT for want of saturation -- it saturates exactly as
#: the hinge does. It is excluded because it is cross-entropy, and the filename guarantee is
#: about family, not about which properties a loss happens to have.
SATURATING: Final[tuple[str, ...]] = ("hinge", "directional")


def run_confined_episode(
    backend: Any,
    *,
    objective: str = "hinge",
    kappa: float = FL.DEFAULT_KAPPA,
    **kwargs: Any,
) -> dict[str, Any]:
    """``ce_monitor_patch_attack.run_confined_episode`` with a saturating default objective.

    Every other keyword is forwarded untouched -- see the core's docstring for the full set.
    Raises ``ValueError`` for a non-saturating ``objective``; use ``ce_monitor_patch_attack``
    for those, so the module a run came from identifies its loss family.
    """
    if objective not in SATURATING:
        raise ValueError(
            f"{objective!r} is not in the margin family; this entry point accepts {SATURATING}. "
            "Run a cross-entropy episode -- including the saturating `ce_saturating` -- from "
            "`ce_monitor_patch_attack` instead."
        )
    return _run_confined_episode(backend, objective=objective, kappa=kappa, **kwargs)


def main() -> None:
    """Env-var driven single episode, mirroring the CE main with a distinct default run dir.

    The default ``MP_RUN_DIR``/tag differ from the CE entry point on purpose: an identical tag
    in an identical directory would let a hinge episode silently overwrite a published CE
    result file.
    """
    run_dir = os.environ.get(
        "MP_RUN_DIR", os.path.join(HOME, "autoresearch/runs/monitor-patch-hinge")
    )
    seed = int(os.environ.get("MP_SEED", "0"))
    rect = (
        int(os.environ.get("MP_R0", "150")), int(os.environ.get("MP_C0", "150")),
        int(os.environ.get("MP_PH", "60")), int(os.environ.get("MP_PW", "60")),
    )
    objective = os.environ.get("MP_OBJECTIVE", "hinge")
    tag = os.environ.get(
        "MP_TAG", f"{objective}_seed{seed}_r{rect[0]}c{rect[1]}_{rect[2]}x{rect[3]}"
    )
    backend = HijackBackend(run_dir=run_dir, max_steps=int(os.environ.get("MP_MAX_STEPS", "200")))
    run_confined_episode(
        backend,
        rect=rect,
        seed=seed,
        max_steps=int(os.environ.get("MP_MAX_STEPS", "200")),
        chunk=int(os.environ.get("MP_CHUNK", "250")),
        k=int(os.environ.get("MP_K", "10")),
        lr=float(os.environ.get("MP_LR", "3e-2")),
        maxtries=int(os.environ.get("MP_MAXTRIES", "6")),
        trial=os.environ.get("MP_TRIAL", "0"),
        run_dir=run_dir,
        tag=tag,
        record_dir=os.environ.get("MP_RECORD_DIR", ""),
        objective=objective,
        kappa=float(os.environ.get("MP_KAPPA", str(FL.DEFAULT_KAPPA))),
    )


if __name__ == "__main__":
    main()
