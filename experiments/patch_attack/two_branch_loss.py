"""The word-gated patch's two-branch training objective (WP3).

One differentiable loss over the patch parameters ``ε``, combining the two conditions a
placed patch must satisfy simultaneously (it is physically present whether or not the
operator utters the magic word ``w``):

    L(ε) = CE(f(P_M(o,ε), c⊕w), aᵀ)   +   λ · CE(f(P_M(o,ε), c), aᵁ)
           └── targeting: force target aᵀ ──┘   └── dormancy: reproduce clean aᵁ ──┘

``λ`` weights the dormancy (word-absent) branch — the dormancy↔potency frontier the
``λ`` sweep (E2.2d) explores. Both cross-entropies mirror the existing per-frame forcing
loss (``monitor_patch_attack.py:244``):
``F.cross_entropy(logits.reshape(7,-1), teacher.reshape(7))``.

This is the *method* objective (agent-editable). It never produces a reported number —
every ``targeted``/``commanded`` verdict comes from the fixed evaluator running the frozen
patch. See ``docs/plans/2026-07-30-word-gated-patch.md`` (Formalization, WP3).
"""
from __future__ import annotations

from typing import Final

import torch
import torch.nn.functional as F

#: OpenVLA emits exactly this many discretized action tokens per step.
ACTION_DIM: Final[int] = 7


def _validate_action_shapes(logits: torch.Tensor, teacher: torch.Tensor, branch: str) -> None:
    """Fail fast (``ValueError``) if a branch's tensors are not 7 action tokens wide.

    Cheaper and clearer than letting ``reshape`` raise a raw ``RuntimeError`` deep in the
    loss — a malformed teacher/logit pair is a caller bug, caught at the boundary.
    """
    n_tokens = int(teacher.numel())
    if n_tokens != ACTION_DIM:
        raise ValueError(
            f"{branch} teacher must have {ACTION_DIM} action tokens, got {n_tokens}"
        )
    n_values = int(logits.numel())
    if n_values % ACTION_DIM != 0:
        raise ValueError(
            f"{branch} logits ({n_values} values) do not reshape into "
            f"{ACTION_DIM} action-token rows"
        )


def two_branch_loss(
    armed_logits: torch.Tensor,
    target_teacher: torch.Tensor,
    dormant_logits: torch.Tensor,
    clean_teacher: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Return the scalar two-branch loss (targeting + ``λ``·dormancy).

    Logits are flattened to ``(ACTION_DIM, vocab)`` and teachers to ``(ACTION_DIM,)`` so
    the caller may pass either flat or batched (``[1, 7, V]`` / ``[1, 7]``) tensors.

    Raises ``ValueError`` if either branch is not shaped as 7 action tokens.
    """
    _validate_action_shapes(armed_logits, target_teacher, "armed")
    _validate_action_shapes(dormant_logits, clean_teacher, "dormant")
    targeting = F.cross_entropy(
        armed_logits.reshape(ACTION_DIM, -1), target_teacher.reshape(ACTION_DIM)
    )
    dormancy = F.cross_entropy(
        dormant_logits.reshape(ACTION_DIM, -1), clean_teacher.reshape(ACTION_DIM)
    )
    return targeting + lam * dormancy
