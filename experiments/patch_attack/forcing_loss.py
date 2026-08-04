"""What one static patch is asked to achieve — the attack's internal optimization objective.

This is the agent-editable *method* side of the invariant: nothing here judges a rollout, and
none of it can reach the evaluator's `attack_score`. It only decides what gradient the patch
receives. Every verdict still comes from the fixed evaluator.

The objective it replaces was `F.cross_entropy` over all 7 action dims of a single frame.
Three properties of this problem make that the wrong shape:

**Only some dims are contested.** The user- and target-instructed policies agree on ~3 of 7
dims on clean frames (measured: they differ on 3.96/7). Cross-entropy over all 7 spends ~43%
of every gradient step on dims that are already correct, and — worse — *anchors* them, so
they compete for the same epsilon budget as the dims that decide the action.

**One artifact must serve many frames.** The measured capacity curve (forcing 1.000 at one
frame, 0.230 at sixteen, 0.168 at sixty-four for a fixed rect) says the binding constraint is
capacity, not search. Cross-entropy never saturates: it keeps widening a logit gap that
already decides the argmax, on a frame already won, using budget a still-unforced frame
needs. `margin_hinge` goes to zero at a margin of `kappa` and stays there, which turns the
multi-frame fit from a sum of log-likelihoods into **constraint satisfaction over frames**.

**The 256 bins are ordered.** The action head is categorical, but its classes are uniform bins
over a continuous range, and the user/target disagreement has a median of ~23 bins (~9% of
range). Cross-entropy charges the same for a one-bin miss as for a hundred-bin miss, so it
buys exact matching where approximate would do. `directional_hinge` scores the *signed
progress* from the user's action toward the target's: it saturates on arrival, does not punish
overshoot in the right direction, and does punish travelling the wrong way. That asymmetry is
the redirection threat model written down as a loss — squared error would penalise a useful
overshoot exactly as hard as a harmful reversal.

Everything here is pure torch on logits: no model, no simulator, CPU-testable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

import torch
import torch.nn.functional as F

#: OpenVLA's action detokenization: `bin = VOCAB_ANCHOR - token_id`, `bin` in [1, N_BINS],
#: and bin index is monotone-increasing in the underlying continuous action value (the bin
#: centres are a uniform linspace). Working in bin index therefore needs no model handle.
VOCAB_ANCHOR: Final[int] = 32000
N_BINS: Final[int] = 256
ACTION_ID_MIN: Final[int] = VOCAB_ANCHOR - N_BINS  # 31744 -> bin 256, the largest action
ACTION_ID_MAX: Final[int] = VOCAB_ANCHOR - 1  # 31999 -> bin 1, the smallest action

ACTION_DIM: Final[int] = 7

#: Logit margin at which a dim counts as won and stops receiving gradient.
#:
#: **6.0, not 3.0** (changed 2026-08-04). A dim that clears `kappa` contributes exactly zero
#: gradient, so too small a margin releases it while it is still fragile: the optimiser wins two
#: dims, stops pushing them, and progress on the third knocks them back under margin — it cycles
#: instead of converging. Measured on aurora, 300 steps, one frame, free budget: forcing 0.667 at
#: kappa in {1, 3}, **1.000 at {6, 12}**, 0.667 again at {25, 50} (too deep a margin never lets
#: any dim go). kappa=3 silently capped every hinge run before this date. Sweep and discussion:
#: `docs/plans/2026-08-04-epsilon-threshold-design.md` section 3.
DEFAULT_KAPPA: Final[float] = 6.0

#: Softmax temperature for the soft action decode. Above 1 keeps gradient alive through a
#: peaked action head, at the cost of biasing the decoded bin toward the distribution's mean.
DEFAULT_TEMPERATURE: Final[float] = 1.0


def token_to_bin(token_ids: torch.Tensor) -> torch.Tensor:
    """Action token ids -> bin indices in [1, N_BINS], monotone-increasing in action value."""
    return VOCAB_ANCHOR - token_ids


def bin_to_token(bin_index: int) -> int:
    """Bin index in [1, N_BINS] -> the action token id that decodes to it."""
    if not 1 <= bin_index <= N_BINS:
        raise ValueError(f"bin must be in [1, {N_BINS}], got {bin_index}")
    return VOCAB_ANCHOR - bin_index


def _zero_like(logits: torch.Tensor) -> torch.Tensor:
    """A differentiable scalar zero — a detached `0.0` would break the batch backward."""
    return logits.sum() * 0.0


def _dim_index(dims: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(tuple(dims), dtype=torch.long, device=device)


def masked_cross_entropy(
    logits: torch.Tensor, teacher: torch.Tensor, dims: Sequence[int]
) -> torch.Tensor:
    """Cross-entropy restricted to `dims` — the previous objective, minus the free agreement.

    `logits` is `[ACTION_DIM, vocab]`, `teacher` is `[ACTION_DIM]` token ids. Passing all
    seven dims reproduces the old `F.cross_entropy(logits, teacher)` exactly.
    """
    if not dims:
        return _zero_like(logits)
    index = _dim_index(dims, logits.device)
    return F.cross_entropy(logits[index], teacher[index])


def margin_hinge(
    logits: torch.Tensor,
    teacher: torch.Tensor,
    dims: Sequence[int],
    kappa: float = DEFAULT_KAPPA,
) -> torch.Tensor:
    """`relu(best_non_teacher_logit - teacher_logit + kappa)`, averaged over `dims`.

    The Carlini-Wagner objective shape. Zero once the teacher's token wins by `kappa`, so a
    won frame stops consuming the patch's capacity. The max runs over the **whole vocab**,
    matching greedy decoding, so a zero loss certifies the dim really decodes to the teacher.
    """
    if not dims:
        return _zero_like(logits)
    index = _dim_index(dims, logits.device)
    selected, target = logits[index], teacher[index].unsqueeze(1)
    teacher_logit = selected.gather(1, target).squeeze(1)
    best_other = selected.scatter(1, target, float("-inf")).max(dim=1).values
    return F.relu(best_other - teacher_logit + kappa).mean()


def soft_bins(logits: torch.Tensor, temperature: float = DEFAULT_TEMPERATURE) -> torch.Tensor:
    """Differentiable expected action bin per dim, `[ACTION_DIM]`, in [1, N_BINS].

    The softmax is taken over the **action-token slice only**: averaging a non-action token
    into a bin index would be meaningless, and the head never emits one in practice.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    slice_logits = logits[:, ACTION_ID_MIN : ACTION_ID_MAX + 1] / temperature
    bins = N_BINS - torch.arange(
        N_BINS, device=logits.device, dtype=slice_logits.dtype
    )  # slice index j <-> bin N_BINS - j
    return (slice_logits.softmax(dim=1) * bins).sum(dim=1)


def directional_hinge(
    logits: torch.Tensor,
    teacher: torch.Tensor,
    clean_user: torch.Tensor,
    dims: Sequence[int],
    temperature: float = DEFAULT_TEMPERATURE,
) -> torch.Tensor:
    """`relu(1 - progress)` from the user's action toward the teacher's, averaged over `dims`.

    `progress = 1` means the decoded action has travelled the full user->teacher gap on that
    dim. Saturating at 1 makes overshoot free (a redirection that pushes past the target is
    still a redirection); going negative makes reversal expensive. Dims where the two
    instructions agree have no direction to travel and are dropped rather than divided by zero.
    """
    if not dims:
        return _zero_like(logits)
    index = _dim_index(dims, logits.device)
    action = soft_bins(logits, temperature)[index]
    target_bin = token_to_bin(teacher[index]).to(action.dtype)
    user_bin = token_to_bin(clean_user[index]).to(action.dtype)

    gap = target_bin - user_bin
    contested = gap != 0
    if not bool(contested.any()):
        return _zero_like(logits)
    # sign(0)=0 keeps an uncontested dim's progress finite; `contested` then drops it.
    progress = (action - user_bin) * torch.sign(gap) / gap.abs().clamp(min=1.0)
    return (F.relu(1.0 - progress) * contested).sum() / contested.sum()


def cvar(values: torch.Tensor, q: float) -> torch.Tensor:
    """Mean of the worst (largest) `q` share of `values` — at `q=1`, the plain mean.

    Optimizing the batch mean lets a few catastrophic frames survive, and those are exactly
    the frames that derail a closed-loop rollout: a patch measured at 0.910 mean forcing still
    had ~10% of decisive dims wrong and produced no behavioural change. At least one element
    is always kept, so a small `q` degrades to worst-case rather than to nothing.
    """
    if not 0.0 < q <= 1.0:
        raise ValueError(f"q must be in (0, 1], got {q}")
    k = max(1, math.ceil(q * values.numel()))
    return values.topk(k).values.mean()


#: The dispatch domain. `ce` is first because it is the behaviour-preserving default: it
#: reproduces the `F.cross_entropy` over all 7 dims that every published closed-loop result was
#: produced with, so an existing caller that names nothing keeps its exact gradient.
OBJECTIVES: Final[tuple[str, ...]] = ("ce", "ce_decisive", "hinge", "directional")


def action_loss(
    logits: torch.Tensor,
    teacher: torch.Tensor,
    clean_user: torch.Tensor,
    dims: Sequence[int],
    *,
    objective: str,
    kappa: float = DEFAULT_KAPPA,
    temperature: float = DEFAULT_TEMPERATURE,
    anchor: float = 0.0,
) -> torch.Tensor:
    """Per-frame scalar loss under `objective` — the one dispatch both optimisers share.

    `logits` is `[ACTION_DIM, vocab]`; `teacher` and `clean_user` are `[ACTION_DIM]` token ids;
    `dims` is the decisive set (the dims on which the two instructions actually disagree).

    Two callers use this: the static EoT optimiser and the per-frame closed-loop attack. Before
    it existed they chose objectives independently — the per-frame path was hard-coded to
    all-7 cross-entropy and recorded no objective at all — which made it impossible to say from
    an artifact which loss had produced it. Selecting and *naming* the objective in one place is
    what fixes that.

    `anchor` mixes a light decisive-dim cross-entropy into a saturating objective. Both
    saturating losses go flat once satisfied, which is their purpose, but `directional` reads
    the head through a softmax whose gradient thins on a peaked distribution, so a small CE term
    keeps a usable signal. Zero by default.
    """
    if objective == "ce":
        # Deliberately unmasked and unsaturating: this is the historical path, kept exact.
        return F.cross_entropy(logits, teacher)
    if objective == "ce_decisive":
        return masked_cross_entropy(logits, teacher, dims)
    if objective == "hinge":
        loss = margin_hinge(logits, teacher, dims, kappa)
    elif objective == "directional":
        loss = directional_hinge(logits, teacher, clean_user, dims, temperature)
    else:
        raise ValueError(f"unknown objective {objective!r}; choose from {OBJECTIVES}")

    if anchor:
        loss = loss + anchor * masked_cross_entropy(logits, teacher, dims)
    return loss
