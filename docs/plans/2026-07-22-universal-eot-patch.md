# Plan (B): Universal (static) EoT patch — from per-frame digital artifact to one fixed patch

**Date:** 2026-07-22 · **Branch:** `monitor-hijack/phase0` · **Scope:** search-side only
**Status:** PLANNED (awaiting go-ahead). Sibling of `2026-07-22-stealth-corner-hijack.md` (C)
and `2026-07-22-outcome-teacher-ablation.md` (A). Time is not a constraint (researcher
decision 2026-07-22); GPU is physically capable (below).

## Goal

Every hijack so far re-optimizes the patch **every frame** with gradients — the cross-task
study measured that the recorded pixels are **inert** (a function of frame+prompt). That is a
*digital per-episode artifact*, not a physical object. (B) asks: can ONE **fixed** patch
(no per-frame re-optimization) hijack across a distribution of episode inits? A static patch is
the step from a per-frame digital adversarial example toward a *static* patch (still camera-space — not physically realizable; see Revisions).

## Novelty positioning (read with the 2026-07-22 scan)

Universality alone is **partially shadowed**: "When Robots Obey the Patch" (2511.21192) already
does universal *transferable* VLA patches (but **untargeted**), and TRAP (2603.23117) uses a
**printed physical** patch — necessarily static — that is **targeted** (on CoT VLAs). So a
universal-targeted patch is not, by itself, clean daylight. The unambiguously-unclaimed cell is
**(B)∧(C): a universal AND stealthy targeted-object patch on a base (non-CoT) VLA.** Therefore
(B) is run in two steps and the *headline* is the stealthy-universal one.

Honest limit: a universal patch in **camera-space** removes per-frame re-optimization but does
NOT cross the render reality-gap (the Exp-2 wall). TRAP (physical) crossed that gap; we have
not. So our claim is "one fixed *camera-space* patch," a step toward — not all the way to — a
printable physical patch. State this plainly.

## Hypotheses

- **H0 (baseline, cheap):** the seed-0 per-frame patch, frozen and replayed across other inits
  of the same task, is **inert** (≈ base-policy behavior) — measures non-universality.
- **H1:** an EoT-optimized single patch hijacks on **held-out** inits (targeted-success-rate
  over inits meaningfully above the frozen-patch and no-patch baselines).
- **H2 (headline):** the universal patch can be made **stealthy** (bounded around the logo, per
  C) and still hijack across inits — the (B)∧(C) cell.

## Method

### One patch, shared across frames/inits

Single parameter `raw` (free-range `sigmoid(raw)` for B1; `logo + eps*tanh(raw)` for B2/stealth)
optimized to force the teacher (policy or, per A, goal-derived reference) over a **distribution**
of frames, not one frame. EoT already partially present (crop-side jitter); extend the
distribution over inits (+ optional placement jitter).

### Closed-loop via DAgger (the correct loop for a moving frame distribution)

A static patch must work under its *own* induced trajectory distribution, not the clean one:
1. optimize the single patch on the current frame buffer (EoT over the buffer);
2. roll out with the **fixed** patch on each of N inits;
3. add the visited frames to the buffer;
4. re-optimize; repeat until held-out targeted-rate converges or budget.
Each DAgger round = N rollouts (this is the long pole — see feasibility).

### Distribution (start small, expand)

- **B1:** N inits of the **same** task (alphabet_soup → salad_dressing), e.g. N=4–8; measure
  targeted-rate on held-out inits.
- **Expansion (later):** placement/camera jitter; cross-task universality (hardest — the
  cross-task study shows even re-optimized transfer is instruction-independent by construction,
  so a single cross-task patch is a separate, harder claim).

### Ordering (one variable at a time)

- **B1 = universal free-range** (isolate the universality variable at max available capacity).
- **B2 = universal + stealth** (compose with C only after B1 works) → the novel headline.

## Feasibility (grounded, 2026-07-22)

- **Memory: fits.** GPU 1 has ~23 GB free; a single patch optimized over sequentially-
  accumulated frames has the same footprint as the current single-frame optimization. Not a
  blocker.
- **Thermal: speed only, not a crash.** GPU 0's reserved task (84 °C/100 %) can heat GPU 1 under
  sustained load (→ up to 2.5–15 min/step worst case). DAgger's many rollouts make wall-clock the
  cost (potentially days). Acceptable per the time decision. Mitigation: aggressive checkpointing
  (the loop already pickles env state → thermal stalls are resumable), temperature monitoring,
  and a small N first.
- **Verdict: physically feasible.** The risk is duration, not impossibility.

## Metrics / deliverables

- **Held-out-init targeted-success-rate** for: no-patch, frozen-seed-0-patch (H0), EoT-universal
  (H1), EoT-universal-stealth (H2). This 4-row table is the result.
- Per-init `latch_step` / `min_target_dist_m` distribution (variance, not a point estimate —
  also fixes the single-seed rigor gap that dogs the corner results).
- The single fixed patch image (+ stealth version) and its L∞ / visual for H2.

## Integrity / scope

Simulation-only, test-time, weights frozen, white-box. New search-side driver + additive knobs
only (a universal/EoT mode reusing `run_confined_episode`'s optimization + adjudication path).
Zero changes to `src/evaluator/`, `src/rendering/`, `experiments/configs/`, budgets, tasks/seeds.

## Risks

- **R1 (main):** EoT may not converge to a single patch that generalizes across inits at a
  confined corner — capacity over a distribution is far less than per-frame. Mitigation: start
  N small; allow larger area for B before shrinking; if it plateaus, document the
  per-frame-vs-universal capacity gap as a boundary (consistent with our render-gap boundary).
- **R2:** wall-clock (days) under thermal throttling. Mitigation: checkpoint + resume; run B1 at
  small N as a go/no-go before scaling N.
- **R3:** cross-task universality likely unreachable (instruction-independent-by-construction
  mechanism) — keep it out of the headline; same-task-init universality is the target.

## Relation to (C) and (A)

- **(C):** B2 is literally (B)∧(C); reuses the stealth parameterization.
- **(A):** the universal patch can target the goal-derived reference teacher, making universality
  goal-directed and objection-proof in one shot.

## Revisions — Codex review round 1 (2026-07-22)

Now the **universality axis** of `2026-07-22-controllability-program.md`; standing methodology binds it.

- **No-grad eval + leakage guard (F9):** the held-out evaluation of a fixed patch must run through
  a **separate no-grad path** with an **assertion that no optimizer executes** during evaluation —
  `run_confined_episode`'s per-frame optimizer must never touch test frames (that would re-optimize
  on the eval set = leakage/cheating). Training frames (DAgger) and held-out eval frames are
  disjoint by construction.
- **Precommit the split (F5):** declare the **train inits** and the **held-out inits** (from the
  shared index set) up front; report targeted-rate on held-out inits with CIs for: no-patch,
  color-matched blank, uniform random, frozen-seed-0-patch (H0), EoT-universal (H1),
  EoT-universal-stealth (H2 = B∧C).
- **Scoring (F10):** held-out numbers via the fixed-backend wrapper (latch-not-terminate).
- **Scope (F3/F4):** camera-space upper bound. A fixed camera-space patch removes per-frame
  re-optimization but does **not** cross the render gap — "toward a thing you could print" is
  **aspiration, not evidence**. No physical/universality-in-the-world claim.
- **Novelty (F3):** headline the narrow honest cell (base non-CoT, targeted, action-token,
  camera-space); concede "When Robots Obey the Patch" owns universal-untargeted and TRAP owns
  physical-targeted.
