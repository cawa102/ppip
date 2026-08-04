# Plan (B): Universal (static) EoT patch — from per-frame digital artifact to one fixed patch

**Date:** 2026-07-22 · **Branch:** `monitor-hijack/phase0` · **Scope:** search-side only
**Status:** **B1 RUN — negative, mechanism identified (2026-07-28).** See
`runs/monitor-stealth/RESULT.md`. H0/H1 answered; **H2 (stealth) never reached** because H1 fails.
DAgger (steps 2-4 of the Method below) is now the necessary next step, not optional.
Sibling of `2026-07-22-stealth-corner-hijack.md` (C)
and `2026-07-22-outcome-teacher-ablation.md` (A). Time is not a constraint (researcher
decision 2026-07-22); GPU is physically capable (below).

## Goal

Every hijack so far re-optimizes the patch **every frame** with gradients — the cross-task
study measured that the recorded pixels are **inert** (a function of frame+prompt). That is a
*digital per-episode artifact*, not a physical object. (B) asks: can ONE **fixed** patch
(no per-frame re-optimization) hijack across a distribution of episode inits? A static patch is
the step from a per-frame digital adversarial example toward a *static* patch (still camera-space — not physically realizable; see Revisions).

## Universality sub-axes — scope clarification (2026-07-24)

The researcher's question is: **does ONE optimised artifact transfer, without re-optimisation, to a
setting it was not optimised on?** Two artifact notions × three settings — keep them distinct so a
result is not mis-billed:

- **Artifact.** (i) *one optimised video* = the recorded per-frame patch sequence replayed
  **verbatim** (what `corner_crosstask_probe.py` does); only literally replayable when the trajectory
  aligns frame-for-frame. (ii) *one fixed patch* = a single **frame-independent** image (H1/H2 below);
  the only "one artifact" that is even *defined* across diverging trajectories.
- **Setting.** (a) **cross-init** — same task, different object *positions* / episode seed.
  (b) **cross-object-setting / cross-scene** — different layout or objects present.
  (c) **cross-user-task** — same scene/pixels, only the commanded language changes.

**What we already have (narrow slice, `corner_crosstask_*`): cross-user-task only**, scene held fixed.
Recorded pixels replayed verbatim **✗ do not transfer** (`..._probe`); a **re-optimised** corner
**✓ retains leverage** (`..._gate`). Caveat: cross-*user-task* is instruction-independent by
construction (7/7 forcing ignores the language), so it is the *least* meaningful slice — it shows the
mechanism is not language-gated, **not** that any artifact is universal.

**The deeper observation still needed (this is the actual universality test):** whether one artifact
transfers across **(a) cross-init** and **(b) cross-object-setting/scene** — the slices that are *not*
instruction-independent-by-construction. H1/H2 below cover exactly the **one-fixed-patch × cross-init**
cell; **one-fixed-patch × cross-object-setting/scene is NEW scope** beyond the current plan, bounded by
adjudicability (the target object must exist in each scene — `salad_dressing` is in only 6/10
`libero_object` scenes), so any cross-scene set must be pre-screened for object presence **and** the
base-policy ceiling, exactly like the multi-pair target set in the program spine. "One optimised video"
(artifact i) is only meaningful within trajectory-aligned settings; across diverging inits/scenes the
testable artifact is the fixed patch (ii).

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

**Merged with C (researcher decision 2026-07-28).** Stealth requires a *static* artifact — a
per-frame re-optimized logo flickers, which kills the perception claim — so Exp C is executed
as B2, and B1 is its ε=1 endpoint under the shared parameterization
`patch = clamp(base + ε·tanh(raw))`. One ε ladder therefore runs B1 (ε=1, capacity ceiling),
the C curve (0.02–0.32), and the pure-logo control (ε=0). See
`2026-07-22-stealth-corner-hijack.md` § "Revisions — researcher decisions 2026-07-28".
The train/held-out split B's F5 precommit requires is now `experiments/patch_attack/shared_inits.py`
(later subdivided three ways: 5 optimize / 3 gate / 12 held-out, init 0 excluded as contaminated).

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


## Results — B1 run 2026-07-28 (full write-up: `runs/monitor-stealth/RESULT.md`)

**H1 is FALSE as specified** (single patch optimised on the *clean* frame buffer), and the reason is
not the one the plan's R1 anticipated. R1 guessed a **capacity** limit ("capacity over a distribution
is far less than per-frame"). Capacity is not the binding constraint:

| region | area | N=1 | N=4 | N=16 (⅓ ep) | N=37 (1 ep) | N=64 (2 ep) |
|---|---|---|---|---|---|---|
| BL 64×64 | 8.2% | 1.000 | 0.800 | 0.230 | — | 0.168 |
| TL 80×80 | 12.8% | — | — | 0.639 | — | — |
| band | 26.3% | — | — | 1.000 | 0.444 | 0.284 |
| two-band (max valid area) | **37.9%** | — | — | — | **0.910** | — |

At 37.9% — the largest region clear of both objects and gripper — a static patch forces a **full
episode at 0.910 / 81% fully-forced**, above the ~78-91% the per-frame corner hijacks ran at. It
still produces **zero** behavioural change in closed loop (`commanded` stays True). The 26.3% patch,
with *worse* forcing (0.444), produces **denial**. The better-forcing patch does less.

**Mechanism: induced-distribution dependence.** The patch reproduces `OpenVLA(clean_frame, target)`
only on frames from the clean trajectory, which the robot leaves as soon as it acts. On those frames
the two instructions differ on 3.96/7 dims (median 23/256 bins), so the teacher is genuinely distinct
— but following it does not lead to the target once you are off the clean path. The per-frame attack
works because it re-optimises on the trajectory it induces.

**Therefore the plan's ordering changes.** "B1 = universal free-range, then B2 = compose with C" is
superseded: B1 as written (clean-buffer EoT) is answered and negative, and B2/stealth is unreachable
through it. The live question is the plan's own **DAgger** loop (Method §"Closed-loop via DAgger"),
which is now the *only* route to H1/H2. If DAgger also fails, "a static camera-space patch cannot
hijack this policy at any non-occluding area" becomes publishable with the mechanism explained.

**Also settled here:** the F5 precommit (`shared_inits.py`, 5 optimize / 3 gate / 12 held-out) and
the F9 no-grad eval guard (`eval_static_patch.frozen_evaluation`, which blocks constructing *and*
stepping every concrete optimizer class) both exist and are tested.
