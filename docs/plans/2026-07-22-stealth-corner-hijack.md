# Plan: Stealthy non-occluding corner hijack (camera-space, pre-render)

**Date:** 2026-07-22 · **Branch:** `monitor-hijack/phase0` · **Scope:** search-side only
**Status:** PLANNED (awaiting go-ahead before GPU spend)

## Goal (the novel hijack case)

Show a *targeted* VLA hijack driven by a patch that (a) **looks like ordinary,
innocuous screen content** — a brand-like solid logo — rather than adversarial noise,
and (b) sits in an **empty corner, measured (per-init) not covering the object**. Commanded
`alphabet_soup`, the robot places `salad_dressing` (`targeted=True`, fixed evaluator).

Prior corner work already established the *non-occluding* half (free-range patch, BL
64×64 = 8.2%). This adds the **stealth** half: how much perturbation budget does the
attack need if the patch must remain a recognizable logo? The headline artifact is the
**capacity↔stealth tradeoff** (targeted-rate vs ε over the shared inits) plus a picture a human would not flag.

This isolates the **stealth** constraint from the **render** constraint on purpose: we
stay in the camera-space regime we already own (BL corner, direct pixel replacement) and
do NOT touch the render reality-gap yet. If a stealthy patch cannot hijack even here,
there is no point fighting the render; if it can, we have the result *and* a substrate to
push toward realizability next.

## Prior-art / novelty positioning (scan 2026-07-22)

The 4-way conjunction (stealthy + non-occluding + targeted-object + base non-CoT
OpenVLA/LIBERO, test-time white-box) is **unclaimed**, but each piece is individually
published — so the novelty is the **conjunction**, foregrounding **stealth/naturalism**
and the **non-CoT base model**. Headline must NOT be "first non-occluding targeted VLA
patch" (TRAP owns that).

- **TRAP (2603.23117, Mar 2026) — the threat.** Owns non-occluding + targeted-object +
  patch + VLA + test-time WB, but attacks **CoT-reasoning** VLAs by corrupting reasoning
  text, with an **unconstrained** patch. We differ: **base non-CoT OpenVLA** (mechanism =
  teacher-forced action tokens, no CoT) + **stealth-constrained** patch.
- **Trajectory-Level Redirection (2606.12978).** Same targeted-object redirect on
  OpenVLA/LIBERO but via **text** edits (language channel). We are visual/camera channel.
- **Exploring Adversarial Vulnerabilities of VLA (2411.13587, ICCV 2025).** Foundational
  non-occluding OpenVLA patch, but "targeted" = trajectory direction, not object
  substitution; unconstrained.
- **Corroboration:** AttackVLA survey (2511.12149) — targeted test-time VLA attacks
  "largely unexplored"; only targeted method is a training-time backdoor.
- **Timing:** most threats are Mar–Jun 2026 arXiv (≤4 months before today) → framable as
  concurrent work.

Defensible daylight = **stealth + non-CoT base OpenVLA** (+ optionally the autoresearch-loop
discovery framing). See memory `stealth-hijack-novelty-vs-trap`.

## Locked design decisions

- **Stealth model:** bounded δ around a fixed base image (L∞ ε-ball).
- **Base:** a clean, low-frequency **brand-like solid logo** (bold mark on a solid fill).
- **Area:** BL corner **80×80 = 12.8%** — a *known-robust* free-range hijack area
  (CLAUDE.md: "robust to 12.8%, 80×80 ✅ latch 122"), so any failure is attributable to
  the stealth constraint alone, not to patch area.
- **Seed/init:** seed-0 first (existence). Multi-init variance is a follow-up, not a gate.

## Hypothesis

The free-range corner hijack spends most of its capacity on high-frequency structure. A
logo-constrained patch has far less high-frequency headroom, so:
- **H1:** a *pure* logo (ε=0) does **not** hijack (it is just a picture) — control.
- **H2:** there exists an ε at which logo+δ **does** hijack; that ε is the stealth cost.
- **H3:** the tradeoff is monotone in ε (more budget → more decisive-frame forcing).

## Method — the surgical change

Today both the closed-loop core (`run_confined_episode` in `monitor_patch_attack.py`)
and the open-loop gate (`probe_cell` in `corner_probe.py`) build the patch identically:

```python
patch01 = torch.sigmoid(raw)          # FREE-RANGE: any pixel value (noise)
```

The stealth variant substitutes exactly that line (in both), leaving the teacher,
adjudication, `commanded_success`, redirection instrumentation and recording untouched:

```python
# STEALTH: patch = logo ± ε, provably |patch − base|_∞ ≤ ε
patch01 = (base01 + eps * torch.tanh(raw)).clamp(0, 1)
```

- `base01`: the logo rendered into the rect, `[1,3,224,224]`, values in [0,1].
- `raw`: the optimized parameter, init 0 → `patch01 = base01` exactly (pure logo).
- The inner Adam loop + escalation optimize `raw` (δ) **within** the fixed ε-ball;
  the **caller** sweeps ε across the ladder (below).

### Implementation shape (search-side, additive)

- Add additive kwargs `stealth_base: Tensor|None = None`, `stealth_eps: float|None = None`
  to `run_confined_episode` and `probe_cell`. When both are `None` (every existing
  caller) behavior is **bit-identical** to today; when provided, the two `patch01` lines
  branch to the stealth form. This reuses the whole instrumented rollout path (DRY) and
  cannot regress the existing corner/crosstask scripts.
- New driver + helper files (new files, no edits to trusted side):
  - `experiments/patch_attack/make_logo.py` — draws the base logo (PIL) into an 80×80
    BL-corner base tensor; deterministic, saved as PNG for inspection.
  - `experiments/patch_attack/corner_stealth_probe.py` — open-loop ε-sweep gate over the
    grasp-window frames; emits the capacity↔stealth curve.
  - `experiments/patch_attack/corner_stealth_attack.py` — closed-loop confirmation at the
    chosen ε (+ the ε=0 control), reusing `run_confined_episode`.

## Controls (make it publishable)

At the **identical** BL 80×80 rect:
- **ε=0 (pure logo):** must NOT hijack ⇒ proves the attack is the sub-perceptual δ, not the
  logo's presence/appearance. *This is the money control for the stealth claim.*
- Existing `patch_mode` controls already available at this rect: `blank` (gray), `random`
  (fresh noise each step), `none` (clean) — all previously `targeted=False`.

## Execution stages (thermal-aware)

1. **Build + draw logo (CPU):** `make_logo.py`, inspect the base PNG.
2. **Open-loop ε-sweep gate (cheap, GPU-light):** on ~8 grasp-window frames from
   `runs/monitor-patch/run2_rec/clean_input/` (131 available), sweep
   ε ∈ {0.00, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24, 0.32}. Metric per ε = fraction of
   **decisive** frames forced 7/7 (the monotone metric; token-match is inflated by
   agreement). Output = the tradeoff curve + the smallest ε that matches free-range forcing.
3. **Closed-loop confirmation (expensive):** seed-0 as the go/no-go gate, then the precommitted shared inits, at the chosen ε, BL 80×80,
   plus the ε=0 control. Reuse `run_confined_episode` verdicts (`targeted`, `latch_step`,
   `min_target_dist_m`, `commanded_success`, redirection).
4. **Demo + write-up:** render the stealth patch standalone + composited; 3-panel demo
   (left = clean seed-0 "expected" from `runs/autoresearch-hijack/demo/baseline/scene/`,
   middle = attacked input, right = δ); `RESULT.md`.

## Metrics / deliverables

- **Stealth:** L∞ bound (= ε, exact/provable) + saved patch PNG for perceptual judgment.
  (LPIPS not installed; L∞ + visual is the honest, sufficient measure. Optional SSIM if
  `skimage` is present.)
- **Capacity:** decisive-frame forcing vs ε (open-loop curve).
- **Hijack:** `targeted`, `latch_step`, `min_target_dist_m`, `commanded_success` (closed-loop).
- **Curve + control table** = the paper figure.

## Integrity / scope

Simulation-only, test-time, weights frozen (`requires_grad_(False)`), white-box. All edits
under `experiments/patch_attack/*` (search side). **Zero** changes to `src/evaluator/`,
`src/rendering/`, `experiments/configs/`, budgets, or task/seed definitions. The stealth
kwargs are additive with behavior-preserving defaults.

## Risks / open questions

- **R1 (likely):** at 12.8% the logo constraint may need a large ε to hijack, so it stops
  looking like a logo. Mitigation: the ε-sweep *is* the experiment — a clean tradeoff curve
  is the result either way. If ε-at-hijack is large, we report the stealth cost honestly.
- **R2:** teacher-forcing caveat inherited (same objective as the corner runs). Acceptable
  for an existence result; swap to an outcome objective as a later hardening step.
- **R3:** GPU-1 thermal throttling makes the closed-loop rollout slow. Mitigation: the
  open-loop gate front-loads the decision so we spend the one rollout only if the gate passes.
- **R4:** seed-0 only for the first confirmation (same rigor gap as all corner results);
  multi-init variance is the immediate follow-up if it hijacks.

## Next (after this)

If stealthy hijack holds in camera-space: (a) multi-init variance; (b) push ε down / area
down; (c) then carry the *low-frequency* stealth patch into the through-render monitor
(Exp 2) — the hypothesis being that low-freq logo content survives the render's low-pass
better than the high-freq noise that failed GATE-B.

## Revisions — Codex review round 1 (2026-07-22)

Now the **stealth axis** of `2026-07-22-controllability-program.md`; its standing methodology
(fixed-evaluator scoring, N≥10 shared inits, control set, honest scope) binds this experiment.
Experiment-specific changes:

- **Metric (F6):** the headline is **targeted-success-rate vs ε over the shared inits + restarts**
  — *not* "ε-at-first-hijack" (optimizer/seed-dependent). Report alongside a **perceptual** stealth
  measure: SSIM + LPIPS (install `lpips`) vs the logo base, TV / high-frequency energy, and
  side-by-side patch images. L∞ = ε is kept only as the provable bound, not *the* stealth claim.
- **Controls (F7):** ε=0 pure-logo is necessary but not sufficient. Add, at the same rect and
  shared inits: **color-matched blank, scrambled-logo, multiple benign logos, δ-around-flat-fill,
  logo-without-δ**. Purpose: rule out the logo acting as a distractor / high-contrast carrier
  rather than the perturbation.
- **Non-occlusion (F8):** replace the seed-0 static keep-out with **measured** per-frame/init
  segmentation-or-bbox overlap for target-obj / user-obj / basket / gripper; report zero overlap
  as evidence across inits. Clarify framing: "stealth" = *looks benign*, **not small** (80×80 =
  12.8% is a visible screen; the stealth is that the perturbation is imperceptible on the logo).
- **Scoring (F10):** `targeted`/`commanded` numbers come from the fixed-backend wrapper
  (latch-not-terminate, run-to-done/max), not the search-side early-break.
- **Scope (F3/F4):** camera-space upper bound; "action-token forcing," no physical claim.
