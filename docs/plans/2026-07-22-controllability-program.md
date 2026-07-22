# Program: Forced controllability of a base VLA under confined visual perturbation

**Date:** 2026-07-22 · **Branch:** `monitor-hijack/phase0` · **Scope:** search-side only
**Status:** FRAMING (parent of plans C/A/B; hardened against Codex review round 1 — see
`2026-07-22-codex-review-log.md`). This is the paper spine; C/A/B are its axes.

## Why this reframe (the top-tier move)

The recurring critique of every hijack result — "at 7/7 token forcing the executed action
IS `OpenVLA(clean,target)`, so the loop is instruction-independent by construction" — is not a
flaw to hide. It is the **object of study**. Our attacks are **targeted adversarial examples in
action-token space**, applied closed-loop from a confined region. Reframing around *forced
controllability* (not "semantic hijack") makes the work honest, mechanistic, and — crucially —
**unscooped**: TRAP (physical, CoT, targeted) and "When Robots Obey the Patch" (universal,
untargeted) each own one axis; **neither gives a systematic, mechanistic controllability map on
a base VLA.**

## Research question

For a **base (non-CoT) VLA** (OpenVLA-7B / LIBERO) under a **test-time, white-box, camera-space**
adversary controlling a **confined** image region, what is the **reachable set of
adversary-imposed behaviors** — no-effect → **denial (DoS)** → **directed redirection** →
**targeted object-substitution hijack** — as a function of the budget axes
{**area**, **occlusion**, **perceptual stealth**, **optimization objective / target-source**,
**cross-episode universality**, **rendering realizability**}, and what **mechanism** governs the
transitions?

## Contributions (claims we will defend)

1. **Mechanistic dichotomy.** Readable scene text → DoS only (the language channel is not
   reachable by pixels-as-text); confined *feature-level* perturbation → **action-token forcing**
   → hijack. Corroborated by the AttackVLA survey (targeted test-time attacks "largely
   unexplored"); explains all prior results.
2. **A controlled controllability map** across the budget axes — multi-init, shared controls,
   **scored only by the fixed evaluator**.
3. **A mechanism characterization** of the hijack as action-token forcing, incl. whether an
   **externally-specified (non-policy) target stream is forceable** (Exp A). This *quantifies how
   much control the adversary truly has* and delimits "action-token forcing" from "semantic
   instruction hijack." This is the intellectual core.
4. **A realizability boundary.** Camera-space forcing succeeds; the through-render in-scene
   monitor fails (render low-pass); we localize the reality-gap as the single blocker. This
   *explains* why a physical attack (TRAP) needs robustness machinery we do not use, and scopes
   our result as a **camera-space upper bound**.
5. *(Method, optional headline)* discovery via an autoresearch red-team loop.

## Positioning (honest, vs prior work)

- **TRAP (arXiv:2603.23117, "Hijacking VLA CoT-Reasoning"):** physical/printed, **CoT-VLA**,
  targeted object-substitution, **and explicitly stealth-aware** — §5.3 "Stealthiness Enhancement"
  uses a content loss + TV loss + DIP (verified on the HTML; my round-1 "no stealth loss" read only
  the abstract). TRAP is therefore stronger on **physicality** and **comparable on stealth intent**.
  We differ on: **base non-CoT OpenVLA** (no CoT to corrupt → mechanism is action-token forcing, not
  reasoning-text corruption), a **different stealth model** (explicit ε-bounded perturbation around a
  recognizable logo vs content/TV/DIP camouflage), and the **controllability map** vs a single demo.
  We **concede TRAP owns physical realizability and has stealth**; **stealth is not our novelty
  headline** — the base-VLA action-token reachability map is.
- **"When Robots Obey the Patch" (2511.21192):** universal/transferable but **untargeted**
  degradation. We differ: **targeted** object substitution + mechanism.
- **AttackVLA survey (2511.12149):** corroborates the dichotomy; predates TRAP.

## Standing methodology (binds C, A, B — do not restate, reference this)

1. **Scoring via the FIXED evaluator only (resolves Codex F10 + the invariant).** Search-side
   scripts produce *diagnostics* (forcing fraction, decisive-frame forcing) and *candidate
   patches*. Every reported `targeted_success_rate` / `commanded_success_rate` comes from a
   **faithful fixed-backend wrapper** that (a) loads the **frozen** patch with **no optimizer**,
   (b) runs the standard rollout with fixed **latch-not-terminate** semantics + `eval_goal_state`,
   (c) runs to user-done / max_steps. It *calls* `src/evaluator/` adjudication **unchanged**. No
   headline number comes from the search-side early-break path.
2. **Multi-init precommit (resolves Codex F5).** A fixed set of **N held-out init indices**
   (N ≥ 10; target 20–30), **identical across every condition and every experiment**. Seed-0 is
   only a cheap go/no-go gate, **never** a claim. Report raw counts + CIs, not just rates.
3. **Shared control set (resolves Codex F7).** `none` / color-matched `blank` / uniform `random`
   / (stealth) `pure-logo (ε=0)` / `scrambled-logo` / `δ-around-flat-fill` — same init indices.
4. **DoS vs hijack reported separately.** `commanded_success` (denial) and `targeted_success`
   (hijack) are distinct axes; report both with counts.
5. **Honest scope (resolves Codex F3/F4).** "camera-space upper bound"; **no** physical /
   realizability claim without through-render success. Say "action-token forcing," not "semantic
   hijack."
6. **Integrity.** Only `experiments/patch_attack/*` edited; evaluator / rendering / configs /
   budgets / tasks unchanged; the fixed-backend wrapper calls unchanged adjudication.

## Axes → experiments (the map)

| Axis | Populated by | Result type |
|---|---|---|
| Area & occlusion | prior corner results + **measured** per-init non-occlusion (upgrade) | reachable vs area |
| Perceptual stealth | **Exp C** (`2026-07-22-stealth-corner-hijack.md`) | targeted-rate vs stealth budget |
| Objective / target-source (mechanism) | **Exp A** (`2026-07-22-outcome-teacher-ablation.md`) | is a non-policy stream forceable? |
| Cross-episode universality | **Exp B** (`2026-07-22-universal-eot-patch.md`) | one fixed patch vs per-frame |
| Rendering realizability | **prior Exp-2** through-render monitor (existing data) | camera-space works, render blocks |

## Out of scope

Physical-robot / printed-patch realization; training-time poisoning; through-render success
(recorded as the boundary, not attempted anew here); semantic-understanding claims.

**Scope of "map" — DECISION (2026-07-22): multi-pair.** C/A/B run across **2–3 task pairs**, not
one, so "controllability map" is earned. Design (uses the existing `scene_task` decoupling so
pixels/init/arm-pose stay identical and only the adjudicated target changes):

- **Fix the scene + user task = `alphabet_soup`** (the policy performs it reliably clean → not
  ceiling-limited) and **vary the attacker TARGET** across objects present in that rich scene
  (`salad_dressing`, `butter`, `cream_cheese`, `milk`, `tomato_sauce`). Each is adjudicable
  (present in the scene), so no `UnevaluableGoalError`.
- **Base-policy-ceiling screen (mandatory, decides the set).** Attack success ≤ base-policy
  success on the target task in that scene (established in `runs/autoresearch-hijack/RELIABILITY.md`
  and the cross-task study). So first run `user = target = <obj>` with **no patch** at the shared
  inits; **include only the 2–3 targets the base policy can actually achieve** (known good:
  `salad_dressing`, `cream_cheese`; screen `butter`/`milk`/`tomato_sauce`). Report each target's
  ceiling alongside its attack rate — the gap *is* the controllability result.
- Compute scales as C/A/B × (#targets) × (#shared inits); acceptable per the no-time-limit decision.
- Varying the *user* task instead is deliberately not the axis: at 7/7 forcing the loop is
  instruction-independent by construction (cross-task study), so target-reachability is the
  meaningful variable, not the commanded instruction.
