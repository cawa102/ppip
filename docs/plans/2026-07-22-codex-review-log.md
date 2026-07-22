# Codex adversarial review log — plans C/A/B

**Date:** 2026-07-22 · Reviewer: OpenAI Codex (via `codex` plugin, read-only) ·
Arbiter: Claude (final decision on each finding). Round 1 complete; round 2 pending.

## Verdict (round 1)

> "Not sound to execute as written **for thesis claims**. Sound only as an exploratory
> camera-space program after (1) tightening evaluator semantics, (2) precommitting multi-init
> controls, (3) narrowing the novelty/physicality language."

Accepted. The reframe to a *forced-controllability map* (`2026-07-22-controllability-program.md`)
+ the fixes below address all three.

## Findings & arbitration

| # | Codex finding | My call | Resolution |
|---|---|---|---|
| F10 | `run_confined_episode` breaks on `targeted` (line 317), violating the fixed evaluator's *latch-not-terminate* contract → undercounts `commanded_success`, overstates pure hijack | **ACCEPT** (most important; touches the invariant) | Program rule 1: all scored numbers via a **fixed-backend wrapper** (frozen patch, no optimizer, latch-not-terminate, run to done/max). Search-side early-break is diagnostic only. |
| F9 | (B) DAgger could re-optimize on held-out test frames = leakage | **ACCEPT** | (B): separate **no-grad fixed-patch eval mode** + assertion that no optimizer runs during eval. |
| F1 | (A) doesn't eliminate instruction-independence; only changes the *source* of forced tokens | **ACCEPT** | Reframe (A) as **"externally-specified action-token forcing,"** not semantic hijack (program contribution 3). |
| F2 | (A) scripted reference may be **off-manifold** / unforceable; round-trip check only proves quantization | **ACCEPT** | (A): add **3 preflight controls** — continuous-controller rollout, decoded-token-controller rollout, OpenVLA-target-instruction rollout — same init. |
| F5 | seed-0/single-init is debugging, not a claim | **ACCEPT** | Program rule 2: **N≥10 (target 20–30) shared held-out inits**, identical across conditions; seed-0 = gate only. |
| F6 | (C) L∞-around-logo is a math bound, not perceptual stealth; ε=0.32 visibly adversarial; "ε-at-hijack" is optimizer/seed-dependent | **ACCEPT** | (C) metric → **targeted-rate vs ε** over inits+restarts + **SSIM/LPIPS + TV/frequency energy + side-by-side images**. |
| F7 | (C) pure-logo (ε=0) control insufficient | **ACCEPT** | Program rule 3: expanded control set (color-matched blank, scrambled logo, multiple logos, δ-around-flat-fill, logo-without-δ). |
| F8 | 80×80 = 12.8% is visually large; keep-out is seed-0-specific, not proof | **ACCEPT** | (C): **measure** per-frame/init overlap (target/user-obj/basket/gripper); clarify stealth = *looks-benign*, **not small**. |
| F4 | camera-space = digital patch; must not be sold as physical | **ACCEPT** | Program rule 5: "camera-space upper bound"; no physical claim without through-render. |
| F3 | novelty thinner; TRAP "has stealth machinery" | **ACCEPT — round 2: I was wrong** | **Round-2 correction: TRAP §5.3 "Stealthiness Enhancement" DOES use content loss + TV loss + DIP (verified on the HTML).** My round-1 "no stealth loss" read only the abstract. Honest distinction is NOT "TRAP lacks stealth" but **base non-CoT OpenVLA + a *different* stealth model (ε-bounded logo vs content/TV/DIP camouflage) + the action-token controllability map**. Concede TRAP owns physicality AND has stealth; stealth is not our headline. |

## Net effect

No experiment discarded; every fix sharpens the design. C/A/B are re-scoped as axes of the
controllability map, scored by the fixed evaluator, precommitted multi-init, honestly bounded to
camera-space. Round 2: re-submit the revised docs to the same Codex thread for verification.
