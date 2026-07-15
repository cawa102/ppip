# Plan Review Log: Physically-realizable in-scene monitor-video hijack of OpenVLA-7B
Act 1 (grill) complete — plan locked with the user. MAX_ROUNDS=4.

## Round 1 — Codex (VERDICT: REVISE)

1. Camera-buffer invariant not enforceable: `hijack_backend.py:41` exposes `_delta`/`_patch`, `adaptive_attack.py:176` executes from perturbed `pu8`. Fix: new backend inheriting `OpenVLARolloutBackend` with runtime hashes/tests proving policy input == rendered env output only.
2. Real-path `7/7` token match is not honesty enforcement — research-log records 7/7 for 150 steps with no target motion. Fix: escalation gate = target-object pose progress/success; token match only a diagnostic.
3. In-place texture upload has no existing seam (only XML reset in `inject.py:102`). Fix: prototype fixed-size texture alloc, texid lookup, `tex_rgb` slice mutation, render-context upload, pixel-hash verify across steps without reset.
4. Ignores fixed compiled texture dims + MuJoCo UV quirks (variable-size textures `text_prompt.py:52`, box mirror pre-flip `inject.py:111`). Fix: one fixed WxH monitor asset + UV/mirror calibration test grid.
5. Projected quad NOT constant — omits center-crop (`vla_diff.py:29`) + dynamic robot occlusion. Fix: compute mask in post-crop policy-input coords; log segmentation/visibility per step.
6. Option A underspecified: additive masked δ in 224px ≠ shaded rendered texture replacing scene pixels. Fix: calibrate a texture→policy-input surrogate from real renders, or optimize texture pixels directly with black-box/finite-difference.
7. Offline target trajectory vs current-frame labeling (`adaptive_attack.py:170`) ambiguity. Fix: define current-state target-policy tokens vs time-indexed reference tokens; add state-matching/DAgger if the latter.
8. Same-seed open-loop replay overstated as deployable; one early drift invalidates later texture timing. Fix: predefine trajectory-divergence metrics + success thresholds; same-seed failure = no-deployment result.
9. "Scoring on the real render" imprecise — `eval_goal_state` scores simulator object state. Fix: reword as fixed object-state predicate; rendered frames logged as policy-input evidence.
10. Phase-1 extended spec conflicts with `attack_candidate.schema.json` (`additionalProperties: false`). Fix: versioned frozen schema/validator for optimizer-design candidates.
11. Tools "fixed" but loop chooses loss/budget/reality-gap/A-B-C. Fix: predeclare a finite action space; equalize optimizer-calls/GPU-hours/seeds across conditions.
12. Baselines don't search the same space (random = text/style/placement only). Fix: all conditions propose the same extended config type.
13. Logging inadequate — backend saves one static `geom.texture` (`openvla_backend.py:599`). Fix: log per-step texture hash/frame, rendered policy image, seg mask, token match, decoded action, target pose, upload status.
14. Ledger append racy under async jobs (`ledger.py:28` checks IDs then appends, no lock). Fix: file locking or single scheduler with idempotent IDs.
15. Precomputed-video artifact path/boundary leaks. Fix: candidates carry parameters only; evaluator-owned code generates/reads artifacts under the run dir; no candidate-supplied paths.

### Claude's response
Accepted 14 of 15 essentially verbatim — all materially strengthen the plan. Detailed edits to PLAN.md:
- **1** → new `MonitorHijackBackend(OpenVLARolloutBackend)`; policy input asserted == rendered env observation via per-step hash (mechanizes the Q4 invariant instead of merely promising it).
- **2** → escalation/verify gate is `min_target_dist` progress; token-7/7 demoted to per-step diagnostic (directly avoids the DoS-at-7/7 trap the research-log already recorded).
- **3,4** → Phase-0 step 1 is now an explicit in-place-upload prototype checklist + a fixed WxH monitor asset + a UV/mirror calibration grid.
- **5** → mask computed in post-crop policy-input coords; per-step visibility/segmentation logged; "quad constant" claim corrected to "constant up to crop + occlusion."
- **6** → Option A restated honestly: additive δ is only a *proposal*; a real-render texture→policy-input **surrogate calibration** step is mandatory, with black-box/finite-difference as the named fallback when the gap won't close.
- **7** → Stage-1 teacher = current-frame target-policy tokens (reactive, as adaptive_attack already does); the recorded sequence is the time-indexed Stage-2 video; open-loop adds state-matching/DAgger.
- **8** → predefined trajectory-divergence metric + threshold; same-seed failure explicitly a no-deployment result. (Minor: plan already scoped L1 as a mechanism study, not deployment — kept, but added the metrics.)
- **9** → scoring reworded: fixed object-state predicate is the score; the invariant is that the **policy input** is a real render (never a camera-buffer write); rendered frames are logged as evidence.
- **10,11,12** → Phase 1 gets a frozen versioned optimizer-design schema; a finite predeclared action space; budget equalized on **optimizer-calls/GPU-hours/seeds** (not candidate count); all baselines emit the same extended config type.
- **13** → per-step logging expanded to texture hash, rendered policy image, seg mask, token match, decoded action, target pose, upload status.
- **14** → ledger locking / single idempotent scheduler adopted as a Phase-1 requirement (conditional on parallel jobs; sequential Claude-in-the-loop is currently safe — noted as such).
- **15** → candidates carry parameters only; evaluator-owned code owns all artifact paths under the run dir.
No rejections. Finding 8's "overstated as deployable" is a slight misread (L1 was already a mechanism study, not a deployment claim) but its metrics ask is adopted.

## Round 2 — Codex (VERDICT: REVISE)
"Most prior findings addressed textually. Remaining problems are new or caused by the revisions."

1. Stale frame: texture uploaded after `obs` produced → policy may see pre-upload frame (`openvla_backend.py:224`). Fix: `MonitorHijackBackend` must fresh-render after every upload and feed that exact post-upload frame.
2. Teacher-token contamination: the current frame now contains the previous adversarial texture; the adaptive path had no in-scene monitor contaminating the teacher. Fix: compute teacher tokens from a **neutral/off-monitor render** of the same simulator state, then optimize the user-policy frame with the attack texture.
3. Object-progress escalation requires stepping the sim → trying multiple textures per step contaminates the oracle trajectory unless exact state/controller/RNG rollback. Fix: cloned shadow envs with validated full-state restore, OR keep per-texture verification **stateless** and make object-progress a committed rollout metric.
4. `min_target_dist` too blunt pre-grasp — correct approach reduces eef-to-target while min_target_dist is unchanged. Fix: **phase-aware** gate (eef-to-target → grasp/contact → target displacement → containment).
5. Hash assertion needs a canonical image stage — raw render, cropped/resized policy input, and processor pixel_values are different artifacts. Fix: hash/log all three; assert policy input derives only from the fresh post-upload render via fixed preprocessing.
6. "Periodic resets on a controller-state-preserving path" fallback contradicts the plan's own premise that resets broke continuity. Fix: validate that path first, or mark reset-based fallback as outside the threat-faithful claim.
7. Controls only "fail to hijack" allows a high-vis DoS to pass while the optimized video barely steers. Fix: controls report commanded success, target displacement, min distance, trajectory divergence — claim is a **margin**, not a binary.

### Claude's response
Accepted all 7 (no rejections). Edits to PLAN.md:
- **1** → Stage-1 step now mandates ordering **upload → fresh render → feed exact post-upload frame**; no stale `obs`.
- **2** → teacher tokens computed from a **neutral/off-monitor render** of the same sim state (two renders per step: monitor-off → teacher; attack-texture-on → user-policy input).
- **3** → reconciled my Round-1 over-correction: **per-texture inner selection is stateless** (target-token CE / logit-margin on the post-upload render, no env step); phased target-progress is the **committed** per-step outcome; optional cloned shadow-env lookahead with validated full-state restore. Token match returns as the *inner* signal — just never as the *claim*.
- **4** → graded metric is now a **phase-aware** progress signal, not bare `min_target_dist`.
- **5** → invariant names three canonical stages (raw render / 224 policy input / processor pixel_values), hashes all three, asserts policy input flows only from the fresh post-upload render.
- **6** → dropped the self-contradictory reset fallback; in-place upload is required for the continuous claim, and any reset-based variant is explicitly marked out-of-claim.
- **7** → controls report the full metric panel; the hijack claim is the **margin** of the optimized video over blank + time-scrambled on target displacement/containment, not just `targeted_success`.

## Round 3 — Codex (VERDICT: REVISE)
Round 2 findings confirmed addressed. Three new precision issues:
1. Stage 2 still blurred by "state-matching/DAgger" inside it — online state-matched frame selection is feedback, not a precomputed video "merely played." Fix: Stage 2 strictly time-indexed; offline DAgger only; state-conditioned selection is a separate closed-loop ablation.
2. "off/neutral" teacher render not equivalent — "off" changes occlusion/background; "neutral" preserves geometry but may itself perturb the target policy. Fix: neutral same-geometry as default + an S0 sanity check that TARGET still succeeds with the neutral monitor present.
3. Upload-prototype "nothing else changed" full-frame pixel-hash too brittle (resolution/AA/crop). Fix: segmentation-mask-local hash + outside-mask tolerance, separate from the S1/S2/S3 invariant hashes.

### Claude's response
Accepted all 3 (no rejections):
- **1** → Stage 2 redefined as strictly `texture_t = video[t]`, no online selection; robustness only via offline precompute-time DAgger/EoT baking one fixed video; state-matched selection split out as a separate labeled closed-loop ablation.
- **2** → teacher view is now **neutral same-geometry** (monitor present, blank content — not "off"), plus an **S0 sanity gate**: verify TARGET=salad_dressing still succeeds with the neutral monitor before trusting any teacher tokens; if not, fix placement/size first.
- **3** → change-detection uses a **mask-local hash + outside-mask tolerance**, kept separate from the S1/S2/S3 invariant hashes.

## Round 4 — Codex (VERDICT: APPROVED)
"No material blockers remain in PLAN.md." All Round 3 fixes confirmed present. One non-blocking implementation caveat: the S0 sanity gate says "run once" — for seeds 0–4, run it across the same evaluated init states or explicitly record seed exclusions before attack evaluation.

### Claude's response
Adopted the caveat: S0 gate now runs across seeds 0–4 (same init states used for evaluation), with explicit recording of any per-seed exclusions.

---
**CONVERGED** — grilled (Act 1, 7 decisions) + survived 4 rounds of Codex adversarial review (Act 2). 24 findings raised across 3 REVISE rounds, 24 accepted, 0 rejected. Plan locked, awaiting user sign-off before implementation.

