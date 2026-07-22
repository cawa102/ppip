# Plan (A): Goal-derived-reference teacher — hardening the hijack vs the teacher-forcing objection

**Date:** 2026-07-22 · **Branch:** `monitor-hijack/phase0` · **Scope:** search-side only
**Status:** PLANNED (awaiting go-ahead). Sibling of `2026-07-22-stealth-corner-hijack.md` (C)
and `2026-07-22-universal-eot-patch.md` (B). Runs on the shared BL-corner machinery.

## The objection this defeats

Every hijack result so far (corner, cross-task, and the upcoming stealth run) optimizes the
patch so `OpenVLA(patched, user)` emits the tokens of `teacher = OpenVLA(clean, target)`. At
7/7 forcing the executed action **is** the target-conditioned policy's own action, so the
closed loop is "instruction-independent by construction." A reviewer objects:

> *"You assumed query access to the target-instructed model to compute the teacher, and you
> only showed the patch can **substitute the target policy's own action stream** — not that it
> can **impose attacker intent** the policy wouldn't already produce. The hijack is an artifact
> of the teacher."*

This is the single biggest validity risk to the whole hijack line. (A) answers it.

## Idea

Replace the **policy-derived** teacher with a **goal-derived reference**: a scripted
pick-and-place controller that drives toward the attacker goal (grasp `salad_dressing` → move
to basket), computed *independently of OpenVLA*. Force the model toward that reference. If it
still hijacks, the patch **imposes an externally-specified attacker trajectory**, not "the
model imitating its own target-instructed self" — and the threat model no longer needs to
query the victim with the target instruction (still white-box for gradients).

## Hypotheses

- **H1:** the reference action is *forceable* — a corner patch can push OpenVLA's tokens to a
  goal-derived reference nearly as well as to the policy teacher. (Open question: the reference
  may be off the policy's action manifold and only partly forceable.)
- **H2:** the reference-teacher patch achieves targeted hijack (`targeted=True`) closed-loop.
- **H3 (either-way finding):** the *gap* between policy-teacher and reference-teacher forcing
  measures how much the patch can impose vs merely substitute — a bound on patch leverage.

## Method

### Scripted reference controller (new helper, search-side)

Per step, from state already exposed in `run_confined_episode`
(`obs["robot0_eef_pos"]`, `obs["robot0_gripper_qpos"]`, `backend._position_for(ostates,
tobj)` for the dressing, and the basket/target drop region from
`backend._target_entities(resolved_target)`), emit a 7-DoF reference action
`[dx,dy,dz, droll,dpitch,dyaw, gripper]` by phase:
1. **approach** — proportional delta of eef toward the dressing xyz, gripper open;
2. **grasp** — close gripper when within a distance threshold;
3. **transport** — proportional delta toward the basket region center;
4. **release** — open gripper over the basket.
Clip deltas to the action bounds (`q01..q99` from `get_action_stats("libero_object")`).

### Continuous reference → 7 action tokens (inverse of `_decode_action`)

`_decode_action` is: `disc = vocab_size - token; disc = clip(disc-1, 0, N-1);
norm = bin_centers[disc]; action = 0.5*(norm+1)*(q99-q01)+q01` (masked dims; gripper uses
`norm`). Invert per dim: `norm_ref = 2*(a-q01)/(q99-q01) - 1` (gripper: `norm_ref = a`) →
`disc = argmin_j |bin_centers[j]-norm_ref|` → `token = vocab_size-(disc+1)`. This yields the
7-token teacher the existing optimizer already knows how to force.

### Ablation wiring (additive, search-side)

Add `teacher_mode: str = "policy"` to `run_confined_episode` and `probe_cell`. Default
`"policy"` = today's `teacher = _real_tokens(model, processor, image, target_task)` (bit-
identical for every existing caller). `"reference"` = the scripted-controller token teacher
above. Everything downstream (optimizer, adjudication, instrumentation) is unchanged, so the
two modes are a clean like-for-like ablation at the **same** BL rect.

## Execution

1. **CPU/build:** reference controller + token-inversion helper; unit-check the inversion
   round-trips (`decode(invert(a)) ≈ a` within one bin) with no GPU.
2. **Open-loop probe (cheap, GPU-light):** on the grasp-window frames, measure per-frame
   forcing of the reference teacher vs the policy teacher (decisive-frame forcing). Answers H1
   before any rollout.
3. **Closed-loop confirmation:** seed-0 gate, then the precommitted shared inits, at the BL corner with
   `teacher_mode="reference"`; compare `targeted`, `latch_step`, `min_target_dist_m`,
   `commanded_success` against the policy-teacher run at the identical rect.

## Metrics / deliverables

- Per-frame **forceability gap**: reference vs policy teacher (decisive-frame forcing).
- Closed-loop **targeted success** under the reference teacher (the headline for H2).
- Interpretation table: forced+hijacked (patch imposes attacker trajectory) / forced-but-not-
  hijacked (reference imperfect / off-manifold accumulation) / not-forceable (leverage bounded
  to on-manifold substitution).

## Controls

Same-rect controls already exist (`none`/`blank`/`random`). Add: policy-teacher run at the
identical rect/seed as the direct comparator (it is our prior result — re-emit or reuse).

## Integrity / scope

Simulation-only, test-time, weights frozen, white-box. All edits under
`experiments/patch_attack/*`; `teacher_mode` additive with behavior-preserving default. Zero
changes to `src/evaluator/`, `src/rendering/`, `experiments/configs/`, budgets, tasks/seeds.

## Relation to (C) and (B)

- Hardens **(C)**: once the reference teacher is validated, re-run the stealth patch under it
  so the stealthy hijack is also free of the teacher-forcing objection.
- Feeds **(B)**: the universal-patch objective can target the reference teacher too, making the
  universal hijack goal-directed rather than policy-substituting.

## Risks

- **R1:** reference may be unforceable/off-manifold → H1 fails. This is itself a documented
  finding (bounds patch leverage), not a dead end.
- **R2:** accumulated small per-step reference errors could diverge the trajectory even at high
  forcing → tune the controller gains; report as a fidelity limit.
- **R3:** teacher-forcing at 7/7 still makes the executed action instruction-independent — but
  now the action is *goal-derived*, which is the point; state this explicitly in the write-up.

## Revisions — Codex review round 1 (2026-07-22)

Now the **mechanism axis** of `2026-07-22-controllability-program.md` — reframed from "rigor
insurance" to the program's **intellectual core**. Standing methodology binds this experiment.

- **Reframe (F1):** (A) does **not** eliminate instruction-independence — at 7/7 forcing the
  action is forced regardless of the user instruction. (A) changes the *source* of the forced
  stream (external goal vs. the target policy). The scientific claim is therefore
  **"can an externally-specified, non-policy action stream be forced?"** — i.e. is the adversary's
  reachable set larger than the policy's own behavioral repertoire? Report as **externally-specified
  action-token forcing**, never "semantic instruction hijack."
- **Preflight controls (F2), same init:** before any attack claim, run and report three
  references — (a) the **continuous** scripted controller executed directly (does the hand-coded
  pick-place even solve the task?); (b) its **decoded-token** version (does quantize→`_decode_action`
  still solve it? — isolates quantization loss); (c) **OpenVLA under the target instruction** (the
  policy-teacher baseline). Only against these three is a forced-reference result interpretable.
  The existing inverse-token round-trip check proves quantization only, **not** forceability.
- **Scoring (F10):** verdicts via the fixed-backend wrapper (latch-not-terminate), not early-break.
- **Rigor (F5):** compare policy-teacher vs reference-teacher forcing over the **shared N inits**,
  not seed-0.
- **Scope (F3/F4):** camera-space upper bound.
