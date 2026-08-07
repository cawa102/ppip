# Word-gate — limitations section (draft)

**Date:** 2026-08-06 · **Revised 2026-08-07** (L1 rewritten, L9 updated) · Draft prose for the paper, written **before** the remaining experiments so the
scope cannot drift to fit whatever comes back. Owed by
`docs/plans/2026-08-06-word-gate-paper-gaps.md` §7.7.

> **Rule this draft enforces:** every limitation is stated as a *measured or structural fact*, not as
> a softener. If an experiment later closes one, delete that paragraph — do not weaken it.

---

## L1. The artifact is a per-init recorded video, not a universal sticker

**Revised 2026-08-07.** The previous version of this paragraph claimed the per-frame patches were
"inert on replay" and that no deployable artifact existed. That is now measured false on this track
and the claim is withdrawn — see the research log entry of 2026-08-07 and
`runs/monitor-stealth/word-gate/replay_init46/`. What follows is the limitation that actually
remains, which is narrower but real.

The gate **does** survive compression into a fixed artifact. One pre-recorded 126-frame patch video,
displayed at the same rectangle and indexed by control step alone with no optimizer in the loop,
drives both conditions at init 46: with the trigger word, `targeted = True` (latch step 125);
without it, the operator's own task **completes** at step 135. A blank rectangle and a
time-scrambled version of the same frames both fail *both* tasks, so the effect requires the video's
content **and** its temporal alignment, not merely an occupied corner. The pixels are byte-identical
across the two conditions; only the surroundings differ, because the robot is doing different things.

Four things this still is not, and a reader should not be allowed to infer any of them.

**It is one init, replicated zero times.** n = 1 at the artifact level. The 12-init held-out evidence
is per-frame; the artifact claim currently rests on a single pair and must be replicated before the
paper leans on it.

**The armed leg is in-distribution by construction.** The video was recorded *from* that rollout at
that init, and replaying it reproduces the original bit-faithfully (`latch_step = 125`,
`min_target_dist_m` identical to 17 digits). That leg is a fidelity check, not evidence. The load
is carried entirely by the dormant leg, where the video meets observations it was never fitted to.

**The gate is initiated by language but sustained by trajectory alignment.** At step 0 both
conditions see the same scene through the same pixels and the word alone changes the emitted action —
that part is genuine cross-modal gating. Thereafter the split self-reinforces: the video is aligned
to the armed trajectory, and off it the arming decays (`mean_armed_forced` 0.999 armed → **0.144**
dormant). We do not claim the frozen policy holds the patch conditional for 160 steps on its own.

**It is not a frame-independent patch, and not physical.** A single static image that gates targeted
forcing remains unbuilt (E2.4), the whole result is camera-space — pixels replaced in the
observation, no perspective, lighting or resampling — and nothing here is a poster on a warehouse
wall. The genuinely open question the video result does *not* answer is transfer: whether a video
recorded at one init gates at another.

The surrounding context stands. The closest static-patch work under a realistic
partial-observability threat model (arXiv 2606.03556) obtains **disruption**, not targeted control,
from a fixed patch: static ⇒ denial looks general, and static + targeted remains open.

*(Narrowed by the 2026-08-07 replay panel. Closes further on artifact-level replication across inits;
fully only if E2.4 lands.)*

## L2. Single task pair, single scene

Every number is `alphabet_soup → salad_dressing` in the LIBERO-object alphabet-soup scene. The pair
was chosen because it is the one with headroom: the ceiling screen shows `salad_dressing` is
reachable by the *clean* policy in 11/12 held-out inits, so a targeted rate can be normalized against
something real. We do not claim the gate generalizes across scenes or object pairs, and the
adjudicability constraint (a target is only adjudicable if its object is in the user task's scene,
and LIBERO-object scenes do not share one object set) makes that generalization non-trivial to test.

*(Closes with E-A7.)*

## L3. Single trigger word, single insertion slot

All closed-loop results use `w = "please"` at slot 0. `please` was chosen for a reason that is itself
a limitation: it is semantically empty, so the patch must do the forcing and the result cannot be
confounded with the model simply obeying a command word. That makes it a clean *mechanism* probe and
a weak *coverage* claim. We do not know whether the gate is a property of this token, of
single-token triggers generally, or of any prompt perturbation.

We also do **not** currently support the claim that the gate is position-free. That property is a
stated differentiator from suffix-based text attacks (which append a trigger at the end by
construction), and it requires the position profile we have not yet measured.

*(Closes with E-A4 and E-A5. Until then, do not write "position-free" anywhere.)*

## L4. The gate is measured, its specificity is not

Dormancy is measured only against the *exact* absence of `please`. We have not measured what happens
under synonyms (`kindly`, `pls`), casing variants, or a corpus of unrelated benign words. So we can
say "the attack fires with this word and not without it", but not yet "the attack fires with this
word and not with other words" — and only the second rules out the reading that any inserted token
perturbs the prompt enough to release the patch.

*(Closes with E-A6.)*

## L5. Attribution is incomplete without the word-alone control

The clean decomposition of the effect is `(patch + word) − (word alone) − (patch alone)`. We measure
`patch alone` — that is the dormant arm, and it is behaviourally indistinguishable from no patch. We
have **not** measured `word alone`: `please` commanded with no patch at all. If `please` on its own
perturbs the policy, the gate margin must be reported net of that baseline.

*(Closes with E-A1 — the cheapest experiment in the whole register.)*

## L6. We do not know whether the dormancy objective was necessary

The patch is optimized against a two-branch loss whose second term pins the word-absent behaviour to
the clean policy. Dormant targeted success is 0/12. But we have not run the λ=0 ablation, so we
cannot distinguish two explanations: (a) the dormancy term produced the dormancy, or (b) a patch
fitted under `c⊕w` is naturally inert under `c` and the term was redundant. (b) would be a *stronger*
statement about the model and a weaker one about our method.

*(Closes with E-A2.)*

## L7. Simulation only; camera-space; white-box

No physical robot, no printed patch, no through-render realization. The perturbation is written
directly into the policy's camera-space input, which is an idealized attacker channel — an upper
bound on what a physically realized patch could achieve. The optimizer needs white-box gradients of
the frozen policy (standard for the adversarial-patch threat model, but worth stating). No weights
are modified and no training data is touched at any point.

We deliberately do **not** claim a "semantic hijack". The mechanism is **action-token forcing**: the
patch drives the discretized action tokens toward those the clean policy would emit if commanded the
attacker's task. Whether the model "understands" anything is not measured and not claimed.

## L8. Statistical scope

n = 12 held-out initializations, one trial per (init, condition), one horizon (240 steps). The design
is paired, which is what carries the significance (exact McNemar p = 0.0020 for targeted, p = 0.0039
for commanded) despite the small n; the marginal Wilson intervals are correspondingly wide
(armed targeted 0.833, 95% CI [0.552, 0.953]). We report raw counts throughout. Two armed episodes
did not latch, and both had near-perfect per-step forcing (0.994 / 0.998) — they are **execution**
failures downstream of the action tokens, not forcing failures, and one of them (init 4) fails even
when the target is directly commanded, i.e. was impossible by construction.

We have not measured within-init variance across repeated trials.

## L9. Condition-blindness is an imposed constraint, not a physical one

The per-frame optimizer physically has the instruction available; we forbid it from branching on
whether the word is present, and rank candidate patches by a deploy-independent score. This is a
deliberate methodological choice, made so the measured gate is a property of the frozen policy rather
than of our search procedure — and it is what a deployed artifact enforces by physics anyway, which
the 2026-08-07 replay panel now demonstrates rather than merely asserts (one video, both
instructions; see L1). A reader should know the constraint is ours, not the setting's.

The justification is **structural, not epistemic**: a deployed patch is a *value*, not a function of
`β = 1[w ∈ c]`, so even a perfectly informed attacker still faces both instructions with one tensor
of pixels. Committing to the armed branch alone is λ = 0, which yields an always-on patch, not a gate.

## L10. No defense evaluated

We do not evaluate mitigations. The obvious candidates — JPEG compression, blur, random resized crop,
input noise — are cheap to test and untested here.

*(Closes with E-A8.)*
