# Design: the ε threshold between user-task-completion, DoS, and hijack

**Date:** 2026-08-04 · **Branch:** `monitor-hijack/phase0` · **Scope:** search-side only
**Status:** DESIGN — agreed in session. **Steps 1–2 implemented and run (2026-08-04); step 3
(the first ladder rung) is the next action.**

> **New session?** Read `2026-08-04-epsilon-threshold-HANDOVER.md` **first** — it carries the
> retracted findings, the regime traps, and the exact next command. This file is the *why*; the
> handover is the *what now*.
**Supersedes the run plan in** `2026-07-22-stealth-corner-hijack.md` (that plan's static ε ladder
stays blocked on DAgger; this is its per-frame fallback, its option 3, taken deliberately).

## 1. What this experiment claims

One sentence: **how much bounded perturbation, hidden in a logo that covers no object, is needed
to move a base VLA from doing the user's task, to doing nothing, to doing the attacker's task.**

Three outcome classes, adjudicated only by the fixed evaluator:

| class | predicate | meaning |
|---|---|---|
| **completed** | `commanded=True`, `targeted=False` | the attack failed; the user's task still finishes |
| **DoS** | `commanded=False`, `targeted=False` | denial — the user's task breaks, no substitution |
| **hijack** | `targeted=True` | the attacker's object is delivered |

The deliverable is the pair of ε values where the class changes, and the perceptual cost at each.

### What "stealth" means here (decided 2026-08-04)

**Spatial stealth only**, and stated as such. The patch is non-occluding (measured per init, not
eyeballed), logo-shaped, provably within ε of a fixed carrier, and LPIPS-quantified. The
**temporal** instability of a per-frame attack is reported as a limitation, not hidden:

> measured at ε=0.06, 220 steps: mean `|patch_t − patch_{t−1}|` = 0.0378 (9.6/255), 91.8% of
> steps change by >5/255, against a total perturbation magnitude of 0.0388 — **97% of the
> perturbation is re-randomised every step.**

Each frame is individually near-indistinguishable from the logo (LPIPS 0.0176), but the sequence
shimmers at 20 Hz, and human vision is far more sensitive to flicker than to static texture. No ε
and no objective fixes this; only a static patch would, and that route is blocked upstream. The
churn number is itself novel — no prior VLA-patch work reports it — so it is a contribution, not
just a caveat.

## 2. Why the threshold needs the strongest attack we can mount

A threshold measured with a wasteful optimizer is **an upper bound wearing a threshold's clothes**.
If our loss squanders budget we report "stealth costs ε ≥ 0.25" when the truth is 0.10, and the
whole classification is an artifact of our own method. This is the standard no-under-attacking norm
in adversarial robustness evaluation, and it is why the objective is chosen on problem-class
grounds rather than on which one we happen to have run.

**Decision: margin hinge, not cross-entropy** (researcher, 2026-08-04).

- The paper reports a **minimum-perturbation** quantity; C&W margin losses were designed for
  exactly that regime, whereas PGD-with-CE is the convention for *fixed-ε robustness curves*.
- CE never saturates: on a dim already decoding to the teacher it keeps buying logit margin that
  changes no action, spending budget the unsolved dims need.
- CE-all-7 (what `run_confined_episode` runs today) is unmasked, and the two instructions already
  agree on ~3 of 7 dims — so ~43% of every gradient step anchors free wins.
- CE prices a 1-bin miss like a 128-bin miss, though the measured user/target disagreement is a
  median of 23/256 bins (~9% of range).

**Honest statement of the evidence at tight budget:** neither objective dominates, and the
measurement we have cannot settle it. At ε=0.06, N=1 frame, 3 decisive dims, forcing can only take
{0, 0.333, 0.667, 1.0}:

| base | hinge (κ=3) | CE |
|---|---|---|
| aurora | 0.667 | 0.000 |
| vertex | 0.667 | 1.000 |
| solstice | 0.667 | 0.667 |

The choice is therefore made **on principle where evidence is absent**, which is the correct way to
break that tie — not on our own provenance, which would be the bias this decision exists to avoid.
The paper reports the comparison honestly and names CE as the alternative considered.

## 3. Bug found while deciding this: `DEFAULT_KAPPA` is too small

`margin_hinge` averages `relu(best_other − teacher_logit + κ)` over decisive dims. A dim that
clears margin κ contributes **exactly zero** gradient, so with κ too small the optimizer wins two
dims, stops pushing them, and progress on the third knocks them back under margin — constraint
cycling rather than convergence. Measured on aurora, 300 steps, N=1:

| κ | ε=1.0 | ε=0.06 |
|---|---|---|
| 1 | 0.667 | 0.667 |
| **3 (current default)** | **0.667** | 0.667 |
| **6** | **1.000** | 0.667 |
| **12** | **1.000** | 0.667 |
| 25 | 0.667 | 0.667 |
| 50 | 0.667 | 0.667 |
| 100 | — | 0.667 |

**κ=3 has silently capped every hinge run this project has done.** At free budget κ ∈ {6,12}
reaches full forcing; too small *and* too large both fail. This also retracts a finding logged on
2026-07-31 ("the objective inverts with ε — hinge wins at 0.06, loses at ε=1"): that was measuring
the κ default, not the loss family. The research log entry must be corrected, not left standing.

Note the ε=0.06 column is unmoved by **any** κ. That is not established as a hinge defect — CE
scored 0.000 on the same cell — but it is the specific risk the probe in §4.2 exists to catch.

## 4. Method

### 4.1 Port the hinge into the closed-loop path (no GPU)

`run_confined_episode` has only ever run `F.cross_entropy` over all 7 dims (line 332); it does not
import `forcing_loss` at all. Add it the same way the stealth kwargs were added — **additive
kwargs with behaviour-preserving defaults**, so every existing caller keeps the CE path
bit-identically and no prior corner result is disturbed:

- `objective: str = "ce"`, `kappa: float`, reusing `forcing_loss` rather than reimplementing.
- The decisive-dim set `dec_dims` is already computed per step for `decisive_boost` and reporting;
  the hinge consumes it directly, so masking costs nothing new.
- Fail-fast on an unknown objective before any policy load, as the gate and stealth kwargs do.
- Record `objective` and `kappa` in the result dict — today's closed-loop JSONs have **no**
  objective field, which is why provenance was ambiguous this session.
- Change `DEFAULT_KAPPA` 3 → 6, citing §3.

Tests, mirroring `test_stealth_confined.py`: default returns the CE path unchanged; unknown
objective raises; κ propagates; decisive masking selects the right dims; a GPU-marked end-to-end
seam.

### 4.2 Per-frame objective probe (~20 min GPU) — validation, not selection

The objective is chosen. This checks it is not pathological **in the regime the ladder runs**, and
it exists because every objective comparison so far used the wrong tool: `stealth_optimize
--max-frames N` fits ONE patch to N frames (the *static* problem), while the ladder optimizes each
frame **independently**.

Take ~20 decisive frames from `runs/monitor-stealth/ceiling/frames/train/`, optimize each
independently at ε=0.06 under `hinge` (κ ∈ {6, 12}), `directional`, and `ce` as the reference.
Report mean decisive forcing per objective — a continuous metric over 20 frames rather than a
4-valued one over 1.

**Go/no-go:** if hinge pins at ≈0.667 here too while CE does not, we have learned it in 20 minutes
rather than after a 9-hour rollout, and the choice is revisited with evidence instead of principle.
`directional` is included at near-zero marginal cost: same saturating family, but it respects the
bins' ordering, which the 23-bin median disagreement suggests matters.

#### RESULT (run 2026-08-04, `runs/monitor-stealth/objective_probe/`) — **PASS, hinge cleared**

8 frames × 5 specs, 240 steps each, ε=0.06, BL 64×64, identical effort:

| spec | mean | dim-weighted | fully-forced |
|---|---|---|---|
| ce_decisive | 0.729 | 0.720 | 0.375 |
| ce | 0.708 | 0.680 | 0.500 |
| **hinge@κ6** | 0.667 | 0.640 | 0.250 |
| hinge@κ12 | 0.604 | 0.600 | 0.125 |
| directional | 0.188 | 0.200 | 0.000 |

**The aggregate ranking is not real.** Paired per frame, `ce_decisive` vs `hinge@κ6` is **1 win,
7 ties, 0 losses** — the whole gap is one frame, (1,40). `ce` vs `ce_decisive` is 1 win each and 6
ties. These objectives are indistinguishable on this evidence, and any table above that is read as a
ranking will mislead.

**What it does establish — which was the point:** the static regime's pathology (hinge pinned at
0.667 for every κ from 1 to 100) **does not reproduce per-frame**. `hinge@κ6` matches CE exactly on
7 of 8 frames. R1 has *not* fired; the objective choice stands on the principle it was made on, and
**κ=6 is the pin** — κ12 is strictly worse (loses frame (1,20), halves fully-forced).

**Two caveats that bound this.**

- `directional`'s collapse is most likely **misconfiguration, not the objective**: it ran at
  `anchor=0.0`, and `forcing_loss`'s own docstring warns its softmax gradient thins on a peaked
  action head and needs a light CE term. It was given none. Not a verdict on `directional`.
- **All 8 frames are init 1, steps 0–40** — `build_frames` returns them in order and the probe took
  the first 8, so they are consecutive steps of one episode, highly correlated, and not a sample
  across the five `OPTIMIZE_INITS`. A firmer answer needs a stratified second pass (~30 min); this
  one is sufficient to clear pathology but not to rank.

### 4.3 The ladder (closed-loop throughout)

Every rung is a full 220-step rollout adjudicated by the fixed evaluator. No open-loop proxy enters
any claim — open-loop degrades exactly where the attack succeeds (the trajectory leaves the clean
one), which is the trap that produced the static track's false-positive gate.

- **Cell:** BL 64×64 (8.2% of frame), init 0, `alphabet_soup` → `salad_dressing`.
- **Effort pinned** at the escalated config (k=30, maxtries=10, restarts=3) that the free-range
  positive required — at default effort that cell scored `targeted=False` free-range, so a weaker
  budget manufactures false negatives.
- **κ and objective pinned** across all rungs, for the same reason P8 pins effort: a ladder where
  the objective varies measures the objective, not ε.
- **ε_hijack first** — the headline. Log-spaced bisection upward from the known DoS point, because
  ε is perceptual and the arithmetic midpoint (0.53) is a patch nobody would call stealthy:
  **0.25 → then 0.12 or 0.5 → then one refinement.** Three rungs bracket it to ~1.4×.
- **ε_dos second**, downward from 0.06 into (0, 0.06).
- **Top rung re-run:** free-range under hinge, so ceiling and rungs share an objective.

Per rung, record: `targeted` / `commanded` / `latch_step` / `min_target_dist_m`, mean decisive
forcing, `linf_measured_max` (re-measured from executed patches), **LPIPS vs the carrier**, and
**per-step churn**.

### 4.4 N-init widening

At the located threshold, run the precommitted `HELDOUT_INITS`. **Framed as init-0 from the start,
not retrofitted:** at ε=0.06 init 0 gave DoS (forcing 0.676) while held-out inits 4 and 7 gave
`commanded=True` and no redirection (forcing 0.556 / 0.580). The threshold may not transfer, and
that must be a stated expectation rather than a later concession.

## 5. Deliverables

- **Threshold table** — outcome class vs ε, with ε_dos and ε_hijack as brackets, not points.
- **Stealth-vs-capability curve** — LPIPS (and ε) against targeted/commanded rate.
- **Patch strip** at every rung, plus the ε=0 pure-logo control.
- **Churn measurement** as the temporal-stealth limitation.
- **Non-occlusion evidence**, measured per init, already built (`occlusion_probe.py`).
- **The 4-panel GIF**, extended with the threshold rungs.

## 6. Controls (unchanged, already run or built)

ε=0 pure logo (`targeted=False`, `commanded=True`, ‖δ‖∞=0.0 exactly — the money control), plus
`blank` / `random` / `none` at the identical rect, and the `scrambled:` / `flat:` / `gray` carriers.

## 7. Redirection is measured, not inferred

A small `min_eef_to_target` alone does not show redirection: the objects are clustered, and
`salad_dressing` sits only 0.110 m from `milk`, so an arm stalling at a neighbour posts a small
target distance with no redirection at all. `nearest_object_probe.py` replays a trace against every
graspable and reports which object was actually nearest. At ε=0.06, init 0: nearest was
`salad_dressing` on **196/220 steps (89%)** and `milk` on **zero**; at closest approach, 0.071 m to
the target vs 0.103 m to the milk. Every redirection claim in the paper cites this probe.

*(Caveat: the probe uses start-of-episode object positions, so it differs from the live per-step
figure — 0.071 m here vs 0.047 m recorded. The ranking is what the claim rests on, and both agree.)*

## 8. Integrity and scope

Simulation-only, test-time, weights frozen, white-box, camera-space. All edits under
`experiments/patch_attack/*`. **Zero** changes to `src/evaluator/`, `src/rendering/`,
`experiments/configs/`, budgets, or task/seed definitions. Every `targeted`/`commanded` verdict
comes from the unmodified `eval_goal_state`. The objective is the agent-editable *method* side of
the invariant — which is exactly why changing it cannot cheat the score.

No physical-realizability claim. Camera-space upper bound. TRAP owns physical + targeted; "When
Robots Obey the Patch" owns universal-untargeted; our cell is base non-CoT OpenVLA, action-token
forcing, **with a provable ε bound TRAP does not have**.

## 9. Risks

- **R1 — hinge pins at tight ε.** Unmoved by κ at ε=0.06, N=1. §4.2 catches it in 20 min. If it
  reproduces per-frame, revisit the objective with evidence.
- **R2 — the threshold is init-specific.** Likely, given seeds 4/7. Mitigated by framing (§4.4).
- **R3 — no ε below free-range hijacks.** Then the result is "bounded per-frame perturbation cannot
  hijack at this area; only unbounded can", which is publishable as a boundary with the forcing
  numbers explaining it. Not a wasted ladder.
- **R4 — wall-clock.** ~9.5 h/rollout measured (worse than the 7h20m first estimate under GPU
  contention). ~6–8 rungs ≈ 3–4 days. Retry + env-state checkpointing already in
  `run_stealth_asr.sh`; this host kills long GPU jobs non-deterministically.
- **R5 — porting risk.** New code on the only path that currently produces hijacks. Mitigated by
  additive-default kwargs + tests, the pattern that has held for the gate and stealth kwargs.

## 10. Sequence

1. κ fix + hinge port + tests (no GPU, ~2 h)
2. Per-frame objective probe (~20 min GPU) — go/no-go
3. ε_hijack bisection, init 0 (3 rungs, ~1.5 days)
4. ε_dos bisection, init 0 (2–3 rungs, ~1 day)
5. Free-range top rung under hinge (1 rollout)
6. N-init widening at the threshold
7. Figures + write-up

Steps 1–2 are cheap and gate everything after them; nothing beyond step 2 should start until the
probe result is in.
