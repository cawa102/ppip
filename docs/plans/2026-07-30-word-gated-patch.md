# Plan: Word-gated adversarial patch — a dormant, language-triggered hijack (camera-space)

**Date:** 2026-07-30 · **Branch:** `monitor-hijack/phase0` · **Scope:** search-side only
**Status:** RUNNING (rev. 2026-07-31 — professor signed off, GPU-1 available; targeted track first).
GPU-free prep is **WP1–WP9**: the 2026-07-31 audit found WP1–WP7 *not* run-ready (a
condition-aware selection rule that would have inflated the headline, and an E2.1 driver still
stubbed), so **WP8/WP9 landed the fixes** — see the prep-gate table.
New **conditionality / gateability** axis of the controllability program
(`2026-07-22-controllability-program.md`); inherits its standing methodology verbatim.
GPU-1 is shared with the stealth (`2026-07-22-stealth-corner-hijack.md`) / universal-EoT
(`2026-07-22-universal-eot-patch.md`) work — pin `CUDA_VISIBLE_DEVICES=1`, check `nvidia-smi`
first, and expect thermal kills (rule 8: checkpoint + resume).

> **Explainer:** `docs/research/word-gate-explained.md` — the self-contained conceptual walkthrough
> (mechanism, code map, the condition-blind selection fix, measured results, limits, glossary).
> Read that for *how it works*; read this file for the plan, work packages, and precommitted gates.

> **📄 Writing the paper? Read `docs/plans/2026-08-06-word-gate-paper-gaps.md` first.** E2.1 landed a
> headline (gate margin 0.833) but a headline is not a paper. That doc holds the evidence ledger, the
> **gap register** (word-alone control and λ=0 never run; one word / one slot / one task pair), and an
> executable per-experiment spec with precommitted interpretations and a tiered GPU budget. It also
> records two reporting defects to fix (`gate_present: false` in the Phase-0 artifact; the
> exactly-zero `mean_dormant_forced`) and corrects the *reason* the objective is two-branch —
> **structural** (a deployed patch is a value, not a function of β), not epistemic.

> **Read this first — the one sentence that must not drift again.** We optimize a **visual
> patch ε** (frozen model, test-time, white-box) whose *effect is gated by a natural word `w`
> in the language instruction*: with `w` present the patch forces a target action `T`; with `w`
> absent the same patch is inert and the robot completes the user task. **ε is optimized; `w`,
> the model, and the evaluator are fixed.** The patch is the high-bandwidth forcing lever; the
> word is only the low-bandwidth gate. This is *not* a text/suffix attack (that is RoboGCG), *not*
> a training-time backdoor (out of scope), and *not* a word that hijacks on its own (our earlier
> analysis showed a bare natural word cannot force a *targeted* action — which is exactly why the
> patch is required).

## Research question

For a **base (non-CoT) VLA** (OpenVLA-7B / LIBERO) under a **test-time, white-box, camera-space**
adversary controlling a **confined** image region, is **action-token forcing gateable by a
language token**? Concretely: does there exist a single patch parameterization ε such that the
frozen policy is **forced to `T` when the instruction contains `w`** and **behaves as the clean
policy when it does not** — and if so, across which targets `T`, how surgical/specific is the gate,
and does it hold regardless of where `w` sits in the instruction?

This adds one axis the spine does not yet have — **conditionality** — on top of {area, occlusion,
stealth, objective, universality, realizability}. It also opens a distinct threat framing: a
**dormant** patch that passes acceptance testing (robot works normally) and fires only on a keyword.

## Why the patch, not the word (settled — do not relitigate)

From the 2026-07-29/30 discussion, three interpretations were killed:

1. **Word-only, nothing injected.** On a frozen model the attacker controls only *which word*, not
   the model's response. A bare natural word can reach DoS (some words degrade the policy) but
   **cannot reliably force a *targeted* action** (a single token is too low-bandwidth). Useful only
   as the lower-bound control (§6).
2. **RoboGCG / GCG suffix.** Reaches targets, but the trigger is a high-perplexity **gibberish
   string** (`"int sm boxes gu x next pu Ру Get给 xエ Setal directlybottom putɯ lowerlap"`), appended
   at the **end**, deliberately added by the attacker. We differ on all three: **natural** word,
   **position-free** (measured), **uttered unconsciously** by the operator.
3. **Training-time backdoor.** The reliable way to install a keyword→behavior map — **out of scope**
   (no training-time poisoning).

The professor's design routes around all three: **the patch carries the forcing; the word only
gates it.** That decouples *naturalness* from *potency* — the word can be perfectly ordinary
("carefully", "please", a color) because it needs to carry no adversarial signal. Naturalness,
which was the hard constraint in interpretation 1, is **free** here.

## Formalization (the formula — the object we optimize)

Notation follows the patch track. Frozen policy `f(o, c)` → 7 discretized action-token logits.
`o` = camera observation (image), `c` = language instruction (token sequence), `w` = the gate word,
`c⊕w` = `c` with `w` inserted, `M` = the confined corner mask, and `P_M(o, ε)` = the observation
with region `M` overwritten by the patch parameterized by `ε` (same direct camera-space pixel
replacement as `run_confined_episode`; `patch01 = sigmoid(raw)` free-range, or
`clamp(base + η·tanh(raw))` for the stealth variant — `ε` denotes the patch parameters `raw`).

Two teachers, both from the **fixed** frozen model (the target-source studied in Exp A):
- target teacher `aᵀ(o) = argmax f(o, c_target)` — the attacker action `T`;
- clean teacher `aᵁ(o) = argmax f(o, c)` — what the unperturbed policy would do.

**Per-frame objective (the regime that is known to force — Exp 2).** For each observation `o`,
solve for one patch that must satisfy *both* word conditions, because at deploy time the patch does
not know whether `w` is present — the model's own cross-modal routing must do the gating:

```
ε*(o) = argmin_ε   CE( f(P_M(o,ε), c⊕w),  aᵀ(o) )        ← word present → force target T
                 + λ · CE( f(P_M(o,ε), c),    aᵁ(o) )        ← word absent  → reproduce clean action
```

> **⚠️ RETRACTED 2026-08-07 — do not cite the paragraph below.** It claimed the per-frame sequence
> is "inert on replay (measured on this project)". The cited measurement is **GATE B**
> (`runs/monitor-hijack/seed0/gate_b_result.json`), whose **oracle itself** scored
> `targeted_success=false, max_phase=0` — no hijack existed for replay to destroy — and which was
> the through-render track besides. On this (camera-space corner) track replay is faithful
> (`runs/monitor-corner/reemit_summary_seed0.json`, `abs_drift_m = 0.0`), and the artifact-level
> panel now measures the gate **from one fixed 126-frame video**: armed `targeted=True` (latch 125),
> dormant `commanded=True` (step 135), blank and time-scrambled controls failing both tasks
> (`runs/monitor-stealth/word-gate/replay_init46/`). Exp 2 is therefore **not** pinned to a live
> optimizer. See the research log entry of 2026-08-07 and limitations L1.

~~**The per-frame sequence is a live procedure, not a replayable video.** `ε*(o_t)` is valid only for
the observation it was fitted to; concatenating `ε₁…ε_H` and replaying it open-loop is **inert on
replay** (measured on this project — per-frame pixels do not reproduce the hijack because deploy
observations do not line up frame-for-frame). So **end-to-end targeted gating (Exp 2) requires the
optimizer to run live each step**, which pins it to per-frame / camera-space scope (R1).~~

**Static / placeable objective (the artifact goal — Exp 2.4, DAgger-gated).** One frame-independent
patch over an EoT frame buffer:

```
ε* = argmin_ε   𝔼_{o∈B_w}[ CE( f(P_M(o,ε), c⊕w), aᵀ(o) ) ]  +  λ · 𝔼_{o∈B}[ CE( f(P_M(o,ε), c), aᵁ(o) ) ]
```

where `B` = clean-trajectory frames (the dormant rollout) and `B_w` = frames the *armed* rollout
induces. `B_w` is the induced distribution — so the static targeted variant **inherits the exact
staticness blocker Exp B is fighting** (a patch fitted to a trajectory the robot leaves does not
hijack; `runs/monitor-stealth/RESULT.md`). See §7 R1.

This is the sketch, made precise: the `(1−β)·L_benign + β·L_malicious` form with `β = 1[w∈c]`, ε
present in **both** branches (because a placed patch is physically there whether or not the word is
spoken), and `L_benign = CE(f(P_M(o,ε),c), aᵁ)` distilling to the **clean policy** (KL≈0 dormancy,
stronger than merely "not `T`").

> **Objective vs measurement (the invariant).** The CE above is the **search-side objective** (the
> method, agent-editable). It **never** produces a reported number. Every `targeted`/`commanded`
> verdict comes from the **fixed** evaluator running the *frozen* patch with no optimizer,
> latch-not-terminate, to done/max — exactly as spine rule 1 requires. A low CE does **not** imply
> closed-loop success (the proxy↔rollout gap already bit the static-patch work).

## The feasibility hinge (state it honestly)

The two terms are **mutually adversarial** — one wants ε silent, the other wants ε to shout —
separated *only* by whether `w` is in the text. It works iff the frozen model's cross-attention
makes ε's effect **conditional** on `w`. Attention is multiplicative, so a gate is *plausible*;
whether the on/off separation is *clean* is empirical. If `w` barely moves the loss landscape, the
optimizer settles for a mushy compromise (weakly-always-on, or never-on). **Diagnostic:** at a fixed
observation, compare `∂/∂ε CE(f(P_M(o,ε),c⊕w), aᵀ)` with and without `w` — magnitude and direction.
Large change ⇒ the gate exists; negligible ⇒ it does not in that config (try another `w`, a bigger
rect, a different teacher). Probe before scaling (§5 Phase 0).

## Decomposition (why this is testable despite the static-hijack blocker)

Gateability is **orthogonal** to the static-vs-per-frame and stealth axes. So isolate it: probe the
gate in the **regime that already forces the wanted effect**, on the **one pair with headroom**
(`alphabet_soup → salad_dressing`, ceiling 11/12), before fighting staticness or stealth.

## Two executable experiments (the DoS / targeted split)

The DoS-vs-targeted split is not two coincidences — it is one fact (a *static* patch has the
capacity for **denial only**; *targeted* forcing needs the per-frame induced-distribution re-fit,
`runs/monitor-stealth/RESULT.md`) seen from four angles at once:

| | **Exp 1** | **Exp 2** |
|---|---|---|
| effect | **DoS** — user task never completes | **targeted** — attacker's action executed |
| patch regime | **static** (one patch suffices) | **per-frame** (a different patch each step) |
| what it is | a **placeable artifact** (a sticker) | a **live search** producing a **replayable per-init video** (artifact-level verified 2026-08-07) |
| threat | weaker effect, **stronger deployment story** | stronger effect, **weaker deployment story** |
| status | **executable now** | **executable now, per-frame scope only** (R1) |

**Exp 1 — word-gated DoS (static, artifact).** Optimize a *word-gated DoS objective* (the two-branch
loss with the malicious teacher = task-failure), inspired by UPA-RFSA's universal-degradation
capability (2511.21192) but **conditional from the start** — not a gate bolted onto their patch.
Static ⇒ a real placeable sticker ⇒ the scary story ("silently fails the task whenever the operator
utters the keyword"). **Make-or-break metric: false-fire** — a DoS patch is disruptive *by design*,
and the dormancy branch needs it inert *without* the word; that tension (R5) is measured first.

**Exp 2 — word-gated targeted (per-frame *search*, replayable *artifact*).** Gate the per-frame
forcing that already hijacks. The **search** runs live — ε_t regenerated each step from the current
observation — but the resulting ε₁…ε_H **is** a deployable video: replayed with no optimizer, time-
indexed, it hijacks under `c⊕w` and lets the user's task complete under `c`
(2026-08-07, `replay_init46/`). The "inert on replay" claim previously stated here is **retracted**
(see the box above). Exp 2 is an existence/mechanism result at **camera-space** scope, now with a
per-init artifact-level demonstration; a **frame-independent** patch still waits on DAgger (E2.4).

**Bridge — word-gated release (static, targeted-lite).** Some targets need only **one decisive
step** — e.g. "open the gripper at the grasp moment" drops the object: a *specific* sabotage, not
generic DoS, and reachable by the **single-step static forcing** that already works. So a word-gated
drop/release is a **static, artifact-shaped, targeted-ish** result *without* DAgger — the sweet spot
between Exp 1 and Exp 2.

## Threat model (fixed / editable / measured)

| | |
|---|---|
| **Attacker capability** | Pre-places **one** patch in the scene (same realizability class as the corner/monitor patches); white-box access to the **frozen** weights for optimization (standard adversarial-patch threat model). **No** weight change, **no** training poisoning, **no** runtime access. |
| **Trigger** | A natural word `w`, attacker-chosen, **fixed**. The operator utters `w` **unconsciously** while phrasing a normal instruction; the attacker does nothing at runtime. (Secondary model — attacker injects `w` via a text channel — is not our headline.) |
| **Editable (method)** | `experiments/patch_attack/*` — new word-gate optimizer + probe; the two-branch objective, `λ`, `w`, position, teacher source. |
| **Fixed (never touch)** | `src/evaluator/*` (predicates, `eval_goal_state`, `attack_score`), `src/rendering/*`, `experiments/configs/*`, budgets, task/seed/init definitions. |
| **Measured (reported, not chosen)** | targeted/commanded/invalid triple both word conditions, gate margin, position profile, gate specificity, `w` perplexity. |

## Positioning (honest, vs prior work) — scan TODO before any claim

Novelty is a **conjunction**, foregrounding **conditionality/dormancy**:

- **TRAP (2603.23117)** — non-occluding, targeted, VLA patch, but **always-on** and attacks a
  **CoT** VLA. We differ: **dormant/language-gated** + **base non-CoT** (mechanism = action-token
  forcing). Dormancy is a *new axis* TRAP does not have.
- **RoboGCG / GCG-for-VLA** — reaches targets via a **gibberish text suffix**, ungated, attacker-
  appended. We differ: **natural word gating a visual patch**; naturalness free.
- **Trajectory-Level Redirection (2606.12978)** — targeted redirect via the **text** channel; we
  are visual + **conditional**.
- **AttackVLA survey (2511.12149)** — the only *targeted* method is a **training-time backdoor**;
  ours is the **test-time** analog of a keyword backdoor with no weight access — likely the open
  corner.

**Prior-art scan — ✅ DONE 2026-08-06: `docs/research/word-gate-prior-art.md`.** It **changed the
positioning above**, so read it before writing related work:

- **Conditionality is NOT unclaimed.** **TPatch** (USENIX Security 2023) owns "adversarial iff
  triggered, benign otherwise" — including a specificity clause equivalent to our E2.2c. We
  differentiate on the *channel* (a natural word in the operator's own instruction, vs an
  attacker-injected acoustic signal), the *victim* (VLA action tokens vs detectors/classifiers), and
  **zero attacker action at runtime** (TPatch's attacker must emit the signal live). Never claim
  conditionality per se.
- **DropVLA** (arXiv 2510.10932) is the closest VLA work: a composite **visual patch + language
  token** trigger forcing a reusable action primitive. It is a **training-time backdoor**
  (data-poisoning, chunked fine-tuning) — out of scope for us by construction. Its own ablation,
  source-verified: *"combining text with vision provides no consistent ASR improvement over
  vision-only attacks"* (text-only transfer 0.72% vs 96.27%). **They had weight access and found the
  language channel inert; we have none and find it decisive.** That contrast is the strongest hook we
  have — but it is only rigorous once E2.2c (specificity) and the λ=0 ablation exist.
- **RoboGCG verified** (arXiv 2506.03350): GCG-style textual suffix, applied once at rollout start,
  "near-complete control over robotic actuators". Our three differentiators hold.
- **arXiv 2606.03556** (partially-observable static patch) gets **disruption, not targeted control** —
  independent corroboration of R1: static ⇒ denial looks field-wide, and static + targeted is open.

## Standing methodology (inherited — see the spine, do not restate)

Rules 1–8 of `2026-07-22-controllability-program.md` bind this plan unchanged: **fixed-evaluator
scoring only** (frozen patch, no optimizer, latch-not-terminate, run-to-done); **N≥10 shared inits**
via the three-way `shared_inits.py` split (5 optimize / 3 gate / 12 held-out, init 0 excluded);
**shared control set**; **full triple + `attack_score`** with `invalid = 0/N` reported explicitly;
**honest scope** ("camera-space [per-frame] upper bound", "action-token forcing" not "semantic
hijack"); **integrity** (only `experiments/patch_attack/*` edited); **checkpoint/resume** under the
thermal hazard (rule 8).

## Phases (each with a GO/NO-GO gate)

**Precommitted GO/NO-GO thresholds (fixed 2026-07-31, before the first probe ran).** Written down
in advance so a marginal result cannot be re-read as a pass afterwards:

| stage | GO requires |
|---|---|
| **A** (open-loop probe) | armed forced-fraction ≥ 0.7 **and** dormant false-fire ≤ 0.2 **and** gate margin ≥ 0.5 on decisive frames **and** gradient hinge `‖g_w − g‖/‖g‖` clearly > 0 |
| **B0** (short-horizon closed-loop) | on ≥ half the decisive steps the two instructions emit **different** actions on the same frame, with armed matching the target more than dormant does |
| **B** (`GATE_INITS`, diagnostic) | ≥ 1 armed targeted latch, 0 dormant targeted, dormant commanded preserved on ≥ 2/3 |
| **C** (`HELDOUT_INITS`) | no threshold — this *is* the reported number (full triple + `attack_score` both conditions, raw counts + CIs, `invalid = 0/N` explicit) |

Sampling rule for A: the frame buffer holds ~1471 train frames, so probe a **stratified sample**
(equal frames per init, evenly spaced within each episode), never the first N of the sorted list —
those are consecutive, highly correlated frames from a single episode.

**Phase 0 — gate-feasibility probe (shared, cheap, open-loop). Run separately per effect.** On the
`frames/train/` buffer, one fixed natural `w`, BL 64×64 (8.2%, measured clear). Per frame, optimize
the two-branch ε and report the **gate diagram**: (armed) fraction of *decisive* frames driven to
the wanted effect *with* `w`, vs (dormant) false-fire *without* `w`. Plus the §"feasibility hinge"
gradient diagnostic. **GO** if armed ≳ the known forcing level *and* dormancy holds (false-fire low);
**NO-GO** ⇒ "the effect is not gateable in this config" (itself a finding) → try `w`/rect/teacher.
The gate may open for DoS and not targeted (or vice-versa) — that asymmetry is a result.

**Exp 1 track — word-gated DoS (static).**
- **E1.1 closed-loop static existence.** Frozen static patch through the **fixed** backend at the
  held-out inits, two conditions: **armed** (`c⊕w`, expect `commanded↓`) and **dormant** (`c`, expect
  `commanded↑` ≈ ceiling, `targeted≈0`). Headline = **DoS gate margin** = `commanded_rate(dormant) −
  commanded_rate(armed)`. **False-fire (R5) is the gate** — if the patch breaks the task without `w`,
  the result is void.

**Exp 2 track — word-gated targeted (per-frame procedure).**
- **E2.1 closed-loop per-frame existence (single pair).** Live per-step optimizer, two rollouts:
  **armed** (`c⊕w`, expect `targeted↑`), **dormant** (`c`, expect `commanded↑`, `targeted≈0`).
  Headline = **gate margin** = `targeted_rate(armed) − targeted_rate(dormant)`, full triple both.
- **E2.2 characterize the gate.** (a) **Target ladder** — halt → directional → gripper →
  object-substitution (`salad_dressing`): where the gate reaches, where it breaks. (b) **Position
  profile** `ASR(w,p)` — all-index (mechanism) + grammatical-only (threat-real). (c) **Gate
  specificity** — false-fire over a benign-word corpus; synonyms of `w` (semantic lock) vs exact `w`
  (lexical). (d) `λ` sweep = dormancy↔potency frontier.
- **E2.3 (bridge) word-gated release — static, targeted-lite.** Single decisive-step gripper-open on
  the static patch (drop-on-keyword). Artifact-shaped targeted result **without** DAgger.
- **E2.4 static targeted gate (DAgger-gated, stretch).** Only after Exp B yields a static patch that
  forces end-to-end; then add the gate via the static objective. Inherits R1 until then.

**Phase S — stealth gate (stretch, either track).** Carry the winning gate onto the ε-ball logo
ladder (stealth plan) once its closed-loop existence holds.

## Metrics / deliverables (all from the fixed evaluator)

- **Armed** (`c⊕w`): targeted `targeted`↑ / DoS `commanded`↓ → `attack_score`.
- **Dormant** (`c`): `commanded` ↑ (≈ clean 10–11/12 ceiling), `targeted` ≈ 0 — the **false-fire**
  legs (targeted-fire and DoS-fire without the word); for Exp 1 this is the make-or-break metric (R5).
- **Gate margin** — targeted: `targeted_rate(armed) − targeted_rate(dormant)`; DoS:
  `commanded_rate(dormant) − commanded_rate(armed)`. The headline, **derived from** fixed-evaluator
  outputs — **not** a new evaluator score (the invariant stays intact).
- **Position profile**, **gate specificity / benign-word corpus**, **`w` perplexity** (report the
  word is natural), **`λ` frontier**. Raw counts + CIs, never rates alone.

## Out of scope

Training-time poisoning; physical/printed realization; through-render success (recorded as the
boundary); "semantic hijack" claims (say "gated action-token forcing"); runtime attacker access.

## Risks / open questions

- **R1 (the big one) — staticness.** A *placeable* patch must be static, and static targeted forcing
  is currently blocked (Exp B). So the **defensible first result is per-frame** (camera-space,
  idealized attacker) — an *existence/mechanism* result on gateability, scoped exactly like the rest
  of the program. Static DoS-gating is the cheapest artifact-shaped result; static targeted-gating
  waits on DAgger.
- **R2 — feasibility hinge** (§): the gate may not exist cleanly. Phase 0 is designed to find that
  cheaply, and a negative is publishable ("forcing is not language-gateable from a confined region").
- **R3 — ceiling.** Only `salad_dressing` is reachable in `alphabet_soup` (11/12); targeted-rung
  results are single-pair unless we screen other scenes. Lower rungs (DoS/halt/directional) do not
  need a reachable *object*, so the ladder is still informative single-scene.
- **R4 — thermal / long-job kills** (rule 8): checkpoint + `setsid` resume.
- **R5 — DoS dormancy tension (Exp 1).** A DoS patch is broadly disruptive *by design*; the dormancy
  branch needs it inert *without* the word. That robustness fights dormancy, so **false-fire — does
  the task break even without `w`? — is the make-or-break metric for Exp 1**, measured in Phase 0
  before any closed-loop spend.

## Open questions for the professor — **signed off 2026-07-31**

1. **Confirm the object** = the boxed sentence at the top (patch optimized, word fixed gate). ✅
2. **`w` fixed vs co-searched?** Start one fixed natural word; add a word search later. ✅ —
   `please` confirmed a **single Llama token** (id 3113) on 2026-07-31; armed prompt 25 tokens vs
   dormant 24, so the two conditions differ by exactly one token.
3. **Target `T` priority.** ✅ **Targeted-first.** The targeted teachers are already fixed
   (`aᵀ = argmax f(o, c_target)`), so the targeted track needs no further decision, while the DoS
   armed teacher (halt vs untargeted divergence) is still an open *formulation* question. Exp 1
   therefore waits: it needs that decision **and** a static gated-DoS optimizer that does not
   exist yet (`run_static_dos_gate` only *scores* a patch handed to it).
4. **Accept the per-frame/camera-space scope for the first result** (R1). ✅

## Files (proposed, additive — no trusted-side edits)

**Landed (GPU-free core, all tested):**
- `experiments/patch_attack/word_gate.py` — WP1/WP4 instruction construction (dormant `c` / armed
  `c⊕w`, position slots, `FIRST_WORD="please"`, contamination guard).
- `experiments/patch_attack/gate_metrics.py` — WP2 gate metrics (`gate_report` → `gate_margin`,
  `armed_forcing_fraction`, `false_fire_rate` + raw counts; targeted **and** DoS `Effect`). Consumes
  the fixed `evaluator.metrics.summarize_rollouts`, so it derives — never re-judges.
- `experiments/patch_attack/two_branch_loss.py` — WP3 objective `two_branch_loss(...)`; CE mirrors
  `ce_monitor_patch_attack.py:244`. *(WP2/WP3 live in their own modules, not `word_gate.py`: cohesion,
  and to keep `torch` out of the pure-string core.)*
- `experiments/patch_attack/word_gate_probe.py` — WP5 Phase-0 open-loop gate diagram + gradient
  diagnostic over `frames/train/`. Pure core (frame listing, decisive-dim/forced classification,
  `aggregate_gate_diagram`, `gradient_gate_signal`) tested; `probe_frame` GPU seam mirrors the
  proven per-frame optimize loop with the dormancy branch added.
- `experiments/patch_attack/word_gated_attack.py` — WP6 closed-loop driver. Pure core
  (`assemble_word_gate_result`, `reportable_inits`, E2.1 deferral) tested; `run_static_dos_gate`
  (E1.1) GPU seam composes the fixed pieces (adjudicate-on-clean, override-the-instruction).
- `experiments/patch_attack/word_gate.py` (`GateSetup`/`resolve_gate_setup`, pure) +
  `experiments/patch_attack/ce_monitor_patch_attack.py` (`run_confined_episode` edit) — WP7 additive
  two-branch kwargs `gate_word` / `word_index` / `dormancy_weight` / `deploy_word`. Ungated path
  bit-identical; gated path is the two-branch optimiser deployed under the armed/dormant condition.
  Unblocks the E2.1 per-frame targeted gate (the `word_gated_attack` seam calls this twice).

**Landed 2026-07-31 (the audit fixes — see WP8/WP9 below):** condition-blind selection +
per-step gate record in `ce_monitor_patch_attack.py` / `word_gate.py`, and the real E2.1 driver in
`word_gated_attack.py`.

**Still owed on the Exp-1 side (GPU-free):** a **static word-gated DoS optimizer** — the two-branch
static objective over the EoT frame buffer with a DoS armed teacher. Blocked on the WP3 teacher
decision (open Q3), not on plumbing.

**Reused unchanged (GPU runs):**
- `shared_inits.py`, the measured-clear BL 64×64 rect (`occlusion_probe.py`;
  `corner_rect("BL", 64) = (160, 0, 64, 64)`), `ceiling_screen.py` frame buffers, and the
  fixed-evaluator wrapper (`eval_static_patch.py` / `hijack_backend.py`).

## Prep gate (GPU-free; nothing hits GPU until these land + tested)

Modeled on the stealth plan's prep gate. Every new file is **additive** — it touches nothing on the
trusted side and nothing the running stealth/EoT session uses, so it is safe to build now, in
parallel with those runs.

| # | item | file(s) | status |
|---|---|---|---|
| WP1 | **Instruction construction** — build the dormant (`c`) and armed (`c⊕w`) conditions; insert one natural word at any slot; enumerate slots for the position sweep. Pure string, fully unit-tested. | `word_gate.py` + `tests/patch_attack/test_word_gate.py` | ✅ done 2026-07-30 — 24 tests, ruff + mypy --strict clean |
| WP2 | **Gate metrics** (pure) — `gate_margin`, `false_fire_rate`, `armed_forcing_fraction` from per-init/frame rows; targeted **and** DoS variants; **derived from fixed-evaluator outputs only** (no new score). | `gate_metrics.py` (+ tests) | ✅ done 2026-07-30 — 9 tests, ruff + mypy --strict clean; built on the fixed `evaluator.metrics.summarize_rollouts` (never re-judges) |
| WP3 | **Two-branch loss** (pure, mock-tensor tested) — `CE(armed_logits, target_teacher) + λ·CE(dormant_logits, clean_teacher)`. Reuses existing teachers: target = `_real_tokens(clean, target_task)`, clean = `_real_tokens(clean, user_task)` (already computed for `dec_dims`, `ce_monitor_patch_attack.py:206`). | `two_branch_loss.py` (+ tests) | ✅ done 2026-07-30 — 8 tests, ruff + mypy --strict clean; CE mirrors `ce_monitor_patch_attack.py:244` |
| WP4 | **Word list / naturalness** — a small curated set of candidate natural magic words + a naturalness (perplexity) note; a decision doc, no code dependency. | plan / research doc | ✅ first word = `please` (2026-07-30); sweep axes recorded below |
| WP5 | **Open-loop probe scaffold** — Phase 0 gate diagram + gradient diagnostic over `frames/train/`, run per effect; `@requires_gpu` seam test, scaffold now / run later. | `word_gate_probe.py` (+ tests) | ✅ done 2026-07-30 — pure core 13 tests, ruff clean, **own-code mypy-strict clean** (GPU-seam file like `ce_monitor_patch_attack`: direct `vla_diff`/`adaptive_attack` imports surface *their* pre-existing debt, 0 errors in this file); `@requires_gpu` seam test. Targeted probe wired; `--effect dos` **defers pending the DoS-teacher decision** (open Q3). |
| WP6 | **Closed-loop driver scaffold** — E1.1 (static DoS) + E2.1 (per-frame targeted): armed/dormant rollout pair via `HijackBackend` / fixed eval; `@requires_gpu`. | `word_gated_attack.py` (+ tests) | ✅ done 2026-07-30 — pure core 5 tests, **ruff + mypy --strict clean** (composes only clean modules). E1.1 static-DoS seam wired via the `ceiling_screen` mechanism (`carrier_candidate` + `set_instruction_override` + `frozen_evaluation`, adjudicate-on-clean / override-the-instruction); E2.1 per-frame path a **loud deferral** pending WP7; `@requires_gpu` seam test. |
| WP7 | **`run_confined_episode` additive kwargs** `gate_word` / `word_index` / `dormancy_weight` / `deploy_word` — the surgical two-branch change in the optimize loop, **behavior-preserving when `gate_word is None`**. | `ce_monitor_patch_attack.py` (edit) + `word_gate.py` (pure resolver) | ✅ done 2026-07-30 — two-branch loss when gated (mirrors `word_gate_probe.probe_frame`); deploy instruction + best-selection teacher follow the armed/dormant condition; scene/adjudication/clean-teacher pinned to plain `user_task`. Pure `resolve_gate_setup`/`GateSetup` in `word_gate.py` (ruff + mypy-strict clean); fail-fast guards (optimize-only, novel trigger) reachable on CPU. Ungated path bit-identical. 14 CPU tests + 3 `@requires_gpu` smoke seams. **No GPU spend yet.** |
| WP8 | **Condition-blind selection** (the 2026-07-31 audit fix) — rank candidate patches by `armed_match + dormant_match`, identically in both rollouts; execute under the deployed instruction only. Plus the free per-step gate record it makes possible. | `word_gate.py` (`gate_step_selection`) + `ce_monitor_patch_attack.py` (edit) | ✅ done 2026-07-31 — WP7 ranked candidates (and early-stopped, and carried `warm_raw`) by the **deployed** condition, so the armed rollout kept the most target-forcing ε and the dormant rollout the most inert one: the gate margin was inflated at both ends by our own selection, contradicting the premise that ε cannot know whether `w` was uttered. Now one deploy-independent score, +1 forward per attempt (~3–5%). Ungated path bit-identical. 9 CPU tests incl. the invariance property over the whole match grid |
| WP9 | **Real E2.1 driver** — `run_perframe_targeted_gate`: one live two-branch episode per (init, condition), fixed-predicate verdicts → `RolloutOutcome` → `gate_report`; resumable, crash-tolerant. | `word_gated_attack.py` | ✅ done 2026-07-31 — WP7 unblocked E2.1 but the driver was still a `NotImplementedError` stub. Now loops inits × {armed, dormant}, appends every finished episode to `rows.jsonl` and skips it on restart (rule 8 — episodes are hours on a thermally shared card), and turns a crashed episode into an **errored** outcome (kept out of the rates, never scored as "the attack failed"). GPU boundary injected (`episode_fn`) so the pairing/resume/mapping logic is CPU-tested: 7 tests |

**Insertion mechanics (confirmed).** The magic word enters the instruction **string** before the
prompt template `In: What action should the robot take to {task.lower()}?\nOut:` (used in
`_prompt_ids` / `_real_tokens`, `adaptive_attack.py:52,88`). The two conditions differ only by that
string; the fixed evaluator commands the **user** task, and the search side supplies the armed vs
dormant `user_ids`. WP7 is the *only* edit to an existing file, and it is additive.

### WP4 — first magic word (decision 2026-07-30)

**First word: `please`.** It is the cleanest word for isolating the *gate*:

- **Natural + unconscious.** Operators say "please" to robots reflexively — it appears in a real
  instruction with no social engineering.
- **Semantically neutral (the deciding property).** Pure politeness carries *zero* action content, so
  the **patch**, not the word, must do the forcing. A task-aligned word (`stop`, `drop`) would
  confound "patch gated by word" with "model simply obeys the word"; `please` removes that confound.
- **Clean baseline.** Absent from the LIBERO instruction (enforced by `assert_trigger_novel`), so the
  dormant condition is genuinely word-absent; likely a single Llama token (verify cheaply later);
  natural at both sentence ends → serves the position sweep (E2.2).
- **Attribution via the margin.** Even if `please` has a small out-of-distribution solo effect, the
  **gate margin** `(patch+word) − (word-alone) − (patch-alone)` isolates the gating; word-alone
  inertness is a Phase-0 *control*, not an assumption.

**First milestone: the DoS gate (Exp 1) with `please`** — cheapest, most likely feasible (static ⇒
denial already works), artifact-shaped. The *same* word is then carried into the targeted probe
(Exp 2) so the **DoS-vs-targeted gate asymmetry is measured with the word held fixed**.

**Eventual sweep (full WP4, later)** — three axes, `please` as the neutral single-English-word origin:
- **length:** single word → short phrase → multi-token chunk;
- **naturalness:** neutral filler → task-relevant word → gibberish (RoboGCG's suffix = high-perplexity endpoint);
- **language:** English → other languages → code-switched.

Codified now: `word_gate.FIRST_WORD` + `assert_trigger_novel` (contamination guard). The broader
candidate list stays the researcher's to design.
