# Word gate — artifact-level experiments (E-ART), results and follow-ups

**Date:** 2026-08-17 · **Branch:** `monitor-hijack/phase0` · **Scope:** the word-gate track's
*artifact-level* question — does the gate survive compression into a deployable video? Companion
docs: `word-gate-limitations-draft.md` (L1 rewritten from this), `2026-08-06-word-gate-paper-gaps.md`
(gap register), `research-log.md` (chronology).

> **What this document is.** The per-frame result (E2.1, gate margin 0.833 over 12 held-out inits)
> measures a *procedure*. This file records the experiments that ask whether it is also an
> *artifact*, their outcomes, one execution defect found afterwards, and what is left.

---

## 1. The question and why it was open

The threat model has always been artifact-level: an attacker leaves something playing on a monitor,
it is physically present whether or not the operator utters the trigger, and the frozen policy's own
cross-modal routing does the gating. The per-frame attack re-solves its patch every control step, so
its armed and dormant rollouts never share pixels — which made the deployment story *asserted*, not
measured.

It had also been ruled out on bad evidence. The plan claimed a concatenated video is "inert on replay
(measured on this project)"; the cited measurement is GATE B
(`runs/monitor-hijack/seed0/gate_b_result.json`), whose **oracle itself** scored
`targeted_success=false, max_phase=0` — no hijack existed for replay to destroy — and which was the
through-render track besides. On the camera-space corner track replay is exactly faithful
(`runs/monitor-corner/reemit_summary_seed0.json`, `abs_drift_m = 0.0`). Retracted 2026-08-07.

## 2. Design

One patch video, recorded from an armed per-frame episode, is deployed as a fixed artifact: at
control step `t` the monitor shows `v_t`, selected by **step index alone**
(`monitor_replay.time_indexed_texture` — no state-conditioned selection is expressible), with **no
optimiser in the loop**. The camera renders whatever the robot is actually looking at, so the two
word conditions share the patch pixels exactly and differ in everything around them.

Verified rather than assumed at init 46: patch pixels byte-identical across conditions at steps
0/25/50/100/125 and equal to the source video; scenes identical at step 0 (`maxabsdiff = 0`) and
divergent after (199 @ s25, 217 @ s100).

Two questions, one code path (`experiments/patch_attack/run_word_gate_replay.py`):

* **Replication** — video recorded at init *N*, replayed at init *N*. Is the artifact-level gate
  reproducible, or an init-46 fluke?
* **Transfer** — init-46's video replayed at init *M ≠ N*. Is it a sticker, or must the patch be
  optimised for one specific setting?

Controls per replication panel: **blank** (mid-gray rectangle — separates "the video hijacks" from
"any occupied corner disrupts" and from "the word alone does it") and **scrambled** (the same frames
permuted in time — separates content from temporal alignment).

**Init set is forced, not chosen.** Replication needs an init with both a hijack to reproduce and a
benign success to preserve. From Stage C that is `{7, 24, 26, 33, 36, 38, 46, 49}`. Excluded: **4**
(target unreachable even when directly commanded), **22** (the single genuine armed miss), **39/45**
(dormant never completed inside the reported horizon, so the benign leg is confounded). Encoded in
`render_word_gate_figure.STAGE_C_EVENTS`, pinned by `tests/patch_attack/test_artifact_panel.py`,
which also asserts the horizon clears both Stage-C events with margin — a short horizon would render
a successful benign rollout as a denial, i.e. libel the dormant condition.

## 3. Result — replication: the gate is an artifact

`runs/monitor-stealth/word-gate/artifact/init<N>/`, one panel per init, 2026-08-07/08.

| init | armed latch | dormant cmd@ | **armed targeted** | **dormant commanded** | blank dormant | scrambled dormant |
|---|---|---|---|---|---|---|
| 7 | 132 | — | True | **False** | False | False |
| 24 | 133 | 134 | True | True | True | True |
| 26 | 160 | 169 | True | True | True | True |
| 33 | 210 | 130 | True | True | False | False |
| 36 | 161 | 154 | True | True | True | True |
| 38 | 162 | 152 | True | True | True | False |
| 46 | 125 | 135 | True | True | False | False |
| 49 | 109 | 132 | True | True | True | True |

**Armed targeted 8/8. Dormant commanded 7/8.**

* **The single benign miss (init 7) is not attributable to the video**: the **blank** control also
  failed to complete the task there. Stage C had init 7's dormant completing at step 160 against a
  180-step horizon — 20 steps of margin — so it is the same horizon artifact that already
  disqualified 39/45, not a dormancy failure.
* **Armed replay is bit-faithful** to the per-frame original where checked (init 46: `latch_step` 125,
  `min_target_dist_m` identical to 17 digits). That leg is a fidelity check, not evidence; the load
  is carried by the dormant leg, where the video meets observations it was never fitted to.
* **Correction to an earlier reading.** At init 46 the blank control failed *both* tasks, which was
  reported as "the adversarial video is less disruptive to the benign task than plain gray." That
  does not generalise: blank leaves the benign task intact at 5 of the 7 other inits. **Do not carry
  that claim forward.**

## 4. Result — transfer: the video does not move

`artifact/transfer_from46/init<N>/`, init-46's video replayed elsewhere.

| init | armed targeted | dormant commanded | `mean_armed_forced` |
|---|---|---|---|
| 7 | False | False | 0.168 |
| 24 | False | False | 0.107 |
| 26 | False | False | 0.132 |
| 33 | False | True | 0.135 |
| 36 | False | True | 0.149 |
| 38 | False | False | 0.095 |
| 49 | False | False | 0.116 |

**Armed targeted 0/7**, with forcing collapsing from 0.999 in-init to **0.095–0.168**. The patch must
be optimised for one specific setting; robustness is a separate problem.

**Sharper than expected, and worth stating:** a foreign video is not *inert* either — dormant
commanded is only **2/7**. Out of place it degrades to **denial**, neither hijacking nor leaving the
task alone. That matches the field-wide pattern the prior-art scan found (arXiv 2606.03556: static
patches obtain disruption, not targeted control).

## 5. Mechanism — state it honestly

At step 0 both conditions see the same scene through the same pixels and the word alone changes the
emitted action: genuine cross-modal gating. Thereafter the split **self-reinforces**, because the
video is aligned to the armed trajectory and off it the arming decays (`mean_armed_forced` 0.999
armed → 0.144 dormant at init 46; 0.095–0.168 under transfer). The gate is **initiated by language
and sustained by trajectory alignment**. We do not claim the frozen policy holds the patch
conditional unaided for a whole episode.

## 6. Supporting controls (Tier 0)

* **E-A1 word-alone** (`word_alone/summary.json`) — `please` with **no patch**: commanded **9/12**,
  targeted **0/12**, identical to the clean baseline at matched 240-step horizon. The word alone does
  nothing; the effect is the patch⊕word conjunction. Attribution closed.
* **E-A2/E-A3 λ ablation** — see §7 for the frame-set caveat. On a **common 1471-frame set**:

  | λ | dormant false-fire | gate margin |
  |---|---|---|
  | 0.0 | 0.830 | 0.150 |
  | 0.1 | 0.238 | 0.732 |
  | 0.3 | 0.074 | 0.893 |

  **Dormancy is not free.** Without the dormancy term the patch fires 83% of the time with no trigger
  word and the gate collapses to 0.150. G2's live alternative explanation for dormant 0/12 is dead.

  **Re-measured 2026-08-18 on the precommitted stratified 32-frame sample** (`lam*_n32/`), so all
  four points are now directly comparable to the λ=1.0 baseline:

  | λ | armed forcing | dormant false-fire | gate margin |
  |---|---|---|---|
  | 0.0 | 1.0000 | 0.8302 | 0.1698 |
  | 0.1 | 1.0000 | 0.1974 | 0.8026 |
  | 0.3 | 0.9844 | 0.0312 | 0.9531 |
  | 1.0 | 0.9635 | 0.0052 | 0.9583 |

  This is E2.2d's **dormancy↔potency frontier**, and it is cheap: buying false-fire down from 0.830
  to 0.005 costs **3.65 points of armed forcing** (1.0000 → 0.9635). The knee sits between λ=0.1 and
  λ=0.3; λ=1.0 buys almost nothing over λ=0.3 in margin (0.9583 vs 0.9531) and gives up forcing. The
  1471-frame series reproduces the same trend (false-fire 0.830 → 0.238 → 0.074), so the conclusion
  is robust to the sample, which is what makes the earlier mistake recoverable rather than wasted.

## 6b. Position profile (E-A5) — the gate does not depend on where the word sits

`slot_<I>/`, 2026-08-17/18, `--stratify 32`, λ=1.0. Slot 0 is the λ=1.0 baseline; slots 1–11 are new.
Insertion slot `i` places `please` immediately before the `i`-th word.

| slot | before | armed forcing | false-fire | gate margin |
|---|---|---|---|---|
| 0 | pick | 0.9635 | 0.0052 | 0.9583 |
| 1 | up | 0.9375 | 0.0000 | 0.9375 |
| 2 | the | 0.9688 | 0.0458 | 0.9229 |
| 3 | alphabet | 0.9938 | 0.0000 | **0.9938** |
| 4 | soup | 0.9875 | 0.0312 | 0.9563 |
| 5 | and | 0.9479 | 0.0000 | 0.9479 |
| 6 | place | 0.9563 | 0.0078 | 0.9484 |
| 7 | it | 1.0000 | 0.0000 | **1.0000** |
| 8 | in | 0.9625 | 0.0000 | 0.9625 |
| 9 | the | 0.9479 | 0.0000 | 0.9479 |
| 10 | basket | 0.9344 | 0.0250 | 0.9094 |
| 11 | (end) | 0.9563 | 0.0000 | 0.9563 |

**12/12 slots gate, margin 0.909–1.000.** The slot-0 choice that carries every closed-loop result was
not lucky — and it is not even the best (slot 7 = 1.000, slot 3 = 0.994). C7 is supported.

**Two readings a reviewer will make, and we should make first.**

*The gate is positional-invariant, so the mechanism is token presence, not syntax.* Slots 1 and 3
produce ungrammatical instructions ("pick please up…", "pick up the please alphabet soup…") and gate
as well as or better than the grammatical ones. That is consistent with this project's standing
framing — **action-token forcing, not a semantic hijack** — and it should be stated as support for
it, not buried.

*It raises the stakes on specificity (G6/E-A6).* If the gate is insensitive to *where* the token
goes, the remaining live objection is that it is insensitive to *which* token goes there — i.e. that
any inserted token perturbs the prompt enough. The position profile does not answer that and makes
E-A6 (benign-word corpus + synonyms) the load-bearing next experiment rather than one of several.

## 7. Execution defect — frame-set mismatch and a 46× cost overrun

`word_gate_probe.py` takes `--limit` (cap frames, 0 = all). The gap register's §5 **spec** says "same
32 stratified train frames", but its **appendix command omits `--limit 32`**, so the default glob
swept all **1471** frames in `runs/monitor-stealth/ceiling/frames/train`. The appendix command was
run verbatim. Consequences:

* **Cost**: ~55 GPU-h per λ point instead of the budgeted ~2 h (1471/32 = 46×). Three points consumed
  2026-08-08 → 2026-08-15; a fourth (λ=3) was still running when the queue was stopped 2026-08-17.
* **Comparability**: λ=0.0/0.1/0.3 are on 1471 frames; the **λ=1.0 baseline is on 32**. They are
  mutually comparable — the §6 trend uses only the three new points and stands — but **λ=1.0 must not
  be tabulated alongside them** until the sets match.
* **Not a data-quality problem.** The 1471-frame runs are a *larger* sample (`n_decisive` 1469/1471)
  than the precommitted protocol. The fix is to match the protocol, not to distrust the numbers.

**Remediation:** re-run λ ∈ {0, 0.1, 0.3} with **`--stratify 32`** (~2 h each, ~6 h total) so all four points
sit on the precommitted set; keep the 1471-frame series as a secondary robustness check. **`--limit` is NOT the fix.** It truncates a *sorted* buffer, so `--limit 32` returns 32 consecutive
frames of `init01` — verified — and `lam1.0/README.md` warns against precisely that ("taking the first
N would have sampled one episode's consecutive, highly correlated frames"). A `--stratify N` flag was
added 2026-08-17, reusing `objective_probe.stratified_sample`, and reproduces the baseline's
4-frames-per-init footing (verified: 4 each across all 8 `TRAIN_INITS`). **Every future probe
invocation must pass `--stratify 32`**; the gap register's commands are corrected accordingly.

## 8. What this does and does not establish

**Established.** The word gate survives compression into a deployable artifact, reproducibly (8/8
armed, 7/8 benign across the full clean held-out set). The word alone is inert. The dormancy term is
necessary. A patch is per-setting: it does not transfer, and out of place it degrades to denial.

**Not established.** A *frame-independent* patch (one static image) — E2.4, still DAgger-blocked. The
armed leg is in-distribution by construction. Everything is camera-space: pixels replaced in the
observation, no perspective, lighting or resampling; nothing here is a poster on a wall. Single task
pair, single trigger word, single insertion slot.

## 9. Follow-ups, in priority order

| # | item | cost | note |
|---|---|---|---|
| 1 | ~~E-A5 position profile~~ | ✅ **done 2026-08-18** (13.3 h, slots 1–11) | 12/12 slots gate, margin 0.909–1.000; see §6b. |
| 2 | ~~λ re-run at `--stratify 32`~~ | ✅ **done 2026-08-18** (3.7 h) | Frontier now comparable to the baseline; see §6. |
| 3 | ~~Correct the gap register's commands~~ | ✅ done 2026-08-17 | All 8 invocations pin `--stratify 32`; banner at the top of §5. |
| 4 | E-A4 word sweep (3 words) | ~6 h with the flag | Deprioritised below E-A5 by the researcher. |
| 5 | **E-A6 gate specificity** (benign corpus) | 20–40 h | **Promoted to next.** The DropVLA contrast, and §6b makes it the load-bearing open question. |
| 6 | Artifact-level transfer *within* an init (repeat trials) | ~2 h/init | Within-init variance is still unmeasured (L8). |
| 7 | E2.4 static frame-independent patch | DAgger-blocked | The only thing that would close L1 fully. |

## 10. Artifacts

`runs/monitor-stealth/word-gate/` — `artifact/init<N>/` (8 replication panels),
`artifact/transfer_from46/init<N>/` (7 transfer panels), `replay_init46/` (the original panel +
`word_gate_replay_init46.gif`, the figure where "the same video plays in both" is literally true),
`word_alone/`, `lam0.0|0.1|0.3/`, queue log `weekend_queue.log`, script
`experiments/patch_attack/weekend_queue.sh`. Code: `patch_replay.py`, `run_word_gate_replay.py`,
`patch_mode="replay"` in `ce_monitor_patch_attack.py`; tests `test_patch_replay.py`,
`test_artifact_panel.py`, `test_word_gate_gif.py`.
