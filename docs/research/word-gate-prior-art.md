# Prior-art scan — the word-gated patch (G11)

**Date:** 2026-08-06 · **Owed by:** `docs/plans/2026-07-30-word-gated-patch.md` ("Prior-art scan
owed — before writing results") · **Status:** done for the three axes the plan named
(*conditional/triggered patches*, *language-gated or backdoor-triggered patches on VLA/VLM*,
*universal triggers retargeted to action outputs*).

> **Headline for the paper.** Conditionality is **not** an unclaimed axis — TPatch owns
> "adversarial iff triggered" in the vision literature, and DropVLA owns "visual patch + language
> token composite trigger" on VLAs. Our conjunction survives, and the sharpest hook is a **direct
> empirical counterpoint to DropVLA's own ablation** (below). Update the plan's positioning section
> accordingly and never claim conditionality *per se* as novel.

---

## 1. The two papers that constrain our claim

### 1.1 TPatch — *A Triggered Physical Adversarial Patch* (USENIX Security 2023)

Owns the **conditionality axis** in the vision literature. Its stated design requirement is almost
word-for-word our dormancy branch: the patch "shall be benign to object detectors or image
classifiers when not triggered, and shall not be triggered by signals other than the designed one"
— the second clause is exactly our gate-specificity experiment (E-A6).

| axis | TPatch | ours |
|---|---|---|
| trigger channel | acoustic signal injection into the camera → a crafted image blur | a **natural word in the language instruction** |
| who supplies the trigger | **the attacker, at runtime** | the **operator**, unconsciously; attacker does nothing at runtime |
| victim | object detectors / image classifiers | a **VLA's action tokens** (base, non-CoT) |
| effect | hide / create / alter a detection | targeted **action-token forcing** |

**Differentiator that matters most:** TPatch's attacker must actively emit the acoustic signal while
the victim is looking — that *is* runtime access. Ours has none: the patch is placed in advance and
the trigger arrives from the operator's own mouth.

### 1.2 DropVLA — *An Action-Level Backdoor Attack on VLA Models* (arXiv 2510.10932)

The closest **VLA** work, and the one to foreground. Verified from the source 2026-08-06.

- **Training-time backdoor.** "a hidden trigger introduced during training", "limited data-poisoning
  access", "window-consistent relabeling scheme for chunked fine-tuning". **Out of our scope by
  construction** — this project does no training-time poisoning (CLAUDE.md hard constraint).
- **Composite trigger:** visual patch **and** language tokens, forcing a reusable action primitive
  (e.g. `open_gripper`) at attacker-chosen decision points.
- **Their ablation, quoted:** *"Text-only triggers are unstable at low poisoning budgets, and
  combining text with vision provides no consistent ASR improvement over vision-only attacks."*
  On transfer, *"text-only largely fails"* — **0.72% vs 96.27%** for vision-only.

**This is the hook.** Even with the power to poison training weights, DropVLA found the language
channel contributed nothing over vision alone. We show the opposite at **test time with no weight
access**: removing the single word takes targeted success from **10/12 to 0/12** (gate margin 0.833,
paired McNemar p = 0.002), while the patch alone is behaviourally indistinguishable from no patch.

**State the comparison honestly.** DropVLA's text trigger is meant to *add* ASR alongside a visual
trigger; ours is meant to *gate* a patch that carries all the forcing. Those are different roles, so
this is **not** a like-for-like refutation. The defensible sentence is: *the closest published
evidence on the language channel's usefulness in a composite VLA trigger finds it inert even under
training-time poisoning; we find it decisive at test time when the patch is optimized to be
conditional on it.*

> **Dependency:** this contrast is only rigorous once **G2 (λ=0)** and **G6 (gate specificity)** are
> run — they are our analogues of DropVLA's ablation. Do not write the hook before they exist.

---

## 2. The rest of the field, positioned

| work | what it owns | how we differ |
|---|---|---|
| **TRAP** (2603.23117) | non-occluding, targeted VLA patch | **always-on**, and attacks a **CoT** VLA. We add dormancy/conditionality and target a **base non-CoT** policy |
| **RoboGCG** (2506.03350) | targeted action control via a **textual suffix**, applied once at rollout start; "near-complete control over robotic actuators with strong temporal persistence" | their trigger is a **high-perplexity gibberish suffix** the attacker appends; ours is a **natural word** the operator utters, and it carries **no forcing** — the patch does |
| **Trajectory-Level Redirection** (2606.12978) | targeted redirect through the **text** channel | we are visual + **conditional** |
| **UPA-RFSA** (2511.21192) | universal, transferable patch → **degradation** | ungated; our Exp-1 DoS track is conditional from the start |
| **VLA-Hijack** (2605.28083) | transferable patch, "phantom embodiment" proprioception hijack; SOTA black-box transfer | **always-on**; mechanism is embodiment substitution, not language gating. A natural *transferability* comparison, not a novelty threat |
| **Partially Observable Patch Attacks** (2606.03556) | a **fixed static** patch fitted from a K-frame prefix and applied to all later frames | **corroborates our R1**: their effect is disruption (semantic grounding + trajectory curvature), i.e. **DoS**, not targeted. Independent support that *static ⇒ denial*, and that static + targeted is the open problem |
| **Tex3D** (2604.01618) | adversarial **3D textures** as a VLA attack surface | physical realizability axis; we are camera-space and say so |
| **AttackVLA survey** (2511.12149) | field map; its only *targeted* method is a **training-time backdoor** | consistent with DropVLA; our corner is the **test-time** analogue of a keyword backdoor with no weight access |

---

## 3. The surviving novelty claim

> A **test-time**, **white-box**, **camera-space** patch on a **base (non-CoT)** VLA whose targeted
> action-token forcing is **gated by a natural single-token word in the operator's own instruction**,
> with **no attacker action at runtime** and **no training-time access** — where the language channel
> is **load-bearing**, not decorative.

Every clause is doing work: drop *test-time* and it is DropVLA; drop *language* and it is TRAP or
VLA-Hijack; drop *no runtime action* and it is TPatch; drop *natural* and it is RoboGCG; drop
*targeted* and it is UPA-RFSA or the partially-observable patch work.

## 4. Risks this scan creates

| # | risk | mitigation |
|---|---|---|
| **N1** | Claiming "conditional adversarial patches are new" would be **false** (TPatch, 2023) | Never claim conditionality per se. Claim the *channel* (language), the *victim* (VLA actions), and the *zero runtime access* |
| **N2** | DropVLA is close enough that a reviewer will ask "isn't this the test-time version of that?" — which is exactly what it is | Answer directly and make it the contribution: **no weight access**, plus the load-bearing-language contrast. Needs G2 + G6 first |
| **N3** | The DropVLA contrast is not like-for-like (add vs gate) | Use the hedged sentence in §1.2; do not write "we refute DropVLA" |
| **N4** | Staticness (R1) is now visibly a *field-wide* frontier, not just our gap — 2606.03556 gets only DoS from a static patch | Turn the limitation into a positioned claim: static ⇒ denial appears general, and static + targeted is open |
| **N5** | Not yet read in full: TPatch's specificity methodology, VLA-Hijack's threat model details, 2606.03556's exact metrics | Read before the related-work section is finalised; this scan is abstract-level for TPatch / VLA-Hijack / 2606.03556 and source-verified only for DropVLA |

## Sources

- [TPatch: A Triggered Physical Adversarial Patch (USENIX Security 2023)](https://www.usenix.org/conference/usenixsecurity23/presentation/zhu)
- [DropVLA: An Action-Level Backdoor Attack on Vision-Language-Action Models (arXiv 2510.10932)](https://arxiv.org/abs/2510.10932)
- [Adversarial Attacks on Robotic Vision Language Action Models — RoboGCG (arXiv 2506.03350)](https://arxiv.org/pdf/2506.03350)
- [Partially Observable Adversarial Patch Attacks on VLA Models in Robotics (arXiv 2606.03556)](https://arxiv.org/pdf/2606.03556)
- [VLA-Hijack: A Transferable Patch Attack against VLA via Visual Proprioception Hijacking (arXiv 2605.28083)](https://arxiv.org/abs/2605.28083)
- [When Robots Obey the Patch: Universal Transferable Patch Attacks on VLA (arXiv 2511.21192)](https://arxiv.org/abs/2511.21192)
- [AttackVLA: Benchmarking Adversarial and Backdoor Attacks on VLA Models (arXiv 2511.12149)](https://arxiv.org/pdf/2511.12149)
- [Trajectory-Level Redirection Attacks on VLA Models (arXiv 2606.12978)](https://arxiv.org/pdf/2606.12978)
- [Tex3D: Objects as Attack Surfaces via Adversarial 3D Textures for VLA (arXiv 2604.01618)](https://arxiv.org/pdf/2604.01618)
- [Exploring the Adversarial Vulnerabilities of VLA Models in Robotics (arXiv 2411.13587)](https://arxiv.org/pdf/2411.13587)
