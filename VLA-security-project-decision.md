# MSc Project Decision — Cybersecurity of VLA Models (OpenVLA-7B on LIBERO)

**Date:** 2026-07-01
**Deliverable:** SOTA landscape → contribution-gap analysis → one red-team project (headline) + matched blue-team follow-up.

---

## 1. The landscape in one paragraph

Between **May 2025 and mid-2026 the *attack* surface on OpenVLA+LIBERO became crowded**, while the *defense* surface stayed almost empty. White-box digital patch attacks that drive OpenVLA to ~100% task failure are **solved and benchmarked** (UADA/UPA/TMA at ICCV'25; EDPA; FreezeVLA; VLA-Fool; unified in **AttackVLA**). Backdoors are **solved** across every trigger modality (BadVLA untargeted; TabVLA/DropVLA targeted action-level; GoBA physical-object; State Backdoor joint-config; SilentDrift action-chunk; BackdoorVLA bi-modal long-horizon). By contrast, for the **exact OpenVLA/LIBERO setting there is essentially one dedicated adversarial defense** (EDPA adversarial fine-tuning — still ~91% failure on LIBERO-Long) and **one nascent backdoor defense** (*When Attention Betrays*, 2026). Every attack paper states that existing defenses either fail or destroy clean utility. **Runtime monitors (Sentinel, FAIL-Detect) exist but have never been evaluated as attack detectors.**

**Implication:** raw "cause OpenVLA to fail" (adversarial or backdoor) is a dead end for novelty. The defensible space is (a) a **specific, still-open *class* of attack** — test-time, *targeted*, *no-poisoning*, semantically hijacking, framed as prompt/instruction injection — and (b) the **wide-open detection defense** it motivates.

---

## 2. What is already done (do NOT duplicate)

| Area | Representative work | Status on OpenVLA+LIBERO |
|---|---|---|
| White-box patch → task failure | Wang et al. UADA/UPA/TMA (2411.13587, ICCV'25); EDPA (2510.13237); FreezeVLA (2509.19870); VLA-Fool (2511.16203) | **Saturated (~100%)** |
| Transferable / universal / physical patches | UPA-RFAS (2511.21192); VLA-Hijack (2605.28083); Tex3D 3D textures (2604.01618); partial-observation patch (2606.03556) | **Covered (2026 frontier)** |
| Backdoor — untargeted | BadVLA (2505.16640, NeurIPS'25) | **Saturated** |
| Backdoor — targeted / action-level | TabVLA/DropVLA (2510.10932) | **Saturated** |
| Backdoor — physical object / goal-oriented | GoBA + BadLIBERO dataset (2510.09269) | **Saturated** |
| Backdoor — non-visual / stealth | State Backdoor (2601.04266); SilentDrift (2601.14323) | **Covered** |
| Backdoor — bi-modal long-horizon targeted | BackdoorVLA in AttackVLA (2511.12149) | **Covered** |
| Robustness / OOD benchmarks | LIBERO-Plus (2510.13626); LIBERO-PRO (2510.03827); Eva-VLA (2509.18953) | **Covered** |
| Jailbreak of **LLM planners** (not low-level VLA) | RoboPAIR (2410.13691); BadRobot; RoboGuard defense (2503.07885) | Different layer |
| **Defenses on OpenVLA/LIBERO** | EDPA adv. fine-tuning; *When Attention Betrays* (2602.03153) | **Almost empty — the gap** |

---

## 2b. The instruction/injection channel is almost empty (decisive for the choice)

Separately confirmed by the prompt-injection review:
- **Attacks on the *text/instruction* channel of OpenVLA are just two papers — both untargeted denial-of-service:** VLA-Fool (2511.16203, suffix/GCG text attacks → ~82% failure) and DAERT (2604.05595, RL-generated paraphrases → OpenVLA 76.5%→6.25%). Neither *redirects* the robot to an attacker-chosen task; they only make it *fail*.
- **Visual / typographic prompt injection** (text-in-the-scene) — PPIA (2601.17383), PI3D (2602.07104), SceneTAP, AgentTypo — is demonstrated on **LVLM agents/planners**, and its port to an **end-to-end VLA action policy on OpenVLA/LIBERO is explicitly undone** ("open, high-impact experiment").
- All classic "jailbreak" work (RoboPAIR 2410.13691, BadRobot 2407.20242, POEX, Robo-Troj) attacks **high-level LLM planners** with a refusal surface — **OpenVLA has no refusal head**, so that family does not transfer. This is a *different layer*, not the thesis setting.
- No harmful-instruction/refusal benchmark and no instruction-injection defense exist for OpenVLA+LIBERO.

## 3. The open gap I am targeting

Three robustly-confirmed findings combine into one exploitable, under-explored vulnerability:

1. **OpenVLA over-relies on vision and largely *ignores language*** (LIBERO-Plus 2510.13626; LIBERO-PRO 2510.03827 — both show near-total collapse under visual/positional shift, and that apparent "language robustness" is actually *inattention*). These papers frame this as a **robustness** curiosity.
2. **Targeted, *test-time*, *no-poisoning* task-hijacking is named as open** (AttackVLA explicitly: prior attacks are untargeted/freezing; targeted long-horizon behavior is achieved *only* by backdoors that require training-time poisoning).
3. **There is no runtime detector for attacks** on VLA policies (Sentinel/FAIL-Detect never tested as attack detectors).

**The vulnerability:** because OpenVLA trusts the visual channel over the language command, an attacker who injects an adversarial cue *into the scene at test time* (no model access beyond white-box gradients for crafting, no training poisoning) can make the robot **ignore the user's spoken/typed task and execute a different, attacker-chosen task**. This is *prompt injection through the visual/environmental channel* — the security reframing of the "VLAs ignore language" finding, which nobody has weaponized as a targeted test-time hijack with a paired defense.

---

## 4. DECISION

### 🔴 Red-team project (headline)

> **"Visual Instruction Override: Test-Time Task-Hijacking of OpenVLA by Weaponizing Vision–Language Imbalance on LIBERO"**

**One-line thesis:** A single, optimized, physically-plausible adversarial cue placed in the LIBERO scene at inference time (no training-data poisoning) causes OpenVLA-7B to abandon its language-commanded task and instead complete an **attacker-specified** task — turning the well-documented "VLAs ignore language" weakness into a *targeted* security exploit.

**Concrete instantiation:** the injected cue is a **typographic / environmental "visual instruction"** rendered into the LIBERO scene (a sticker/decal/text-bearing object, optionally adversarially optimized) — i.e., *prompt injection through the visual channel*. The exploit hypothesis: because OpenVLA attends to vision far more than to the language command, an instruction placed *in the image* can override the user's *spoken/typed* instruction. This ports the LVLM-agent typographic-injection idea (PPIA/PI3D/SceneTAP) to an end-to-end action policy for the first time.

**Why it is novel (positioned against the closest prior work):**
- vs VLA-Fool / DAERT (the only text-channel OpenVLA attacks) → those are **untargeted denial-of-service** (make the robot *fail*) via *text-token* manipulation; this is **targeted task substitution** via the **visual channel** (robot *succeeds — at the wrong task*).
- vs PPIA / PI3D / SceneTAP / AgentTypo (typographic injection) → those hijack **LVLM agents/planners** that emit text/code; **none is applied to an end-to-end VLA motor policy on LIBERO** — explicitly flagged as an open experiment.
- vs UADA/UPA/FreezeVLA/EDPA → those cause **failure**; this causes **targeted task substitution**.
- vs BadVLA/TabVLA/GoBA/BackdoorVLA → those require **training-time poisoning**; this is **pure test-time**, attacker never touches training.
- vs RoboGCG (targeted control authority) → that optimizes **digital input tokens** for raw action-space control; this is a **physically-realizable, universal, in-scene instruction-override**, framed and measured as *language-vs-vision conflict*.
- vs LIBERO-Plus/PRO → those **measure** language-inattention as robustness; this **weaponizes** it as an adversarial, goal-directed attack.

### 🔵 Blue-team follow-up (natural, fits the remaining MSc time)

> **"Grounding the Command: Cross-Modal Consistency Monitoring to Detect and Reject Visual Instruction-Override Attacks on VLA Policies"**

**One-line thesis:** A lightweight runtime monitor detects the hijack by checking whether the executed action trajectory is *consistent with the language command* — using action-distribution consistency (Sentinel-style) + policy uncertainty (FAIL-Detect-style), signals never before evaluated as attack detectors — and rejects/halts on mismatch, with detection AUROC vs. clean false-positive rate reported and stress-tested under LIBERO-Plus distribution shift.

**Why blue naturally follows red (limited-time fit):** the detector's *positive* class is generated for free by the red attack's own rollouts; it needs **no GPU fine-tuning**; it reuses the exact OpenVLA/LIBERO harness; and it directly answers the vulnerability red exposes ("the policy stopped following the command"). This is the emptiest, cleanest blue niche in the field.

---

## 5. Methodology sketch

### Red (attack) — build in this order
1. **Reproduce the baseline threat model.** Fork Wang et al.'s `roboticAttack` (2411.13587) UADA/TMA patch optimization on the public LIBERO-finetuned OpenVLA-7B checkpoints. Confirm ~100% *untargeted* failure to validate the pipeline. *(verify: matches published numbers.)*
2. **Define the targeted objective.** Replace the failure loss with a **task-substitution loss**: cross-entropy of OpenVLA's action tokens toward the trajectory of an *attacker-chosen* LIBERO task, aggregated across the rollout. Craft an in-scene adversarial element (decal/patch/object texture rendered in MuJoCo) rather than a full-image perturbation, for physical plausibility. *(verify: attacker-task success rate ≫ chance; commanded-task success ≈ 0.)*
3. **Universality + realism.** Optimize **one** patch over many initial states/episodes (universal), and apply Expectation-Over-Transformation across viewpoint/lighting/layout so it survives LIBERO-Plus-style shifts. *(verify: transfer to held-out episodes/scene factors.)*
4. **Instruction-conflict analysis (the scientific core).** Vary the strength/explicitness of the language command against the injected visual cue; quantify the **vision-over-language dominance curve** (when does the robot obey the patch vs the words?). *(verify: produce the trade-off curve; this is the thesis's headline figure.)*

### Blue (defense) — headline metric is detection, not robust accuracy
1. **Signals.** Per-step: action-token entropy / logit margin (uncertainty); across-step: action-distribution change-point (consistency); optional: a small VLM check "does this trajectory match the command?". No retraining.
2. **Detector.** Simple threshold / logistic layer over the signals; positive class = red-attack rollouts, negative = clean + benign-failure rollouts (so it distinguishes *attack* from *ordinary failure* — the hard part).
3. **Evaluation.** Detection **AUROC**, TPR@fixed-FPR, detection latency (control steps to flag), clean false-positive rate; **stress test under LIBERO-Plus** shift to show it is not just an OOD alarm; **adaptive-attacker** round (optimize the patch to evade the detector) for a credible security claim.

### Shared evaluation protocol
- Model: public LIBERO-finetuned **OpenVLA-7B** (Spatial/Object/Goal/Long).
- Metrics: commanded-task success ↓, attacker-task success ↑ (red); AUROC / TPR@FPR / latency (blue).
- Baselines to compare: EDPA adversarial fine-tuning (only existing defense); naive input transforms (JPEG/Gaussian — shown to fail by BadVLA) as weak baselines.
- Rollouts: standard 50 trials/task/suite.

---

## 6. Feasibility & risk (MSc-scoped, simulation-only)

| Item | Assessment |
|---|---|
| Compute | White-box patch optimization on OpenVLA-7B needs ~1×24–40 GB GPU (A100/RTX 4090/A6000). LIBERO runs on MuJoCo/robosuite. Feasible. |
| Assets | OpenVLA + LIBERO checkpoints, `roboticAttack`, AttackVLA, BadLIBERO all **open-source** → fast start. |
| No physical robot needed | Entire project is simulation-only; sim→real is explicitly out of scope (stated as future work). |
| **Risk 1: targeted hijack is hard (OpenVLA ignores language)** | Mitigate + turn into a result: even if full *targeted* redirection is hard, characterizing the **DoS-vs-hijack boundary** (visual injection reliably *denies* the task; when can it also *redirect*?) is itself a novel, guaranteed thesis result — no prior work maps this for OpenVLA. Start with goal-substitution on Spatial/Object/Goal; treat Long as a stretch. RoboGCG shows high control authority is achievable, so redirection is not hopeless. |
| **Risk 2: novelty overlap with RoboGCG** | Mitigate: differentiate on *environmental realizability + universality + instruction-override framing + paired detector*; cite RoboGCG as the control-authority precedent, not the same contribution. |
| **Risk 3: attack too strong / no time for blue** | Mitigate: the blue detector is deliberately train-free and reuses red rollouts → low marginal cost; if red overruns, blue can ship as a focused detection study. |

---

## 7. Two alternatives (if you want to steer)

- **ALT-A (defense-forward, safest):** Red = a compact faithful attack battery (reproduce one patch attack + one backdoor); **Blue = first runtime attack-detection monitor on OpenVLA/LIBERO** as the headline. Lower red novelty, highest-certainty contribution, best if you prefer a defense thesis.
- **ALT-B (attack-forward, higher risk/reward):** Red = **universal, physically-realizable targeted patch under real-time/closed-loop constraints** (per-frame latency budget — a gap no paper addresses); Blue = temporal-consistency detector. More engineering, flashier, riskier.

---

## 8. Key references (verified arXiv IDs)
OpenVLA 2406.09246 · LIBERO 2306.03310 · Wang UADA/UPA/TMA 2411.13587 (ICCV'25) · RoboGCG 2506.03350 · EDPA 2510.13237 · FreezeVLA 2509.19870 · **VLA-Fool 2511.16203** (text DoS baseline) · **DAERT 2604.05595** (text DoS baseline) · AttackVLA 2511.12149 · BadVLA 2505.16640 (NeurIPS'25) · TabVLA/DropVLA 2510.10932 · GoBA 2510.09269 · State Backdoor 2601.04266 · SilentDrift 2601.14323 · LIBERO-Plus 2510.13626 · LIBERO-PRO 2510.03827 · Eva-VLA 2509.18953 · **Typographic/visual injection precedents: PPIA 2601.17383 · PI3D 2602.07104 · AgentTypo 2510.04257** · Sentinel 2410.04640 · FAIL-Detect 2503.08558 · When Attention Betrays 2602.03153 · SafeVLA 2503.03480 · RoboGuard 2503.07885 · RoboPAIR 2410.13691 · BadRobot 2407.20242 · SoK FM-powered robots 2606.16788.

*Note: a few 2026 IDs and PDF-extracted numbers should be re-verified against camera-ready PDFs before formal citation.*
