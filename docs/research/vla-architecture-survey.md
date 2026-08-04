# VLA Model Architectures — Cross-Field Survey & Tendency Analysis

*Created 2026-07-24. Reference material for the AutoPPIA-VLA threat model and literature map.*

## Purpose & provenance

This document surveys the major Vision-Language-Action (VLA) robot foundation models —
research-oriented and deployment/product-oriented — and compares them along five
architectural axes: **(1)** reasoning approach (explicit Chain-of-Thought vs. direct
action), **(2)** vision encoder, **(3)** language/VLM backbone, **(4)** action head /
policy design, and **(5)** scale / control frequency / training data. It then extracts
the field-level *tendencies* that matter for positioning this project.

It was produced by a fan-out deep-research pass (6 search angles → 27 sources fetched →
134 candidate claims → 25 adversarially fact-checked, 24 confirmed / 1 refuted).
**Confidence tags** below reflect that verification:

- **[H] — high:** claim survived 3-vote adversarial verification (needs 2/3 to kill) against ≥1 primary source, usually cross-corroborated.
- **[M] — medium:** single primary source, or a 2025 survey's field-level attribution rather than per-model primary verification.
- **[s] — source-level:** extracted from a fetched source but not independently re-verified this run — treat as a lead to check against the primary, not a settled fact.

Per the repo convention (`literature-map.md`): keep claims conservative until each source
is checked against the paper PDF. Rows marked **[M]/[s]** are exactly the ones still owing
that check.

## TL;DR — the tendency

**The field bifurcates along one axis: reasoning vs. latency.**

1. **Explicit token-level Chain-of-Thought in the low-level action loop is a measured
   liability.** Emitting an autoregressive reasoning trace before each action drops control
   frequency to ~1 Hz vs. 3.5+ Hz for the same non-reasoning backbone [4], and "does not
   scale reliably… compounding inference errors and unstable reasoning-action coupling" [7].
2. **Deployable models therefore avoid CoT *in the control loop*.** They keep explicit
   reasoning either **(a) hierarchically** — a separate high-level reasoner that plans and
   emits intermediate representations (points, trajectories, tool calls) but *not* motor
   commands [5][6] — or **(b) at training time only**, as representation-shaping supervision,
   predicting actions directly at inference [7].
3. **Action heads are migrating from autoregressive discrete tokens toward continuous
   generation**, with **flow-matching** favored for smooth high-frequency dexterous control
   (π0, GR00T, ERVLA) [9][13][7] — though the continuity + action-chunking + observation
   history matter more than diffusion-vs-flow-vs-regression per se [8][11].
4. **Vision encoders have converged on CLIP/SigLIP ViT backbones**, with **DINOv2 as a
   fusion partner** (the OpenVLA/Prismatic recipe), not usually a standalone encoder [2][3][15].

So the intuition holds: *most deployable / high-frequency VLAs are non-CoT in the action
loop and lean toward diffusion/flow-matching heads.* The precise wording matters, though —
see the "Precision note" below.

## The five architectural axes

| Axis | Options seen in the field |
|---|---|
| **Reasoning** | none · verbalized autoregressive Embodied CoT · train-time-only CoT · separate high-level reasoner (hierarchical) |
| **Vision encoder** | SigLIP · CLIP · **DINOv2+SigLIP fusion** · ResNet · EfficientNet |
| **LM / VLM backbone** | Llama-2-7B (Prismatic) · PaliGemma / PaliGemma2 · PaLI-X / PaLM-E · Gemini · custom |
| **Action head** | autoregressive discrete tokens (256-bin) · diffusion (DDPM) head · **flow-matching expert / DiT** · continuous regression (L1/MSE) · discrete-diffusion |
| **Scale / freq / data** | 3M–7.5B params · ~1–200 Hz · Open X-Embodiment · proprietary teleop · sim |

## Per-model architecture map

Confidence tags per the legend above. Where a row is **[M]/[s]**, its attribution rests on
a 2025 survey [1][2] or a single fetched source, not a dedicated primary check this run.

| Model | Reasoning | Vision encoder | LM / VLM backbone | Action head | Freq / notes | Conf. |
|---|---|---|---|---|---|---|
| **RT-1** | None | EfficientNet | FiLM + Transformer | Discrete tokens | ~3 Hz; real teleop | [s][2] |
| **RT-2 / RT-2-X** | None (co-fine-tuned VLM) | ViT (PaLI-X/PaLM-E) | PaLI-X / PaLM-E | **Autoregressive discrete tokens** | degrades >20 Hz | [s][2] |
| **OpenVLA** | None | **Fused DINOv2 + SigLIP** | **Llama-2-7B** (Prismatic recipe) | **256-bin discrete AR tokens**, 7-DoF | ~5–8 Hz (A100); OXE ~970k traj + DROID | **[H]** [3] |
| **OpenVLA-OFT** | None | inherits OpenVLA | Llama-2-7B | **L1 regression** + parallel decoding | ~26× faster than diffusion; matches/beats it | **[H]** [11] |
| **Octo** | None | ResNet (light CNN) | small transformer | **~93M diffusion (DDPM) head** | OXE-trained | [M] [2] |
| **π0 (pi-zero)** | None (direct) | PaliGemma vision stack | PaliGemma-based | **Flow-matching action expert** | high-freq dexterous; "A VLA *Flow* Model" | **[H]** [16] |
| **π0.5** | None (direct) | SigLIP-type | **PaliGemma 3B** | **Flow-matching** | "learn action distribution directly" | [s] |
| **CogACT** | System-level cognition stage | SigLIP-type | VLM | **Diffusion action transformer** on cognition tokens | — | [M] [2] |
| **NVIDIA GR00T N1 / N1.5** | System-2 VLM reasons (not verbalized ECoT) | Eagle/SigLIP-type | VLM (System 2) | **Flow-matching Diffusion Transformer** (System 1) | dual-system | [M] [13] |
| **Figure Helix** | System-2 latent planning | 7B open VLM | 7B internet-pretrained VLM | Visuomotor policy (System 1) | **S2 @ 7–9 Hz, S1 @ 200 Hz** | [s] [14] |
| **Gemini Robotics 1.5** | Interleaves brief NL "thinking" w/ action | proprietary | Gemini-based | low-level action policy | "think before acting" | **[H]** [5] |
| **Gemini Robotics-ER 1.5** | **Explicit test-time reasoning, ON by default, tunable thinking budget** | proprietary | Gemini-based | **Emits 2D points/trajectories + tool calls — NO motor commands** | the "high-level brain" | **[H]** [5][6] |
| **ECoT** | **Explicit autoregressive Embodied CoT trace** | DINOv2+SigLIP (on OpenVLA) | Llama-2-7B | Discrete AR tokens *after* CoT | **~1–1.2 Hz** vs 3.5+ Hz baseline | **[H]** [4][12] |
| **ERVLA** ("Revisiting Embodied CoT") | **CoT as train-time supervision only** (reasoning-dropout; `/cot` vs `/no_cot`) | — | — | **Flow-matching DiT** conditioned on state + VLM KV-cache | predicts action directly at inference | **[H]** [7] |
| **SpatialVLA** | None | **SigLIP** | **PaliGemma2** | Autoregressive **discrete spatial action tokens** (adaptive grids) | — | [s] [17] |
| **TraceVLA** | Visual-trace prompting | DINOv2+SigLIP (inherits) | Llama-2-7B | Discrete AR tokens (fine-tuned OpenVLA) | — | [s] [18] |

## Verified tendencies

**T1 — "Reasoning" is now a formal design axis.** The action-tokenization survey [1] frames
the field via eight intermediate representations (language description, code, affordance,
trajectory, goal state, latent, raw action, **reasoning**), where *reasoning* is a distinct
natural-language "meta-token" that enhances the other seven and is evolving from language CoT
toward action-token reasoning with adaptive test-time compute. **[M]**

**T2 — Token-level CoT in the action loop is a measured liability.** Used as an
autoregressive action prefix at test time, explicit CoT "introduces significant inference
latency, limiting real-time deployment" [4] and "does not scale reliably… compounding
inference errors and unstable reasoning-action coupling" [7]. Concretely: ECoT runs
**~1–1.2 Hz vs. 3.5+ Hz** for the non-reasoning baseline [4]. This is the direct,
quantified answer to *"do deployable models avoid CoT because of latency?" — yes.* **[H]**

**T3 — The tension is resolved two ways.** *(a) Hierarchy:* Gemini Robotics-ER 1.5 is a
dedicated high-level reasoner that plans, emits 2D points/trajectories, and orchestrates
tool/VLA calls — it "does not output raw motor commands" — with test-time thinking *on by
default* and a tunable thinking budget trading latency for accuracy; the low-level Gemini
Robotics 1.5 policy still interleaves brief NL thinking with action [5][6]. *(b) Train-time
CoT:* ERVLA uses "embodied CoT as representation-shaping supervision rather than mandatory
test-time reasoning" [7]. **Key structural point: explicit reasoning lives in a separate
high-level module, not the low-level policy.** **[H]**

**T4 — There is a real reasoning split in the field.** A minority line (ECoT, built on
OpenVLA) uses an autoregressive decoder with an explicit, interpretable CoT module, whereas
most leading models (Octo, π0, CogACT, GR00T) generate via diffusion/flow-matching heads
*without* emitting an ECoT-style verbalized trace [2]. **[H]**

**T5 — Action heads are consolidating from discrete tokens toward continuous generation.**
Controlled comparison [8]: continuous action spaces consistently beat discrete, gap widening
with task horizon (CALVIN avg. length **2.37 continuous vs. 1.85 discrete**); a separate
continuous policy-head beats interleaved decoding (**4.49 vs. 4.12 vs. 4.09**). Flagships
(π0 [16], ERVLA [7]) use continuous flow-matching experts conditioned on the VLM KV-cache
instead of discrete token decoding. **[H]**

**T6 — But the generative *mechanism* is not the decisive factor.** In the same ablation [8],
diffusion policies do *not* significantly beat a simple MSE(pose)+BCE(gripper) regression
head, and flow-matching gives only a small, non-significant edge. OpenVLA-OFT corroborates:
L1 regression matches/beats diffusion while being **~26× faster** [11]. What matters is
**continuity + action chunking + observation history**, not diffusion-vs-flow-vs-regression
per se. *(Scoped to CALVIN/SimplerEnv; diffusion may retain an edge in OOD / cross-embodiment
robustness those suites under-measure.)* **[H]**

**T7 — Why flow-matching, then?** It yields "smooth, temporally coherent, high-frequency
action segments… essential for dexterous and long-horizon manipulation" [9] — the advantage
is in dexterous high-frequency multimodal control that standard benchmarks under-measure,
not raw benchmark score. **[H]**

**T8 — A newer line questions both prevailing action-decoders.** Discrete Diffusion VLA [10]
argues autoregressive left-to-right discrete tokens give "poor performance" while separate
diffusion heads "bolted outside the backbone… fragment information pathways," and instead
discretizes action chunks and models them with a discrete-diffusion (iterative masked-
refinement) process kept *inside* a single unified transformer backbone. **[M]**

**T9 — Vision encoders: CLIP/SigLIP dominate, DINOv2 is a fusion partner.** Across the
surveyed field, "the majority of models used CLIP and SigLIP-based ViT backbones" [2];
DINOv2 is a secondary self-supervised choice that typically *augments* SigLIP rather than
standing alone, and **DINOv2+SigLIP fusion (OpenVLA / Prismatic) is the canonical fused
approach** [2][3][15]. Counter-examples (Octo/Diffusion-Policy: ResNet; RT-1: EfficientNet)
are consistent with a "majority," not "all." **[H]**

**T10 — OpenVLA is the canonical open reference stack.** Fused DINOv2+SigLIP → Llama-2-7B
(Prismatic-7B) → 256-bin discretized autoregressive action tokens, 7-DoF, trained on Open
X-Embodiment (~970k trajectories) plus DROID [3][15]. This is exactly this project's target
model, so it is the load-bearing citation for the threat model. **[H]**

## Precision note (one claim was refuted)

The blanket statement that diffusion/flow models generate actions "without *any* intermediate
reasoning" was **refuted 0-3** against [2]. It is false at the *system* level — GR00T's
System-2 VLM and CogACT's cognition stage *do* reason. "Direct action" refers only to the
**action-decoder mechanism** and the **absence of a verbalized ECoT trace**, not the absence
of any reasoning module. Preserve this distinction in any writeup: *non-CoT* here means
"no interpretable, verbalized, autoregressive reasoning trace in the action loop."

## Caveats & confidence

1. **Field-level tendency claims lean on 2025 surveys** ([1]; [2], a review of 102 models)
   rather than exhaustive per-model primary verification — surveys can lag or mis-summarize
   a specific model's internals.
2. **Uneven model coverage.** Verification is strong on OpenVLA [3], π0 [16], Gemini
   Robotics/ER [5][6], ECoT/ERVLA [4][7][12], RoboVLMs [8], OpenVLA-OFT [11]. It is thinner
   (survey- or single-source only) on RT-1, RT-2/RT-X, Octo, GR00T N1.5 internals, Figure
   Helix, π0.5, CogACT, SpatialVLA, TraceVLA — the **[M]/[s]** rows.
3. **T6 ("head type doesn't matter") is benchmark-bounded** (CALVIN / SimplerEnv); diffusion's
   OOD-robustness edge is not captured there.
4. **T1 and T8 rest on a single primary paper each** ([1]; [10]) — unanimous but not
   cross-validated across independent groups.
5. **Fast-moving field.** Gemini Robotics 1.5 / ER 1.5 (Oct 2025) and ERVLA (Jun 2026) are
   very recent; specifics such as the tunable thinking-budget API may already be superseded.

## Open questions

1. Where is the control-frequency threshold above which token-level CoT becomes infeasible,
   and do caching / parallel-decoding methods (Fast ECoT reasoning-reuse [4]; OpenVLA-OFT
   parallel decoding [11]) actually *close* that latency gap in deployment, or only narrow it?
2. Is the high-level-reasoner + low-level-policy split strictly necessary for production, or
   can a single model interleave NL "thinking" with action at true control frequency as
   Gemini Robotics 1.5 claims [5] — and what is its measured control rate under interleaving?
3. If diffusion/flow heads don't beat simple continuous regression on CALVIN/SimplerEnv [8][11],
   why have production models (π0, GR00T, Gemini Robotics) converged on flow-matching — is the
   advantage confined to OOD robustness, action multimodality, and dexterous bimanual tasks?
4. Which vision encoders do the newest *closed* product VLAs (Gemini Robotics, Figure Helix,
   GR00T N1.5) actually use? The "DINOv2+SigLIP fusion is the norm" evidence [2][3] leans on
   open OpenVLA-lineage models; the fusion may be an open-source pattern while product models
   favor single SigLIP/PaliGemma-style or proprietary encoders.

## Relevance to AutoPPIA-VLA

This survey sharpens the project's positioning (cf. `literature-map.md` Novelty Boundary and
the `stealth-hijack-novelty-vs-trap` line):

- **OpenVLA — the target — is the canonical *non-CoT, discrete-token, base* VLA** [3][10]:
  no interpretable reasoning trace to interfere with visual action-token forcing. This is the
  mechanism the controllability-map work exploits (action-token forcing, not semantic
  reasoning).
- **The field split (T4) is the positioning.** The explicit-reasoning branch (ECoT [12],
  Gemini Robotics-ER [5][6]) is where a verbalized CoT could act as a *defense/detection
  surface* — and is exactly the victim class TRAP (arXiv 2603.23117) targets. This project
  deliberately attacks the *other* branch: the non-CoT low-level policy, which T4 confirms is
  where **most** deployable VLAs actually run. That strengthens the "attacks a realistic
  deployment target" argument.
- **OpenVLA-OFT [11]** (L1 regression head, ~26× faster) is worth noting as the
  deployment-optimized variant of the exact target — a candidate for a robustness/transfer
  extension.
- **Threat-model precision:** use "non-CoT" to mean *no verbalized reasoning trace in the
  action loop* (per the Precision note), so reviewers can't conflate it with "no reasoning
  module anywhere."

## References

Sources verified in the 2026-07-24 deep-research run (primary unless noted). arXiv IDs given
where available.

1. Zhong et al. (2025). *A Survey on Vision-Language-Action Models: An Action Tokenization Perspective.* arXiv:2507.01925. https://arxiv.org/abs/2507.01925
2. *A systematic review of 102 VLA models* (2025). arXiv:2507.10672. https://arxiv.org/pdf/2507.10672
3. Kim et al. (2024). *OpenVLA: An Open-Source Vision-Language-Action Model.* arXiv:2406.09246. https://arxiv.org/abs/2406.09246
4. *Fast ECoT: Efficient Embodied Chain-of-Thought via Thoughts Reuse* (2025). arXiv:2506.07639. https://arxiv.org/abs/2506.07639
5. *Gemini Robotics 1.5 Technical Report* (2025). arXiv:2510.03342. https://arxiv.org/pdf/2510.03342
6. Google DeepMind (2025). *Building the next generation of physical agents with Gemini Robotics-ER 1.5.* Google Developers Blog. https://developers.googleblog.com/building-the-next-generation-of-physical-agents-with-gemini-robotics-er-15/
7. *Revisiting Embodied Chain-of-Thought* (ERVLA) (2026). arXiv:2606.03784. https://arxiv.org/pdf/2606.03784
8. *Towards Generalist Robot Policies: What Matters in Building VLMs for Robotics* (RoboVLMs) (2024). arXiv:2412.14058. https://arxiv.org/html/2412.14058
9. *Reinforcement Fine-Tuning of Flow-Matching Policies for Vision-Language-Action Models* (2025). arXiv:2510.09976. https://arxiv.org/html/2510.09976
10. *Discrete Diffusion VLA* (2025). arXiv:2508.20072 / OpenReview `YWeNCMxdhM`. https://openreview.net/forum?id=YWeNCMxdhM
11. *OpenVLA-OFT: Optimized Fine-Tuning for Vision-Language-Action Models* (2025). arXiv:2502.19645. https://arxiv.org/html/2502.19645v1
12. *Embodied Chain-of-Thought Reasoning* (ECoT) (2024). arXiv:2407.08693.
13. NVIDIA (2025). *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots.* arXiv:2503.14734. https://arxiv.org/pdf/2503.14734
14. Figure AI (2025). *Helix: A Vision-Language-Action Model for Generalist Humanoid Control.* https://www.figure.ai/news/helix
15. Karamcheti et al. (2024). *Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models.* arXiv:2402.07865.
16. Black et al. / Physical Intelligence (2024). *π0: A Vision-Language-Action Flow Model for General Robot Control.* arXiv:2410.24164. https://www.pi.website/download/pi0.pdf
17. *SpatialVLA: Exploring Spatial Representations for Vision-Language-Action Models* (2025). arXiv:2501.15830. https://arxiv.org/html/2501.15830v5
18. *TraceVLA: Visual Trace Prompting for Spatial-Temporal Awareness in VLA Models* (2024). arXiv:2412.10345. https://arxiv.org/pdf/2412.10345

**Further primaries still owing a per-model check** (canonical IDs from prior knowledge, *not*
verified in this run — flagged per the repo's conservative-until-checked rule): RT-1
(arXiv:2212.06817), RT-2 (arXiv:2307.15818), Open X-Embodiment / RT-X (arXiv:2310.08864),
Octo (arXiv:2405.12213), CogACT (arXiv:2411.19650), π0.5 (arXiv:2504.16054). Verify each
against its PDF before citing specifics in the thesis.
