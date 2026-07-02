# AutoPPIA-VLA Literature Map

This file tracks the nearest prior work for the AutoPPIA-VLA direction. Keep claims conservative until each source is checked against the paper PDF.

## Core System

- **OpenVLA** — target VLA model family for simulation experiments. Source: https://arxiv.org/abs/2406.09246
- **LIBERO** — target simulated manipulation benchmark. Source: https://arxiv.org/abs/2306.03310

## Vision-Layer Vulnerability Landscape (the two motivating papers)

Both reference papers attack the **vision input layer** of embodied/vision-language
models, but via two distinct sub-types. AutoPPIA-VLA targets the first sub-type and
cites the second as the adversarial-patch cousin (see Novelty Boundary).

- **PPIA / Physical Prompt Injection** (Ling et al.) — *semantic / typographic* sub-type.
  A black-box, query-agnostic attack that prints a **readable** malicious instruction on a
  common container (paper bag, book, poster) placed in view. Evaluated on **LVLMs** (VQA /
  planning / navigation) in **Embodied City / Habitat** simulators and on a real camera
  vehicle — **not** LIBERO/VLA. In simulation the prompt is inserted by 2D image compositing
  of a transformed container image; success = a target keyword appears in the model's text
  output. AutoPPIA-VLA ports this *attack class* to end-to-end VLA action policies
  (OpenVLA on LIBERO), with a far stricter simulator-predicate success metric.
  Source: https://arxiv.org/abs/2601.17383
- **TRAP** (Hijacking VLA CoT-Reasoning via Adversarial Patches) — *adversarial-patch* sub-type.
  An **optimized, non-legible** patch (printed as e.g. a tablecloth) that hijacks the
  **chain-of-thought reasoning trace** of CoT-reasoning VLAs to force targeted action
  substitution. Different mechanism (pixel optimization, not readable text) and a different
  victim class (models with an explicit CoT trace). Out of scope as an attack to *reproduce*
  — OpenVLA has no CoT trace — but it is the key related-work boundary that defines what
  AutoPPIA-VLA deliberately does not do. Source: https://arxiv.org/abs/2603.23117

## Autonomous Research / Red-Team Context

- **karpathy/autoresearch** — lightweight inspiration for a fixed-metric AI research loop. The project should borrow the pattern, not assume the repository is directly reusable for VLA. Source: https://github.com/karpathy/autoresearch
- **AI Scientist-v2** — broader context for agentic tree-search scientific discovery. Source: https://arxiv.org/abs/2504.08066
- **AIRTBench** — autonomous AI red-teaming benchmark context. Source: https://arxiv.org/abs/2506.14682

## Closest VLA Security Prior

- **SABER** — agentic black-box VLA attack prior. Current differentiation: SABER focuses on agentic instruction perturbation, while AutoPPIA-VLA focuses on environmental visual prompt discovery and scaffold comparison. Source: https://arxiv.org/abs/2603.24935

## Scope Decision (locked)

**Scope (a): autoresearch discovery of PPIA-class typographic injections on OpenVLA/LIBERO.**
The attack class under search is **readable/typographic** visual prompts (PPIA sub-type),
injected as a **3D visual-only textured geom** in the LIBERO/MuJoCo scene (see threat model).
TRAP's adversarial-patch-on-CoT-VLA vulnerability is **not** reproduced: OpenVLA has no CoT
trace, and raw-pixel patch optimization is a different candidate representation and budget.
The `hybrid_prompt_object` level-3 stretch is the only place a less-legible / patch-like
variant may later be gestured at, without committing the thesis to patch optimization.

## Novelty Boundary

AutoPPIA-VLA should not claim that physical prompt injection itself is new. Neither PPIA nor
TRAP is an autonomous-discovery method — each presents a single hand-crafted / optimized
attack. The novelty target is:

1. applying physical/visual (typographic) prompt injection to OpenVLA+LIBERO-style VLA rollouts;
2. measuring **autonomous AI loop capability** at *discovering* such attacks under fixed budgets
   (the discovery method is the contribution, not any single attack);
3. comparing scaffold variants such as no memory, memory, and skill/program-file guidance;
4. reporting failure modes and metric-gaming risks in autonomous red-team research.

## Notes To Verify

- **PPIA (checked 2026-07):** LVLM victims; Embodied City / Habitat + real vehicle; typographic
  prompt on a container; 2D image-insertion in sim; keyword-ASR success. No LIBERO/VLA policy
  experiment — re-confirm exact numbers against the PDF before the dissertation cites them.
- **TRAP (checked 2026-07):** adversarial-patch, CoT-reasoning VLA victims, targeted hijack.
  Confirm exact victim model names and whether any evaluation is on LIBERO before citing.
- Exact SABER threat model, allowed query access, and whether visual scene modifications are in scope.
- Whether any 2026 VLA paper already evaluates autonomous environmental prompt discovery.
