# AutoPPIA-VLA Literature Map

This file tracks the nearest prior work for the AutoPPIA-VLA direction. Keep claims conservative until each source is checked against the paper PDF.

## Core System

- **OpenVLA** — target VLA model family for simulation experiments. Source: https://arxiv.org/abs/2406.09246
- **LIBERO** — target simulated manipulation benchmark. Source: https://arxiv.org/abs/2306.03310

## Base Attack Class

- **PPIA / physical prompt injection** — visual/physical prompt injection precedent for large vision-language models. The proposed project ports the attack class toward end-to-end VLA policies and tests whether autonomous loops can discover stronger prompt objects. Source: https://arxiv.org/abs/2601.17383

## Autonomous Research / Red-Team Context

- **karpathy/autoresearch** — lightweight inspiration for a fixed-metric AI research loop. The project should borrow the pattern, not assume the repository is directly reusable for VLA. Source: https://github.com/karpathy/autoresearch
- **AI Scientist-v2** — broader context for agentic tree-search scientific discovery. Source: https://arxiv.org/abs/2504.08066
- **AIRTBench** — autonomous AI red-teaming benchmark context. Source: https://arxiv.org/abs/2506.14682

## Closest VLA Security Prior

- **SABER** — agentic black-box VLA attack prior. Current differentiation: SABER focuses on agentic instruction perturbation, while AutoPPIA-VLA focuses on environmental visual prompt discovery and scaffold comparison. Source: https://arxiv.org/abs/2603.24935

## Novelty Boundary

AutoPPIA-VLA should not claim that physical prompt injection itself is new. The novelty target is:

1. applying physical/visual prompt injection to OpenVLA+LIBERO-style VLA rollouts;
2. measuring autonomous AI loop capability under fixed budgets;
3. comparing scaffold variants such as no memory, memory, and skill/program-file guidance;
4. reporting failure modes and metric-gaming risks in autonomous red-team research.

## Notes To Verify

- Exact PPIA benchmark settings and whether any robotic/VLA policy experiments are included.
- Exact SABER threat model, allowed query access, and whether visual scene modifications are in scope.
- Whether any 2026 VLA paper already evaluates autonomous environmental prompt discovery.
