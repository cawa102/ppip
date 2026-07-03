# AutoPPIA-VLA Research Workspace

This workspace supports an MSc research project on simulation-only VLA red teaming.

Working question:

> Can autoresearch-style AI loops discover and improve physical prompt injection attacks against OpenVLA on LIBERO under fixed evaluation budgets?

## Structure

- `VLA-security-project-decision.md` — original project decision memo.
- `docs/plans/` — implementation/research execution plans.
- `docs/research/` — literature notes, threat model, experiment protocol, research log, and risk register.
- `programs/` — fixed program files for agent-loop conditions.
- `experiments/configs/` — schemas and experiment configuration.
- `experiments/candidates/` — candidate visual prompt JSON files.
- `experiments/results/` — summaries and aggregation scripts.
- `src/evaluator/` — fixed evaluator boundary.
- `src/autoresearch_loop/` — editable/autonomous search loop scaffold.
- `src/rendering/` — LIBERO scene prompt insertion code.
- `data/` — external and processed data placeholders.
- `runs/` — local run artifacts, logs, and generated metrics.
- `third_party/` — external repositories such as OpenVLA, LIBERO, and autoresearch.

## Current Status

The harness is implemented and tested, including the OpenVLA rollout body behind
the injectable `RolloutBackend` seam. `pilot-001` completed as a four-condition
diagnostic run and found visible prompt-induced denial, not targeted hijack. The
next runnable scaffold is `pilot-002`, a cheap exploratory discovery pass over
broader PPIA prompt families. The lightweight test environment runs without
loading the OpenVLA/LIBERO stack; GPU tests and real rollouts are opt-in in the
configured GPU rollout environment. Current local verification: 129 passed / 5
skipped, `ruff` clean, and `mypy --strict` clean. See `docs/research/research-log.md`
for the current status.

### Running the tests

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest            # full suite
.venv/bin/ruff check src tests experiments/results
.venv/bin/mypy src experiments/results/aggregate_results.py
```

The OpenVLA/LIBERO/torch stack is intentionally NOT required for lightweight harness
tests. This repository is on a GPU-capable machine; real rollout work uses the
configured GPU environment described in `third_party/README.md`.

## Important Autoresearch Assumption

This project does not use the original `karpathy/autoresearch` 5-minute nanochat training cap.

`autoresearch` is only the design inspiration for an iterative loop:

```text
candidate proposal -> fixed OpenVLA/LIBERO evaluator -> metrics -> ledger -> next candidate
```

OpenVLA+LIBERO evaluations may be long-running jobs. Budgets are defined by candidate count, task pairs, seeds, rollout count, and recorded GPU/wall-clock usage. See `experiments/configs/evaluation_budgets.yaml` and `docs/research/experiment-protocol.md`.

## Autoresearch File Mapping

This project does not keep the exact `prepare.py` / `train.py` / `program.md` contract from `karpathy/autoresearch`. The roles are split so the evaluator cannot be edited by the search agent.

| `karpathy/autoresearch` role | AutoPPIA-VLA location | Status |
|---|---|---|
| `program.md` | `programs/autoppia-vla/program.md` | Exists. This is the instruction file for the candidate-proposal agent. |
| `prepare.py` | `src/evaluator/eval_attack.py`, `src/evaluator/metrics.py`, `src/rendering/`, `experiments/configs/evaluation_budgets.yaml` | Implemented. Fixed benchmark/evaluation side; real OpenVLA rollouts run in the configured GPU environment. |
| `train.py` | `src/autoresearch_loop/run_loop.py`, `src/autoresearch_loop/candidate_writer.py`, `src/autoresearch_loop/memory.py`, plus generated files in `experiments/candidates/` | Implemented. Editable/search side that proposes candidates and reads results. |

Interpretation:

- `prepare.py` equivalent = trusted setup/evaluation code. The agent must not modify it during benchmark comparisons.
- `train.py` equivalent = candidate search logic. The agent or loop may revise this side depending on the experimental condition.
- `program.md` equivalent = fixed natural-language research instructions for the agent.
