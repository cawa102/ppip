# Repository Guidelines

## Project Structure & Module Organization

This repository is an AutoPPIA-VLA research workspace for simulation-only VLA red teaming. Core planning documents live in `docs/plans/` and research notes in `docs/research/`. Source code is planned under `src/`: `src/evaluator/` is the fixed OpenVLA/LIBERO evaluation boundary, `src/autoresearch_loop/` is the editable candidate-search loop, and `src/rendering/` will contain LIBERO visual prompt insertion utilities. Experiment schemas and budgets live in `experiments/configs/`; generated candidate JSON files belong in `experiments/candidates/`; lightweight summaries go in `experiments/results/`. Use `runs/` for local run artifacts and `third_party/` for external repositories.

## Build, Test, and Development Commands

There is no implemented Python package or test suite yet. Useful current checks:

- `find . -maxdepth 3 -type f | sort` - inspect repository layout.
- `python -c "import json; [json.load(open(p)) for p in ['experiments/configs/attack_candidate.schema.json']]"` - validate JSON syntax.
- `rg "5-minute|evaluation_budgets|candidate evaluation job"` - confirm long-running evaluation assumptions remain documented.

When code is added, prefer `python -m pytest` for tests and document exact commands in `README.md`.

## Coding Style & Naming Conventions

Use Python with clear module boundaries. Keep evaluator code deterministic and separate from agent-editable search code. Use snake_case for Python files and functions, kebab-case for Markdown plan filenames, and lowercase descriptive run IDs such as `pilot-001`. JSON candidates should use stable IDs like `baseline_human_ppia` or `loop_with_memory_003`.

## Testing Guidelines

Tests should first cover schema validation, metric calculation, ledger append/resume behavior, and the rule that search agents cannot alter evaluator outputs. Name tests by behavior, for example `test_invalid_candidate_is_penalized.py`. Heavy OpenVLA/LIBERO rollouts should be staged as smoke, pilot, and full evaluations using `experiments/configs/evaluation_budgets.yaml`.

## Commit & Pull Request Guidelines

No Git history exists in this workspace yet, so no local convention is established. Use concise conventional-style commits such as `docs: clarify evaluation budget model` or `feat: add fixed evaluator contract`. Pull requests should state the research task, list changed paths, describe validation performed, and note any large artifacts kept out of version control.

## Agent-Specific Instructions

Do not copy `karpathy/autoresearch`'s 5-minute cap. Treat it only as a loop pattern: candidate proposal, fixed evaluator, metrics, ledger, next candidate. During benchmark comparisons, never modify evaluator code, scoring code, seed lists, prior metrics, or ledger history to improve results.
