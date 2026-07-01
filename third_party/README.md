# Third-Party Repositories & Pinned GPU Stack

The OpenVLA rollout backend (`src/evaluator/openvla_backend.py`) and the LIBERO
render path (`src/rendering/`) depend on an external GPU stack. On this GPU host we
**reuse the proven, already-working environment** from the sibling project rather
than rebuilding it (rebuilding the OpenVLA/LIBERO stack is slow and error-prone).

## Environment (reused)

- **Interpreter:** `~/vla-injection/.venv/bin/python` — a uv-managed venv,
  Python 3.10.20. Guaranteed to have OpenVLA + LIBERO working.
- **Run invocation for anything that touches rollouts / rendering:**
  ```bash
  PYTHONPATH=~/LIBERO MUJOCO_GL=egl ~/vla-injection/.venv/bin/python ...
  ```
  `PYTHONPATH=~/LIBERO` makes `libero` importable (it is *not* installed into
  site-packages); `MUJOCO_GL=egl` selects headless off-screen rendering (verified
  booting `libero_spatial` `OffScreenRenderEnv` and rendering a 256×256 frame).
- ppip's own CPU tests need none of this: `~/vla-injection/.venv/bin/python -m pytest`
  from the repo root (imports resolve via `pyproject` `pythonpath`).

## Pinned source repositories (commit hashes)

| Repo | Location | Remote | Commit | Wiring |
|---|---|---|---|---|
| OpenVLA | `~/openvla` | github.com/openvla/openvla.git | `c8f03f48af692657d3060c19588038c7220e9af9` (2025-03-23) | editable install → `import prismatic`; reference eval harness at `experiments/robot/libero/` |
| LIBERO | `~/LIBERO` | github.com/Lifelong-Robot-Learning/LIBERO.git | `8f1084e3132a39270c3a13ebe37270a43ece2a01` (2025-03-15) | on `PYTHONPATH` → `from libero.libero import benchmark` |

Reference rollout code to port in Phase C:
- `~/openvla/experiments/robot/libero/libero_utils.py` — `get_libero_env`,
  `get_libero_image`, `quat2axisangle`, dummy action.
- `~/openvla/experiments/robot/libero/run_libero_eval.py` — the closed-loop episode.
- `~/vla-injection/src/evasion_tax/` — the sibling project's OpenVLA+LIBERO wiring.

## Model checkpoints (HF cache, ~15 GB each — not tracked)

- `openvla/openvla-7b-finetuned-libero-spatial` (the default `model_id`).
- `openvla/openvla-7b`.
Both already present under `~/.cache/huggingface/hub/`, so no download is needed.

## Pinned package versions (in the reused venv)

torch 2.2.0+cu121 · transformers 4.40.1 · tokenizers 0.19.1 · timm 0.9.10 ·
robosuite 1.4.1 · mujoco 3.9.0 · numpy 1.26.4 · draccus 0.8.0 · accelerate 0.30.1.
These are mirrored (commented) in the `gpu` optional-dependency group of
`pyproject.toml` for documentation; the canonical install is the reused venv above.
