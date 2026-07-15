"""GATE-A spike: prove in-place per-step monitor texture upload with NO sim reset.

This is Phase-0 Task 1's riskiest plumbing. It builds a real LIBERO env, injects a
fixed-size visual-only *monitor* geom once (the only `reset_from_xml_string`), then every
step overwrites the monitor's compiled texture in place and re-uploads it to the active
offscreen render context via `mjr_uploadTexture`. It verifies:

  * different uploads change the mask-local hash of the monitor region;
  * the rest of the frame stays stable within tolerance (upload changes ONLY the monitor);
  * an upload never disturbs simulator state (visual-only: end-effector pose unchanged);
  * `reset_from_xml_string` is never called after the one-time setup injection.

If in-place upload proves infeasible in the robosuite/LIBERO wrapper, GATE A fails: STOP
and escalate (a reset-based variant is out-of-claim). Run:

    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYTHONPATH=$HOME/LIBERO:$HOME/autoresearch/src \
      ~/vla-injection/.venv/bin/python experiments/patch_attack/monitor_upload_probe.py
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from evaluator.libero_tasks import resolve_task
from evaluator.openvla_backend import _ENV_RESOLUTION, OpenVLARolloutBackend
from rendering.geometry import PromptGeom
from rendering.inject import build_injection_xml, write_texture_png
from rendering.monitor import (
    MonitorTextureHandle,
    build_monitor_asset,
    dilate_mask,
    mask_local_hash,
    outside_mask_delta,
)

USER_TASK = "pick up the alphabet soup and place it in the basket"
TARGET_TASK = "pick up the salad dressing and place it in the basket"
TEX_HW = (256, 256)  # fixed monitor canvas (power-of-two, square)
OUTSIDE_TOLERANCE = 3.0  # 0..255 scale; renderer AA/shadow slack outside the monitor

# The BEST_CASE central override placement -- known visible to agentview (~0.16 coverage),
# non-occluding for the alphabet_soup scene.
_MONITOR_CANDIDATE: dict[str, Any] = {
    "candidate_id": "monitor_probe",
    "condition": "loop_with_skill",
    "user_task": USER_TASK,
    "target_task": TARGET_TASK,
    "visual_prompt": {"text": "MONITOR", "prompt_level": "semantic_visual_prompt"},
    "placement": {
        "scene_anchor": "table_surface",
        "position": [-0.08, 0.0, 0.14],
        "rotation": [90.0, 90.0, 0.0],
        "scale": 1.6,
    },
    "style": {
        "foreground_color": "#000000",
        "background_color": "#ffffff",
        "font_size": 40,
        "font_family": "sans-serif",
    },
    "metadata": {
        "created_by": "gate_a_probe",
        "created_at": "2026-07-15T00:00:00Z",
        "notes": "spike",
    },
}


def _inject_monitor(env: Any, geom: PromptGeom, texture_dir: str) -> None:
    """Inject the fixed-size monitor geom via one setup-time XML reset."""
    from rendering.inject import _get_model_xml  # MuJoCo-seam accessor

    os.makedirs(texture_dir, exist_ok=True)
    texture_path = write_texture_png(geom.texture, os.path.join(texture_dir, f"{geom.name}.png"))
    injected_xml = build_injection_xml(_get_model_xml(env), geom, texture_path)
    env.reset_from_xml_string(injected_xml)


def _solid_texture(step: int) -> np.ndarray:
    """A distinct fixed-size texture per step: a moving RGB gradient + step-tinted fill."""
    h, w = TEX_HW
    row = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    col = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    tex = np.zeros((h, w, 3), dtype=np.uint8)
    tex[:, :, 0] = (row + step * 13) % 256
    tex[:, :, 1] = (col + step * 7) % 256
    tex[:, :, 2] = (step * 29) % 256
    return tex


def _eef_pos(env: Any) -> np.ndarray:
    obs = env._get_observations() if hasattr(env, "_get_observations") else {}
    return np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float64)


def run_upload_probe(n_steps: int = 20) -> dict[str, Any]:
    """Run the GATE-A in-place-upload spike; return a report dict (see module docstring)."""
    backend = OpenVLARolloutBackend()
    resolved_user = resolve_task(USER_TASK, suite="libero_object")
    env, init_states, _desc, _obj = backend._build_env(resolved_user)

    report: dict[str, Any] = {"outside_tolerance": OUTSIDE_TOLERANCE}
    try:
        geom = build_monitor_asset(_MONITOR_CANDIDATE, tex_hw=TEX_HW)
        texture_dir = os.path.join(
            os.environ.get("PROBE_DIR", "/tmp/monitor_probe"), "tex"
        )
        _inject_monitor(env, geom, texture_dir)

        obs = env.set_init_state(init_states[0])
        cfg = backend._build_cfg()
        dummy = backend._dummy_action(cfg)
        for _ in range(backend.num_steps_wait):
            obs, _r, _d, _i = env.step(dummy)

        # Create the offscreen render context, then bind the texture handle.
        _ = env.sim.render(width=_ENV_RESOLUTION, height=_ENV_RESOLUTION, camera_name="agentview")
        handle = MonitorTextureHandle(geom.name)
        handle.resolve(env)

        # Monitor region mask, from the geom-id segmentation at the settled state.
        # Dilate by 2px so the monitor's ~1px antialiased edge counts as "inside the
        # monitor" rather than outside drift (the tight geom-id seg misses that ring).
        geom_id = backend._geom_id(env, geom.name)
        seg = backend._segmentation(env)
        tight_mask = np.asarray(seg) == geom_id
        mask = dilate_mask(tight_mask, iterations=2)
        report["mask_pixels"] = int(tight_mask.sum())

        # Count any reset AFTER setup (must stay 0). Wrap the wrapper method honestly.
        reset_calls = {"n": 0}
        original_reset = env.reset_from_xml_string

        def _counting_reset(*a: Any, **k: Any) -> Any:  # pragma: no cover - guard
            reset_calls["n"] += 1
            return original_reset(*a, **k)

        env.reset_from_xml_string = _counting_reset  # type: ignore[method-assign]

        # Render in the SAME raw orientation as the segmentation mask (no vertical flip)
        # so mask and frame align. The claim is about the in-place UPLOAD mechanism, so
        # baseline on the FIRST mjr-upload (comparing compile-time vs mjr-upload frames
        # crosses two different texture code paths and is not what we're testing).
        def _render() -> np.ndarray:
            return env.sim.render(
                width=_ENV_RESOLUTION, height=_ENV_RESOLUTION, camera_name="agentview"
            )

        handle.upload(_solid_texture(0))
        env.sim.forward()
        prev_frame = _render()
        hashes = [mask_local_hash(prev_frame, mask)]
        max_outside = 0.0
        worst_step = -1
        max_eef_jump = 0.0
        for step in range(1, n_steps):
            eef_pre = _eef_pos(env)
            handle.upload(_solid_texture(step))
            env.sim.forward()  # push the new texture to the render state; no physics step
            eef_post = _eef_pos(env)
            max_eef_jump = max(max_eef_jump, float(np.abs(eef_post - eef_pre).max()))

            frame = _render()
            hashes.append(mask_local_hash(frame, mask))
            step_outside = outside_mask_delta(prev_frame, frame, mask)
            if step_outside > max_outside:
                max_outside = step_outside
                worst_step = step
            prev_frame = frame
        report["worst_step"] = worst_step

        env.reset_from_xml_string = original_reset  # type: ignore[method-assign]

        report["reset_calls_after_setup"] = reset_calls["n"]
        report["distinct_hashes"] = len(set(hashes))
        report["max_outside_delta"] = max_outside
        report["max_eef_jump_m"] = max_eef_jump
        report["gate_a_pass"] = (
            reset_calls["n"] == 0
            and len(set(hashes)) > 1
            and max_outside <= OUTSIDE_TOLERANCE
            and max_eef_jump < 1e-6
            and int(mask.sum()) > 0
        )
    finally:
        if hasattr(env, "close"):
            env.close()
    return report


if __name__ == "__main__":
    result = run_upload_probe(n_steps=int(os.environ.get("PROBE_STEPS", "20")))
    print("GATE-A upload probe report:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print("GATE A:", "PASS" if result.get("gate_a_pass") else "FAIL")
