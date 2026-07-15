"""In-scene *monitor* asset + in-place per-step texture upload (Task 1).

A "monitor" is a fixed-size, visual-only textured geom placed in the LIBERO scene whose
texture is re-uploaded **in place** every control step -- no `reset_from_xml_string`, so
the OSC controller state is never disturbed (the reset is exactly what stalled the
chunked adaptive runs). In-place `mjr_uploadTexture` cannot change a texture's compiled
dimensions, so the asset allocates one fixed ``H x W`` canvas up front and every later
frame is resized into it.

CPU-pure here (unit-tested without MuJoCo): `build_monitor_asset` and the
change-detection primitives `mask_local_hash` / `outside_mask_delta`. The MuJoCo-bound
upload seam (`MonitorTextureHandle`) is the GATE-A spike, verified in the GPU rollout env.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from rendering.geometry import (
    BASE_HALF_HEIGHT_M,
    PANEL_THICKNESS_M,
    PromptGeom,
    _euler_deg_to_quat,
    _mujoco_safe_name,
)
from rendering.text_prompt import render_prompt_from_candidate


def build_monitor_asset(candidate: dict[str, Any], tex_hw: tuple[int, int]) -> PromptGeom:
    """Build a fixed ``tex_hw = (H, W)`` monitor geom for a validated candidate.

    Unlike `build_prompt_geom` (whose texture is sized to its text), the monitor texture
    is always exactly ``(H, W, 3)`` so the compiled MuJoCo texture has stable dimensions
    an in-place upload can overwrite. The panel half-width follows the fixed ``W/H`` aspect
    so the rendered content is never distorted.
    """
    tex_h, tex_w = int(tex_hw[0]), int(tex_hw[1])
    content = render_prompt_from_candidate(candidate)
    texture = np.asarray(
        Image.fromarray(np.ascontiguousarray(content), mode="RGB").resize(
            (tex_w, tex_h), Image.Resampling.BILINEAR
        ),
        dtype=np.uint8,
    )

    placement = candidate["placement"]
    scale = float(placement.get("scale", 1.0))
    half_h = BASE_HALF_HEIGHT_M * scale
    half_w = half_h * (tex_w / tex_h)
    x, y, z = (float(v) for v in placement["position"])

    return PromptGeom(
        name=_mujoco_safe_name(candidate["candidate_id"]),
        pos=(x, y, z),
        quat=_euler_deg_to_quat([float(r) for r in placement["rotation"]]),
        half_extents=(half_w, half_h, PANEL_THICKNESS_M),
        texture=texture,
    )


def mask_local_hash(image: NDArray[np.uint8], mask: NDArray[np.bool_]) -> str:
    """SHA-256 of the pixels of ``image`` where ``mask`` is True.

    Depends only on the in-mask content, so it detects that a monitor-region upload
    changed while ignoring incidental changes elsewhere in the frame.
    """
    selected = np.ascontiguousarray(np.asarray(image)[np.asarray(mask, dtype=bool)])
    return hashlib.sha256(selected.tobytes()).hexdigest()


def outside_mask_delta(
    a: NDArray[np.uint8], b: NDArray[np.uint8], mask: NDArray[np.bool_]
) -> float:
    """Max absolute per-pixel difference between ``a`` and ``b`` OUTSIDE ``mask``.

    On the 0..255 scale. Full-frame exact equality is too brittle under render
    resolution / antialiasing, so the upload spike asserts this stays under a tolerance
    (the frame outside the monitor is stable) rather than demanding bit-identity.
    """
    outside = ~np.asarray(mask, dtype=bool)
    diff = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    outside_pixels = diff[outside]
    if outside_pixels.size == 0:
        return 0.0
    return float(outside_pixels.max())


def dilate_mask(mask: NDArray[np.bool_], iterations: int = 1) -> NDArray[np.bool_]:
    """Grow a boolean mask by ``iterations`` pixels with 8-connectivity (pure numpy).

    The monitor's rendered footprint includes a ~1px antialiased edge that the tight
    geom-id segmentation misses; dilating a step or two absorbs that edge so it counts as
    "inside the monitor" rather than outside drift. Never mutates the input.
    """
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(int(iterations)):
        grown = out.copy()
        grown[:-1, :] |= out[1:, :]
        grown[1:, :] |= out[:-1, :]
        grown[:, :-1] |= out[:, 1:]
        grown[:, 1:] |= out[:, :-1]
        grown[:-1, :-1] |= out[1:, 1:]
        grown[:-1, 1:] |= out[1:, :-1]
        grown[1:, :-1] |= out[:-1, 1:]
        grown[1:, 1:] |= out[:-1, :-1]
        out = grown
    return out


def _sim_of(env: Any) -> Any:  # pragma: no cover - MuJoCo/GPU seam
    """Best-effort access to the live robosuite ``MjSim`` across LIBERO wrappers."""
    candidate_env = env
    for _ in range(4):
        sim = getattr(candidate_env, "sim", None)
        if sim is not None:
            return sim
        candidate_env = getattr(candidate_env, "env", None)
        if candidate_env is None:
            break
    raise AttributeError("could not obtain sim from env")


def _raw_model(sim: Any) -> Any:  # pragma: no cover - MuJoCo/GPU seam
    """The underlying ``mujoco.MjModel`` behind robosuite's model wrapper."""
    model = sim.model
    return getattr(model, "_model", model)


class MonitorTextureHandle:  # pragma: no cover - MuJoCo/GPU seam, verified in GPU env
    """A live handle to one compiled monitor texture, for in-place per-step re-upload.

    After `resolve`, `upload(rgb)` mutates the texture's ``tex_data`` slice and re-uploads
    it to the active offscreen render context via `mjr_uploadTexture` -- **without** any
    `reset_from_xml_string`, so the OSC controller state is never disturbed. In-place
    upload cannot change the compiled dimensions, so `rgb` must match `dims` exactly.
    """

    def __init__(self, geom_name: str) -> None:
        # `inject.build_injection_xml` names the texture "<geom>__tex".
        self._tex_name = f"{geom_name}__tex"
        self._texid: int | None = None
        self._adr: int | None = None
        self._size: int | None = None
        self._hw: tuple[int, int] | None = None
        self._nchannel: int = 3
        self._model: Any = None
        self._sim: Any = None

    def resolve(self, env: Any) -> None:
        """Locate the monitor texture in the live model and cache its data slice."""
        import mujoco

        sim = _sim_of(env)
        model = _raw_model(sim)
        texid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, self._tex_name))
        if texid < 0:
            raise ValueError(f"monitor texture {self._tex_name!r} not found in model")
        height = int(model.tex_height[texid])
        width = int(model.tex_width[texid])
        nchannel = int(model.tex_nchannel[texid])
        self._texid = texid
        self._adr = int(model.tex_adr[texid])
        self._size = height * width * nchannel
        self._hw = (height, width)
        self._nchannel = nchannel
        self._model = model
        self._sim = sim

    def upload(self, rgb: NDArray[np.uint8]) -> None:
        """Overwrite the monitor texture with ``rgb`` and re-upload to the render context."""
        import mujoco

        if self._texid is None or self._hw is None:
            raise RuntimeError("MonitorTextureHandle.upload called before resolve()")
        arr = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        if arr.shape != (self._hw[0], self._hw[1], self._nchannel):
            raise ValueError(
                f"upload expects {(self._hw[0], self._hw[1], self._nchannel)}, got {arr.shape}"
            )
        assert self._adr is not None and self._size is not None
        self._model.tex_data[self._adr : self._adr + self._size] = arr.reshape(-1)
        context = self._sim._render_context_offscreen
        if context is None:
            raise RuntimeError("no offscreen render context; render once before uploading")
        mujoco.mjr_uploadTexture(self._model, context.con, self._texid)

    @property
    def dims(self) -> tuple[int, int]:
        if self._hw is None:
            raise RuntimeError("MonitorTextureHandle.dims read before resolve()")
        return self._hw
