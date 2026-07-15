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
from dataclasses import dataclass
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


# OpenVLA's inference preprocessing center-crops area 0.9 (side sqrt(0.9)) then resizes to
# 224. Mirrors vla_diff._CROP_SIDE (kept here so monitor.py stays torch-free).
CROP_SIDE = 0.9**0.5


def center_crop_mask(mask: NDArray[np.bool_], side: float = CROP_SIDE) -> NDArray[np.bool_]:
    """Map a pre-crop 224 mask into post-crop policy-input coords (what the model samples).

    Replicates `vla_diff.center_crop_resize` (align_corners crop of the central ``side^2``
    area, resized to 224) with nearest-neighbour sampling, so a monitor mask lands in the
    exact 224 space Task-6's masked-δ optimisation and the invariant hashing operate on.
    """
    m = np.asarray(mask, dtype=bool)
    n = m.shape[0]
    lin = np.linspace(0.0, 1.0, n)
    coords = (1.0 - side) / 2.0 + lin * side  # normalized [0,1] input positions
    idx = np.clip(np.round(coords * (n - 1)).astype(int), 0, n - 1)
    return m[np.ix_(idx, idx)]


@dataclass(frozen=True)
class UVMap:
    """Corner correspondences between the monitor texture and its projected image quad.

    ``texture_corners`` and ``image_corners`` are each ``(4, 2)`` arrays in the SAME order
    (top-left, top-right, bottom-right, bottom-left). The mapping is discovered empirically
    by `calibrate_uv` (a rendered calibration grid), so it already absorbs MuJoCo's box-face
    horizontal mirror -- no flip is hardcoded.
    """

    texture_corners: NDArray[np.float64]
    image_corners: NDArray[np.float64]


@dataclass(frozen=True)
class Homography:
    """A 2D projective transform; ``apply`` maps ``(N, 2)`` points to ``(N, 2)``."""

    matrix: NDArray[np.float64]

    def apply(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        homog = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
        projected = homog @ self.matrix.T
        return np.asarray(projected[:, :2] / projected[:, 2:3], dtype=np.float64)


def homography_from_correspondences(
    src: NDArray[np.float64], dst: NDArray[np.float64]
) -> Homography:
    """Fit the projective transform mapping ``src`` points onto ``dst`` (DLT, >=4 pairs)."""
    src_arr = np.asarray(src, dtype=np.float64).reshape(-1, 2)
    dst_arr = np.asarray(dst, dtype=np.float64).reshape(-1, 2)
    if src_arr.shape[0] < 4 or src_arr.shape != dst_arr.shape:
        raise ValueError("need >=4 matched (src, dst) point pairs of equal shape")
    rows = []
    for (x, y), (u, v) in zip(src_arr, dst_arr, strict=True):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    h = vh[-1].reshape(3, 3)
    return Homography(matrix=h / h[2, 2])


def homography_quad_to_texture(uv_map: UVMap) -> Homography:
    """Image(policy-input) -> texture homography, for synthesizing a texture from an
    image-space pattern (Task 6 inverts the projection to lay pixels on the monitor)."""
    return homography_from_correspondences(uv_map.image_corners, uv_map.texture_corners)


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


def _fresh_obs(env: Any) -> Any:  # pragma: no cover - GPU seam
    """Force a fresh observation (re-rendering agentview) across LIBERO wrappers.

    `_get_observations` lives on the nested robosuite env, not the OffScreenRenderEnv
    wrapper, so walk the chain like the backend's `_object_states` does."""
    candidate_env = env
    for _ in range(5):
        getter = getattr(candidate_env, "_get_observations", None)
        if getter is not None:
            return getter()
        candidate_env = getattr(candidate_env, "env", None)
        if candidate_env is None:
            break
    raise AttributeError("could not obtain a fresh observation from env")


def _policy_input_frame(
    env: Any, resize_size: int = 224, render_size: int = 256
) -> NDArray[np.uint8]:  # pragma: no cover - GPU seam
    """The 224 uint8 image the policy consumes, from a FRESH render that reflects the upload.

    Critical: robosuite's ``obs['agentview_image']`` does NOT reflect an in-place
    `mjr_uploadTexture` (it renders via a separate/cached path), while ``sim.render`` does
    and is byte-identical to the obs image otherwise. So the monitor threat model requires
    building the policy input from a fresh ``sim.render`` -- exactly the Task-4 invariant.
    ``get_libero_image`` then applies the same rot180 + resize the real rollout uses.
    """
    from experiments.robot.libero.libero_utils import get_libero_image

    fresh = env.sim.render(width=render_size, height=render_size, camera_name="agentview")
    return np.asarray(
        get_libero_image({"agentview_image": np.asarray(fresh)}, resize_size), dtype=np.uint8
    )


def monitor_mask_224(
    env: Any,
    handle: MonitorTextureHandle,
    *,
    resize_size: int = 224,
    threshold: float = 12.0,
    dilate: int = 2,
) -> NDArray[np.bool_]:  # pragma: no cover - GPU seam, verified in GPU env
    """Monitor region as a bool mask in **post-crop** policy-input coords (self-calibrating).

    Instead of projecting the geom and tracking every flip/resize/crop, measure directly
    which policy-input pixels the monitor controls by contrasting an all-black vs all-white
    upload. Robust to render orientation by construction. Mutates the monitor texture (the
    caller re-uploads afterwards)."""
    h, w = handle.dims
    handle.upload(np.zeros((h, w, 3), dtype=np.uint8))
    env.sim.forward()
    black = _policy_input_frame(env, resize_size)
    handle.upload(np.full((h, w, 3), 255, dtype=np.uint8))
    env.sim.forward()
    white = _policy_input_frame(env, resize_size)

    precrop = (
        np.abs(white.astype(np.int64) - black.astype(np.int64)).max(axis=2) > threshold
    )
    precrop = dilate_mask(precrop, iterations=dilate)
    return center_crop_mask(precrop)


def calibrate_uv(
    env: Any,
    handle: MonitorTextureHandle,
    *,
    resize_size: int = 224,
    threshold: float = 12.0,
) -> UVMap:  # pragma: no cover - GPU seam, verified in GPU env
    """Discover the texture(u,v) -> policy-image correspondence of the 4 monitor corners.

    Lights one corner patch white-on-black at a time and locates its centroid in the
    policy image (high-contrast diff, robust to surface shading and MuJoCo's mirror flip).
    Returns the 4 (texture_corner, image_corner) pairs in TL, TR, BR, BL order. Mutates the
    monitor texture (the caller re-uploads afterwards)."""
    h, w = handle.dims
    patch = max(8, min(h, w) // 4)
    # Patch centres in texture pixel coords, order TL, TR, BR, BL.
    texture_corners = np.array(
        [
            [patch / 2, patch / 2],
            [w - patch / 2, patch / 2],
            [w - patch / 2, h - patch / 2],
            [patch / 2, h - patch / 2],
        ],
        dtype=np.float64,
    )
    slices = [
        (slice(0, patch), slice(0, patch)),
        (slice(0, patch), slice(w - patch, w)),
        (slice(h - patch, h), slice(w - patch, w)),
        (slice(h - patch, h), slice(0, patch)),
    ]

    handle.upload(np.zeros((h, w, 3), dtype=np.uint8))
    env.sim.forward()
    baseline = _policy_input_frame(env, resize_size).astype(np.int64)

    image_corners = np.zeros((4, 2), dtype=np.float64)
    for i, (rs, cs) in enumerate(slices):
        tex = np.zeros((h, w, 3), dtype=np.uint8)
        tex[rs, cs, :] = 255
        handle.upload(tex)
        env.sim.forward()
        lit = _policy_input_frame(env, resize_size).astype(np.int64)
        diff = np.abs(lit - baseline).max(axis=2) > threshold
        ys, xs = np.where(diff)
        if ys.size == 0:
            raise RuntimeError(f"calibrate_uv: monitor corner {i} not visible in policy image")
        image_corners[i] = [xs.mean(), ys.mean()]  # (x=col, y=row)

    return UVMap(texture_corners=texture_corners, image_corners=image_corners)


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
