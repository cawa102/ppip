"""Map a candidate's placement + style to MuJoCo geom parameters (CPU-pure).

`build_prompt_geom` produces a `PromptGeom`: the world pose, half-extents, and texture
of a thin, visual-only textured panel that represents the attacker's in-scene label.
Inserting that panel into a live MuJoCo model is a separate, MuJoCo-dependent step
(`rendering.inject`); everything here is pure math + PIL and is unit-tested on CPU.

Conventions:
  * ``placement.position`` is the panel centre in world metres (already bounds-checked
    by `evaluator.validation.PLACEMENT_BOUNDS`).
  * ``placement.rotation`` is XYZ intrinsic Euler angles in **degrees**; identity
    rotation leaves the panel lying in the world xy-plane, facing +z.
  * ``placement.scale`` multiplies the panel's base half-height; the half-width follows
    from the rendered texture's aspect ratio so text is never distorted.

The panel's local **+Z is the readable front face** (text-up = local +Y). To read as
an upright billboard, rotate that front toward the viewing camera and keep text-up
aligned with world +Z. For LIBERO ``libero_object`` (agentview camera on +x looking at
the table), ``rotation = [90, 90, 0]`` stands the panel upright facing +x. ``inject.py``
pre-flips the MuJoCo texture horizontally so the front face reads un-mirrored to the
camera (MuJoCo maps a box's 2D texture mirrored on its outward face).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rendering.text_prompt import render_prompt_from_candidate

# A readable tabletop label is ~10 cm tall at scale 1.0; the panel is ~2 mm thick so it
# reads as a flat decal/sign rather than a block. Both are visual-only design constants.
BASE_HALF_HEIGHT_M = 0.05
PANEL_THICKNESS_M = 0.002

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


@dataclass(frozen=True, eq=False)
class PromptGeom:
    """MuJoCo parameters for a thin visual-only textured panel (label)."""

    name: str
    pos: Vec3
    quat: Quat  # (w, x, y, z)
    half_extents: Vec3  # (half_width, half_height, half_thickness) in metres
    texture: NDArray[np.uint8]  # (H, W, 3) uint8 RGB


def _axis_quat(axis: Vec3, degrees: float) -> Quat:
    half = math.radians(degrees) / 2.0
    s = math.sin(half)
    return (math.cos(half), s * axis[0], s * axis[1], s * axis[2])


def _quat_mul(a: Quat, b: Quat) -> Quat:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _euler_deg_to_quat(rotation: list[float]) -> Quat:
    """XYZ intrinsic Euler degrees -> unit quaternion (w, x, y, z)."""
    rx, ry, rz = rotation
    qx = _axis_quat((1.0, 0.0, 0.0), rx)
    qy = _axis_quat((0.0, 1.0, 0.0), ry)
    qz = _axis_quat((0.0, 0.0, 1.0), rz)
    return _quat_mul(_quat_mul(qx, qy), qz)


def _mujoco_safe_name(candidate_id: str) -> str:
    return f"ppia_prompt__{candidate_id}"


def build_prompt_geom(candidate: dict[str, Any]) -> PromptGeom:
    """Build the MuJoCo panel spec for a validated candidate's visual prompt."""
    texture = render_prompt_from_candidate(candidate)
    tex_h, tex_w = int(texture.shape[0]), int(texture.shape[1])
    aspect = tex_w / tex_h

    placement = candidate["placement"]
    scale = float(placement.get("scale", 1.0))
    half_h = BASE_HALF_HEIGHT_M * scale
    half_w = half_h * aspect

    x, y, z = (float(v) for v in placement["position"])

    return PromptGeom(
        name=_mujoco_safe_name(candidate["candidate_id"]),
        pos=(x, y, z),
        quat=_euler_deg_to_quat([float(r) for r in placement["rotation"]]),
        half_extents=(half_w, half_h, PANEL_THICKNESS_M),
        texture=texture,
    )
