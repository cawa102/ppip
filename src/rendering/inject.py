"""Inject a rendered prompt label into a LIBERO/MuJoCo scene (Option A).

The attacker's label enters the scene as a thin, **visual-only** textured box geom
(`contype=0`, `conaffinity=0`): visible to every camera with correct perspective and
occlusion, but inert to physics so it never perturbs the task. This module has two
parts:

  * `build_injection_xml` / `write_texture_png` — pure, CPU-testable: they add the
    texture/material assets and the geom to the model XML and persist the texture.
  * `inject_prompt` — the thin MuJoCo-dependent seam: it reads the live env's model
    XML, injects, and re-initialises the sim via `reset_from_xml_string`.

Editing the model XML and re-initialising (rather than mutating a compiled `MjModel`)
is the approach robosuite/LIBERO support directly (`get_xml` + `reset_from_xml_string`).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from rendering.geometry import PromptGeom, build_prompt_geom


def _fmt(values: tuple[float, ...]) -> str:
    return " ".join(f"{v:.9g}" for v in values)


def write_texture_png(texture: NDArray[np.uint8], path: str) -> str:
    """Persist an ``(H, W, 3)`` uint8 texture as a PNG MuJoCo can load; return `path`."""
    Image.fromarray(np.ascontiguousarray(texture), mode="RGB").save(path)
    return path


def build_injection_xml(base_xml: str, geom: PromptGeom, texture_path: str) -> str:
    """Return `base_xml` with the label's texture/material asset and geom added.

    Pure string/ElementTree editing — no MuJoCo. The geom is a visual-only box so it
    renders the label without colliding with the robot or task objects.
    """
    root = ET.fromstring(base_xml)

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    texture_name = f"{geom.name}__tex"
    material_name = f"{geom.name}__mat"
    ET.SubElement(
        asset,
        "texture",
        {"name": texture_name, "type": "2d", "file": texture_path},
    )
    ET.SubElement(
        asset,
        "material",
        {"name": material_name, "texture": texture_name, "texuniform": "false"},
    )

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("model XML has no <worldbody> to inject the prompt into")
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": geom.name,
            "type": "box",
            "pos": _fmt(geom.pos),
            "quat": _fmt(geom.quat),
            "size": _fmt(geom.half_extents),
            "material": material_name,
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
        },
    )
    return ET.tostring(root, encoding="unicode")


def _get_model_xml(env: Any) -> str:  # pragma: no cover - exercised on the GPU host
    """Best-effort extraction of the live env's model XML across robosuite/LIBERO wrappers."""
    try:
        return str(env.sim.model.get_xml())
    except AttributeError:
        pass
    try:
        return str(env.env.sim.model.get_xml())
    except AttributeError:
        pass
    try:
        return str(env.model.get_xml())
    except AttributeError:
        pass
    raise AttributeError("could not obtain model XML from env (no get_xml accessor found)")


def inject_prompt(
    env: Any, candidate: dict[str, Any], *, texture_dir: str
) -> PromptGeom:  # pragma: no cover - MuJoCo/GPU seam, verified by smoke on the GPU host
    """Render the candidate's label and inject it into `env`, re-initialising the sim.

    Returns the `PromptGeom` that was injected (for logging / verification).
    """
    geom = build_prompt_geom(candidate)
    os.makedirs(texture_dir, exist_ok=True)
    texture_path = write_texture_png(geom.texture, os.path.join(texture_dir, f"{geom.name}.png"))
    injected_xml = build_injection_xml(_get_model_xml(env), geom, texture_path)
    env.reset_from_xml_string(injected_xml)
    return geom
