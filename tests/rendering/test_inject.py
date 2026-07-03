"""Contract of the MuJoCo prompt-injection XML builder (CPU-pure).

The risky, deterministic part of Option A injection is editing the model XML: adding
the label's texture/material assets and a visual-only textured box geom to the
worldbody. That is pure ElementTree work and is unit-tested here. Actually handing the
edited XML to a live MuJoCo sim (`reset_from_xml_string`) is a thin MuJoCo-dependent
seam verified separately in the configured GPU rollout environment.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from rendering.geometry import build_prompt_geom
from rendering.inject import build_injection_xml, write_texture_png
from tests.support import make_candidate

_BASE_XML = """
<mujoco model="libero">
  <asset>
    <texture name="floor" type="2d" builtin="checker"/>
  </asset>
  <worldbody>
    <geom name="table" type="box" size="0.5 0.5 0.02"/>
  </worldbody>
</mujoco>
"""


def test_write_texture_png_round_trips_the_image(tmp_path):
    from PIL import Image

    texture = build_prompt_geom(make_candidate()).texture
    out = write_texture_png(texture, str(tmp_path / "label.png"))

    loaded = np.asarray(Image.open(out).convert("RGB"), dtype=np.uint8)
    assert loaded.shape == texture.shape
    assert np.array_equal(loaded, texture)


def test_injection_adds_a_visual_only_textured_geom():
    geom = build_prompt_geom(make_candidate())
    xml = build_injection_xml(_BASE_XML, geom, "/tmp/label.png")
    root = ET.fromstring(xml)

    added = root.find(f".//worldbody/geom[@name='{geom.name}']")
    assert added is not None
    assert added.get("type") == "box"
    # visual-only: no collision with robot / task objects.
    assert added.get("contype") == "0"
    assert added.get("conaffinity") == "0"


def test_injection_preserves_the_existing_scene():
    geom = build_prompt_geom(make_candidate())
    xml = build_injection_xml(_BASE_XML, geom, "/tmp/label.png")
    root = ET.fromstring(xml)

    # The original table geom must survive.
    assert root.find(".//worldbody/geom[@name='table']") is not None


def test_injected_geom_carries_pose_and_size_from_the_spec():
    geom = build_prompt_geom(make_candidate())
    xml = build_injection_xml(_BASE_XML, geom, "/tmp/label.png")
    added = ET.fromstring(xml).find(f".//worldbody/geom[@name='{geom.name}']")

    pos = tuple(float(v) for v in added.get("pos").split())
    size = tuple(float(v) for v in added.get("size").split())
    assert pos == geom.pos
    assert all(abs(a - b) < 1e-6 for a, b in zip(size, geom.half_extents, strict=True))


def test_material_links_the_texture_file():
    geom = build_prompt_geom(make_candidate())
    xml = build_injection_xml(_BASE_XML, geom, "/tmp/my_label.png")
    root = ET.fromstring(xml)

    added = root.find(f".//worldbody/geom[@name='{geom.name}']")
    material_name = added.get("material")
    material = root.find(f".//asset/material[@name='{material_name}']")
    texture = root.find(f".//asset/texture[@name='{material.get('texture')}']")
    assert texture is not None
    assert texture.get("file") == "/tmp/my_label.png"


def test_injection_is_valid_parseable_xml_and_keeps_prior_assets():
    geom = build_prompt_geom(make_candidate())
    xml = build_injection_xml(_BASE_XML, geom, "/tmp/label.png")
    root = ET.fromstring(xml)  # must not raise

    assert root.find(".//asset/texture[@name='floor']") is not None
