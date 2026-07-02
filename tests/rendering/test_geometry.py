"""Contract of the prompt-geom spec builder (CPU-pure, GPU-independent).

`build_prompt_geom` maps a validated candidate's placement + style into the MuJoCo
parameters for a thin, visual-only textured panel: world position, orientation
quaternion, half-extents (sized from the rendered texture's aspect ratio and the
candidate scale), and the texture itself. This is pure math + PIL; the actual
insertion into a live MuJoCo model is a separate, MuJoCo-dependent step.
"""

from __future__ import annotations

import math

import numpy as np

from rendering.geometry import (
    PANEL_THICKNESS_M,
    PromptGeom,
    build_prompt_geom,
)
from rendering.text_prompt import render_prompt_from_candidate
from tests.support import make_candidate


def _norm(quat):
    return math.sqrt(sum(c * c for c in quat))


def test_position_is_taken_from_candidate_placement():
    candidate = make_candidate()
    candidate["placement"]["position"] = [0.2, -0.1, 0.05]

    geom = build_prompt_geom(candidate)

    assert isinstance(geom, PromptGeom)
    assert geom.pos == (0.2, -0.1, 0.05)


def test_identity_rotation_gives_identity_quaternion():
    candidate = make_candidate()
    candidate["placement"]["rotation"] = [0.0, 0.0, 0.0]

    geom = build_prompt_geom(candidate)

    assert geom.quat == (1.0, 0.0, 0.0, 0.0)


def test_ninety_degrees_about_z_gives_expected_quaternion():
    candidate = make_candidate()
    candidate["placement"]["rotation"] = [0.0, 0.0, 90.0]

    w, x, y, z = build_prompt_geom(candidate).quat

    assert math.isclose(w, math.sqrt(0.5), abs_tol=1e-6)
    assert math.isclose(x, 0.0, abs_tol=1e-6)
    assert math.isclose(y, 0.0, abs_tol=1e-6)
    assert math.isclose(z, math.sqrt(0.5), abs_tol=1e-6)


def test_quaternion_is_unit_length():
    candidate = make_candidate()
    candidate["placement"]["rotation"] = [30.0, 45.0, 60.0]

    assert math.isclose(_norm(build_prompt_geom(candidate).quat), 1.0, abs_tol=1e-6)


def test_panel_is_thin_along_its_normal():
    geom = build_prompt_geom(make_candidate())

    assert geom.half_extents[2] == PANEL_THICKNESS_M
    assert geom.half_extents[2] < geom.half_extents[0]
    assert geom.half_extents[2] < geom.half_extents[1]


def test_half_extents_match_texture_aspect_ratio():
    candidate = make_candidate()
    texture = render_prompt_from_candidate(candidate)
    height, width = texture.shape[0], texture.shape[1]

    geom = build_prompt_geom(candidate)
    half_w, half_h, _ = geom.half_extents

    assert math.isclose(half_w / half_h, width / height, rel_tol=1e-6)


def test_scale_grows_the_panel_linearly():
    small = build_prompt_geom(make_candidate(placement={
        "scene_anchor": "table_surface", "position": [0.1, 0.0, 0.02],
        "rotation": [0.0, 0.0, 0.0], "scale": 1.0,
    }))
    large = build_prompt_geom(make_candidate(placement={
        "scene_anchor": "table_surface", "position": [0.1, 0.0, 0.02],
        "rotation": [0.0, 0.0, 0.0], "scale": 2.0,
    }))

    assert math.isclose(large.half_extents[1], 2.0 * small.half_extents[1], rel_tol=1e-6)


def test_texture_is_the_rendered_prompt():
    candidate = make_candidate()

    geom = build_prompt_geom(candidate)

    assert np.array_equal(geom.texture, render_prompt_from_candidate(candidate))


def test_geom_name_is_a_deterministic_mujoco_safe_identifier():
    candidate = make_candidate(candidate_id="human_ppia_001")

    name = build_prompt_geom(candidate).name

    assert name == build_prompt_geom(candidate).name  # deterministic
    assert " " not in name and candidate["candidate_id"] in name
