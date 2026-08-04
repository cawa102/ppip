"""The off-camera carrier candidate every clean / frozen-patch evaluation is run under.

The fixed evaluator's episode path always injects a prompt geom, so a "clean" or
"patch-only" run is one whose geom sits **off-camera** (placement `[0.9, 0.5, 1.9]`, proven
visibility ~0 by `s0_reachability`). Keeping the injection in place rather than bypassing it
means every baseline, ceiling and attack number is measured under bit-identical scene
construction — nothing about the pipeline changes between conditions except the thing under
test.

This exists because the same ~30-line schema-shaped dict was being retyped in every driver;
a drifting copy (a different placement, a different `prompt_level`) would silently make two
conditions non-comparable.
"""

from __future__ import annotations

from typing import Any

#: Off-camera placement. Do NOT change: every recorded run used it, so a different value
#: would make new numbers incomparable with the existing ledger.
OFFCAMERA_POSITION: tuple[float, float, float] = (0.9, 0.5, 1.9)
OFFCAMERA_ROTATION: tuple[float, float, float] = (90.0, 90.0, 0.0)


def carrier_candidate(
    *,
    candidate_id: str,
    user_task: str,
    target_task: str,
    notes: str,
    created_at: str = "2026-07-28T00:00:00Z",
    created_by: str = "patch_attack",
) -> dict[str, Any]:
    """Build a schema-shaped candidate whose visual prompt is off-camera (vis ~0).

    `user_task` builds the scene and supplies the commanded instruction (unless a caller
    overrides it on the backend); `target_task` selects the predicate `targeted_success`
    is adjudicated against.
    """
    return {
        "candidate_id": candidate_id,
        "condition": "loop_with_skill",
        "user_task": user_task,
        "target_task": target_task,
        "visual_prompt": {
            "text": "(offcamera carrier)",
            "prompt_level": "optimized_typographic_prompt",
        },
        "placement": {
            "scene_anchor": "table_surface",
            "position": list(OFFCAMERA_POSITION),
            "rotation": list(OFFCAMERA_ROTATION),
            "scale": 0.6,
        },
        "style": {
            "foreground_color": "#000000",
            "background_color": "#ffffff",
            "font_size": 40,
            "font_family": "sans-serif",
        },
        "metadata": {"created_by": created_by, "created_at": created_at, "notes": notes},
    }
