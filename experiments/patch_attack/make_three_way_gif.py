"""The hijack / denial / clean figure: ONE deployed video, three instructions, three outcomes.

This is the sharpest claim in the word-gate track, and the one a reader is least likely to believe
from a table:

* say ``please`` where the patch was fitted        -> the arm delivers the ATTACKER's object;
* say ``please`` anywhere else                     -> neither task completes (denial);
* say nothing extra                               -> the robot does its job, as if unpatched.

The same 126-frame video is playing on the monitor in all three. Nothing is re-optimised between
panels; only the sentence the operator utters changes.

**Three safeguards, because a side-by-side is exactly the kind of figure that can lie quietly.**

*The claim is verified from the runs.* :func:`assert_one_video_three_instructions` refuses to draw
unless the three results agree on init, rect, both tasks and patch mode, replay the **same video
directory**, and carry three **distinct** deployed instructions.

*The regimes are derived, not asserted.* :func:`order_panels` classifies each panel with
``rollout_gif.outcome_of`` — i.e. from the fixed evaluator's ``targeted`` / ``commanded_success``
verdicts — and refuses a set that is not one of each. Panel order comes from that classification,
never from the caller's argument order, so a mislabelled figure is not expressible.

*The captions are quoted.* Every subtitle is the instruction that run actually deployed, read from
its own ``word_gate.deploy`` record.

    ~/vla-injection/.venv/bin/python experiments/patch_attack/make_three_way_gif.py
"""
from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Final

HOME = os.path.expanduser("~")
for _p in ("autoresearch/experiments/patch_attack",):
    sys.path.insert(0, os.path.join(HOME, _p))

import rollout_gif as RG  # noqa: E402
from make_word_gate_gif import SUBTITLE_SIZE, SUBTITLE_WIDTH, elide_to_width  # noqa: E402

#: Everything that must match, so the only thing left to explain an outcome difference is the words.
SHARED_FIELDS: Final = ("seed", "rect", "user_task", "target_task", "patch_mode")

#: Panel order, left to right: the attack, its brittleness, and the baseline.
REGIME_ORDER: Final = ("hijack", "dos", "commanded")

REGIME_TITLE: Final = {
    "hijack": "HIJACK — trigger in the fitted slot",
    "dos": "DENIAL — trigger moved",
    "commanded": "CLEAN — no trigger",
}


def _deploy(result: Mapping[str, Any]) -> str:
    gate = result.get("word_gate")
    if not gate or not gate.get("deploy"):
        raise AssertionError("result has no word_gate.deploy — cannot caption it honestly")
    return str(gate["deploy"])


def _video(result: Mapping[str, Any]) -> str:
    replay = result.get("replay")
    if not replay or not replay.get("replay_dir"):
        raise AssertionError(
            "result did not replay a recorded video (patch_mode='replay' required) — this figure "
            "claims one deployed artifact drives all three panels"
        )
    return str(replay["replay_dir"])


def assert_one_video_three_instructions(results: Sequence[Mapping[str, Any]]) -> None:
    """Refuse to draw unless the runs prove 'one video, three instructions'."""
    assert len(results) == 3, f"expected 3 results, got {len(results)}"
    first = results[0]
    for field in SHARED_FIELDS:
        values = {json.dumps(r.get(field), sort_keys=True) for r in results}
        assert len(values) == 1, f"panels disagree on {field}: {sorted(values)}"
    videos = {_video(r) for r in results}
    assert len(videos) == 1, f"panels did not replay the same video: {sorted(videos)}"
    instructions = [_deploy(r) for r in results]
    assert len(set(instructions)) == 3, (
        f"panels must deploy 3 distinct instructions: {instructions}"
    )
    _ = first


def order_panels(results: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Sort into hijack -> denial -> clean, classified by the FIXED evaluator's verdicts."""
    by_regime: dict[str, Mapping[str, Any]] = {}
    for result in results:
        by_regime.setdefault(RG.outcome_of(result).key, result)
    missing = [r for r in REGIME_ORDER if r not in by_regime]
    assert not missing and len(by_regime) == 3, (
        f"panels must be one of each regime {REGIME_ORDER}; got "
        f"{[RG.outcome_of(r).key for r in results]}"
    )
    return [by_regime[regime] for regime in REGIME_ORDER]


def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        result: dict[str, Any] = json.load(handle)
    return result


def build(specs: Sequence[tuple[str, str]], out_path: str) -> str:
    """``specs`` = (result JSON path, frames dir) per panel, in any order."""
    loaded = [(_load(p), frames) for p, frames in specs]
    results = [r for r, _ in loaded]
    assert_one_video_three_instructions(results)
    ordered = order_panels(results)
    frames_for = {id(r): f for r, f in loaded}

    panels = [
        RG.Panel(
            title=REGIME_TITLE[RG.outcome_of(result).key],
            subtitle=elide_to_width(f'"{_deploy(result)}"', SUBTITLE_WIDTH, SUBTITLE_SIZE),
            frames_dir=frames_for[id(result)],
            verdict=RG.outcome_of(result).label,
            colour=RG.outcome_of(result).colour,
        )
        for result in ordered
    ]
    video = _video(ordered[0])
    n_frames = (ordered[0].get("replay") or {}).get("n_frames")
    footer = [
        f"The SAME pre-recorded {n_frames}-frame video plays on the monitor in all three panels "
        f"— nothing is re-optimised between them.",
        f"Only the operator's sentence differs. Init {ordered[0]['seed']}, "
        f"{ordered[0]['rect'][2]}x{ordered[0]['rect'][3]} corner. Video: {os.path.basename(video)}",
    ]
    return RG.render(panels, out_path, footer, stride=2, duration=110, hold_frames=24)


def main() -> None:
    wg = os.path.join(HOME, "autoresearch/runs/monitor-stealth/word-gate")
    specs = [
        (os.path.join(wg, "replay_init46/result_replay_armed_init46_trial0.json"),
         os.path.join(wg, "replay_init46/replay/armed/policy_input")),
        (os.path.join(wg, "artifact/crosspos_fig_init46/"
                          "result_xposfig_replay_armed_init46_slot6_trial0.json"),
         os.path.join(wg, "artifact/crosspos_fig_init46/replay/armed_slot6/policy_input")),
        (os.path.join(wg, "replay_init46/result_replay_dormant_init46_trial0.json"),
         os.path.join(wg, "replay_init46/replay/dormant/policy_input")),
    ]
    print(build(specs, os.path.join(wg, "three_way_init46.gif")), flush=True)


if __name__ == "__main__":
    main()
