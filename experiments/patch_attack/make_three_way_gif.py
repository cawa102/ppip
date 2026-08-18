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

#: Kept short deliberately: ``rollout_gif`` draws titles WITHOUT elision at ``x + 12``, so an
#: overlong title runs straight into the next panel. The first build rendered
#: "HIJACK — trigger in the fDENIALot trigger moved". A test pins the width.
REGIME_TITLE: Final = {
    "hijack": "HIJACK — fitted slot",
    "dos": "DENIAL — moved",
    "commanded": "CLEAN — no trigger",
}


def trigger_window(instruction: str, word: str, max_width: int, size: int) -> str:
    """A window of ``instruction`` centred on ``word``, sized to fit ``max_width``.

    The two-panel figure could elide from the end, because there the trigger always sat at the
    front and the panels differed in their first word. Here the trigger MOVES, so head-elision
    renders all three panels as "pick up the alphabet soup ..." — identical, hiding the single
    thing the figure exists to show. This expands outward from the trigger word instead, marking
    each truncated side, and falls back to head-elision for the no-trigger panel.

    The text stays a verbatim quote of what the run deployed; only the window moves.
    """
    words = instruction.split()
    lowered = [w.lower() for w in words]
    if word.lower() not in lowered:
        return elide_to_width(instruction, max_width, size)

    centre = lowered.index(word.lower())
    lo = hi = centre

    def rendered(lo: int, hi: int) -> str:
        body = " ".join(words[lo : hi + 1])
        return ("..." if lo > 0 else "") + body + ("..." if hi < len(words) - 1 else "")

    while True:
        grew = False
        for nxt in ((lo, hi + 1), (lo - 1, hi)):  # bias right, so the trigger sits left-of-centre
            new_lo, new_hi = nxt
            if not (new_lo >= 0 and new_hi < len(words)):
                continue
            if RG.text_width(rendered(new_lo, new_hi), size) <= max_width:
                lo, hi = new_lo, new_hi
                grew = True
        if not grew:
            break
    return rendered(lo, hi)


def _deploy(result: Mapping[str, Any]) -> str:
    gate = result.get("word_gate")
    if not gate or not gate.get("deploy"):
        raise AssertionError("result has no word_gate.deploy — cannot caption it honestly")
    return str(gate["deploy"])


def _video(result: Mapping[str, Any]) -> str:
    """The replayed video directory, RESOLVED.

    The driver records whatever path string it was invoked with, so the same directory appears
    absolute in one run and relative in another. Comparing the raw text rejected a legitimate
    figure whose panels replayed the identical video, so the identity check is on the resolved
    path — normalised enough to see through spelling, not so much that two different videos pass.
    """
    replay = result.get("replay")
    if not replay or not replay.get("replay_dir"):
        raise AssertionError(
            "result did not replay a recorded video (patch_mode='replay' required) — this figure "
            "claims one deployed artifact drives all three panels"
        )
    return os.path.realpath(str(replay["replay_dir"]))


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
    trigger = str((ordered[0].get("word_gate") or {}).get("word") or "please")

    panels = [
        RG.Panel(
            title=REGIME_TITLE[RG.outcome_of(result).key],
            subtitle=trigger_window(_deploy(result), trigger, SUBTITLE_WIDTH, SUBTITLE_SIZE),
            frames_dir=frames_for[id(result)],
            verdict=RG.outcome_of(result).label,
            colour=RG.outcome_of(result).colour,
        )
        for result in ordered
    ]
    n_frames = (ordered[0].get("replay") or {}).get("n_frames")
    footer = [
        f"The SAME pre-recorded {n_frames}-frame video plays on the monitor in all three panels "
        f"— nothing is re-optimised between them.",
        f"Only the operator's sentence differs — where \"{trigger}\" sits, or whether it is "
        f"said at all. Init {ordered[0]['seed']}, "
        f"{ordered[0]['rect'][2]}x{ordered[0]['rect'][3]} corner.",
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
