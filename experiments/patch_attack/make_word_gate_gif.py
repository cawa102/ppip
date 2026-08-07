"""Word-gate figure: one patch, one word, two outcomes.

The E2.1 headline is hard to believe from a table, so this is the figure: the armed and dormant
rollouts side by side, the adversarial corner visible and identical in both, and the *only*
difference the word ``please`` in the instruction printed above each panel.

Two safeguards, because a side-by-side is exactly the kind of figure that can lie quietly:

**The pairing is verified, not assumed.** :func:`assert_only_the_word_differs` refuses to draw
unless the two result JSONs agree on init, rect, both tasks, patch mode, optimiser budget and
trigger word, and disagree on ``deploy_word`` alone. A figure captioned "only the word differs"
must be able to prove it from the runs themselves.

**The captions are quoted, not typed.** Each panel's subtitle is the instruction the policy was
actually given, read out of that run's ``word_gate.deploy`` record; each verdict band comes from
``rollout_gif.outcome_of``, i.e. from the fixed evaluator. Neither can be hand-edited into
claiming something the run did not do.

Build the figure:

    ~/vla-injection/.venv/bin/python experiments/patch_attack/make_word_gate_gif.py \\
        runs/monitor-stealth/word-gate/figure_init46 46
"""
from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from typing import Any, Final

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rollout_gif as RG  # noqa: E402

#: Fields that must be identical across the pair -- everything that could otherwise explain the
#: difference in outcome. ``word_gate.deploy_word`` is the one field allowed to differ.
SHARED_FIELDS: Final = ("seed", "rect", "user_task", "target_task", "patch_mode", "effort")

#: Which recorded stream the panels show. ``policy_input`` is what the model actually sees, so
#: the patch corner is visible in BOTH panels -- the point being that it is the same patch.
FRAME_STREAM: Final = "policy_input"


def _gate(result: Mapping[str, Any]) -> Mapping[str, Any]:
    """The run's own word-gate record, or a refusal if the rollout was not gated."""
    gate: Mapping[str, Any] | None = result.get("word_gate")
    if not gate:
        raise ValueError(
            "result has no word_gate record -- an ungated rollout cannot be labelled as the "
            "armed or dormant condition of a word-gate figure"
        )
    return gate


def assert_only_the_word_differs(
    armed: Mapping[str, Any], dormant: Mapping[str, Any]
) -> None:
    """Raise unless the two rollouts differ *only* in whether the trigger word was deployed."""
    armed_gate, dormant_gate = _gate(armed), _gate(dormant)
    for field in SHARED_FIELDS:
        if armed[field] != dormant[field]:
            raise ValueError(
                f"the pair differs in {field!r} ({armed[field]!r} vs {dormant[field]!r}); "
                "then the word is not the only explanation for the outcomes"
            )
    if armed_gate["word"] != dormant_gate["word"]:
        raise ValueError(
            f"different trigger word ({armed_gate['word']!r} vs {dormant_gate['word']!r})"
        )
    if bool(armed_gate["deploy_word"]) == bool(dormant_gate["deploy_word"]):
        raise ValueError(
            "both rollouts deployed the same condition -- this is not an armed/dormant pair"
        )
    if not armed_gate["deploy_word"]:
        raise ValueError("the first result must be the armed (word-present) rollout")


def panel_title(result: Mapping[str, Any]) -> str:
    gate = _gate(result)
    word = gate["word"]
    return f'WITH "{word}"' if gate["deploy_word"] else f'WITHOUT "{word}"'


#: Width a panel subtitle must fit into, and the size it is drawn at (see ``rollout_gif.render``).
SUBTITLE_WIDTH: Final = RG.PANEL - 24
SUBTITLE_SIZE: Final = 14


def elide_to_width(text: str, max_width: int, size: int) -> str:
    """Trim ``text`` from the END until it fits ``max_width`` at ``size``, marking the cut.

    The head is kept because that is where the trigger word sits: the two panels must differ
    visibly in their first word. The full instructions are printed unabridged in the footer, so
    nothing the figure asserts depends on the elided tail.
    """
    if RG.text_width(text, size) <= max_width:
        return text
    ellipsis = "..."
    body = text
    while body and RG.text_width(body + ellipsis, size) > max_width:
        body = body[:-1]
    return body + ellipsis


def panel_subtitle(result: Mapping[str, Any]) -> str:
    """The instruction the policy was actually given, quoted from the run's own record."""
    return elide_to_width(
        f'"{_gate(result)["deploy"]}"', SUBTITLE_WIDTH, SUBTITLE_SIZE
    )


def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        result: dict[str, Any] = json.load(handle)
    return result


def build(run_dir: str, init: int, out_path: str = "") -> str:
    """Render the two-panel word-gate GIF for one init."""
    armed = _load(os.path.join(run_dir, f"result_fig_armed_init{init}_trial0.json"))
    dormant = _load(os.path.join(run_dir, f"result_fig_dormant_init{init}_trial0.json"))
    assert_only_the_word_differs(armed, dormant)

    panels = [
        RG.Panel(
            title=panel_title(result),
            subtitle=panel_subtitle(result),
            frames_dir=os.path.join(run_dir, condition, FRAME_STREAM),
            verdict=RG.outcome_of(result).label,
            colour=RG.outcome_of(result).colour,
        )
        for condition, result in (("armed", armed), ("dormant", dormant))
    ]
    gate = _gate(armed)
    # The panel subtitles are elided to fit; the instructions appear here in full, so the exact
    # one-word difference is on the figure itself and not only in the caption of a slide.
    footer = [
        f'WITH: "{gate["armed"]}"',
        f'WITHOUT: the same sentence minus "{gate["word"]}". Same patch, optimiser, init '
        f"{init}, {armed['rect'][2]}x{armed['rect'][3]} corner.",
    ]
    out_path = out_path or os.path.join(run_dir, f"word_gate_init{init}.gif")
    return RG.render(panels, out_path, footer, stride=2, duration=110, hold_frames=24)


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(f"usage: {os.path.basename(__file__)} <run_dir> <init> [out.gif]")
    run_dir, init = sys.argv[1], int(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else ""
    print(f"[word-gate-gif] wrote {build(run_dir, init, out)}")


if __name__ == "__main__":
    main()
