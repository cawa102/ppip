"""Shared rendering for rollout comparison GIFs.

Every figure in the epsilon-threshold study is a side-by-side of rollouts that differ only in
what sits in the patch corner. Two things about those figures could silently lie, so both live
here, tested, instead of being retyped per figure:

**The outcome label.** Each panel is banded with which of the three outcome classes the
rollout landed in -- user task completed / DoS / hijack. That label is *derived* from the fixed
evaluator's own ``targeted`` and ``commanded_success`` fields by :func:`outcome_of`; it is never
written by hand. A result missing a verdict field raises rather than defaulting, because a
default of ``False`` would render an unjudged rollout as a confident "DENIED (DoS)".

**Frame alignment.** Rollouts have different lengths (a hijack ends at its latch). Panels are
aligned by *step index* and a finished rollout holds its last frame, so the GIF never shows
step 200 of one condition beside step 90 of another.

Colours encode the class, consistently across every figure: green = the user's task survived,
amber = denied, red = hijacked.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from PIL import Image, ImageDraw, ImageFont

HOME = os.path.expanduser("~")
RUNS = os.path.join(HOME, "autoresearch/runs")

PANEL: Final = 260  # per-panel side, px
HEADER: Final = 78  # title + subtitle + verdict band
FOOTER: Final = 52  # step counter + standing caption
BG: Final = (18, 18, 20)
FG: Final = (238, 238, 238)
DIM: Final = (150, 150, 155)
GOOD: Final = (110, 200, 130)  # user task survived
WARN: Final = (235, 175, 70)  # denied / redirected
BAD: Final = (235, 100, 100)  # hijacked


@dataclass(frozen=True)
class Outcome:
    """Which of the three outcome classes a rollout landed in, per the fixed evaluator."""

    key: str  # "commanded" | "dos" | "hijack"
    label: str
    colour: tuple[int, int, int]


@dataclass(frozen=True)
class Panel:
    """One rollout panel: where its frames live and how it is banded."""

    title: str
    subtitle: str
    frames_dir: str
    verdict: str
    colour: tuple[int, int, int]


def _verdict_flag(result: Mapping[str, Any], key: str) -> bool:
    if key not in result:
        raise KeyError(
            f"result has no {key!r} field -- it was not adjudicated by the evaluator, "
            "and defaulting it would mislabel the figure"
        )
    value = result[key]
    if not isinstance(value, bool):
        raise ValueError(f"{key!r} is {value!r}, not a bool -- the rollout was not judged")
    return value


def outcome_of(result: Mapping[str, Any]) -> Outcome:
    """Map a result JSON's evaluator verdicts onto the three outcome classes."""
    targeted = _verdict_flag(result, "targeted")
    commanded = _verdict_flag(result, "commanded_success")

    if targeted:
        # Both predicates latch independently, so both can fire; hijack is the headline class
        # but the label must not hide that the user's task also completed.
        label = (
            "HIJACKED (user task also done)" if commanded else "HIJACKED - target delivered"
        )
        return Outcome("hijack", label, BAD)
    if commanded:
        return Outcome("commanded", "user task DONE", GOOD)
    return Outcome("dos", "DENIED (DoS) - neither task", WARN)


def caption_for(result: Mapping[str, Any]) -> str:
    """One-line quantitative caption: the perturbation budget and how hard it forced."""
    stealth = result.get("stealth")
    if not stealth:
        budget = "unbounded (free-range)"
    else:
        eps = stealth["eps"]
        budget = f"eps = {eps:g}, measured Linf = {stealth['linf_measured_max']:.4f}"
        if not stealth.get("bound_holds", True):
            budget += "  [BOUND VIOLATED]"
    forcing = result.get("mean_decisive_forcing")
    forcing_text = "n/a" if forcing is None else f"{forcing:.3f}"
    return f"{budget}     mean decisive forcing {forcing_text}"


def aligned_index(step: int, n_frames: int) -> int:
    """Frame to show for a global `step`, holding the last one once the rollout has ended."""
    if n_frames <= 0:
        raise ValueError("condition has no frames to show")
    return min(step, n_frames - 1)


def sampled_steps(total: int, stride: int) -> list[int]:
    """Global step indices to render, from 0 up to `total`."""
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    return list(range(0, total, stride))


MIN_FONT: Final = 9  # below this the caption is present but unreadable


def font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def text_width(text: str, size: int) -> int:
    return int(font(size).getbbox(text)[2])


def fitted_size(text: str, max_width: int, start: int) -> int:
    """Largest font size <= `start` whose rendered `text` fits in `max_width`.

    Captions carry the epsilon and the measured Linf bound; a caption clipped by the canvas
    edge silently drops exactly the numbers the figure exists to report. Panel count varies
    per figure, so the width to fit into is not known when the caption is written.
    """
    if max_width <= 0:
        raise ValueError(f"max_width must be positive, got {max_width}")
    size = start
    while size > MIN_FONT and text_width(text, size) > max_width:
        size -= 1
    return size


def frame_files(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        raise SystemExit(f"missing frames dir: {directory}")
    files = sorted(
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".png")
    )
    if not files:
        raise SystemExit(f"no frames in: {directory}")
    return files


def render(
    panels: Sequence[Panel],
    out_path: str,
    footer_lines: Sequence[str],
    *,
    stride: int = 3,
    duration: int = 120,
    hold_frames: int = 18,
) -> str:
    """Render the panels side by side into an animated GIF."""
    if not panels:
        raise ValueError("no panels to render")
    files_per_panel = [frame_files(p.frames_dir) for p in panels]
    total = max(len(f) for f in files_per_panel)
    width = PANEL * len(panels)
    height = HEADER + PANEL + FOOTER

    title_font, sub_font = font(19), font(14)
    verdict_font, foot_font = font(13), font(15)
    rendered: list[Image.Image] = []

    for step in sampled_steps(total, stride):
        canvas = Image.new("RGB", (width, height), BG)
        draw = ImageDraw.Draw(canvas)
        for i, (panel, files) in enumerate(zip(panels, files_per_panel, strict=True)):
            x = i * PANEL
            idx = aligned_index(step, len(files))
            frame = Image.open(files[idx]).convert("RGB").resize((PANEL, PANEL), Image.NEAREST)
            canvas.paste(frame, (x, HEADER))

            draw.text((x + 12, 10), panel.title, font=title_font, fill=FG)
            # Fitted like the footer: a subtitle wider than its panel does not clip, it runs
            # into the NEXT panel's header and reads as that panel's caption.
            draw.text(
                (x + 12, 33),
                panel.subtitle,
                font=font(fitted_size(panel.subtitle, PANEL - 24, sub_font.size)),
                fill=DIM,
            )
            draw.text((x + 12, 55), panel.verdict, font=verdict_font, fill=panel.colour)
            # A finished rollout is marked so a still panel reads as "done", not "frozen".
            if idx == len(files) - 1 and step > idx:
                draw.text(
                    (x + PANEL - 52, HEADER + PANEL - 20), "ended", font=sub_font, fill=DIM
                )
            if i:
                draw.line([(x, 0), (x, height)], fill=(52, 52, 56), width=1)

        for row, line in enumerate(footer_lines):
            colour = FG if row == 0 else DIM
            text = line.replace("{step}", f"{step:3d}")
            draw.text(
                (12, height - 44 + row * 21),
                text,
                font=font(fitted_size(text, width - 24, foot_font.size)),
                fill=colour,
            )
        rendered.append(canvas)

    # Hold the final comparison so the ending is readable rather than a flash.
    rendered.extend([rendered[-1]] * hold_frames)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    first, *rest = [im.quantize(colors=128, method=Image.MEDIANCUT) for im in rendered]
    first.save(
        out_path,
        save_all=True,
        append_images=rest,
        duration=duration,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return out_path
