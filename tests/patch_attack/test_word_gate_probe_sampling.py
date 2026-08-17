"""Frame selection for the open-loop word-gate probe.

The λ=1.0 baseline was measured over a **stratified** 32-frame sample — 4 evenly spaced frames from
each of the 8 `TRAIN_INITS` — and `runs/monitor-stealth/word-gate/lam1.0/README.md` warns why:
"taking the first N would have sampled one episode's consecutive, highly correlated frames".

`--limit` does exactly that wrong thing (`frames[:n]` over a sorted glob, so `--limit 32` is 32
consecutive frames of `init01`). It was used for the λ=0/0.1/0.3 runs' *replacement* commands by
mistake; these tests pin the stratified path so a probe can be put back on the baseline's footing.

The sampler itself is `objective_probe.stratified_sample` (already tested there) — reused, not
reimplemented. What is tested here is the path adapter that lets the word-gate probe use it.
"""

from __future__ import annotations

import pytest
from word_gate_probe import parse_frame_path, stratified_frame_paths


def _buffer(inits=(1, 13, 14, 18, 20, 34, 41, 44), steps: int = 180) -> list[str]:
    """A frame buffer shaped like `ceiling_screen._dump_frames` writes it."""
    return [f"/runs/frames/train/init{i:02d}/f{s:04d}.png" for i in inits for s in range(steps)]


def test_parse_frame_path_reads_init_and_step() -> None:
    frame = parse_frame_path("/runs/frames/train/init14/f0076.png")

    assert (frame.init, frame.step) == (14, 76)


def test_parse_frame_path_rejects_an_unparseable_path() -> None:
    # Silently dropping such a frame would shrink the sample without saying so.
    with pytest.raises(ValueError, match="cannot parse"):
        parse_frame_path("/runs/frames/train/whatever.png")


def test_stratified_selection_covers_every_init_not_just_the_first() -> None:
    # The exact defect: `--limit 32` over a sorted glob returns 32 consecutive frames of init01.
    paths = _buffer()

    chosen = stratified_frame_paths(paths, 32)

    assert len(chosen) == 32
    assert len({parse_frame_path(p).init for p in chosen}) == 8, "must span all 8 train inits"
    assert chosen != paths[:32]


def test_stratified_selection_spreads_within_each_init() -> None:
    """Four frames per init must not all be that episode's opening."""
    chosen = stratified_frame_paths(_buffer(), 32)

    by_init: dict[int, list[int]] = {}
    for p in chosen:
        frame = parse_frame_path(p)
        by_init.setdefault(frame.init, []).append(frame.step)
    for init, steps in by_init.items():
        assert max(steps) - min(steps) > 40, f"init {init} sampled a single window: {steps}"


def test_stratified_selection_is_deterministic() -> None:
    """Resampling between λ points would compare them on different frames."""
    paths = _buffer()

    assert stratified_frame_paths(paths, 32) == stratified_frame_paths(paths, 32)


def test_requesting_more_than_the_buffer_returns_everything() -> None:
    paths = _buffer(inits=(1,), steps=5)

    assert stratified_frame_paths(paths, 32) == paths
