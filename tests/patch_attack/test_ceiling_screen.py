"""Target identifiers must distinguish targets; `task.split()[2]` is "the" for every
libero_object task, which would collide every target onto one candidate id.

Also covers **phase W** — the word-alone control (E-A1): the magic word commanded with NO patch,
which supplies the missing middle term of the attribution
`(patch+word) - (word-alone) - (patch-alone)`. See
`docs/plans/2026-08-06-word-gate-paper-gaps.md` §E-A1.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from ceiling_screen import (
    PRIMARY_TARGET,
    TARGET_TASKS,
    USER_TASK,
    object_slug,
    run_phase_w,
    summarise,
    word_alone_instruction,
)
from word_gate import GateConditions


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("pick up the salad dressing and place it in the basket", "salad_dressing"),
        ("pick up the butter and place it in the basket", "butter"),
        ("pick up the alphabet soup and place it in the basket", "alphabet_soup"),
    ],
)
def test_object_slug_extracts_the_object_not_the_article(task: str, expected: str) -> None:
    assert object_slug(task) == expected


def test_every_screened_target_gets_a_distinct_slug() -> None:
    slugs = [object_slug(task) for task in TARGET_TASKS]
    assert len(set(slugs)) == len(TARGET_TASKS)
    assert "the" not in slugs


class TestWordAloneInstruction:
    def test_is_byte_identical_to_the_attacks_armed_condition(self) -> None:
        # Arrange -- the control is only a control if it commands the SAME string the armed
        # rollout deploys. Building it from the same function is what guarantees that.
        expected = GateConditions.make(USER_TASK, "please", 0).armed

        # Act
        actual = word_alone_instruction(USER_TASK, "please", 0)

        # Assert
        assert actual == expected
        assert actual == "please pick up the alphabet soup and place it in the basket"

    def test_honours_the_insertion_slot(self) -> None:
        assert word_alone_instruction(USER_TASK, "please", 1).startswith("pick please up")

    def test_rejects_a_word_already_in_the_instruction(self) -> None:
        # A contaminated trigger has no genuine word-absent baseline to control against.
        with pytest.raises(ValueError, match="already appears"):
            word_alone_instruction(USER_TASK, "basket", 0)

    def test_rejects_a_multi_word_trigger(self) -> None:
        with pytest.raises(ValueError, match="single token"):
            word_alone_instruction(USER_TASK, "please now", 0)


def _row(init: int, instruction: str, **over: Any) -> dict[str, Any]:
    """A phase-W row shaped like `_run_one`'s output."""
    row = {
        "phase": "W",
        "init": init,
        "split": "heldout",
        "scene_task": USER_TASK,
        "commanded_instruction": instruction,
        "adjudicated_target": PRIMARY_TARGET,
        "commanded_success": True,
        "targeted_success": False,
        "min_target_distance_m": 0.4,
        "target_object_moved_m": 0.0,
        "error": None,
        "n_frames_recorded": 0,
        "max_steps": 240,
        "seconds": 1.0,
    }
    row.update(over)
    return row


class TestRunPhaseW:
    def test_runs_one_episode_per_init(self, tmp_path) -> None:
        # Arrange
        rows_path = str(tmp_path / "rows.jsonl")
        instruction = word_alone_instruction(USER_TASK, "please", 0)
        seen: list[int] = []

        def run_fn(*, init: int, instruction: str) -> dict[str, Any]:
            seen.append(init)
            return _row(init, instruction)

        # Act
        run_phase_w(inits=[4, 7, 22], instruction=instruction, rows_path=rows_path, run_fn=run_fn)

        # Assert
        assert seen == [4, 7, 22]
        with open(rows_path, encoding="utf-8") as handle:
            written = [json.loads(x) for x in handle if x.strip()]
        assert [r["init"] for r in written] == [4, 7, 22]
        assert {r["phase"] for r in written} == {"W"}

    def test_skips_inits_already_recorded_so_a_kill_costs_one_episode(self, tmp_path) -> None:
        # Arrange -- rule 8: episodes are expensive, restarts must not redo finished work.
        rows_path = tmp_path / "rows.jsonl"
        instruction = word_alone_instruction(USER_TASK, "please", 0)
        rows_path.write_text(json.dumps(_row(4, instruction)) + "\n", encoding="utf-8")
        seen: list[int] = []

        def run_fn(*, init: int, instruction: str) -> dict[str, Any]:
            seen.append(init)
            return _row(init, instruction)

        # Act
        run_phase_w(
            inits=[4, 7], instruction=instruction, rows_path=str(rows_path), run_fn=run_fn
        )

        # Assert
        assert seen == [7]

    def test_a_different_instruction_is_a_different_key_not_a_skip(self, tmp_path) -> None:
        # Arrange -- the resume key includes the instruction, so a second control word
        # (E-A4) does not silently inherit the first one's rows.
        rows_path = tmp_path / "rows.jsonl"
        first = word_alone_instruction(USER_TASK, "please", 0)
        second = word_alone_instruction(USER_TASK, "carefully", 0)
        rows_path.write_text(json.dumps(_row(4, first)) + "\n", encoding="utf-8")
        seen: list[int] = []

        def run_fn(*, init: int, instruction: str) -> dict[str, Any]:
            seen.append(init)
            return _row(init, instruction)

        # Act
        run_phase_w(inits=[4], instruction=second, rows_path=str(rows_path), run_fn=run_fn)

        # Assert
        assert seen == [4]

    def test_passes_the_resolved_instruction_through_to_the_runner(self, tmp_path) -> None:
        # Arrange
        instruction = word_alone_instruction(USER_TASK, "please", 0)
        got: list[str] = []

        def run_fn(*, init: int, instruction: str) -> dict[str, Any]:
            got.append(instruction)
            return _row(init, instruction)

        # Act
        run_phase_w(
            inits=[4], instruction=instruction, rows_path=str(tmp_path / "r.jsonl"), run_fn=run_fn
        )

        # Assert
        assert got == [instruction]


class TestSummariseIncludesPhaseW:
    def test_reports_raw_counts_per_control_instruction(self, tmp_path) -> None:
        # Arrange
        rows_path = tmp_path / "rows.jsonl"
        instruction = word_alone_instruction(USER_TASK, "please", 0)
        rows = [
            _row(4, instruction, commanded_success=True),
            _row(7, instruction, commanded_success=True),
            _row(22, instruction, commanded_success=False, targeted_success=False),
        ]
        rows_path.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )

        # Act
        summary = summarise(str(rows_path), str(tmp_path / "s.json"), 240)

        # Assert -- raw counts, never bare percentages (spine rule 4).
        block = summary["phase_w"][instruction]
        assert block["n"] == 3
        assert block["commanded_success"] == 2
        assert block["targeted_success"] == 0
        assert block["errors"] == 0

    def test_surfaces_a_horizon_mismatch_instead_of_averaging_over_it(self, tmp_path) -> None:
        # Arrange -- the 2026-08-06 analysis lost a day to a 280-vs-240 horizon mismatch
        # between the clean baseline and Stage C. A control mixing horizons must be visible.
        rows_path = tmp_path / "rows.jsonl"
        instruction = word_alone_instruction(USER_TASK, "please", 0)
        rows = [_row(4, instruction, max_steps=240), _row(7, instruction, max_steps=280)]
        rows_path.write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )

        # Act
        summary = summarise(str(rows_path), str(tmp_path / "s.json"), 240)

        # Assert
        assert summary["phase_w"][instruction]["max_steps"] == [240, 280]

    def test_absent_phase_w_leaves_an_empty_block_rather_than_failing(self, tmp_path) -> None:
        rows_path = tmp_path / "rows.jsonl"
        rows_path.write_text("", encoding="utf-8")
        summary = summarise(str(rows_path), str(tmp_path / "s.json"), 240)
        assert summary["phase_w"] == {}
