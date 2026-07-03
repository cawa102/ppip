"""Tests for the autoresearch-style results.tsv experiment log helper."""

from __future__ import annotations

from pathlib import Path

from autoresearch_loop.results_tsv import (
    CRASH,
    DISCARD,
    KEEP,
    RESULTS_HEADER,
    append_results_row,
    decide_status,
    init_results_tsv,
)


def test_init_writes_header_row_once(tmp_path: Path) -> None:
    # Arrange
    tsv = tmp_path / "results.tsv"

    # Act
    init_results_tsv(tsv)

    # Assert
    assert tsv.read_text(encoding="utf-8") == "\t".join(RESULTS_HEADER) + "\n"


def test_init_is_idempotent_and_preserves_existing_rows(tmp_path: Path) -> None:
    # Arrange
    tsv = init_results_tsv(tmp_path / "results.tsv")
    append_results_row(
        tsv,
        candidate_id="c0",
        attack_score=0.5,
        targeted=1,
        commanded=0,
        completed=1,
        visibility=0.06,
        memory_gb=14.5,
        status=KEEP,
        description="baseline",
    )
    before = tsv.read_text(encoding="utf-8")

    # Act — re-initializing must not truncate the recorded row
    init_results_tsv(tsv)

    # Assert
    assert tsv.read_text(encoding="utf-8") == before


def test_append_writes_tab_separated_cells_with_none_as_dash(tmp_path: Path) -> None:
    # Arrange
    tsv = init_results_tsv(tmp_path / "results.tsv")

    # Act
    append_results_row(
        tsv,
        candidate_id="c1",
        attack_score=-1.0,
        targeted=0,
        commanded=2,
        completed=2,
        visibility=None,
        memory_gb=None,
        status=DISCARD,
        description="seen but not hijacked",
    )

    # Assert
    row = tsv.read_text(encoding="utf-8").splitlines()[1]
    cells = row.split("\t")
    assert cells == ["c1", "-1.0000", "0", "2", "2", "-", "-", DISCARD, "seen but not hijacked"]


def test_append_sanitizes_tabs_and_newlines_in_description(tmp_path: Path) -> None:
    # Arrange
    tsv = init_results_tsv(tmp_path / "results.tsv")

    # Act
    append_results_row(
        tsv,
        candidate_id="c2",
        attack_score=0.0,
        targeted=0,
        commanded=0,
        completed=1,
        visibility=0.01,
        memory_gb=1.0,
        status=KEEP,
        description="line one\twith tab\nand newline",
    )

    # Assert — the row must remain exactly 9 tab-separated columns
    row = tsv.read_text(encoding="utf-8").splitlines()[1]
    assert len(row.split("\t")) == len(RESULTS_HEADER)
    assert "line one with tab and newline" in row


def test_decide_status_crash_when_invalid() -> None:
    assert (
        decide_status(valid=False, completed_rollouts=0, attack_score=0.0, incumbent_score=None)
        == CRASH
    )


def test_decide_status_crash_when_no_completed_rollout() -> None:
    assert (
        decide_status(valid=True, completed_rollouts=0, attack_score=0.0, incumbent_score=-0.5)
        == CRASH
    )


def test_decide_status_keep_when_first_result() -> None:
    assert (
        decide_status(valid=True, completed_rollouts=1, attack_score=-1.0, incumbent_score=None)
        == KEEP
    )


def test_decide_status_keep_when_improved_on_incumbent() -> None:
    assert (
        decide_status(valid=True, completed_rollouts=2, attack_score=0.2, incumbent_score=-0.1)
        == KEEP
    )


def test_decide_status_discard_when_tied_or_worse() -> None:
    assert (
        decide_status(valid=True, completed_rollouts=2, attack_score=-0.1, incumbent_score=-0.1)
        == DISCARD
    )
    assert (
        decide_status(valid=True, completed_rollouts=2, attack_score=-0.3, incumbent_score=-0.1)
        == DISCARD
    )
