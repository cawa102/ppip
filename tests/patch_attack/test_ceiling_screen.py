"""Target identifiers must distinguish targets; `task.split()[2]` is "the" for every
libero_object task, which would collide every target onto one candidate id."""

from __future__ import annotations

import pytest
from ceiling_screen import TARGET_TASKS, object_slug


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
