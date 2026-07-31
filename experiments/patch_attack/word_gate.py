"""Instruction construction for the word-gated patch experiment.

Pure, GPU-free core: builds the two language-instruction conditions the attack
compares —

* **dormant** — the operator's plain instruction ``c`` (no magic word); and
* **armed** — the same instruction with a natural "magic word" ``w`` inserted (``c ⊕ w``).

Everything here is string-level, so it is fully unit-testable without a model. The
downstream OpenVLA prompt template ``In: What action should the robot take to
{task.lower()}?\\nOut:`` (``adaptive_attack.py``) lowercases the task, so casing is
deliberately left to the caller.

The only new primitive the word-gated experiment needs on top of the existing patch
track: both teachers (target = ``_real_tokens(clean, target_task)``, clean =
``_real_tokens(clean, user_task)``) already exist. See
``docs/plans/2026-07-30-word-gated-patch.md`` (Exp 1 / Exp 2, WP1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

#: The magic word for the first milestone — a semantically neutral, unconscious word,
#: so any hijack is attributable to the *patch* being gated, not to the model obeying
#: the word. See docs/plans/2026-07-30-word-gated-patch.md (WP4).
FIRST_WORD: Final[str] = "please"


def normalize_instruction(instruction: str) -> str:
    """Collapse runs of whitespace to single spaces and strip the ends.

    Raises ``ValueError`` on a blank instruction — fail fast at the boundary rather
    than silently produce an empty prompt.
    """
    words = instruction.split()
    if not words:
        raise ValueError("instruction is blank after whitespace normalization")
    return " ".join(words)


def _validate_word(word: str) -> str:
    """Return the trimmed magic word, or raise if it is blank or multi-token.

    The threat model is a single *magic word*, not a phrase (a phrase would be the
    RoboGCG-style multi-token trigger this design deliberately avoids).
    """
    stripped = word.strip()
    if not stripped:
        raise ValueError("magic word is blank")
    if len(stripped.split()) != 1:
        raise ValueError(f"magic word must be a single token, got {word!r}")
    return stripped


def word_positions(instruction: str) -> tuple[int, ...]:
    """Valid insertion slots ``0..n`` inclusive, where ``n`` is the word count.

    ``n + 1`` slots: index ``0`` prepends, index ``n`` appends. This is the domain of
    the position sweep (E2.2).
    """
    n = len(normalize_instruction(instruction).split())
    return tuple(range(n + 1))


def insert_word(instruction: str, word: str, index: int) -> str:
    """Return ``instruction`` with ``word`` inserted at word-slot ``index``.

    ``index`` is a word gap: ``0`` prepends, ``len(words)`` appends. The instruction is
    normalized first; ``word`` must be a single token. Raises ``ValueError`` on an
    out-of-range index or an invalid word.
    """
    words = normalize_instruction(instruction).split()
    token = _validate_word(word)
    if not 0 <= index <= len(words):
        raise ValueError(f"index {index} out of range 0..{len(words)}")
    return " ".join(words[:index] + [token] + words[index:])


def is_trigger_novel(word: str, instruction: str) -> bool:
    """True if ``word`` does not already appear in ``instruction`` (case-insensitive).

    A contaminated dormant baseline — the magic word already present in ``c`` — has no
    genuine word-absent condition, so the probe/driver must reject such a pair.
    """
    token = _validate_word(word).lower()
    return token not in normalize_instruction(instruction).lower().split()


def assert_trigger_novel(word: str, instruction: str) -> None:
    """Raise ``ValueError`` if the magic word already appears in the instruction."""
    if not is_trigger_novel(word, instruction):
        raise ValueError(
            f"magic word {word!r} already appears in instruction {instruction!r}; "
            "the dormant condition would be contaminated"
        )


def dormant_instruction(instruction: str) -> str:
    """The condition with no magic word — just the normalized instruction."""
    return normalize_instruction(instruction)


def armed_instruction(instruction: str, word: str, index: int) -> str:
    """The condition with the magic word inserted at ``index``."""
    return insert_word(instruction, word, index)


@dataclass(frozen=True)
class GateConditions:
    """The two instruction strings a word-gated rollout pair is scored on.

    Immutable by design (coding-style: never mutate; a condition pair is a value).
    """

    dormant: str
    armed: str
    word: str
    index: int

    @classmethod
    def make(cls, instruction: str, word: str, index: int) -> GateConditions:
        """Build both conditions from a base instruction, word, and insertion slot."""
        return cls(
            dormant=dormant_instruction(instruction),
            armed=armed_instruction(instruction, word, index),
            word=_validate_word(word),
            index=index,
        )


def all_placements(instruction: str, word: str) -> tuple[GateConditions, ...]:
    """One :class:`GateConditions` per valid insertion slot — the position sweep (E2.2).

    Every element shares the same ``dormant`` string; only the armed insertion moves.
    """
    return tuple(
        GateConditions.make(instruction, word, i) for i in word_positions(instruction)
    )


@dataclass(frozen=True)
class GateSetup:
    """The resolved instructions for one word-gated confined rollout (WP7).

    Wraps a :class:`GateConditions` with which condition is *deployed* — the instruction the
    executed action is actually taken under. The per-frame two-branch optimizer is IDENTICAL in
    the armed and dormant rollouts (it must satisfy both word conditions, because a placed patch
    is physically present whether or not the operator utters ``w``); only ``deploy`` — what drives
    the environment this rollout — differs. See ``docs/plans/2026-07-30-word-gated-patch.md``.
    """

    conditions: GateConditions
    deploy_word: bool

    @property
    def armed(self) -> str:
        """The word-present branch prompt ``c⊕w`` (its teacher is the attacker target action)."""
        return self.conditions.armed

    @property
    def dormant(self) -> str:
        """The word-absent branch prompt ``c`` (its teacher is the clean policy action)."""
        return self.conditions.dormant

    @property
    def deploy(self) -> str:
        """Instruction the executed action is taken under: armed if ``w`` uttered, else ``c``."""
        return self.conditions.armed if self.deploy_word else self.conditions.dormant


def resolve_gate_setup(
    user_task: str, gate_word: str | None, word_index: int, deploy_word: bool
) -> GateSetup | None:
    """Resolve the word gate for one rollout, or ``None`` for the ungated default.

    ``gate_word is None`` returns ``None`` — the signal for the caller to take its original
    single-branch path unchanged (behavior-preserving default). Otherwise the trigger is validated
    as novel (an uncontaminated dormant baseline, ``assert_trigger_novel``) and both conditions are
    built at ``word_index``; ``deploy_word`` records whether this rollout deploys the worded
    instruction (armed) or the plain one (dormant).
    """
    if gate_word is None:
        return None
    assert_trigger_novel(gate_word, user_task)
    return GateSetup(GateConditions.make(user_task, gate_word, word_index), deploy_word)
