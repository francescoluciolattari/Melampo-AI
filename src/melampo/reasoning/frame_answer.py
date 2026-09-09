"""Compare two models' answers by frame slots rather than by characters.

A measurement in `root_model_cross_check` showed why character comparison
cannot work here. On this bench's own vocabulary, "pulmonary embolism" and
"pulmonary oedema" -- clinically opposite findings -- score 0.706, while
"40 mg daily" and "prednisone 40 mg daily" -- the same answer, one more
verbose -- score 0.667. The mechanics are purely arithmetic: the first pair
shares 12 characters (the word "pulmonary ", plus "em" appearing by
coincidence in embolism and oedema) out of 34 total; the second shares 11 --
the entire shorter answer -- out of 33, penalised because "prednisone"
inflates the denominator. A character counter cannot know that "pulmonary"
discriminates nothing in a medical vocabulary (it prefixes embolism, oedema,
fibrosis, hypertension alike) while "embolism" versus "oedema" discriminates
everything.

Containment patched the specific pair. This module removes the need for the
patch: if both models answer by filling the slots of a named frame, the
comparison is slot against slot, and two different findings land in the same
slot with different values -- visibly, regardless of what words they share.

**This is the project's own theory applied one level up, not a new idea.**
`memory/assertion.py` already encodes Fauconnier's mental spaces as
POLARITY_AFFIRMED / POLARITY_NEGATED: "pulmonary embolism, excluded" and
"pulmonary embolism, confirmed" name the same finding in different mental
spaces, and a character comparison sees them as nearly identical while a
polarity slot sees them as opposite. `reasoning/illness_script.py` already
structures a diagnosis as a frame with roles rather than as prose. The
frames here reuse those constants directly rather than defining a parallel
vocabulary that could drift from them.

What this does not do: infer a frame from free text. That would be extraction,
which is the sub-model's job and a different problem. These frames are what a
root model is *asked* to produce -- the answer format itself becomes
structured, so no interpretation stands between the two answers and their
comparison.
"""

from dataclasses import dataclass, field
from typing import Any

from ..memory.assertion import POLARITY_AFFIRMED, POLARITY_NEGATED

# Frame names. Each names a kind of question this bench actually asks, and
# carries the slots an answer to that kind of question has to fill.
FRAME_MEDICATION = "medication"
FRAME_FINDING = "finding"
FRAME_MEASUREMENT = "measurement"
FRAME_FREE_TEXT = "free_text"

# Slot definitions per frame, in the order a model is asked to state them.
# Order matters for the prompt (a fixed order is what makes the format
# learnable and checkable) but not for comparison, which is by slot name.
FRAME_SLOTS: dict[str, tuple[str, ...]] = {
    # "farmaco dose posologia" -- drug, dose, frequency -- plus the polarity
    # slot every clinical frame needs: a drug named as "not given" is not a
    # near-match for the same drug given.
    FRAME_MEDICATION: ("drug", "dose", "frequency", "polarity"),
    # The frame that motivated all of this. "site" separates "pulmonary
    # embolism" into finding=embolism, site=pulmonary -- so the shared word
    # that made the character comparison fail lands in its own slot, where
    # sharing it is correctly uninformative.
    FRAME_FINDING: ("finding", "site", "polarity"),
    FRAME_MEASUREMENT: ("quantity", "unit", "direction", "polarity"),
    # Deliberate escape hatch: not every question a bench asks decomposes
    # into slots ("how does the family history bear on today's measurement?").
    # Rather than forcing a frame that does not fit and getting a worse
    # answer, free_text falls back to whole-answer comparison, and says so.
    FRAME_FREE_TEXT: ("text",),
}

# Slots whose values are compared as a controlled vocabulary rather than as
# free strings: only these two values are meaningful, and anything else is
# treated as unstated rather than as a third option.
POLARITY_VALUES = (POLARITY_AFFIRMED, POLARITY_NEGATED)

SLOT_SEPARATOR = "|"
UNSTATED = ""


@dataclass(frozen=True)
class FrameAnswer:
    """One model's answer, decomposed into the slots of a named frame."""

    frame: str
    slots: dict[str, str] = field(default_factory=dict)
    raw: str = ""

    def value(self, slot: str) -> str:
        return normalise_slot_value(self.slots.get(slot, UNSTATED))

    def as_dict(self) -> dict[str, Any]:
        return {"frame": self.frame, "slots": dict(self.slots), "raw": self.raw}


def normalise_slot_value(value: str | None) -> str:
    """Lowercase, collapse whitespace, strip trailing punctuation.

    Minimal, and deliberately so: a slot value is already short and
    structured, so aggressive normalisation would only risk collapsing
    genuinely different values into agreement.
    """
    if not value:
        return UNSTATED
    return " ".join(str(value).lower().split()).strip().rstrip(".,;:!?")


def frame_prompt_instruction(frame: str) -> str:
    """The instruction to give a root model so its answer fills these slots.

    Returned rather than embedded in a prompt template here, so the caller
    composes it into whatever system prompt it already uses instead of this
    module owning prompt construction it has no other stake in.
    """
    slots = FRAME_SLOTS.get(frame)
    if not slots:
        raise ValueError(f"unknown frame {frame!r}; known frames: {sorted(FRAME_SLOTS)}")
    if frame == FRAME_FREE_TEXT:
        return "State your answer as a single short sentence."
    slot_list = f" {SLOT_SEPARATOR} ".join(slots)
    return (
        f"State your answer as {frame} slots in exactly this order, separated by "
        f"'{SLOT_SEPARATOR}': {slot_list}. Write '{UNSTATED or 'unknown'}' for any slot the "
        f"documents do not state. For 'polarity' write exactly "
        f"'{POLARITY_AFFIRMED}' or '{POLARITY_NEGATED}'."
    )


def parse_frame_answer(frame: str, answer: str | None) -> FrameAnswer:
    """Split a pipe-separated answer into the frame's slots.

    Tolerant by design about arity: an answer with fewer parts than the frame
    has slots leaves the rest unstated, and extra parts are dropped. A model
    that omits a trailing slot has still answered the ones it stated, and
    discarding the whole answer over a formatting slip would throw away the
    information the comparison needs.
    """
    slots = FRAME_SLOTS.get(frame)
    if not slots:
        raise ValueError(f"unknown frame {frame!r}; known frames: {sorted(FRAME_SLOTS)}")
    raw = answer or ""
    if frame == FRAME_FREE_TEXT:
        return FrameAnswer(frame=frame, slots={"text": raw.strip()}, raw=raw)

    parts = [part.strip() for part in raw.split(SLOT_SEPARATOR)]
    filled = {slot: (parts[index] if index < len(parts) else UNSTATED) for index, slot in enumerate(slots)}
    return FrameAnswer(frame=frame, slots=filled, raw=raw)


@dataclass
class SlotComparison:
    """How two answers compared on one slot."""

    slot: str
    primary_value: str
    secondary_value: str

    @property
    def both_stated(self) -> bool:
        return bool(self.primary_value) and bool(self.secondary_value)

    @property
    def agrees(self) -> bool:
        """Exact match on normalised values, or one side containing the other.

        Containment is kept here, at slot level, where it is safe: "40" inside
        "40" is trivially the same, and "embolism" is not contained in
        "oedema". At whole-answer level it was a patch over a broken
        comparison; at slot level it only forgives verbosity within one field.
        """
        if not self.both_stated:
            return False
        if self.primary_value == self.secondary_value:
            return True
        return self.primary_value in self.secondary_value or self.secondary_value in self.primary_value

    @property
    def conflicts(self) -> bool:
        """Both stated a value and they differ -- the case that matters most.

        Distinct from "does not agree": a slot one model left unstated is a
        gap, not a contradiction, and conflating the two would report a
        partial answer as a disagreement about substance.
        """
        return self.both_stated and not self.agrees

    def as_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "primary_value": self.primary_value,
            "secondary_value": self.secondary_value,
            "agrees": self.agrees,
            "conflicts": self.conflicts,
            "both_stated": self.both_stated,
        }


@dataclass
class FrameComparison:
    """The result of comparing two frame answers slot by slot."""

    frame: str
    slots: list[SlotComparison] = field(default_factory=list)

    @property
    def conflicting_slots(self) -> list[str]:
        return [item.slot for item in self.slots if item.conflicts]

    @property
    def agreeing_slots(self) -> list[str]:
        return [item.slot for item in self.slots if item.agrees]

    @property
    def unstated_slots(self) -> list[str]:
        return [item.slot for item in self.slots if not item.both_stated]

    @property
    def polarity_conflict(self) -> bool:
        """Whether the two answers disagree about affirmed vs negated.

        Surfaced on its own because it is categorically worse than any other
        slot conflict: two models naming the same finding but disagreeing on
        whether it is present are not partially agreeing, they are stating
        opposites. This is Fauconnier's mental spaces made checkable -- the
        same finding in an affirmed space and a negated space is not a near
        match, however similar the words.
        """
        return any(item.slot == "polarity" and item.conflicts for item in self.slots)

    @property
    def agrees(self) -> bool:
        """No slot conflicts, and at least one slot agreed on substance.

        Requiring at least one positive agreement matters: two answers that
        left every slot unstated have no conflicts either, and calling that
        agreement would make two non-answers look like corroboration.
        """
        return not self.conflicting_slots and bool(self.agreeing_slots)

    @property
    def agreement_ratio(self) -> float:
        """Agreeing slots as a fraction of slots both models actually stated."""
        comparable = [item for item in self.slots if item.both_stated]
        return len(self.agreeing_slots) / len(comparable) if comparable else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "agrees": self.agrees,
            "agreement_ratio": round(self.agreement_ratio, 3),
            "conflicting_slots": self.conflicting_slots,
            "agreeing_slots": self.agreeing_slots,
            "unstated_slots": self.unstated_slots,
            "polarity_conflict": self.polarity_conflict,
            "slots": [item.as_dict() for item in self.slots],
        }


def compare_frame_answers(frame: str, primary: str | None, secondary: str | None) -> FrameComparison:
    """Compare two answers to the same question, slot by slot."""
    left = parse_frame_answer(frame, primary)
    right = parse_frame_answer(frame, secondary)
    comparison = FrameComparison(frame=frame)
    for slot in FRAME_SLOTS[frame]:
        comparison.slots.append(
            SlotComparison(slot=slot, primary_value=left.value(slot), secondary_value=right.value(slot))
        )
    return comparison
