"""Tests for comparing two models' answers by frame slots rather than characters."""

import pytest

from melampo.memory.assertion import POLARITY_AFFIRMED, POLARITY_NEGATED
from melampo.reasoning.frame_answer import (
    FRAME_FINDING,
    FRAME_FREE_TEXT,
    FRAME_MEASUREMENT,
    FRAME_MEDICATION,
    FRAME_SLOTS,
    compare_frame_answers,
    frame_prompt_instruction,
    normalise_slot_value,
    parse_frame_answer,
)

# --------------------------------------------------------------------------
# The measurement that motivated this module
# --------------------------------------------------------------------------


def test_clinically_opposite_findings_conflict_on_the_finding_slot():
    """The case character comparison got exactly backwards: "pulmonary
    embolism" vs "pulmonary oedema" scored 0.706 -- higher than a correct
    paraphrase -- because they share the word "pulmonary" plus the letters
    "em" appearing by coincidence in embolism and oedema. Split into slots,
    the shared word lands in `site` where sharing it is correctly
    uninformative, and the conflict is localised to `finding`."""
    comparison = compare_frame_answers(
        FRAME_FINDING,
        f"embolism | pulmonary | {POLARITY_AFFIRMED}",
        f"oedema | pulmonary | {POLARITY_AFFIRMED}",
    )
    assert comparison.agrees is False
    assert comparison.conflicting_slots == ["finding"]
    assert "site" in comparison.agreeing_slots, "the shared word agrees where it is uninformative"


def test_the_same_answer_stated_more_verbosely_agrees():
    """The other half of the same measurement: "40 mg daily" vs "prednisone
    40 mg daily" scored only 0.667 on characters because the longer answer
    inflated the denominator, despite the shorter being contained entirely."""
    comparison = compare_frame_answers(
        FRAME_MEDICATION,
        f"prednisone | 40 mg | daily | {POLARITY_AFFIRMED}",
        f"prednisone | 40 mg | once daily | {POLARITY_AFFIRMED}",
    )
    assert comparison.agrees is True
    assert comparison.conflicting_slots == []


# --------------------------------------------------------------------------
# Polarity: Fauconnier's mental spaces, made checkable
# --------------------------------------------------------------------------


def test_the_same_finding_with_opposite_polarity_is_a_conflict():
    """"pulmonary embolism, confirmed" and "pulmonary embolism, excluded"
    name the same finding in different mental spaces. Character comparison
    sees them as nearly identical; a polarity slot sees them as opposite."""
    comparison = compare_frame_answers(
        FRAME_FINDING,
        f"embolism | pulmonary | {POLARITY_AFFIRMED}",
        f"embolism | pulmonary | {POLARITY_NEGATED}",
    )
    assert comparison.agrees is False
    assert comparison.polarity_conflict is True
    assert comparison.conflicting_slots == ["polarity"]


def test_polarity_conflict_is_surfaced_separately_from_other_conflicts():
    """Categorically worse than any other slot disagreement, so it gets its
    own flag rather than being one entry in a list."""
    same_polarity = compare_frame_answers(
        FRAME_FINDING,
        f"embolism | pulmonary | {POLARITY_AFFIRMED}",
        f"oedema | pulmonary | {POLARITY_AFFIRMED}",
    )
    assert same_polarity.conflicting_slots  # there IS a conflict
    assert same_polarity.polarity_conflict is False  # but not this kind


def test_polarity_agreement_alone_does_not_make_answers_agree():
    comparison = compare_frame_answers(
        FRAME_FINDING,
        f"embolism | pulmonary | {POLARITY_AFFIRMED}",
        f"pneumothorax | apical | {POLARITY_AFFIRMED}",
    )
    assert comparison.agrees is False


# --------------------------------------------------------------------------
# Unstated slots are gaps, not contradictions
# --------------------------------------------------------------------------


def test_a_slot_one_model_left_unstated_is_a_gap_not_a_conflict():
    """Conflating "did not say" with "said something different" would report
    a partial answer as a disagreement about substance."""
    comparison = compare_frame_answers(
        FRAME_MEDICATION,
        f"prednisone | 40 mg | daily | {POLARITY_AFFIRMED}",
        f"prednisone |  | daily | {POLARITY_AFFIRMED}",
    )
    assert comparison.agrees is True
    assert comparison.conflicting_slots == []
    assert "dose" in comparison.unstated_slots


def test_two_entirely_unstated_answers_do_not_count_as_agreement():
    """No conflicts, but nothing agreed either -- calling that agreement
    would make two non-answers look like corroboration."""
    comparison = compare_frame_answers(FRAME_MEDICATION, " |  |  | ", " |  |  | ")
    assert comparison.conflicting_slots == []
    assert comparison.agrees is False, "no conflicts is not the same as agreement"


def test_agreement_ratio_counts_only_slots_both_models_stated():
    comparison = compare_frame_answers(
        FRAME_MEDICATION,
        f"prednisone | 40 mg | daily | {POLARITY_AFFIRMED}",
        f"prednisone |  |  | {POLARITY_AFFIRMED}",
    )
    assert comparison.agreement_ratio == 1.0, "2 of 2 comparable slots agreed"


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_parsing_splits_on_the_separator_in_slot_order():
    parsed = parse_frame_answer(FRAME_MEDICATION, f"prednisone | 40 mg | daily | {POLARITY_AFFIRMED}")
    assert parsed.slots["drug"] == "prednisone"
    assert parsed.slots["dose"] == "40 mg"
    assert parsed.slots["frequency"] == "daily"
    assert parsed.slots["polarity"] == POLARITY_AFFIRMED


def test_an_answer_with_too_few_parts_leaves_the_rest_unstated():
    """A model omitting a trailing slot has still answered the ones it
    stated; discarding the whole answer over a formatting slip would throw
    away the information the comparison needs."""
    parsed = parse_frame_answer(FRAME_MEDICATION, "prednisone | 40 mg")
    assert parsed.slots["drug"] == "prednisone"
    assert parsed.slots["frequency"] == ""
    assert parsed.slots["polarity"] == ""


def test_an_answer_with_too_many_parts_drops_the_extras():
    parsed = parse_frame_answer(FRAME_FINDING, "embolism | pulmonary | affirmed | extra | more")
    assert set(parsed.slots) == set(FRAME_SLOTS[FRAME_FINDING])


def test_an_empty_answer_parses_to_all_unstated():
    parsed = parse_frame_answer(FRAME_MEDICATION, None)
    assert all(value == "" for value in parsed.slots.values())


def test_slot_values_are_normalised_for_case_and_whitespace():
    comparison = compare_frame_answers(
        FRAME_MEDICATION,
        "  PREDNISONE  |  40 MG  | Daily. | affirmed",
        f"prednisone | 40 mg | daily | {POLARITY_AFFIRMED}",
    )
    assert comparison.agrees is True


def test_normalise_handles_none_and_empty():
    assert normalise_slot_value(None) == ""
    assert normalise_slot_value("   ") == ""


# --------------------------------------------------------------------------
# Free text: the deliberate escape hatch
# --------------------------------------------------------------------------


def test_free_text_frame_keeps_the_whole_answer_in_one_slot():
    """Not every question decomposes into slots; forcing a frame that does
    not fit would get a worse answer, not a better comparison."""
    parsed = parse_frame_answer(FRAME_FREE_TEXT, "The family history raises the aortic threshold concern.")
    assert parsed.slots["text"] == "The family history raises the aortic threshold concern."


def test_free_text_answers_compare_as_whole_strings():
    same = compare_frame_answers(FRAME_FREE_TEXT, "the same thing", "the same thing")
    different = compare_frame_answers(FRAME_FREE_TEXT, "one thing", "a completely other thing")
    assert same.agrees is True
    assert different.agrees is False


# --------------------------------------------------------------------------
# Prompt instructions
# --------------------------------------------------------------------------


def test_the_instruction_names_every_slot_in_order():
    instruction = frame_prompt_instruction(FRAME_MEDICATION)
    for slot in FRAME_SLOTS[FRAME_MEDICATION]:
        assert slot in instruction
    assert instruction.index("drug") < instruction.index("dose") < instruction.index("frequency")


def test_the_instruction_constrains_polarity_to_the_two_valid_values():
    instruction = frame_prompt_instruction(FRAME_FINDING)
    assert POLARITY_AFFIRMED in instruction
    assert POLARITY_NEGATED in instruction


def test_an_unknown_frame_raises_rather_than_silently_producing_nothing():
    with pytest.raises(ValueError, match="unknown frame"):
        frame_prompt_instruction("not-a-frame")
    with pytest.raises(ValueError, match="unknown frame"):
        parse_frame_answer("not-a-frame", "anything")


# --------------------------------------------------------------------------
# Frame definitions reuse the project's existing vocabulary
# --------------------------------------------------------------------------


def test_polarity_values_come_from_the_assertion_module_not_a_parallel_vocabulary():
    """assertion.py already encodes mental spaces as affirmed/negated;
    defining a second set here would let the two drift apart."""
    from melampo.reasoning.frame_answer import POLARITY_VALUES

    assert POLARITY_VALUES == (POLARITY_AFFIRMED, POLARITY_NEGATED)


def test_every_clinical_frame_carries_a_polarity_slot():
    """A drug named as "not given" is not a near-match for the same drug
    given -- every clinical frame needs the affirmed/negated distinction."""
    for frame in (FRAME_MEDICATION, FRAME_FINDING, FRAME_MEASUREMENT):
        assert "polarity" in FRAME_SLOTS[frame]


def test_the_finding_frame_separates_site_from_finding():
    """The specific fix for the measurement that motivated this module: the
    shared word "pulmonary" must land in its own slot."""
    assert "site" in FRAME_SLOTS[FRAME_FINDING]
    assert "finding" in FRAME_SLOTS[FRAME_FINDING]
