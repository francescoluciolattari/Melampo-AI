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


# --------------------------------------------------------------------------
# Frame recognition: Fillmore's frame-evoking lexical units
# --------------------------------------------------------------------------


def test_the_two_questions_that_were_falling_through_are_now_recognised():
    """These are the actual FRAME_FREE_TEXT cases from the advanced bench set
    that prompted this analysis -- one extractive, one requiring a link the
    documents never state."""
    from melampo.reasoning.frame_answer import (
        FRAME_ATTRIBUTION,
        FRAME_RELEVANCE,
        recognise_frame,
    )

    assert recognise_frame(
        "What laboratory abnormality supports the imaging impression, and which document reports it?"
    ) == FRAME_ATTRIBUTION
    assert recognise_frame(
        "Does the family history have any bearing on today's aortic measurement, and why?"
    ) == FRAME_RELEVANCE


def test_relevance_wins_when_a_question_evokes_both():
    """A question with both an extractive and a relevance half must route to
    relevance: the extraction-answerable half would succeed silently while the
    other failed, which is the worse failure."""
    from melampo.reasoning.frame_answer import FRAME_RELEVANCE, recognise_frame

    assert recognise_frame(
        "Which document reports it, and does it have any bearing on the measurement?"
    ) == FRAME_RELEVANCE


def test_yes_no_finding_questions_are_recognised_as_the_finding_frame():
    from melampo.reasoning.frame_answer import FRAME_FINDING, recognise_frame

    assert recognise_frame("Is fever present according to the report?") == FRAME_FINDING
    assert recognise_frame("Is a pericardial effusion present, and how does it differ?") == FRAME_FINDING


def test_an_unrecognised_question_falls_back_to_free_text_rather_than_guessing():
    """Honest "no frame recognised" rather than forcing an ill-fitting one --
    a missed frame degrades to the old behaviour, while a wrong frame would
    parse an answer into slots it was never asked to fill."""
    from melampo.reasoning.frame_answer import FRAME_FREE_TEXT, recognise_frame

    assert recognise_frame("How long has the dyspnoea been present?") == FRAME_FREE_TEXT
    assert recognise_frame("Summarise the case.") == FRAME_FREE_TEXT


def test_recognition_is_case_and_whitespace_insensitive():
    from melampo.reasoning.frame_answer import FRAME_RELEVANCE, recognise_frame

    assert recognise_frame("  DOES THIS  HAVE ANY BEARING ON that?  ") == FRAME_RELEVANCE


def test_recognition_consults_no_model_and_is_deterministic():
    """Inspectable and unable to hallucinate its own classification -- the
    same question always routes the same way."""
    from melampo.reasoning.frame_answer import recognise_frame

    question = "Does the family history have any bearing on today's aortic measurement?"
    assert recognise_frame(question) == recognise_frame(question) == recognise_frame(question)


# --------------------------------------------------------------------------
# The relevance frame: a judgment conflict is its own kind of contradiction
# --------------------------------------------------------------------------


def test_two_models_disagreeing_on_whether_a_link_exists_is_a_judgment_conflict():
    """The relevance analogue of polarity_conflict: disagreeing about whether
    a factor bears on a target at all is contradiction, not two descriptions."""
    from melampo.reasoning.frame_answer import FRAME_RELEVANCE

    comparison = compare_frame_answers(
        FRAME_RELEVANCE,
        "marfan syndrome | aortic root 4.8 cm | yes | connective tissue weakness",
        "marfan syndrome | aortic root 4.8 cm | no | ",
    )
    assert comparison.agrees is False
    assert comparison.judgment_conflict is True
    assert comparison.contradicts is True


def test_agreeing_that_a_link_exists_while_naming_different_mechanisms_is_not_a_contradiction():
    """Both assert the link; they differ on how it works. That is a
    disagreement worth reviewing but not the same class as one saying yes and
    the other no."""
    from melampo.reasoning.frame_answer import FRAME_RELEVANCE

    comparison = compare_frame_answers(
        FRAME_RELEVANCE,
        "marfan syndrome | aortic root | yes | connective tissue weakness",
        "marfan syndrome | aortic root | yes | inherited aortopathy",
    )
    assert comparison.judgment_conflict is False
    assert comparison.contradicts is False
    assert "mechanism" in comparison.conflicting_slots


def test_contradicts_covers_both_kinds_of_direct_opposition():
    from melampo.reasoning.frame_answer import FRAME_FINDING, FRAME_RELEVANCE

    polarity = compare_frame_answers(
        FRAME_FINDING, "embolism | pulmonary | affirmed", "embolism | pulmonary | negated"
    )
    judgment = compare_frame_answers(
        FRAME_RELEVANCE, "a | b | yes | m", "a | b | no | m"
    )
    assert polarity.contradicts is True
    assert judgment.contradicts is True


# --------------------------------------------------------------------------
# The attribution frame
# --------------------------------------------------------------------------


def test_attribution_separates_the_finding_from_the_document_reporting_it():
    """The Statement frame's Source role is what distinguishes this from
    FRAME_FINDING: the question asks not only what, but which document."""
    from melampo.reasoning.frame_answer import FRAME_ATTRIBUTION, FRAME_SLOTS

    assert "source_document" in FRAME_SLOTS[FRAME_ATTRIBUTION]

    comparison = compare_frame_answers(
        FRAME_ATTRIBUTION,
        "elevated CRP | infective process | report_3b | affirmed",
        "elevated CRP | infective process | report_3a | affirmed",
    )
    assert comparison.conflicting_slots == ["source_document"]


# --------------------------------------------------------------------------
# Instructions match the frame's actual slots
# --------------------------------------------------------------------------


def test_the_relevance_instruction_does_not_mention_polarity_which_it_has_no_slot_for():
    """Telling a model to fill a slot its frame does not contain invites it to
    invent one, or to distrust the rest of the instruction."""
    from melampo.reasoning.frame_answer import FRAME_RELEVANCE

    instruction = frame_prompt_instruction(FRAME_RELEVANCE)
    assert "polarity" not in instruction
    assert "bears_on" in instruction


def test_the_finding_instruction_still_mentions_polarity():
    instruction = frame_prompt_instruction(FRAME_FINDING)
    assert "polarity" in instruction


def test_the_relevance_instruction_asks_for_a_mechanism_not_a_restatement():
    from melampo.reasoning.frame_answer import FRAME_RELEVANCE

    assert "not to restate the question" in frame_prompt_instruction(FRAME_RELEVANCE)
