import pytest

from melampo.memory.assertion import (
    CERTAINTY_FACTUAL,
    CERTAINTY_HYPOTHETICAL,
    CERTAINTY_POSSIBLE,
    EXPERIENCER_OTHER,
    EXPERIENCER_PATIENT,
    ITALIAN_CUES,
    POLARITY_AFFIRMED,
    POLARITY_NEGATED,
    SOURCE_OBJECTIVE,
    SOURCE_SUBJECTIVE,
    STATE_DOCUMENTED,
    STATE_DOCUMENTED_EXCLUSION,
    STATE_GAP,
    STATE_WEAK_NEGATION,
    TEMPORALITY_CURRENT,
    TEMPORALITY_HISTORICAL,
    AssertionDetector,
    AssertionStatus,
)
from melampo.reasoning.findings_boundary import (
    REJECT_HYPOTHETICAL,
    REJECT_NEGATED,
    REJECT_OTHER_EXPERIENCER,
    REJECT_SCREENING,
    REJECT_SYNTHETIC,
    assemble,
    assert_findings_only,
)


def _detect(text: str, term: str, cues=None) -> AssertionStatus:
    start = text.lower().index(term.lower())
    detector = AssertionDetector(cues=cues) if cues else AssertionDetector()
    return detector.detect(text, start, start + len(term))


# --------------------------------------------------------------------------
# The defect this module exists to close
# --------------------------------------------------------------------------


def test_a_denied_symptom_is_negated_not_present():
    """Previously 'denies fever' extracted Fever as a present finding."""
    status = _detect("The patient denies fever today.", "fever")
    assert status.polarity == POLARITY_NEGATED
    assert status.is_patient_finding is True
    assert status.state() == STATE_WEAK_NEGATION


def test_an_objective_exclusion_is_stronger_than_a_reported_one():
    reported = _detect("The patient denies hepatomegaly.", "hepatomegaly")
    observed = _detect("On examination there is no hepatomegaly.", "hepatomegaly")

    assert reported.state() == STATE_WEAK_NEGATION
    assert observed.state() == STATE_DOCUMENTED_EXCLUSION
    assert observed.bounds()[1] < reported.bounds()[1]


def test_a_rule_out_is_an_open_question_not_evidence():
    status = _detect("Chest CT ordered to rule out pneumonia.", "pneumonia")
    assert status.certainty == CERTAINTY_HYPOTHETICAL
    assert status.state() == STATE_GAP
    assert status.bounds() == (0.0, 1.0), "an open question is evidence of neither presence nor absence"


def test_family_history_is_not_a_finding_of_this_patient():
    status = _detect("Family history of diabetes mellitus.", "diabetes mellitus")
    assert status.experiencer == EXPERIENCER_OTHER
    assert status.is_patient_finding is False


def test_a_past_finding_is_not_current():
    current = _detect("Presents with seizure.", "seizure")
    past = _detect("History of seizure in 2019.", "seizure")

    assert current.temporality == TEMPORALITY_CURRENT
    assert past.temporality == TEMPORALITY_HISTORICAL
    assert past.bounds()[1] < current.bounds()[1]


def test_a_plain_affirmed_finding_is_unaffected():
    status = _detect("Presents with progressive dyspnea.", "dyspnea")
    assert status.polarity == POLARITY_AFFIRMED
    assert status.certainty == CERTAINTY_FACTUAL
    assert status.experiencer == EXPERIENCER_PATIENT
    assert status.bounds()[0] > 0.5


# --------------------------------------------------------------------------
# Scope termination
# --------------------------------------------------------------------------


def test_negation_stops_at_an_adversative_terminator():
    """Without termination one cue would negate the rest of the sentence."""
    text = "The patient denies fever but reports cough."
    assert _detect(text, "fever").polarity == POLARITY_NEGATED
    assert _detect(text, "cough").polarity == POLARITY_AFFIRMED


def test_negation_does_not_cross_a_sentence_boundary():
    text = "No fever. Cough is present."
    assert _detect(text, "fever").polarity == POLARITY_NEGATED
    assert _detect(text, "cough").polarity == POLARITY_AFFIRMED


def test_the_objective_observation_is_recognised_across_the_adversative():
    text = "The patient denies palpitations, however the ECG demonstrates arrhythmia."
    palpitations = _detect(text, "palpitations")
    arrhythmia = _detect(text, "arrhythmia")

    assert palpitations.polarity == POLARITY_NEGATED
    assert palpitations.source == SOURCE_SUBJECTIVE
    assert arrhythmia.polarity == POLARITY_AFFIRMED
    assert arrhythmia.source == SOURCE_OBJECTIVE
    assert arrhythmia.state() == STATE_DOCUMENTED


def test_possible_is_distinguished_from_hypothetical_and_from_factual():
    assert _detect("Findings suspicious for pneumonia.", "pneumonia").certainty == CERTAINTY_POSSIBLE
    assert _detect("Evaluate for pneumonia.", "pneumonia").certainty == CERTAINTY_HYPOTHETICAL
    assert _detect("Imaging demonstrates pneumonia.", "pneumonia").certainty == CERTAINTY_FACTUAL


# --------------------------------------------------------------------------
# Language is not hard-coded
# --------------------------------------------------------------------------


def test_italian_cues_detect_the_same_distinctions():
    negated = _detect("Il paziente nega febbre.", "febbre", cues=ITALIAN_CUES)
    assert negated.polarity == POLARITY_NEGATED
    assert negated.source == SOURCE_SUBJECTIVE

    familial = _detect("Familiarita per diabete.", "diabete", cues=ITALIAN_CUES)
    assert familial.experiencer == EXPERIENCER_OTHER

    objective = _detect("All esame obiettivo si riscontra aritmia.", "aritmia", cues=ITALIAN_CUES)
    assert objective.source == SOURCE_OBJECTIVE
    assert objective.polarity == POLARITY_AFFIRMED


def test_italian_negation_stops_at_the_adversative():
    text = "Il paziente nega febbre ma riferisce tosse."
    assert _detect(text, "febbre", cues=ITALIAN_CUES).polarity == POLARITY_NEGATED
    assert _detect(text, "tosse", cues=ITALIAN_CUES).polarity == POLARITY_AFFIRMED


# --------------------------------------------------------------------------
# Intervals, never a scalar
# --------------------------------------------------------------------------


def test_the_four_zero_cases_stay_distinguishable():
    """A scalar score would flatten all of these to 0.0."""
    reported_denial = _detect("The patient denies cough.", "cough")
    objective_exclusion = _detect("On examination there is no cough.", "cough")
    open_question = _detect("Evaluate for cough.", "cough")

    states = {reported_denial.state(), objective_exclusion.state(), open_question.state()}
    assert len(states) == 3
    assert open_question.bounds()[1] == 1.0, "an open question keeps its full interval"


def test_cues_are_reported_so_a_decision_can_be_audited():
    status = _detect("The patient denies fever.", "fever")
    assert any(cue.startswith("negation:") for cue in status.cues)
    assert status.as_dict()["cues"]


# --------------------------------------------------------------------------
# The boundary is enforced, not conventional
# --------------------------------------------------------------------------


def _candidate(label: str, **kwargs) -> dict:
    return {"label": label, **kwargs}


def test_only_current_asserted_patient_findings_are_admitted():
    result = assemble(
        [
            _candidate("Dyspnea", assertion=_detect("Presents with dyspnea.", "dyspnea")),
            _candidate("Fever", assertion=_detect("Denies fever.", "fever")),
            _candidate("Pneumonia", assertion=_detect("Rule out pneumonia.", "pneumonia")),
            _candidate("Diabetes", assertion=_detect("Family history of diabetes.", "diabetes")),
        ]
    )
    assert result.concepts == ["Dyspnea"]
    reasons = {item.label: item.reason for item in result.rejected}
    assert reasons["Fever"] == REJECT_NEGATED
    assert reasons["Pneumonia"] == REJECT_HYPOTHETICAL
    assert reasons["Diabetes"] == REJECT_OTHER_EXPERIENCER


def test_every_rejection_names_where_the_item_belongs_instead():
    result = assemble([_candidate("Fever", assertion=_detect("Denies fever.", "fever"))])
    assert result.rejected[0].route == "documented_exclusion"


def test_a_synthetic_dream_hypothesis_cannot_enter_the_findings():
    result = assemble([_candidate("Amyloidosis", role="exclusion_hypothesis")])
    assert result.admitted == []
    assert result.rejected[0].reason == REJECT_SYNTHETIC
    assert result.rejected[0].route == "differential_as_exclusion_hypothesis"


def test_a_family_history_screening_item_cannot_enter_the_findings():
    result = assemble([_candidate("Long QT syndrome", role="screening_hypothesis")])
    assert result.rejected[0].reason == REJECT_SCREENING


def test_role_disqualifies_regardless_of_how_the_item_is_phrased():
    """An item marked unusable as evidence is rejected even with a clean assertion."""
    result = assemble(
        [
            _candidate(
                "Amyloidosis",
                usable_as_evidence=False,
                assertion=_detect("Presents with amyloidosis.", "amyloidosis"),
            )
        ]
    )
    assert result.admitted == []


def test_the_guard_raises_on_the_production_path():
    with pytest.raises(ValueError) as excinfo:
        assert_findings_only([_candidate("Amyloidosis", role="exclusion_hypothesis")])
    assert "differential_as_exclusion_hypothesis" in str(excinfo.value)


def test_the_guard_passes_clean_findings():
    assert_findings_only([_candidate("Dyspnea", assertion=_detect("Presents with dyspnea.", "dyspnea"))])


def test_candidates_without_an_assertion_are_admitted_unchanged():
    """Assertion detection is optional; its absence must not silently drop findings."""
    result = assemble([_candidate("Dyspnea", term_id="HP:0002094", modifiers=["Progressive"])])
    assert result.concepts == ["Dyspnea"]
    assert result.admitted[0].modifiers == ("Progressive",)


def test_malformed_candidates_are_skipped_rather_than_raising():
    result = assemble([{"label": ""}, "not a dict", {"no_label": 1}])
    assert result.admitted == []
    assert result.rejected == []
