from melampo.memory.assertion import (
    DECIDED_BY_DEFAULT,
    DECIDED_BY_FALLBACK,
    DECIDED_BY_RULES,
    ITALIAN_CUES,
    POLARITY_AFFIRMED,
    POLARITY_NEGATED,
    STATE_DOCUMENTED_EXCLUSION,
    AssertionDetector,
    AssertionStatus,
    LayeredAssertionResolver,
    measure_residue,
)


def _detect(text: str, term: str, cues=None) -> AssertionStatus:
    start = text.lower().index(term.lower())
    detector = AssertionDetector(cues=cues) if cues else AssertionDetector()
    return detector.detect(text, start, start + len(term))


# --------------------------------------------------------------------------
# Absence carried by the clinical term, with no negation marker
# --------------------------------------------------------------------------


def test_a_paraphrase_of_absence_negates_without_a_negation_word():
    """Nothing in the sentence signals negation except what the term means."""
    status = _detect("The chest is unremarkable.", "chest")
    assert status.polarity == POLARITY_NEGATED
    assert any("unremarkable" in cue for cue in status.cues)


def test_several_paraphrases_are_covered():
    for text, term in [
        ("Cardiac exam within normal limits.", "cardiac"),
        ("The abdomen is grossly normal.", "abdomen"),
        ("Neurological findings non contributory.", "neurological"),
    ]:
        assert _detect(text, term).polarity == POLARITY_NEGATED, text


def test_an_objective_paraphrase_reads_as_a_documented_exclusion():
    status = _detect("On examination the chest is unremarkable.", "chest")
    assert status.state() == STATE_DOCUMENTED_EXCLUSION


def test_italian_paraphrases_are_covered_too():
    assert _detect("Torace nella norma.", "torace", cues=ITALIAN_CUES).polarity == POLARITY_NEGATED
    assert _detect("Esame cardiaco nei limiti.", "cardiaco", cues=ITALIAN_CUES).polarity == POLARITY_NEGATED


def test_a_paraphrase_does_not_reach_a_finding_in_another_sentence():
    """Sentence boundaries still bound the scope."""
    status = _detect("Fever persisted. The rest of the exam was unremarkable.", "fever")
    assert status.polarity == POLARITY_AFFIRMED


def test_a_paraphrase_does_not_cross_a_semicolon():
    status = _detect("Dyspnea noted; cardiac exam within normal limits.", "dyspnea")
    assert status.polarity == POLARITY_AFFIRMED


# --------------------------------------------------------------------------
# Measuring the residue before building anything for it
# --------------------------------------------------------------------------


def test_the_residue_is_what_the_rules_leave_undecided():
    samples = [
        ("The patient denies fever.", 20, 25),
        ("The chest is unremarkable.", 4, 9),
        ("Fever was never documented during admission.", 0, 5),
        ("Dyspnea noted on exertion.", 0, 7),
    ]
    report = measure_residue(AssertionDetector(), samples)
    assert report.total == 4
    assert report.decided_by_rules == 2
    assert report.coverage == 0.5
    assert report.residue_fraction == 0.5


def test_the_residue_carries_examples_so_the_gap_is_inspectable():
    report = measure_residue(AssertionDetector(), [("Fever was never documented.", 0, 5)])
    assert "never documented" in report.residue[0]


def test_an_empty_sample_reports_zero_rather_than_dividing():
    report = measure_residue(AssertionDetector(), [])
    assert report.coverage == 0.0 and report.residue_fraction == 0.0


def test_full_coverage_leaves_no_residue():
    report = measure_residue(AssertionDetector(), [("The patient denies fever.", 20, 25)])
    assert report.coverage == 1.0
    assert report.residue == []


# --------------------------------------------------------------------------
# Layering: rules decide where they fire, fallback only on the residue
# --------------------------------------------------------------------------


def _fallback(text: str, start: int, end: int) -> AssertionStatus | None:
    return AssertionStatus(polarity=POLARITY_NEGATED) if "never" in text.lower() else None


def test_where_a_rule_fires_the_rule_decides_and_the_explanation_survives():
    resolver = LayeredAssertionResolver(fallback=_fallback)
    resolved = resolver.resolve("The patient denies fever.", 20, 25)
    assert resolved.decided_by == DECIDED_BY_RULES
    assert resolved.is_explained is True
    assert resolved.status.cues


def test_the_fallback_runs_only_where_no_rule_fired():
    """There the alternative was not a worse explanation but no detection at all."""
    resolver = LayeredAssertionResolver(fallback=_fallback)
    resolved = resolver.resolve("Fever was never documented during admission.", 0, 5)
    assert resolved.decided_by == DECIDED_BY_FALLBACK
    assert resolved.is_explained is False
    assert resolved.status.polarity == POLARITY_NEGATED


def test_a_fallback_that_declines_leaves_the_default():
    resolver = LayeredAssertionResolver(fallback=_fallback)
    resolved = resolver.resolve("Dyspnea noted on exertion.", 0, 7)
    assert resolved.decided_by == DECIDED_BY_DEFAULT
    assert resolved.status.polarity == POLARITY_AFFIRMED


def test_without_a_fallback_the_behaviour_is_the_rules_alone():
    resolver = LayeredAssertionResolver()
    assert resolver.resolve("Fever was never documented.", 0, 5).decided_by == DECIDED_BY_DEFAULT


def test_a_fallback_cannot_override_a_rule():
    """Adding a fallback must not make an explained case opaque."""

    def _always(text, start, end):
        return AssertionStatus(polarity=POLARITY_AFFIRMED)

    resolver = LayeredAssertionResolver(fallback=_always)
    resolved = resolver.resolve("The patient denies fever.", 20, 25)
    assert resolved.status.polarity == POLARITY_NEGATED
    assert resolved.decided_by == DECIDED_BY_RULES


def test_the_resolution_payload_reports_which_layer_decided():
    payload = LayeredAssertionResolver(fallback=_fallback).resolve(
        "Fever was never documented.", 0, 5
    ).as_dict()
    assert payload["decided_by"] == DECIDED_BY_FALLBACK
    assert payload["explained"] is False
    assert payload["polarity"] == POLARITY_NEGATED
