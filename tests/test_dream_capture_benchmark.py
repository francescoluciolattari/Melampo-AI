from melampo.evaluation.case_corpus import (
    REJECT_LEAKED_DIAGNOSIS,
    REJECT_MISSING_FIELD,
    REJECT_TOO_SHORT,
    load_jsonl,
    load_medcasereasoning,
    load_records,
    split_presentation,
)
from melampo.evaluation.dream_capture_benchmark import (
    CaseOutcome,
    EvaluationCase,
    capture_at_k,
    estimate_base_rate,
    evaluate,
)

PRESENTATION = (
    "A 59-year-old woman presents with progressive dyspnea over three weeks, bilateral "
    "pleural effusion on chest radiograph, and marked hepatomegaly on examination. "
    "Laboratory studies show elevated natriuretic peptide."
)


def _outcome(case_id, diagnosis, differential, hypotheses, density=0.8, tests=()):
    return CaseOutcome(
        case_id=case_id,
        documented_diagnosis=diagnosis,
        differential=list(differential),
        hypotheses=list(hypotheses),
        density=density,
        discriminating_tests=list(tests),
    )


# --------------------------------------------------------------------------
# The measure isolates the value of asking "what else"
# --------------------------------------------------------------------------


def test_a_capture_is_a_miss_by_the_differential_caught_by_the_branch():
    outcome = _outcome("c1", "amyloidosis", ["cardiac failure"], ["amyloidosis"])
    assert outcome.differential_hit is False
    assert outcome.hypothesis_hit is True
    assert outcome.is_capture is True


def test_a_diagnosis_the_differential_already_had_is_not_a_capture():
    """The branch adds nothing where the differential was already right."""
    outcome = _outcome("c1", "cardiac failure", ["cardiac failure"], ["cardiac failure"])
    assert outcome.is_capture is False


def test_a_case_neither_caught_is_not_a_capture():
    assert _outcome("c1", "amyloidosis", ["cardiac failure"], ["sarcoidosis"]).is_capture is False


def test_the_rate_is_computed_over_the_cases_the_differential_missed():
    report = evaluate(
        [
            _outcome("c1", "amyloidosis", ["cardiac failure"], ["amyloidosis"]),
            _outcome("c2", "sarcoidosis", ["cardiac failure"], ["amyloidosis"]),
            _outcome("c3", "cardiac failure", ["cardiac failure"], []),
        ]
    )
    assert len(report.missed_by_differential) == 2
    assert report.capture_rate == 0.5, "the case the differential got right is excluded"


def test_matching_ignores_case_and_spacing():
    assert _outcome("c1", "Cardiac  Failure", ["x"], ["cardiac failure"]).is_capture is True


# --------------------------------------------------------------------------
# Stratification, cost, and the floor to beat
# --------------------------------------------------------------------------


def test_capture_is_reported_per_density_band():
    report = evaluate(
        [
            _outcome("c1", "amyloidosis", ["x"], ["amyloidosis"], density=0.9),
            _outcome("c2", "sarcoidosis", ["x"], ["sarcoidosis"], density=0.8),
            _outcome("c3", "amyloidosis", ["x"], ["other"], density=0.1),
        ]
    )
    bands = report.by_band()
    assert bands["dense"]["capture_rate"] == 1.0
    assert bands["sparse"]["capture_rate"] == 0.0


def test_attention_cost_counts_hypotheses_per_diagnosis_caught():
    report = evaluate(
        [
            _outcome("c1", "amyloidosis", ["x"], ["amyloidosis", "a", "b"]),
            _outcome("c2", "sarcoidosis", ["x"], ["c", "d", "e"]),
        ]
    )
    assert report.attention_cost == 6.0


def test_a_branch_that_never_captures_has_infinite_cost():
    report = evaluate([_outcome("c1", "amyloidosis", ["x"], ["other"])])
    assert report.as_dict()["attention_cost"] is None


def test_the_base_rate_is_the_floor_a_capture_rate_must_clear():
    """A small candidate set makes any selector look effective."""
    cases = [
        EvaluationCase("c1", PRESENTATION, "amyloidosis", ("amyloidosis", "sarcoidosis")),
        EvaluationCase("c2", PRESENTATION, "sarcoidosis", ("amyloidosis", "sarcoidosis")),
    ]
    rate = estimate_base_rate(cases, hypotheses_per_case=1, trials=200)
    assert 0.35 < rate < 0.65, "one draw from two candidates lands about half the time"


def test_the_base_rate_is_deterministic():
    cases = [EvaluationCase("c1", PRESENTATION, "a", tuple("abcdefgh"))]
    assert estimate_base_rate(cases, 2) == estimate_base_rate(cases, 2)


def test_a_report_states_whether_it_beat_chance():
    report = evaluate([_outcome("c1", "amyloidosis", ["x"], ["amyloidosis"])], base_rate=0.2)
    assert report.exceeds_base_rate() is True
    assert evaluate([_outcome("c1", "a", ["x"], ["b"])], base_rate=0.2).exceeds_base_rate() is False


def test_without_a_base_rate_the_verdict_is_withheld():
    assert evaluate([_outcome("c1", "a", ["x"], ["a"])]).exceeds_base_rate() is None


def test_capture_at_k_requires_the_hypothesis_to_be_actionable():
    """Present but unactionable has not done the work."""
    actionable = _outcome("c1", "amyloidosis", ["x"], ["amyloidosis"], tests=["tissue biopsy"])
    inert = _outcome("c2", "amyloidosis", ["x"], ["amyloidosis"], tests=["chest radiograph"])
    assert capture_at_k(actionable, "tissue biopsy") is True
    assert capture_at_k(inert, "tissue biopsy") is False


# --------------------------------------------------------------------------
# Loading: a case that leaks its answer measures nothing
# --------------------------------------------------------------------------


def test_a_presentation_containing_its_own_diagnosis_is_rejected():
    report = load_records(
        [{"case_id": "c1", "presentation": PRESENTATION + " Cardiac amyloidosis was confirmed.", "diagnosis": "cardiac amyloidosis"}]
    )
    assert report.cases == []
    assert report.rejected[0][1] == REJECT_LEAKED_DIAGNOSIS


def test_a_clean_case_loads():
    report = load_records([{"case_id": "c1", "presentation": PRESENTATION, "diagnosis": "cardiac amyloidosis"}])
    assert len(report.cases) == 1
    assert report.cases[0].documented_diagnosis == "cardiac amyloidosis"


def test_shared_vocabulary_does_not_reject_a_usable_case():
    """A diagnosis and a presentation naturally share words."""
    report = load_records(
        [{"case_id": "c1", "presentation": PRESENTATION, "diagnosis": "pleural effusion of cardiac origin"}]
    )
    assert len(report.cases) == 1


def test_missing_and_short_records_are_rejected_with_their_reason():
    report = load_records(
        [
            {"case_id": "c1", "presentation": PRESENTATION},
            {"case_id": "c2", "presentation": "Too short.", "diagnosis": "x"},
        ]
    )
    reasons = dict(report.rejected)
    assert reasons["c1"] == REJECT_MISSING_FIELD
    assert reasons["c2"] == REJECT_TOO_SHORT


def test_jsonl_and_named_corpora_load_through_the_same_discipline():
    line = f'{{"pmcid": "PMC1", "case_prompt": "{PRESENTATION}", "final_diagnosis": "cardiac amyloidosis"}}'
    report = load_medcasereasoning([line])
    assert report.cases[0].case_id == "PMC1"
    assert report.cases[0].source == "medcasereasoning"


def test_malformed_json_lines_are_skipped_rather_than_raising():
    valid = f'{{"case_id":"c1","presentation":"{PRESENTATION}","diagnosis":"x y z"}}'
    report = load_jsonl(["not json", "", valid])
    assert len(report.cases) == 1


def test_the_load_report_summarises_what_was_rejected():
    report = load_records([{"case_id": "c1", "presentation": "short", "diagnosis": "x"}])
    assert report.as_dict()["rejected_by_reason"][REJECT_TOO_SHORT] == 1


def test_a_case_report_is_split_before_the_section_that_reveals_the_outcome():
    text = PRESENTATION + "\n\nFinal diagnosis: cardiac amyloidosis confirmed on biopsy."
    presentation, revealed = split_presentation(text)
    assert "amyloidosis" not in presentation.lower()
    assert "amyloidosis" in revealed.lower()
