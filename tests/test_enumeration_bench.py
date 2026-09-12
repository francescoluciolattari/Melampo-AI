"""Tests for the enumeration bench -- measuring the differential, not one claim."""

from melampo.evaluation.enumeration_bench import (
    DIFFERENTIAL_CASES,
    DifferentialCase,
    EnumerationResult,
    bench_enumeration,
    default_enumerator,
    differential_graph,
)
from melampo.memory.concept_paths import ConceptEdge, InMemoryConceptGraph
from melampo.training.mechanism_enumeration import (
    MODE_HYPOTHESES,
    MODE_KNOWLEDGE_GAP,
    MechanismEnumerator,
)

# --------------------------------------------------------------------------
# The four properties, measured separately because they fail independently
# --------------------------------------------------------------------------


def test_the_confirmed_condition_appears_in_the_spread():
    """Recall: a differential that omits the right answer has failed however
    well it ordered the rest."""
    result = bench_enumeration(default_enumerator(), DIFFERENTIAL_CASES)
    assert result.recall == 1.0


def test_the_confirmed_condition_is_ranked_first():
    result = bench_enumeration(default_enumerator(), DIFFERENTIAL_CASES)
    assert result.top_rank_rate == 1.0
    assert result.mean_rank == 1.0


def test_the_system_declines_to_rank_where_the_graph_cannot_support_it():
    """Restraint: the property most easily scored well on by accident and
    most dangerous to fail -- a system that always ranks something looks
    strong on recall and is wrong exactly where a clinician needs it to say
    so."""
    result = bench_enumeration(default_enumerator(), DIFFERENTIAL_CASES)
    assert result.restraint_rate == 1.0


def test_open_questions_name_findings_and_conditions_actually_at_issue():
    """A question naming neither is filler that a count-only metric could not
    tell apart from a substantive one."""
    result = bench_enumeration(default_enumerator(), DIFFERENTIAL_CASES)
    assert result.question_quality == 1.0


# --------------------------------------------------------------------------
# Scoring choices that were deliberate
# --------------------------------------------------------------------------


def test_recall_and_ranking_are_reported_separately():
    """Present-but-eleventh is a different and lesser failure than absent;
    one number would hide which happened."""
    graph = InMemoryConceptGraph.from_edges(
        [
            ConceptEdge("common thing", "manifests_as", "shared finding", 0.9),
            ConceptEdge("right answer", "manifests_as", "shared finding", 0.3),
        ]
    )
    cases = (
        DifferentialCase(
            "ranked_but_not_first",
            findings=("shared finding",),
            candidate_conditions=("common thing", "right answer"),
            confirmed_condition="right answer",
        ),
    )
    result = bench_enumeration(MechanismEnumerator(graph=graph), cases)
    assert result.recall == 1.0, "it is in the spread"
    assert result.top_rank_rate == 0.0, "but not at the top -- a distinct, lesser failure"
    assert result.mean_rank > 1.0


def test_mean_rank_averages_only_over_cases_where_the_condition_appeared():
    """Averaging in a sentinel for absent cases would blend two different
    failures into one number that recall already reports more honestly."""
    graph = InMemoryConceptGraph.from_edges([ConceptEdge("something else", "manifests_as", "a finding", 0.9)])
    cases = (
        DifferentialCase(
            "absent_entirely",
            findings=("a finding",),
            candidate_conditions=("something else", "never linked"),
            confirmed_condition="never linked",
        ),
    )
    result = bench_enumeration(MechanismEnumerator(graph=graph), cases)
    assert result.recall == 0.0
    assert result.mean_rank == 0.0, "no positions recorded, not a sentinel averaged in"


def test_a_restraint_case_is_not_counted_against_recall():
    """A case that exists to test restraint has no right answer to recall;
    counting it as a recall miss would penalise correct behaviour."""
    result = bench_enumeration(default_enumerator(), DIFFERENTIAL_CASES)
    assert result.conclusive_expected == 3
    assert result.restraint_expected == 1
    assert result.conclusive_expected + result.restraint_expected == result.runs


def test_an_empty_result_reports_zero_rather_than_dividing_by_zero():
    empty = EnumerationResult()
    assert empty.recall == 0.0
    assert empty.restraint_rate == 0.0
    assert empty.question_quality == 0.0
    assert empty.mean_rank == 0.0


# --------------------------------------------------------------------------
# The fixture itself must exercise both registers
# --------------------------------------------------------------------------


def test_the_case_set_exercises_both_registers():
    """A fixture where every neighbourhood is well covered would never test
    restraint at all -- the property most worth measuring."""
    enumerator = default_enumerator()
    modes = {enumerator.run(list(case.findings), list(case.candidate_conditions)).mode for case in DIFFERENTIAL_CASES}
    assert MODE_HYPOTHESES in modes
    assert MODE_KNOWLEDGE_GAP in modes


def test_the_restraint_case_findings_are_genuinely_absent_from_the_graph():
    """Two earlier attempts at this fixture got it wrong -- a weak edge and an
    unknown-strength edge both still give density 1.0. Local density measures
    whether the findings themselves are mapped, so restraint needs findings
    the graph has never heard of."""
    graph = differential_graph()
    restraint_cases = [case for case in DIFFERENTIAL_CASES if not case.graph_should_support_conclusion]
    assert restraint_cases, "the fixture must contain at least one restraint case"
    for case in restraint_cases:
        for finding in case.findings:
            assert not graph.edges_from(finding), f"{finding!r} must be absent for this case to test restraint"


def test_every_conclusive_case_names_a_confirmed_condition():
    for case in DIFFERENTIAL_CASES:
        if case.graph_should_support_conclusion:
            assert case.confirmed_condition
            assert case.confirmed_condition in case.candidate_conditions


def test_as_dict_carries_what_a_reviewer_needs():
    payload = bench_enumeration(default_enumerator(), DIFFERENTIAL_CASES).as_dict()
    for key in ("recall", "top_rank_rate", "restraint_rate", "question_quality", "per_case"):
        assert key in payload
