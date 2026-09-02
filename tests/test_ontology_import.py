import pytest

from melampo.memory.concept_paths import (
    STATE_DOCUMENTED,
    STATE_DOCUMENTED_EXCLUSION,
    STATE_GAP,
    ConceptEdge,
    InMemoryConceptGraph,
)
from melampo.memory.graph_coverage import (
    OUTCOME_ABSENT,
    OUTCOME_GAP,
    OUTCOME_PRESENT,
    ReferenceRelation,
    evaluation_is_interpretable,
    measure_coverage,
)
from melampo.memory.ontology_import import (
    FREQUENCY_TERMS,
    RELATION_HAS_PHENOTYPE,
    build_edges,
    build_graph,
    import_summary,
    parse_frequency,
    parse_hpoa,
    wilson_interval,
)

# Real rows from phenotype.hpoa v2026-06-23, tab separated, header included.
HPOA_SAMPLE = (
    "#description: HPO annotations for rare diseases\n"
    "#version: 2026-06-23\n"
    "database_id\tdisease_name\tqualifier\thpo_id\treference\tevidence\tonset\tfrequency\tsex\tmodifier\taspect\tbiocuration\n"
    "OMIM:613094\tMicrophthalmia isolated 4\t\tHP:0000568\tPMID:20207978\tPCS\t\tHP:0040281\t\t\tP\tHPO:probinson\n"
    "OMIM:614102\tImmunoglobulin kappa light chain deficiency\t\tHP:0002014\tPMID:6801190\tPCS\t\tHP:0040283\t\t\tP\tHPO:probinson\n"
    "OMIM:619340\tDevelopmental and epileptic encephalopathy 96\t\tHP:0002187\tPMID:31675180\tPCS\t\t1/1\t\t\tP\tHPO:probinson\n"
    "OMIM:621224\tEctodermal dysplasia 17\t\tHP:0001171\tPMID:39820051\tPCS\t\t3/12\t\t\tP\tHPO:probinson\n"
    "ORPHA:199310\tTetragametic chimerism syndrome\tNOT\tHP:0001263\tORPHA:199310\tTAS\t\t\t\t\tP\tORPHA:orphadata\n"
    "OMIM:100100\tSome disease\t\tHP:0000001\tPMID:1\tPCS\t\t\t\t\tP\tHPO:curator\n"
)


# --------------------------------------------------------------------------
# Published ranges, not invented bands
# --------------------------------------------------------------------------


def test_frequency_terms_match_the_published_ranges():
    """HPO publishes these as ranges; a point estimate would be the lossy step."""
    assert FREQUENCY_TERMS["HP:0040280"] == (1.00, 1.00)
    assert FREQUENCY_TERMS["HP:0040281"] == (0.80, 0.99)
    assert FREQUENCY_TERMS["HP:0040282"] == (0.30, 0.79)
    assert FREQUENCY_TERMS["HP:0040283"] == (0.05, 0.29)
    assert FREQUENCY_TERMS["HP:0040284"] == (0.01, 0.04)
    assert FREQUENCY_TERMS["HP:0040285"] == (0.00, 0.00)


def test_a_frequency_term_becomes_its_range_not_its_midpoint():
    assert parse_frequency("HP:0040283") == (0.05, 0.29)


def test_a_percentage_becomes_a_point():
    assert parse_frequency("35%") == pytest.approx((0.35, 0.35))


def test_an_empty_frequency_is_not_a_zero():
    assert parse_frequency("") is None
    assert parse_frequency("   ") is None


def test_malformed_frequencies_are_rejected_rather_than_guessed():
    for value in ("abc", "5/0", "7/3", "-1/4", "x%"):
        assert parse_frequency(value) is None


# --------------------------------------------------------------------------
# Sample size becomes epistemic width
# --------------------------------------------------------------------------


def test_one_observation_of_one_patient_stays_wide():
    lower, upper = wilson_interval(1, 1)
    assert upper == 1.0
    assert lower < 0.5, "a single observation is not knowledge"


def test_the_same_proportion_narrows_as_the_sample_grows():
    small = wilson_interval(9, 10)
    large = wilson_interval(90, 100)
    assert (large[1] - large[0]) < (small[1] - small[0])
    assert large[0] > small[0]


def test_a_fraction_becomes_an_interval_around_its_proportion():
    lower, upper = parse_frequency("3/12")
    assert lower < 0.25 < upper


# --------------------------------------------------------------------------
# Parsing real rows
# --------------------------------------------------------------------------


def test_comments_and_header_are_skipped():
    annotations = list(parse_hpoa(HPOA_SAMPLE.splitlines()))
    assert len(annotations) == 6
    assert annotations[0].disease_id == "OMIM:613094"
    assert annotations[0].phenotype_id == "HP:0000568"


def test_the_not_qualifier_marks_an_exclusion():
    annotations = {item.disease_id: item for item in parse_hpoa(HPOA_SAMPLE.splitlines())}
    assert annotations["ORPHA:199310"].is_excluded is True
    assert annotations["OMIM:613094"].is_excluded is False


def test_edges_carry_the_state_the_annotation_expresses():
    edges = {edge.source: edge for edge in build_edges(parse_hpoa(HPOA_SAMPLE.splitlines()))}

    very_frequent = edges["Microphthalmia isolated 4"]
    assert very_frequent.bounds == (0.80, 0.99)
    assert very_frequent.state == STATE_DOCUMENTED

    excluded = edges["Tetragametic chimerism syndrome"]
    assert excluded.bounds == (0.0, 0.0)
    assert excluded.state == STATE_DOCUMENTED_EXCLUSION

    unstated = edges["Some disease"]
    assert unstated.state == STATE_GAP
    assert unstated.is_gap


def test_an_exclusion_and_an_unstated_frequency_are_not_the_same_edge():
    """The distinction the whole interval representation exists to carry."""
    edges = {edge.source: edge for edge in build_edges(parse_hpoa(HPOA_SAMPLE.splitlines()))}
    excluded = edges["Tetragametic chimerism syndrome"]
    unknown = edges["Some disease"]

    assert excluded.upper == 0.0 and unknown.upper == 1.0
    assert excluded.is_gap is False and unknown.is_gap is True


def test_edges_keep_their_source_reference():
    edges = build_edges(parse_hpoa(HPOA_SAMPLE.splitlines()))
    assert all(edge.provenance and edge.provenance.startswith("hpoa:") for edge in edges)
    assert any("PMID:20207978" in (edge.provenance or "") for edge in edges)


def test_unstated_frequencies_can_be_dropped_instead_of_imported_as_gaps():
    kept = build_edges(parse_hpoa(HPOA_SAMPLE.splitlines()))
    dropped = build_edges(parse_hpoa(HPOA_SAMPLE.splitlines()), unstated_frequency_is_gap=False)
    assert len(dropped) == len(kept) - 1


def test_phenotype_labels_can_be_substituted_for_identifiers():
    graph = build_graph(
        HPOA_SAMPLE.splitlines(), label_for={"HP:0000568": "microphthalmia"}
    )
    assert "microphthalmia" in graph.concepts()


def test_the_import_reports_what_it_produced_by_state():
    summary = import_summary(build_edges(parse_hpoa(HPOA_SAMPLE.splitlines())))
    assert summary["edges"] == 6
    assert summary["by_state"][STATE_GAP] == 1
    assert summary["by_state"][STATE_DOCUMENTED_EXCLUSION] == 1
    assert 0.0 < summary["gap_fraction"] < 1.0


def test_the_imported_graph_is_traversable():
    graph = build_graph(HPOA_SAMPLE.splitlines())
    edges = graph.edges_from("Microphthalmia isolated 4")
    assert any(edge.relation == RELATION_HAS_PHENOTYPE for edge in edges)


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------


def _coverage_graph() -> InMemoryConceptGraph:
    return InMemoryConceptGraph.from_edges(
        [
            ConceptEdge("disease a", "has_phenotype", "finding a", 0.9),
            ConceptEdge.unknown("disease b", "has_phenotype", "finding b"),
        ]
    )


def test_coverage_separates_present_gap_and_absent():
    report = measure_coverage(
        _coverage_graph(),
        [
            ReferenceRelation("disease a", "finding a"),
            ReferenceRelation("disease b", "finding b"),
            ReferenceRelation("disease c", "finding c"),
        ],
    )
    outcomes = {item.relation.target: item.outcome for item in report.results}
    assert outcomes["finding a"] == OUTCOME_PRESENT
    assert outcomes["finding b"] == OUTCOME_GAP
    assert outcomes["finding c"] == OUTCOME_ABSENT

    assert report.coverage == pytest.approx(1 / 3)
    assert report.reachability == pytest.approx(2 / 3)


def test_the_two_queues_are_kept_apart_because_they_are_different_work():
    report = measure_coverage(
        _coverage_graph(),
        [ReferenceRelation("disease b", "finding b"), ReferenceRelation("disease c", "finding c")],
    )
    assert [item.target for item in report.calibration_queue()] == ["finding b"]
    assert [item.target for item in report.completion_queue()] == ["finding c"]


def test_an_attested_path_is_preferred_over_one_crossing_a_gap():
    graph = InMemoryConceptGraph.from_edges(
        [
            ConceptEdge.unknown("d", "has_phenotype", "f"),
            ConceptEdge("d", "has_phenotype", "m", 0.9),
            ConceptEdge("m", "has_phenotype", "f", 0.9),
        ]
    )
    report = measure_coverage(graph, [ReferenceRelation("d", "f")])
    assert report.results[0].outcome == OUTCOME_PRESENT
    assert report.results[0].hops == 2, "the two-hop attested path beats the one-hop unknown"


def test_low_coverage_marks_an_evaluation_as_uninterpretable():
    report = measure_coverage(
        _coverage_graph(),
        [ReferenceRelation("disease c", "finding c"), ReferenceRelation("disease d", "finding d")],
    )
    verdict = evaluation_is_interpretable(report)
    assert verdict["interpretable"] is False
    assert "knowledge base" in verdict["reason"]


def test_sufficient_coverage_marks_an_evaluation_as_interpretable():
    graph = InMemoryConceptGraph.from_edges([ConceptEdge("d", "has_phenotype", "f", 0.9)])
    report = measure_coverage(graph, [ReferenceRelation("d", "f")])
    assert evaluation_is_interpretable(report)["interpretable"] is True


def test_an_empty_reference_set_reports_zero_rather_than_dividing_by_zero():
    report = measure_coverage(_coverage_graph(), [])
    assert report.total == 0
    assert report.coverage == 0.0
    assert report.reachability == 0.0
