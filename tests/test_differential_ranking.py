"""Tests for IC-weighted differential ranking and the term-history store."""

import tempfile
from pathlib import Path

from melampo.evaluation.enumeration_bench import differential_graph
from melampo.memory.concept_paths import ConceptEdge, InMemoryConceptGraph
from melampo.memory.differential_ranking import rank_differential
from melampo.memory.information_content import InformationContentTable
from melampo.memory.term_history import (
    TermHistoryStore,
    TermObsoletion,
    TermRename,
    diff_releases,
)

# --------------------------------------------------------------------------
# Differential ranking: specificity, not raw overlap
# --------------------------------------------------------------------------


def _graph_with_specificity():
    """A common finding shared by many diseases, and a rare one shared by few."""
    return InMemoryConceptGraph.from_edges(
        [
            ConceptEdge("common cold", "has_phenotype", "fatigue", 0.5),
            ConceptEdge("condition a", "has_phenotype", "fatigue", 0.5),
            ConceptEdge("condition b", "has_phenotype", "fatigue", 0.5),
            ConceptEdge("condition c", "has_phenotype", "fatigue", 0.5),
            ConceptEdge("condition d", "has_phenotype", "fatigue", 0.5),
            ConceptEdge("rare disease x", "has_phenotype", "fatigue", 0.5),
            ConceptEdge("rare disease x", "has_phenotype", "distinctive rare finding", 0.5),
            ConceptEdge("condition e", "has_phenotype", "distinctive rare finding", 0.5),
        ]
    )


def test_a_candidate_matching_only_a_rare_finding_outranks_one_matching_only_a_common_finding():
    graph = _graph_with_specificity()
    table = InformationContentTable.from_graph_structure(graph)

    ranked = rank_differential(
        ["distinctive rare finding"], ["condition e", "condition a"], graph, table
    )

    assert ranked[0].condition == "condition e"
    assert ranked[0].matched_findings == ("distinctive rare finding",)


def test_two_candidates_sharing_the_same_findings_tie_on_score():
    """The enumeration bench's fixture uses `manifests_as`, not HPO's own
    `has_phenotype` -- an earlier version of this test used the default
    relation against this fixture and silently scored both candidates zero,
    with the alphabetical tie-break deciding an order that looked like a
    real result."""
    graph = differential_graph()
    table = InformationContentTable.from_graph_structure(graph)

    ranked = rank_differential(
        ["bilateral hilar lymphadenopathy", "hypercalcaemia", "erythema nodosum"],
        ["sarcoidosis", "lymphoma"],
        graph,
        table,
        relation="manifests_as",
    )

    assert ranked[0].condition == "sarcoidosis", "sarcoidosis shares all three, lymphoma shares fewer"


def test_using_the_wrong_relation_name_scores_everything_zero_not_an_error():
    """The exact failure a wrong relation name produces: silent zero scores
    everywhere, with the alphabetical tie-break deciding an order that looks
    like a real result. Documented here so the failure mode is recognisable,
    not just avoided."""
    graph = differential_graph()
    table = InformationContentTable.from_graph_structure(graph)

    ranked = rank_differential(
        ["bilateral hilar lymphadenopathy"], ["sarcoidosis", "lymphoma"], graph, table, relation="has_phenotype"
    )

    assert all(item.specificity_score == 0.0 for item in ranked)


def test_the_relation_parameter_is_what_makes_a_non_hpo_graph_usable():
    graph = differential_graph()
    table = InformationContentTable.from_graph_structure(graph)

    ranked = rank_differential(
        ["bilateral hilar lymphadenopathy", "hypercalcaemia", "erythema nodosum"],
        ["sarcoidosis"],
        graph,
        table,
        relation="manifests_as",
    )

    assert ranked[0].specificity_score > 0.0


def test_coverage_and_specificity_are_reported_separately():
    """A candidate explaining every finding with common ones and one
    explaining half with rare ones can score similarly on specificity alone
    -- a reader needs both numbers."""
    graph = _graph_with_specificity()
    table = InformationContentTable.from_graph_structure(graph)

    ranked = rank_differential(["fatigue", "distinctive rare finding"], ["rare disease x"], graph, table)

    assert ranked[0].coverage == 1.0
    assert len(ranked[0].matched_findings) == 2


def test_unmatched_findings_are_reported_not_silently_dropped():
    graph = _graph_with_specificity()
    table = InformationContentTable.from_graph_structure(graph)

    ranked = rank_differential(["fatigue", "distinctive rare finding"], ["condition a"], graph, table)

    assert ranked[0].unmatched_findings == ("distinctive rare finding",)


def test_profile_size_is_reported_not_folded_into_the_score():
    """Normalising by a candidate's total phenotype count would reward a
    sparsely-annotated disease purely for being under-curated -- reported
    instead, so a reader can judge it directly."""
    graph = _graph_with_specificity()
    table = InformationContentTable.from_graph_structure(graph)

    ranked = rank_differential(["distinctive rare finding"], ["rare disease x", "condition e"], graph, table)

    sizes = {item.condition: item.profile_size for item in ranked}
    assert sizes["rare disease x"] == 2
    assert sizes["condition e"] == 1


def test_a_candidate_matching_nothing_still_appears_with_a_zero_score():
    graph = _graph_with_specificity()
    table = InformationContentTable.from_graph_structure(graph)

    ranked = rank_differential(["distinctive rare finding"], ["common cold"], graph, table)

    assert ranked[0].specificity_score == 0.0
    assert ranked[0].coverage == 0.0


def test_an_unresolvable_finding_does_not_crash_ranking():
    graph = _graph_with_specificity()
    table = InformationContentTable.from_graph_structure(graph)
    ranked = rank_differential(["nothing the graph has heard of"], ["condition a"], graph, table)
    assert ranked[0].specificity_score == 0.0


# --------------------------------------------------------------------------
# Term history: nothing renamed or obsoleted is ever lost
# --------------------------------------------------------------------------


def test_a_rename_is_detected_between_two_releases():
    renames, obsoletions = diff_releases(
        {"HP:0000256": "Macrocephaly"}, {"HP:0000256": "Macrocephaly, congenital"}, release="v2026-09-01"
    )
    assert renames[0].old_name == "Macrocephaly"
    assert renames[0].new_name == "Macrocephaly, congenital"
    assert obsoletions == []


def test_an_unchanged_term_produces_no_rename():
    renames, _ = diff_releases({"HP:1": "Same name"}, {"HP:1": "Same name"}, release="v1")
    assert renames == []


def test_an_obsoleted_term_is_recorded_with_its_replacement():
    renames, obsoletions = diff_releases(
        {"HP:0009999": "Old finding"}, {}, current_obsolete={"HP:0009999": "HP:0000257"}, release="v2026-09-01"
    )
    assert renames == []
    assert obsoletions[0].replaced_by == "HP:0000257"


def test_history_accumulates_across_more_than_one_rename():
    """A term renamed twice must keep both historical names, not only the
    most recent one."""
    with tempfile.TemporaryDirectory() as directory:
        store = TermHistoryStore(Path(directory))
        store.append_renames([TermRename("HP:1", "First name", "Second name", "v1")])
        store.append_renames([TermRename("HP:1", "Second name", "Third name", "v2")])

        names = store.synonyms_by_term_id()["HP:1"]

    assert set(names) == {"First name", "Second name"}


def test_current_id_for_follows_an_obsoletion_to_its_replacement():
    with tempfile.TemporaryDirectory() as directory:
        store = TermHistoryStore(Path(directory))
        store.append_obsoletions([TermObsoletion("HP:0009999", "Old finding", "HP:0000257", "v1")])

        assert store.current_id_for("Old finding") == "HP:0000257"
        assert store.current_id_for("HP:0009999") == "HP:0000257"


def test_an_obsoletion_with_no_replacement_returns_none_not_a_crash():
    with tempfile.TemporaryDirectory() as directory:
        store = TermHistoryStore(Path(directory))
        store.append_obsoletions([TermObsoletion("HP:0009999", "Old finding", None, "v1")])
        assert store.current_id_for("Old finding") is None


def test_appending_nothing_creates_no_file():
    with tempfile.TemporaryDirectory() as directory:
        store = TermHistoryStore(Path(directory))
        assert store.append_renames([]) == 0
        assert not store.renames_path.exists()


def test_history_is_empty_for_a_store_that_was_never_written():
    with tempfile.TemporaryDirectory() as directory:
        store = TermHistoryStore(Path(directory))
        assert store.load_renames() == []
        assert store.synonyms_by_term_id() == {}


# --------------------------------------------------------------------------
# The bridge: historical names resolve through the cascade, at tier 1
# --------------------------------------------------------------------------


def test_a_historical_name_resolves_through_the_cascade_deterministically():
    from melampo.memory.concept_normalisation import NormalisationCascade
    from melampo.memory.concept_resolution import TermIndex

    graph = InMemoryConceptGraph.from_edges(
        [ConceptEdge("some syndrome", "has_phenotype", "Macrocephaly, congenital", 0.8)]
    )
    index = TermIndex.from_obo(
        "[Term]\nid: HP:0000256\nname: Macrocephaly, congenital\n".splitlines()
    )
    with tempfile.TemporaryDirectory() as directory:
        history = TermHistoryStore(Path(directory))
        history.append_renames([TermRename("HP:0000256", "Macrocephaly", "Macrocephaly, congenital", "v1")])

        cascade = NormalisationCascade(graph=graph, synonym_index=index, term_history=history)
        result = cascade.resolve("macrocephaly", candidates=["Macrocephaly, congenital"])

    assert result.concept == "Macrocephaly, congenital"
    assert result.tier == "lexical"
    assert result.is_deterministic is True
