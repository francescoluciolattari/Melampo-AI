import pytest

from melampo.memory.concept_resolution import (
    MATCH_NAME,
    MATCH_SYNONYM,
    OUTCOME_AMBIGUOUS,
    OUTCOME_UNRESOLVED,
    SCOPE_BROAD,
    SCOPE_EXACT,
    ConceptResolver,
    TermIndex,
    diagnose_empty_result,
    normalise_surface,
    parse_obo,
)

# Real stanzas from hp.obo v2026-06-23, plus two constructed cases.
OBO_SAMPLE = (
    "format-version: 1.2\n"
    "\n"
    "[Term]\n"
    "id: HP:0000002\n"
    "name: Abnormality of body height\n"
    'def: "Deviation from the norm of height." [https://orcid.org/0000-0002-0736-9199]\n'
    'synonym: "Abnormality of body height" EXACT layperson []\n'
    "xref: UMLS:C4025901\n"
    "is_a: HP:0001507 ! Growth abnormality\n"
    "\n"
    "[Term]\n"
    "id: HP:0000003\n"
    "name: Multicystic kidney dysplasia\n"
    "alt_id: HP:0004715\n"
    'synonym: "Multicystic dysplastic kidney" EXACT []\n'
    'synonym: "Kidney abnormality" BROAD []\n'
    "\n"
    "[Term]\n"
    "id: HP:0012735\n"
    "name: Cough\n"
    'synonym: "Coughing" EXACT []\n'
    "\n"
    "[Term]\n"
    "id: HP:0002098\n"
    "name: Respiratory distress\n"
    'synonym: "Coughing" EXACT []\n'
    "\n"
    "[Term]\n"
    "id: HP:0009999\n"
    "name: Retired concept\n"
    "is_obsolete: true\n"
)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_terms_names_synonyms_and_alt_ids_are_read():
    terms = {term.term_id: term for term in parse_obo(OBO_SAMPLE.splitlines())}
    assert len(terms) == 5

    kidney = terms["HP:0000003"]
    assert kidney.name == "Multicystic kidney dysplasia"
    assert kidney.alt_ids == ("HP:0004715",)
    assert ("Multicystic dysplastic kidney", SCOPE_EXACT) in kidney.synonyms
    assert ("Kidney abnormality", SCOPE_BROAD) in kidney.synonyms


def test_obsolete_terms_are_flagged_not_dropped():
    """Retired and never-existed are different, as elsewhere in the graph."""
    terms = {term.term_id: term for term in parse_obo(OBO_SAMPLE.splitlines())}
    assert terms["HP:0009999"].obsolete is True
    assert terms["HP:0012735"].obsolete is False


def test_a_header_without_terms_yields_nothing():
    assert list(parse_obo(["format-version: 1.2", "data-version: x"])) == []


# --------------------------------------------------------------------------
# Index construction
# --------------------------------------------------------------------------


def test_obsolete_terms_are_excluded_from_the_index_by_default():
    index = TermIndex.from_obo(OBO_SAMPLE.splitlines())
    assert "HP:0009999" not in index.by_id
    assert TermIndex.from_obo(OBO_SAMPLE.splitlines(), include_obsolete=True).by_id["HP:0009999"]


def test_broad_synonyms_are_not_indexed_by_default():
    """A broad synonym matches more than the term means; opting in is deliberate."""
    narrow = TermIndex.from_obo(OBO_SAMPLE.splitlines())
    assert narrow.lookup("kidney abnormality") == []

    wide = TermIndex.from_obo(OBO_SAMPLE.splitlines(), scopes={SCOPE_EXACT, SCOPE_BROAD})
    assert wide.lookup("kidney abnormality") == ["HP:0000003"]


def test_alternate_identifiers_resolve_to_the_same_term():
    index = TermIndex.from_obo(OBO_SAMPLE.splitlines())
    assert index.by_id["HP:0004715"].term_id == "HP:0000003"


def test_the_label_map_closes_the_import_gap():
    """This is what turns HP identifiers into graph nodes a clinician can name."""
    labels = TermIndex.from_obo(OBO_SAMPLE.splitlines()).label_map()
    assert labels["HP:0012735"] == "Cough"
    assert labels["HP:0000003"] == "Multicystic kidney dysplasia"


def test_normalisation_ignores_case_and_punctuation():
    assert normalise_surface("Multicystic  Kidney-Dysplasia.") == "multicystic kidney dysplasia"


# --------------------------------------------------------------------------
# Resolving findings
# --------------------------------------------------------------------------


def _resolver() -> ConceptResolver:
    return ConceptResolver(index=TermIndex.from_obo(OBO_SAMPLE.splitlines()))


def test_a_name_and_a_synonym_both_resolve():
    report = _resolver().resolve_findings(["Cough", "Multicystic dysplastic kidney"])
    assert [item.term_id for item in report.resolved] == ["HP:0012735", "HP:0000003"]
    assert report.resolved[0].match_kind == MATCH_NAME
    assert report.resolved[1].match_kind == MATCH_SYNONYM


def test_an_unknown_finding_is_reported_not_silently_dropped():
    report = _resolver().resolve_findings(["Cough", "bibasilar opacities"])
    assert [item.surface for item in report.unresolved] == ["bibasilar opacities"]
    assert report.unresolved[0].outcome == OUTCOME_UNRESOLVED
    assert report.resolution_rate == pytest.approx(0.5)


def test_an_ambiguous_surface_is_reported_rather_than_chosen():
    """Two terms share the synonym 'Coughing'; picking one would be a silent judgement."""
    report = _resolver().resolve_findings(["Coughing"])
    assert report.resolved == []
    assert report.unresolved[0].outcome == OUTCOME_AMBIGUOUS
    assert report.unresolved[0].candidates == ("HP:0002098", "HP:0012735")


def test_resolution_rate_answers_what_can_be_asked_of_the_graph():
    report = _resolver().resolve_findings(["Cough", "unknown a", "unknown b", "unknown c"])
    assert report.resolution_rate == pytest.approx(0.25)
    assert report.concepts == ["Cough"]


def test_empty_and_blank_findings_are_ignored():
    report = _resolver().resolve_findings(["", "   ", "Cough"])
    assert report.total == 1


# --------------------------------------------------------------------------
# Resolving free text
# --------------------------------------------------------------------------


def test_mentions_in_free_text_carry_character_offsets():
    text = "The patient reports Cough and no fever."
    found = _resolver().resolve_text(text)
    assert [item.term_id for item in found] == ["HP:0012735"]
    match = found[0]
    assert text[match.char_start : match.char_end].lower() == "cough"


def test_the_longest_matching_concept_wins():
    text = "Findings consistent with Multicystic kidney dysplasia today."
    found = _resolver().resolve_text(text)
    assert [item.term_id for item in found] == ["HP:0000003"]


def test_ambiguous_surfaces_are_skipped_in_free_text():
    found = _resolver().resolve_text("Persistent Coughing overnight.")
    assert found == []


def test_text_without_known_concepts_resolves_to_nothing():
    assert _resolver().resolve_text("Routine administrative note.") == []


def test_empty_text_is_handled():
    assert _resolver().resolve_text("") == []


# --------------------------------------------------------------------------
# Telling the two failure modes apart
# --------------------------------------------------------------------------


def test_unresolvable_findings_are_diagnosed_as_a_resolution_gap_not_a_sparse_graph():
    """The insidious case: the graph holds 285k edges and none is reachable."""
    report = _resolver().resolve_findings(["bibasilar opacities", "pleural effusion"])
    verdict = diagnose_empty_result(report, density=0.0)

    assert verdict["cause"] == "resolution_gap"
    assert "not the graph" in verdict["action"]
    assert verdict["unresolved"] == ["bibasilar opacities", "pleural effusion"]


def test_resolvable_findings_in_a_sparse_region_are_diagnosed_as_a_coverage_gap():
    report = _resolver().resolve_findings(["Cough"])
    verdict = diagnose_empty_result(report, density=0.1)
    assert verdict["cause"] == "coverage_gap"
    assert "extend the graph" in verdict["action"]


def test_a_covered_region_with_no_candidate_is_diagnosed_as_genuinely_absent():
    report = _resolver().resolve_findings(["Cough"])
    verdict = diagnose_empty_result(report, density=0.9)
    assert verdict["cause"] == "genuinely_absent"


def test_no_findings_at_all_is_its_own_cause():
    report = _resolver().resolve_findings([])
    assert diagnose_empty_result(report, density=None)["cause"] == "no_findings"


# --------------------------------------------------------------------------
# End to end: resolution makes the imported graph reachable
# --------------------------------------------------------------------------


def test_resolution_turns_an_unreachable_graph_into_a_traversable_one():
    """The missing step, closed.

    Imported without labels the graph is keyed on identifiers, so a finding
    named the way a clinician names it reaches nothing. The same graph imported
    with the label map is traversable from the same finding.
    """
    from melampo.memory.concept_paths import InMemoryConceptGraph, find_paths
    from melampo.memory.ontology_import import build_edges, parse_hpoa

    hpoa = (
        "database_id\tdisease_name\tqualifier\thpo_id\treference\tevidence\tonset\tfrequency\tsex\tmodifier\taspect\tbiocuration\n"
        "OMIM:1\tExample disease\t\tHP:0012735\tPMID:1\tPCS\t\tHP:0040281\t\t\tP\tcurator\n"
        "OMIM:1\tExample disease\t\tHP:0000003\tPMID:1\tPCS\t\tHP:0040282\t\t\tP\tcurator\n"
    )
    index = TermIndex.from_obo(OBO_SAMPLE.splitlines())

    unlabelled = InMemoryConceptGraph.from_edges(build_edges(parse_hpoa(hpoa.splitlines())))
    assert find_paths(unlabelled, "Cough", "Multicystic kidney dysplasia", max_hops=2) == []

    labelled = InMemoryConceptGraph.from_edges(
        build_edges(parse_hpoa(hpoa.splitlines()), label_for=index.label_map())
    )
    report = ConceptResolver(index=index).resolve_findings(["Cough", "Multicystic dysplastic kidney"])
    assert report.resolution_rate == 1.0

    paths = find_paths(labelled, report.concepts[0], report.concepts[1], max_hops=2)
    assert paths, "the same graph is now reachable from clinician-facing names"
    assert "Example disease" in paths[0].describe()
    assert paths[0].gap_count == 0
