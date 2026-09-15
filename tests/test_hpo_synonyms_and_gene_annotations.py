"""Tests for layperson synonym capture, its bridge into the normalisation
cascade, the gene-annotation parsers, and the hp.obo loader."""

import tempfile
from pathlib import Path

import pytest

from melampo.memory.concept_normalisation import NormalisationCascade
from melampo.memory.concept_paths import ConceptEdge, InMemoryConceptGraph
from melampo.memory.concept_resolution import (
    SCOPE_BROAD,
    SCOPE_EXACT,
    TYPE_LAYPERSON,
    TermIndex,
    parse_obo,
)
from melampo.memory.gene_annotations import (
    RELATION_ASSOCIATED_GENE,
    RELATION_CAUSES_DISEASE,
    gene_disease_edges,
    gene_phenotype_edges,
    parse_genes_to_disease,
    parse_genes_to_phenotype,
)
from melampo.memory.graph_sources import find_hp_obo_file, load_synonym_index

_MACROCEPHALY_OBO = (
    "[Term]\n"
    "id: HP:0000256\n"
    "name: Macrocephaly\n"
    'synonym: "Big head" BROAD layperson [ORCID:0000-0001-5889-4463]\n'
    'synonym: "Big skull" BROAD layperson\n'
    'synonym: "Enlarged cranium" EXACT []\n'
)


# --------------------------------------------------------------------------
# Synonym parsing: the type tag, previously discarded, is now captured
# --------------------------------------------------------------------------


def test_the_layperson_type_tag_is_captured_not_discarded():
    term = next(parse_obo(_MACROCEPHALY_OBO.splitlines()))
    assert ("Big head", SCOPE_BROAD, TYPE_LAYPERSON) in term.synonyms


def test_an_ordinary_synonym_has_no_type_tag():
    term = next(parse_obo(_MACROCEPHALY_OBO.splitlines()))
    assert ("Enlarged cranium", SCOPE_EXACT, "") in term.synonyms


def test_layperson_synonyms_are_excluded_by_default():
    """SAFE_SCOPES stays EXACT-only; a layperson synonym is BROAD-scoped and
    must not appear unless explicitly asked for."""
    term = next(parse_obo(_MACROCEPHALY_OBO.splitlines()))
    forms = dict(term.surface_forms())
    assert "Big head" not in forms
    assert "Enlarged cranium" in forms


def test_layperson_synonyms_appear_when_explicitly_included():
    term = next(parse_obo(_MACROCEPHALY_OBO.splitlines()))
    forms = dict(term.surface_forms(include_layperson=True))
    assert "Big head" in forms
    assert "Big skull" in forms


def test_a_generic_broad_synonym_is_not_admitted_by_include_layperson():
    """include_layperson is opt-in for the layperson category specifically,
    not a wider door for every BROAD synonym -- a genuinely broader, less
    precise concept must stay excluded even when layperson synonyms are
    wanted."""
    obo = (
        "[Term]\nid: HP:0000003\nname: Multicystic kidney dysplasia\n"
        'synonym: "Kidney abnormality" BROAD []\n'
    )
    term = next(parse_obo(obo.splitlines()))
    forms = dict(term.surface_forms(include_layperson=True))
    assert "Kidney abnormality" not in forms


def test_term_index_threads_include_layperson_through():
    index = TermIndex.from_obo(_MACROCEPHALY_OBO.splitlines(), include_layperson=True)
    assert "big head" in index.by_surface
    assert index.by_surface["big head"] == ["HP:0000256"]


# --------------------------------------------------------------------------
# The bridge: curated synonyms enrich the cascade's lexical tier, staying
# fully deterministic
# --------------------------------------------------------------------------


def test_a_layperson_phrase_resolves_via_the_lexical_tier_not_embedding():
    graph = InMemoryConceptGraph.from_edges(
        [ConceptEdge("some syndrome", "has_phenotype", "Macrocephaly", 0.8)]
    )
    index = TermIndex.from_obo(_MACROCEPHALY_OBO.splitlines())

    cascade = NormalisationCascade(graph=graph, synonym_index=index)
    result = cascade.resolve("big head", candidates=["Macrocephaly"])

    assert result.concept == "Macrocephaly"
    assert result.tier == "lexical"
    assert result.is_deterministic is True


def test_without_a_synonym_index_the_lexical_tier_behaves_as_before():
    graph = InMemoryConceptGraph.from_edges(
        [ConceptEdge("some syndrome", "has_phenotype", "Macrocephaly", 0.8)]
    )
    cascade = NormalisationCascade(graph=graph)
    assert cascade.resolve("big head", candidates=["Macrocephaly"]).resolved is False


def test_the_bare_graph_label_still_matches_with_a_synonym_index_present():
    graph = InMemoryConceptGraph.from_edges(
        [ConceptEdge("some syndrome", "has_phenotype", "Macrocephaly", 0.8)]
    )
    index = TermIndex.from_obo(_MACROCEPHALY_OBO.splitlines())
    cascade = NormalisationCascade(graph=graph, synonym_index=index)

    result = cascade.resolve("macrocephaly", candidates=["Macrocephaly"])
    assert result.concept == "Macrocephaly"
    assert "curated synonym" not in result.detail


# --------------------------------------------------------------------------
# Gene annotation parsers: header-driven, alias-tolerant, never silently
# misattribute a field
# --------------------------------------------------------------------------


def test_genes_to_phenotype_parses_the_confirmed_real_header():
    lines = [
        "entrez_gene_id\tentrez_gene_symbol\thpo_term_id\thpo_term_name\tfrequency_raw\tfrequency_hpo",
        "10\tNAT2\tHP:0001939\tAbnormality of metabolism\tNA\tNA",
    ]
    associations = list(parse_genes_to_phenotype(lines))
    assert associations[0].gene_symbol == "NAT2"
    assert associations[0].hpo_id == "HP:0001939"
    assert associations[0].hpo_term_name == "Abnormality of metabolism"


def test_genes_to_phenotype_also_accepts_the_alternate_column_names():
    """phenotype_to_genes.txt and some releases use gene_symbol/hpo_id."""
    lines = ["hpo_id\tgene_symbol\thpo_name", "HP:0001939\tNAT2\tAbnormality of metabolism"]
    associations = list(parse_genes_to_phenotype(lines))
    assert associations[0].gene_symbol == "NAT2"


def test_genes_to_phenotype_raises_on_an_unrecognised_header():
    with pytest.raises(ValueError, match="unrecognised genes_to_phenotype header"):
        list(parse_genes_to_phenotype(["column_a\tcolumn_b", "x\ty"]))


def test_genes_to_disease_parses_and_accepts_alternate_columns():
    lines = ["gene_symbol\tdisease_id\tdisease_name", "FBN1\tOMIM:154700\tMarfan syndrome"]
    associations = list(parse_genes_to_disease(lines))
    assert associations[0].gene_symbol == "FBN1"
    assert associations[0].disease_name == "Marfan syndrome"

    lines_alt = ["entrez_gene_symbol\tdatabase_id\tdisease_name", "FBN1\tOMIM:154700\tMarfan syndrome"]
    associations_alt = list(parse_genes_to_disease(lines_alt))
    assert associations_alt[0].gene_symbol == "FBN1"


def test_genes_to_disease_raises_on_an_unrecognised_header():
    with pytest.raises(ValueError, match="unrecognised genes_to_disease header"):
        list(parse_genes_to_disease(["column_a\tcolumn_b", "x\ty"]))


def test_a_row_missing_a_required_field_is_skipped_not_crashed():
    lines = [
        "entrez_gene_id\tentrez_gene_symbol\thpo_term_id\thpo_term_name",
        "10\t\tHP:0001939\tsomething",  # blank gene symbol
    ]
    assert list(parse_genes_to_phenotype(lines)) == []


# --------------------------------------------------------------------------
# Edge construction: uniform weight, never a fabricated number
# --------------------------------------------------------------------------


def test_gene_phenotype_edges_carry_the_associated_gene_relation():
    from melampo.memory.gene_annotations import GenePhenotypeAssociation

    edges = list(
        gene_phenotype_edges([GenePhenotypeAssociation("NAT2", "HP:0001939", "Abnormality of metabolism")])
    )
    assert edges[0].relation == RELATION_ASSOCIATED_GENE
    assert edges[0].source == "NAT2"
    assert edges[0].target == "Abnormality of metabolism"


def test_gene_disease_edges_carry_the_causes_disease_relation():
    from melampo.memory.gene_annotations import GeneDiseaseAssociation

    edges = list(gene_disease_edges([GeneDiseaseAssociation("FBN1", "OMIM:154700", "Marfan syndrome")]))
    assert edges[0].relation == RELATION_CAUSES_DISEASE
    assert edges[0].target == "Marfan syndrome"


def test_an_association_with_no_term_name_produces_no_edge():
    """A gene-phenotype row missing the phenotype's plain name has nothing
    usable as an edge target -- skipped rather than emitted with an empty
    string a caller would have to filter out downstream."""
    from melampo.memory.gene_annotations import GenePhenotypeAssociation

    edges = list(gene_phenotype_edges([GenePhenotypeAssociation("NAT2", "HP:0001939", "")]))
    assert edges == []


# --------------------------------------------------------------------------
# hp.obo loading: mirrors the phenotype.hpoa loader's honesty
# --------------------------------------------------------------------------


def test_no_obo_file_yields_none_not_an_empty_index(monkeypatch, tmp_path):
    """None means 'no data available', distinct from an index with zero
    terms, which would mean something parsed but found nothing -- a genuine
    problem the two must not look alike."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MELAMPO_HP_OBO_PATH", raising=False)

    assert load_synonym_index() is None


def test_a_real_obo_file_builds_a_usable_index():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "hp.obo"
        path.write_text(_MACROCEPHALY_OBO)

        index = load_synonym_index(explicit_path=path, include_layperson=True)

    assert index is not None
    assert "big head" in index.by_surface


def test_the_env_var_is_honoured_for_obo_too(monkeypatch, tmp_path):
    path = tmp_path / "somewhere.obo"
    path.write_text(_MACROCEPHALY_OBO)
    monkeypatch.setenv("MELAMPO_HP_OBO_PATH", str(path))

    assert find_hp_obo_file() == path


# --------------------------------------------------------------------------
# Wiring gene edges into the real graph: two defects the fixture could not
# have surfaced, both found by loading the actual HPO release
# --------------------------------------------------------------------------


def test_gene_disease_edges_survive_a_file_with_no_disease_name_column():
    """The real genes_to_disease.txt carries ncbi_gene_id, gene_symbol,
    association_type, disease_id, source -- and no name at all. An earlier
    version of this function required a disease_name field and silently
    dropped every row."""
    from melampo.memory.gene_annotations import GeneDiseaseAssociation

    edges = list(gene_disease_edges([GeneDiseaseAssociation("FBN1", "OMIM:154700", "")]))

    assert len(edges) == 1, "a row with no name must still produce an edge"
    assert edges[0].target == "OMIM:154700", "falling back to the id keeps it traversable and visibly an id"


def test_a_supplied_name_map_turns_disease_ids_into_clinical_text():
    """Every comparison downstream works on text; OMIM:154700 matches
    nothing a clinician writes. phenotype.hpoa indexes the same ids and does
    carry the names."""
    from melampo.memory.gene_annotations import GeneDiseaseAssociation

    edges = list(
        gene_disease_edges(
            [GeneDiseaseAssociation("FBN1", "OMIM:154700", "")],
            name_for_disease_id={"OMIM:154700": "Marfan syndrome"},
        )
    )

    assert edges[0].target == "Marfan syndrome"


def test_the_real_genes_to_disease_header_parses():
    """Confirmed against the actual file shipped in data/."""
    lines = [
        "ncbi_gene_id\tgene_symbol\tassociation_type\tdisease_id\tsource",
        "NCBIGene:64170\tCARD9\tMENDELIAN\tOMIM:212050\tftp://example",
    ]
    associations = list(parse_genes_to_disease(lines))

    assert associations[0].gene_symbol == "CARD9"
    assert associations[0].disease_id == "OMIM:212050"


def test_a_gene_is_never_proposed_as_a_diagnosis():
    """Gene edges point gene -> phenotype, so traversing one backwards from a
    finding lands on a gene -- which passes the reached-by-reverse test but
    is not something to rank in a differential. Verified against the real
    graph, where retrieval for an aortic finding returned AEBP1 and ALG9
    alongside the actual syndromes."""
    from melampo.memory.candidate_retrieval import retrieve_candidates
    from melampo.memory.concept_paths import ConceptEdge, InMemoryConceptGraph
    from melampo.memory.gene_annotations import (
        RELATION_ASSOCIATED_GENE,
        RELATION_CAUSES_DISEASE,
    )

    graph = InMemoryConceptGraph.from_edges(
        [
            ConceptEdge("FBN1", RELATION_ASSOCIATED_GENE, "aortic root aneurysm", 1.0),
            ConceptEdge("FBN1", RELATION_CAUSES_DISEASE, "marfan syndrome", 1.0),
            ConceptEdge("marfan syndrome", "has_phenotype", "aortic root aneurysm", 0.8),
        ]
    )

    report = retrieve_candidates(["aortic root aneurysm"], graph)

    assert "FBN1" not in report.condition_names
    assert "marfan syndrome" in report.condition_names


def test_the_traversal_still_passes_through_a_gene_to_reach_what_it_causes():
    """Excluding genes as candidates must not cut the path: a gene reached
    from a finding leads on to the diseases it causes, which are candidates.
    That is the whole point of adding the gene layer."""
    from melampo.memory.candidate_retrieval import retrieve_candidates
    from melampo.memory.concept_paths import ConceptEdge, InMemoryConceptGraph
    from melampo.memory.gene_annotations import (
        RELATION_ASSOCIATED_GENE,
        RELATION_CAUSES_DISEASE,
    )

    # The only route from finding to disease runs through the gene.
    graph = InMemoryConceptGraph.from_edges(
        [
            ConceptEdge("FBN1", RELATION_ASSOCIATED_GENE, "ectopia lentis", 1.0),
            ConceptEdge("FBN1", RELATION_CAUSES_DISEASE, "marfan syndrome", 1.0),
        ]
    )

    report = retrieve_candidates(["ectopia lentis"], graph)

    assert "marfan syndrome" in report.condition_names, "the gene must be a waypoint, not a dead end"
    assert "FBN1" not in report.condition_names
