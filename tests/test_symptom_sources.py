"""Tests for memory/symptom_sources.py -- every source turned into one SymptomLink shape."""

import io

import pytest

from melampo.memory.concept_resolution import TermIndex, parse_obo
from melampo.memory.ontology_import import Annotation
from melampo.memory.symptom_sources import (
    MAPPING_LABEL,
    MAPPING_SOURCE,
    MAPPING_XREF,
    ORIENTATION_QUERIED_IS_SUBJECT,
    ORIENTATION_RELATED_IS_SUBJECT,
    ORIENTATION_UNDETERMINED,
    RELATION_EXCLUDES_FINDING,
    RELATION_HAS_FINDING,
    RELATION_MAY_HAVE_FINDING,
    SOURCE_DO,
    SOURCE_DO_TEXT,
    SOURCE_NCIT,
    SOURCE_WIKIDATA,
    TIER_COMMUNITY_REFERENCED,
    TIER_COMMUNITY_UNREFERENCED,
    TIER_CURATED,
    TIER_CURATED_POSSIBLE,
    DiseaseIndex,
    SymptomLink,
    assign_mondo,
    code_from_uts_url,
    hpo_xref_map,
    infer_orientation,
    label_counts,
    links_from_doid,
    links_from_hpoa,
    links_from_ncit_relations,
    links_from_wikidata,
    normalise_to_hpo,
    normalise_xref,
    parse_doid_owl,
    parse_mondo,
    symptoms_in_definition,
)

MONDO = """format-version: 1.2

[Term]
id: MONDO:0005812
name: influenza
synonym: "flu" EXACT []
synonym: "grippe" RELATED []
xref: DOID:8469 {source="MONDO:equivalentTo"}
xref: NCIT:C53482 {source="MONDO:equivalentTo"}
xref: UMLS:C0021400 {source="MONDO:equivalentTo"}
xref: ICD9:487 {source="DOID:8469"}
xref: OMIM:614680 {source="MONDO:obsoleteEquivalent"}
is_a: MONDO:0005550 ! infectious disease

[Term]
id: MONDO:0100001
name: avian influenza
xref: Orphanet:999 {source="MONDO:equivalentTo"}
is_a: MONDO:0005812 ! influenza

[Term]
id: MONDO:0000002
name: two claim this
xref: OMIM:100100 {source="MONDO:equivalentTo"}

[Term]
id: MONDO:0000003
name: also claims it
xref: OMIM:100100 {source="MONDO:equivalentTo"}

[Term]
id: MONDO:0000004
name: obsolete thing
is_obsolete: true
xref: DOID:4 {source="MONDO:equivalentTo"}
"""


@pytest.fixture
def index():
    return DiseaseIndex.from_mondo(parse_mondo(MONDO.splitlines()))


# --------------------------------------------------------------------------
# Mondo
# --------------------------------------------------------------------------


def test_only_equivalent_xrefs_are_kept(index):
    influenza = index.diseases["MONDO:0005812"]
    assert set(influenza.equivalent_xrefs) == {"DOID:8469", "NCIT:C53482", "UMLS:C0021400"}
    assert index.mondo_for("OMIM:614680") == set()  # obsoleteEquivalent is not followed
    assert index.mondo_for("ICD9:487") == set()  # a provenance pointer is not an equivalence


def test_exact_synonyms_only(index):
    assert index.diseases["MONDO:0005812"].exact_synonyms == ("flu",)


def test_orphanet_and_orpha_are_the_same_id(index):
    assert index.mondo_for("ORPHA:999") == {"MONDO:0100001"}
    assert normalise_xref("Orphanet:999") == "ORPHA:999"
    assert normalise_xref("UMLS_CUI:C1") == "UMLS:C1"


def test_obsolete_diseases_are_not_indexed(index):
    assert "MONDO:0000004" not in index.diseases
    assert index.mondo_for("DOID:4") == set()


def test_descendants(index):
    assert index.descendants("MONDO:0005812") == {"MONDO:0100001"}


def test_xrefs_of_by_prefix(index):
    assert index.xrefs_of("MONDO:0005812", "NCIT") == ["NCIT:C53482"]


def test_an_id_claimed_by_two_mondo_diseases_is_not_assigned(index):
    link = SymptomLink("hpo", "OMIM:100100", "x", "HP:1", "", RELATION_HAS_FINDING, TIER_CURATED)
    assert assign_mondo([link], index)[0].mondo_id == ""


def test_an_unambiguous_id_is_assigned(index):
    link = SymptomLink(SOURCE_DO, "DOID:8469", "influenza", "SYMP:1", "fever", RELATION_HAS_FINDING, 2)
    assert assign_mondo([link], index)[0].mondo_id == "MONDO:0005812"


# --------------------------------------------------------------------------
# HPO annotations
# --------------------------------------------------------------------------


def _annotation(aspect="P", qualifier="", frequency=""):
    return Annotation("OMIM:100100", "x", "HP:0001945", frequency, qualifier, "PMID:1", aspect)


def test_hpoa_keeps_phenotypic_abnormality_rows_only():
    links = list(links_from_hpoa([_annotation("P"), _annotation("I"), _annotation("C")]))
    assert len(links) == 1
    assert links[0].hpo_id == "HP:0001945"
    assert links[0].mapping == MAPPING_SOURCE
    assert links[0].tier == TIER_CURATED


def test_hpoa_not_qualifier_becomes_an_exclusion():
    link = next(links_from_hpoa([_annotation(qualifier="NOT")]))
    assert link.relation == RELATION_EXCLUDES_FINDING
    assert link.is_exclusion


# --------------------------------------------------------------------------
# Disease Ontology
# --------------------------------------------------------------------------

DOID_OWL = b"""<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:obo="http://purl.obolibrary.org/obo/">
  <owl:Class rdf:about="http://purl.obolibrary.org/obo/DOID_8469">
    <rdfs:label>influenza</rdfs:label>
    <obo:IAO_0000115>A viral infectious disease. The infection has_symptom fever, has_symptom dry cough, and has_symptom muscle aches.</obo:IAO_0000115>
    <rdfs:subClassOf>
      <owl:Restriction>
        <owl:onProperty rdf:resource="http://purl.obolibrary.org/obo/RO_0002452"/>
        <owl:someValuesFrom rdf:resource="http://purl.obolibrary.org/obo/SYMP_0000613"/>
      </owl:Restriction>
    </rdfs:subClassOf>
    <rdfs:subClassOf>
      <owl:Restriction>
        <owl:onProperty rdf:resource="http://purl.obolibrary.org/obo/RO_0004026"/>
        <owl:someValuesFrom rdf:resource="http://purl.obolibrary.org/obo/UBERON_0001004"/>
      </owl:Restriction>
    </rdfs:subClassOf>
  </owl:Class>
  <owl:Class rdf:about="http://purl.obolibrary.org/obo/DOID_1">
    <rdfs:label>retired</rdfs:label>
    <owl:deprecated rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</owl:deprecated>
  </owl:Class>
  <owl:Class rdf:about="http://purl.obolibrary.org/obo/SYMP_0000613"><rdfs:label>fever</rdfs:label></owl:Class>
</rdf:RDF>
"""


def test_parse_doid_owl_reads_has_symptom_axioms_only():
    classes = {cls.doid: cls for cls in parse_doid_owl(io.BytesIO(DOID_OWL))}
    assert set(classes) == {"DOID:8469", "DOID:1"}
    assert classes["DOID:8469"].symptom_ids == ("SYMP:0000613",)  # the located_in restriction is ignored
    assert classes["DOID:1"].deprecated


def test_doid_links_axioms_at_tier_2_and_new_text_symptoms_at_tier_4():
    classes = list(parse_doid_owl(io.BytesIO(DOID_OWL)))
    links = list(links_from_doid(classes, {"SYMP:0000613": "fever"}))
    axiom = [link for link in links if link.source == SOURCE_DO]
    text = [link for link in links if link.source == SOURCE_DO_TEXT]
    assert [(link.symptom_label, link.tier) for link in axiom] == [("fever", TIER_CURATED_POSSIBLE)]
    # "fever" is already an axiom, so the text copy is not repeated
    assert [link.symptom_label for link in text] == ["dry cough", "muscle aches"]
    assert all(link.tier == TIER_COMMUNITY_UNREFERENCED for link in text)
    assert not any(link.disease_id == "DOID:1" for link in links)  # deprecated


def test_symptoms_in_a_real_disease_ontology_definition():
    definition = (
        "A viral infectious disease that results_in infection located_in joint, has_material_basis_in "
        "Chikungunya virus, which is transmitted_by Aedes mosquito bite. The infection has_symptom fever, "
        "has_symptom arthralgia, and has_symptom maculopapular rash."
    )
    assert symptoms_in_definition(definition) == ["fever", "arthralgia", "maculopapular rash"]


def test_a_definition_without_has_symptom_yields_nothing():
    assert symptoms_in_definition("A disease that results_in inflammation located_in liver.") == []


# --------------------------------------------------------------------------
# NCIt through UMLS
# --------------------------------------------------------------------------


def _row(label, code="C3038", name="Fever", source="NCI"):
    return {
        "additionalRelationLabel": label,
        "relatedId": f"https://uts-ws.nlm.nih.gov/rest/content/2026AA/source/NCI/{code}",
        "relatedIdName": name,
        "rootSource": source,
    }


def test_orientation_is_inferred_from_the_dominant_label_family():
    forward = label_counts([_row("disease_has_finding")] * 30 + [_row("is_finding_of_disease")] * 3)
    inverse = label_counts([_row("is_finding_of_disease")] * 30 + [_row("disease_has_finding")] * 3)
    assert infer_orientation(forward) == ORIENTATION_QUERIED_IS_SUBJECT
    assert infer_orientation(inverse) == ORIENTATION_RELATED_IS_SUBJECT


def test_orientation_is_undetermined_on_thin_or_tied_evidence():
    assert infer_orientation(label_counts([_row("disease_has_finding")] * 5)) == ORIENTATION_UNDETERMINED
    tied = label_counts([_row("disease_has_finding")] * 15 + [_row("is_finding_of_disease")] * 15)
    assert infer_orientation(tied) == ORIENTATION_UNDETERMINED


def test_label_counts_ignores_unrelated_labels():
    assert label_counts([_row("disease_has_finding"), _row("has_associated_site")]) == {"disease_has_finding": 1}


def test_ncit_links_follow_the_orientation():
    rows = [_row("disease_has_finding"), _row("is_finding_of_disease", code="C2", name="Some disease")]
    forward = list(
        links_from_ncit_relations(rows, queried_code="C53482", disease_label="Influenza", orientation=ORIENTATION_QUERIED_IS_SUBJECT)
    )
    inverse = list(
        links_from_ncit_relations(rows, queried_code="C53482", disease_label="Influenza", orientation=ORIENTATION_RELATED_IS_SUBJECT)
    )
    assert [link.symptom_id for link in forward] == ["NCIT:C3038"]
    assert [link.symptom_id for link in inverse] == ["NCIT:C2"]


def test_ncit_undetermined_orientation_produces_no_links():
    rows = [_row("disease_has_finding")]
    assert list(links_from_ncit_relations(rows, queried_code="C1", disease_label="x", orientation=ORIENTATION_UNDETERMINED)) == []


def test_ncit_relation_kinds_and_tiers():
    rows = [
        _row("disease_has_finding", code="C1"),
        _row("disease_may_have_finding", code="C2"),
        _row("disease_excludes_finding", code="C3"),
    ]
    links = {
        link.symptom_id: link
        for link in links_from_ncit_relations(
            rows, queried_code="C9", disease_label="x", orientation=ORIENTATION_QUERIED_IS_SUBJECT, mondo_id="MONDO:1"
        )
    }
    assert links["NCIT:C1"].relation == RELATION_HAS_FINDING and links["NCIT:C1"].tier == TIER_CURATED
    assert links["NCIT:C2"].relation == RELATION_MAY_HAVE_FINDING and links["NCIT:C2"].tier == TIER_CURATED_POSSIBLE
    assert links["NCIT:C3"].is_exclusion
    assert all(link.source == SOURCE_NCIT and link.mondo_id == "MONDO:1" for link in links.values())


def test_rows_from_another_source_are_ignored():
    rows = [_row("disease_has_finding", source="MTH")]
    assert list(links_from_ncit_relations(rows, queried_code="C1", disease_label="x", orientation=ORIENTATION_QUERIED_IS_SUBJECT)) == []


def test_code_from_uts_url():
    assert code_from_uts_url("https://uts-ws.nlm.nih.gov/rest/content/2026AA/source/NCI/C3038") == "C3038"
    assert code_from_uts_url("") == ""


# --------------------------------------------------------------------------
# Wikidata
# --------------------------------------------------------------------------


def _wd(disease="Q2840", symptom="Q38933", refs="2", hpo="", mondo="0005812", doid=""):
    row = {
        "disease": {"value": f"http://www.wikidata.org/entity/{disease}"},
        "diseaseLabel": {"value": "influenza"},
        "symptom": {"value": f"http://www.wikidata.org/entity/{symptom}"},
        "symptomLabel": {"value": "fever"},
        "refs": {"value": refs},
    }
    if hpo:
        row["hpo"] = {"value": hpo}
    if mondo:
        row["mondo"] = {"value": mondo}
    if doid:
        row["doid"] = {"value": doid}
    return row


def test_wikidata_referenced_statement_is_tier_3_and_unreferenced_tier_4():
    referenced = next(links_from_wikidata([_wd(refs="2")]))
    unreferenced = next(links_from_wikidata([_wd(refs="0")]))
    assert referenced.tier == TIER_COMMUNITY_REFERENCED
    assert unreferenced.tier == TIER_COMMUNITY_UNREFERENCED
    assert referenced.source == SOURCE_WIKIDATA


def test_wikidata_mondo_value_without_prefix_is_normalised():
    assert next(links_from_wikidata([_wd(mondo="0005812")])).mondo_id == "MONDO:0005812"
    assert next(links_from_wikidata([_wd(mondo="MONDO:0005812")])).mondo_id == "MONDO:0005812"


def test_wikidata_doid_key_is_mapped_to_mondo():
    link = next(links_from_wikidata([_wd(mondo="", doid="DOID:8469")], mondo_for_doid={"DOID:8469": "MONDO:0005812"}))
    assert link.mondo_id == "MONDO:0005812"


def test_wikidata_hpo_id_on_the_symptom_item_is_used_directly():
    link = next(links_from_wikidata([_wd(hpo="HP:0001945")]))
    assert link.hpo_id == "HP:0001945"
    assert link.mapping == MAPPING_SOURCE


def test_wikidata_duplicate_rows_are_one_link():
    assert len(list(links_from_wikidata([_wd(), _wd()]))) == 1


# --------------------------------------------------------------------------
# Normalisation to HPO
# --------------------------------------------------------------------------

HP_OBO = """format-version: 1.2

[Term]
id: HP:0001945
name: Fever
synonym: "Pyrexia" EXACT []
xref: NCIT:C3038

[Term]
id: HP:0012735
name: Cough

[Term]
id: HP:0000001
name: Ambiguous one
synonym: "shared" EXACT []

[Term]
id: HP:0000002
name: Ambiguous two
synonym: "shared" EXACT []
xref: NCIT:C9
[Term]
id: HP:0000003
name: Also claims C9
xref: NCIT:C9
"""


@pytest.fixture
def hpo_terms():
    return list(parse_obo(HP_OBO.splitlines()))


def test_parse_obo_now_keeps_xrefs(hpo_terms):
    assert hpo_terms[0].xrefs == ("NCIT:C3038",)


def test_hpo_xref_map_drops_ids_claimed_twice(hpo_terms):
    assert hpo_xref_map(hpo_terms, "NCIT") == {"NCIT:C3038": "HP:0001945"}


def _link(symptom_id="", label="", hpo_id=""):
    return SymptomLink(SOURCE_NCIT, "NCIT:C1", "x", symptom_id, label, RELATION_HAS_FINDING, 1, hpo_id=hpo_id)


def test_normalisation_prefers_source_then_xref_then_label(hpo_terms):
    index = TermIndex.from_terms(hpo_terms)
    xrefs = hpo_xref_map(hpo_terms, "NCIT")
    links = normalise_to_hpo(
        [
            _link("NCIT:C3038", "anything", hpo_id="HP:9999999"),
            _link("NCIT:C3038", "Fever"),
            _link("SYMP:1", "pyrexia"),
            _link("SYMP:2", "cough"),
        ],
        term_index=index,
        xref_map=xrefs,
    )
    assert links[0].hpo_id == "HP:9999999"
    assert (links[1].hpo_id, links[1].mapping) == ("HP:0001945", MAPPING_XREF)
    assert (links[2].hpo_id, links[2].mapping) == ("HP:0001945", MAPPING_LABEL)
    assert (links[3].hpo_id, links[3].mapping) == ("HP:0012735", MAPPING_LABEL)


def test_an_ambiguous_label_stays_unmapped(hpo_terms):
    index = TermIndex.from_terms(hpo_terms)
    [link] = normalise_to_hpo([_link("SYMP:3", "shared")], term_index=index)
    assert link.hpo_id == ""
