
from melampo.data.document_processing import ClinicalDocumentProcessor
from melampo.memory.concept_resolution import ConceptResolver, TermIndex, parse_babelon

OBO_SAMPLE = (
    "[Term]\n"
    "id: HP:0012823\n"
    "name: Clinical modifier\n"
    "\n"
    "[Term]\n"
    "id: HP:0003676\n"
    "name: Progressive\n"
    "is_a: HP:0012823 ! Clinical modifier\n"
    "\n"
    "[Term]\n"
    "id: HP:0002094\n"
    "name: Dyspnea\n"
    "\n"
    "[Term]\n"
    "id: HP:0001945\n"
    "name: Fever\n"
    "\n"
    "[Term]\n"
    "id: HP:0002090\n"
    "name: Pneumonia\n"
)

BABELON_SAMPLE = (
    "source_language\tsource_value\tsubject_id\tpredicate_id\ttranslation_language\ttranslation_value\ttranslation_status\n"
    "en\tDyspnea\tHP:0002094\trdfs:label\tit\tDispnea\tOFFICIAL\n"
    "en\tFever\tHP:0001945\trdfs:label\tit\tFebbre\tOFFICIAL\n"
)


def _index(with_italian: bool = True) -> TermIndex:
    index = TermIndex.from_obo(OBO_SAMPLE.splitlines())
    if with_italian:
        index.add_translations(parse_babelon(BABELON_SAMPLE.splitlines()))
    return index


def _processor(language: str = "en", with_italian: bool = True) -> ClinicalDocumentProcessor:
    return ClinicalDocumentProcessor(
        concept_resolver=ConceptResolver(index=_index(with_italian), max_text_matches=10),
        language=language,
    )


# --------------------------------------------------------------------------
# Backward compatibility
# --------------------------------------------------------------------------


def test_without_a_resolver_the_original_lexicon_is_unchanged():
    result = ClinicalDocumentProcessor().extract_clinical_entities("cough and fever today")
    assert result["extraction_mode"] == "lexicon"
    assert result["ontology_refs"] == ["Symptom:Cough", "Symptom:Fever"]
    assert "patient_findings" not in result


def test_the_processor_reports_which_mode_it_is_in():
    assert ClinicalDocumentProcessor().describe()["extraction_mode"] == "lexicon"
    assert _processor().describe()["extraction_mode"] == "ontology_index"


# --------------------------------------------------------------------------
# Mentions and findings answer different questions
# --------------------------------------------------------------------------


def test_a_negated_mention_stays_retrievable_but_is_not_a_finding():
    """A chunk stating "denies fever" should be findable when searching fever.

    The negation is what a reader needs to find, so the mention is kept. It is
    the findings list, never the mentions, that supplies graph entry points.
    """
    result = _processor().extract_clinical_entities("The patient denies fever.")

    assert [item["normalized"] for item in result["clinical_entities"]] == ["Fever"]
    assert result["patient_findings"] == []
    assert result["excluded_mentions"][0]["reason"] == "negated"
    assert result["excluded_mentions"][0]["route"] == "documented_exclusion"


def test_a_hypothetical_mention_is_excluded_with_its_route():
    result = _processor().extract_clinical_entities("Imaging ordered to rule out pneumonia.")
    assert result["patient_findings"] == []
    assert result["excluded_mentions"][0]["reason"] == "hypothetical"
    assert result["excluded_mentions"][0]["route"] == "open_question"


def test_an_asserted_finding_is_admitted_with_its_modifier():
    result = _processor().extract_clinical_entities("Presents with progressive dyspnea.")
    findings = result["patient_findings"]

    assert [item["label"] for item in findings] == ["Dyspnea"]
    assert findings[0]["modifiers"] == ["Progressive"]
    assert "Progressive" not in [item["normalized"] for item in result["clinical_entities"]]


def test_mixed_text_separates_the_three_outcomes():
    text = "Presents with progressive dyspnea. The patient denies fever. Rule out pneumonia."
    result = _processor().extract_clinical_entities(text)

    assert [item["label"] for item in result["patient_findings"]] == ["Dyspnea"]
    reasons = {item["label"]: item["reason"] for item in result["excluded_mentions"]}
    assert reasons == {"Fever": "negated", "Pneumonia": "hypothetical"}
    assert len(result["clinical_entities"]) == 3, "every mention is retained for retrieval"


# --------------------------------------------------------------------------
# Ontology references and provenance
# --------------------------------------------------------------------------


def test_ontology_references_become_identifiers_rather_than_invented_strings():
    result = _processor().extract_clinical_entities("Presents with dyspnea.")
    assert result["ontology_refs"] == ["HP:0002094"]


def test_each_mention_carries_its_assertion_and_offsets():
    text = "The patient denies fever."
    entity = _processor().extract_clinical_entities(text)["clinical_entities"][0]

    assert entity["assertion"]["polarity"] == "negated"
    assert entity["assertion"]["state"] == "weak_negation"
    assert text[entity["char_start"] : entity["char_end"]].lower() == "fever"


def test_match_verification_travels_with_the_mention():
    english = _processor().extract_clinical_entities("Presents with dyspnea.")
    assert english["clinical_entities"][0]["verified_match"] is True


# --------------------------------------------------------------------------
# Both languages
# --------------------------------------------------------------------------


def test_italian_text_resolves_to_the_same_identifiers():
    result = _processor(language="it").extract_clinical_entities("Riferisce dispnea.")
    assert result["ontology_refs"] == ["HP:0002094"]
    assert [item["label"] for item in result["patient_findings"]] == ["Dyspnea"]
    assert result["extraction_language"] == "it"


def test_italian_negation_is_detected_with_the_italian_cues():
    result = _processor(language="it").extract_clinical_entities("Il paziente nega febbre.")
    assert result["patient_findings"] == []
    assert result["excluded_mentions"][0]["reason"] == "negated"


def test_the_same_finding_in_both_languages_yields_one_identifier():
    english = _processor().extract_clinical_entities("Presents with dyspnea.")
    italian = _processor(language="it").extract_clinical_entities("Riferisce dispnea.")
    assert english["ontology_refs"] == italian["ontology_refs"]


def test_an_untranslated_term_simply_does_not_resolve():
    """Italian coverage is partial; the gap is visible rather than guessed at."""
    result = _processor(language="it").extract_clinical_entities("Riscontro di polmonite.")
    assert result["ontology_refs"] == []
    assert result["patient_findings"] == []


def test_removing_the_translations_removes_the_italian_path():
    processor = _processor(language="it", with_italian=False)
    assert processor.extract_clinical_entities("Riferisce dispnea.")["ontology_refs"] == []


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------


def test_empty_text_produces_empty_lists_rather_than_raising():
    result = _processor().extract_clinical_entities("")
    assert result["clinical_entities"] == []
    assert result["patient_findings"] == []


def test_text_without_known_concepts_yields_nothing():
    result = _processor().extract_clinical_entities("Routine administrative note filed.")
    assert result["ontology_refs"] == []


def test_an_isolated_modifier_is_reported_as_collapsed():
    result = _processor().extract_clinical_entities("The course has been progressive.")
    assert result["patient_findings"] == []
    assert result["collapsed_modifiers"] == ["Progressive"]


def test_chunking_still_attaches_extraction_metadata(tmp_path):
    path = tmp_path / "report.txt"
    path.write_text("Presents with progressive dyspnea. The patient denies fever.", encoding="utf-8")
    result = _processor().process_document(path, metadata={"document_id": "doc_1"})

    documents = result["documents"]
    assert documents
    metadata = documents[0]["metadata"]
    assert metadata["extraction_mode"] == "ontology_index"
    assert [item["label"] for item in metadata["patient_findings"]] == ["Dyspnea"]
    assert "HP:0002094" in metadata["ontology_refs"]
