from melampo.memory.assertion import ITALIAN_CUES
from melampo.memory.concept_paths import ConceptEdge, InMemoryConceptGraph
from melampo.memory.concept_resolution import ConceptResolver, TermIndex, parse_babelon
from melampo.reasoning.clinical_discourse import (
    ITALIAN_CONNECTIVES,
    ClinicalDiscourseReader,
    read_and_verify,
)
from melampo.reasoning.illness_script import (
    ORIGIN_MODEL,
    VERDICT_GROUNDED_IN_CASE,
    VERDICT_KNOWLEDGE_MEDIATED,
    ScriptVerifier,
)

OBO = (
    "[Term]\nid: HP:0002094\nname: Dyspnea\n\n"
    "[Term]\nid: HP:0012764\nname: Orthopnea\n\n"
    "[Term]\nid: HP:0001945\nname: Fever\n\n"
    "[Term]\nid: HP:0001635\nname: Congestive heart failure\n\n"
    "[Term]\nid: HP:0002090\nname: Pneumonia\n\n"
    "[Term]\nid: HP:0002105\nname: Hemoptysis\n"
)
BABELON = (
    "source_language\tsource_value\tsubject_id\tpredicate_id\ttranslation_language\ttranslation_value\ttranslation_status\n"
    "en\tDyspnea\tHP:0002094\trdfs:label\tit\tDispnea\tOFFICIAL\n"
    "en\tCongestive heart failure\tHP:0001635\trdfs:label\tit\tScompenso cardiaco\tOFFICIAL\n"
    "en\tPneumonia\tHP:0002090\trdfs:label\tit\tPolmonite\tOFFICIAL\n"
)

PRESENTATION = (
    "The patient presents with dyspnea and orthopnea. The picture is most consistent with "
    "congestive heart failure, though the fever bothers me. I would rule out pneumonia "
    "before committing. No evidence of hemoptysis."
)


def _reader(italian: bool = False) -> ClinicalDiscourseReader:
    index = TermIndex.from_obo(OBO.splitlines())
    if italian:
        index.add_translations(parse_babelon(BABELON.splitlines()))
        return ClinicalDiscourseReader(
            resolver=ConceptResolver(index=index), cues=ITALIAN_CUES, connectives=dict(ITALIAN_CONNECTIVES)
        )
    return ClinicalDiscourseReader(resolver=ConceptResolver(index=index))


# --------------------------------------------------------------------------
# Reading a presentation the way a colleague would
# --------------------------------------------------------------------------


def test_observed_findings_are_separated_from_proposed_explanations():
    reading = _reader().read(PRESENTATION)
    assert [item.label for item in reading.findings] == ["Dyspnea", "Orthopnea", "Fever"]
    assert [item.label for item, _, _ in reading.candidates] == ["Congestive heart failure", "Pneumonia"]


def test_commitment_is_read_from_the_connective_and_becomes_rank():
    reading = _reader().read(PRESENTATION)
    by_label = {item.label: (rank, commitment) for item, rank, commitment in reading.candidates}
    assert by_label["Congestive heart failure"] == (2, "consistent with")
    assert by_label["Pneumonia"] == (4, "rule out")


def test_a_connective_binds_only_the_concept_immediately_after_it():
    """'consistent with X, though the Y' must not carry X's connective onto Y."""
    reading = _reader().read(PRESENTATION)
    assert "Fever" not in [item.label for item, _, _ in reading.candidates]
    assert "Fever" in [item.label for item in reading.findings]


def test_a_finding_the_presenter_flags_as_not_fitting_is_marked_discordant():
    reading = _reader().read(PRESENTATION)
    assert [item.label for item in reading.discordant] == ["Fever"]


def test_a_denied_concept_is_neither_a_finding_nor_a_candidate():
    reading = _reader().read(PRESENTATION)
    assert [item.label for item in reading.denied] == ["Hemoptysis"]
    assert "Hemoptysis" not in [item.label for item in reading.findings]


def test_a_hedged_concept_without_a_connective_is_still_a_candidate():
    reading = _reader().read("Findings suspicious for pneumonia.")
    assert [item.label for item, _, _ in reading.candidates] == ["Pneumonia"]


def test_empty_discourse_reads_as_nothing():
    reading = _reader().read("")
    assert reading.findings == [] and reading.candidates == []


# --------------------------------------------------------------------------
# The reading becomes a script the verifier can check
# --------------------------------------------------------------------------


def test_the_script_orders_the_differential_by_commitment():
    script = _reader().to_script(PRESENTATION)
    assert [entry.condition for entry in script.differential] == ["Congestive heart failure", "Pneumonia"]
    assert script.differential[0].rank < script.differential[1].rank
    assert all(entry.origin == ORIGIN_MODEL for entry in script.differential)


def test_the_script_keeps_the_discordant_note():
    script = _reader().to_script(PRESENTATION)
    notes = {item.label: item.note for item in script.consequences}
    assert notes["Fever"] == "discordant"
    assert notes["Dyspnea"] is None


def test_the_leading_hypothesis_is_the_one_the_presenter_committed_to():
    assert _reader().to_script(PRESENTATION).leading.condition == "Congestive heart failure"


def test_the_full_path_from_presentation_to_grounding():
    graph = InMemoryConceptGraph.from_edges(
        [
            ConceptEdge("Dyspnea", "manifests", "Congestive heart failure", 0.8, "cardiology"),
            ConceptEdge("Fever", "manifests", "Pneumonia", 0.8, "infectious"),
        ]
    )
    out = read_and_verify(
        PRESENTATION, _reader(), ScriptVerifier(graph=graph), case_findings=["Dyspnea", "Orthopnea", "Fever"]
    )
    verdicts = {item["element"]: item["verdict"] for item in out["verification"]["differential"]}
    assert verdicts["Congestive heart failure"] == VERDICT_KNOWLEDGE_MEDIATED
    assert verdicts["Pneumonia"] == VERDICT_KNOWLEDGE_MEDIATED
    consequences = {item["element"]: item["verdict"] for item in out["verification"]["consequences"]}
    assert consequences["Dyspnea"] == VERDICT_GROUNDED_IN_CASE


# --------------------------------------------------------------------------
# Both languages
# --------------------------------------------------------------------------


def test_italian_presentation_reads_with_italian_cues_and_connectives():
    reading = _reader(italian=True).read(
        "Il paziente riferisce dispnea. Quadro compatibile con scompenso cardiaco. Da escludere polmonite."
    )
    assert [item.label for item in reading.findings] == ["Dyspnea"]
    by_label = {item.label: commitment for item, _, commitment in reading.candidates}
    assert by_label["Congestive heart failure"] == "compatibile con"
    assert by_label["Pneumonia"] == "da escludere"
