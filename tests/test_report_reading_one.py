"""A patient's documents read with the one procedure (25 November 2026): the same card, interpreters and comparison
as the publications, the interpreters on this machine only, the findings in the graph of the case."""

import json

from melampo.clinical import case_graph as cg
from melampo.clinical.case_graph_builder import add_cards
from melampo.data.report_reading import ONE, ReportReaders, build_linker
from melampo.memory import disease_profiles as dp
from melampo.reading.normalize import HpoIndex

NOTE = ("Dolore in fossa iliaca destra da ieri, con nausea. Non febbre. PCR 52 mg/L. "
        "Possibile appendicite. Madre con diabete.")
HPO = HpoIndex(dp.Terms.from_obo("""[Term]
id: HP:0001945
name: Fever

[Term]
id: HP:0002018
name: Nausea

[Term]
id: HP:0100281
name: Right lower quadrant pain
""".splitlines()))


def base(**kw):
    out = {"words": "", "name": "", "category": "manifestation", "specific": True, "what": "", "interpretation":
           "present", "value": None, "threshold": None, "site": "", "side": "", "observed_by": "history",
           "time": "", "nature": "pathological", "state": "present", "temporality": "current", "subject": "patient",
           "quote": ""}
    return {**out, **kw}


CARDS = [
    base(words="Dolore in fossa iliaca destra", name="right lower quadrant pain", what="abdominal pain",
         site="fossa iliaca destra", side="right", quote="Dolore in fossa iliaca destra da ieri, con nausea."),
    base(words="nausea", name="nausea", what="nausea", quote="Dolore in fossa iliaca destra da ieri, con nausea."),
    base(words="Non febbre", name="fever", what="body temperature", observed_by="examination", state="absent",
         quote="Non febbre."),
    base(words="Madre con diabete", name="diabetes mellitus", what="diabetes", subject="family_member",
         quote="Madre con diabete."),
]


def interpreter(cards, sees_crp):
    def chat(prompt):
        if prompt.startswith("You read"):
            assert "document of one patient" in prompt and "give no code" in prompt
            crp = [base(words="PCR 52 mg/L", name="elevated C-reactive protein" if sees_crp == "long" else
                        "elevated CRP", what="C-reactive protein", interpretation="increased",
                        value={"number": 52, "unit": "mg/L"}, observed_by="laboratory", quote="PCR 52 mg/L.")]
            return json.dumps(cards + crp)
        if prompt.startswith("You and another reader"):
            return '[{"item": 1, "same_as": "a"}]'
        return "[]"
    chat.local_only = True
    return chat


def readers(**kw):
    linker, _ = build_linker()
    return ReportReaders(linker, interpreters={"nemotron": interpreter(CARDS, "long"),
                                               "muse": interpreter(CARDS, "short")}, hpo=HPO, **kw)


def test_every_document_of_the_case_is_read_with_the_one_procedure():
    reading = readers().read("c1", NOTE)
    assert reading["procedure"] == ONE and reading["status"] == "read"
    [entry] = reading["texts"]
    by = {c["name"]: c for c in entry["cards"]}
    assert by["fever"]["state"] == "absent" and by["fever"]["concept"]["code"] == "HP:0001945"
    assert by["right lower quadrant pain"]["concept"]["code"] == "HP:0100281"
    crp = by["elevated C-reactive protein"]                      # other words for one finding: paired in the sentence
    assert crp["paired_by"] == "same_in_context" and crp["doubts"] == [] and crp["value"]["number"] == 52
    assert entry["questions"] == 2 and "readings" not in crp     # one look at the other's findings, by each


def test_the_cards_become_the_findings_of_the_graph_of_the_case():
    reading = readers().read("c1", NOTE)
    graph = cg.CaseGraph(case_id="c1")
    found, notes = add_cards(graph, "note", NOTE, reading["texts"][0]["cards"])
    by = {f.details["name"]: f for f in found}
    assert by["fever"].polarity == cg.NEGATED and by["fever"].concept.curie == "HP:0001945"
    assert by["fever"].provenance.words == "Non febbre." and NOTE[by["fever"].provenance.start:].startswith("Non")
    assert "diabetes mellitus" not in by and "family member" in notes[0]       # not the patient's
    assert by["elevated C-reactive protein"].certainty == "certain"


def test_without_interpreters_on_this_machine_nothing_is_read_and_the_case_says_so():
    linker, _ = build_linker()
    reading = ReportReaders(linker).read("c1", NOTE)
    assert reading["status"] == "interpreters_not_configured"
    assert reading["texts"][0]["read"] is False and "no interpreter" in reading["to_a_person"][0]["reason"]


def test_a_model_off_this_machine_never_reads_a_patients_document():
    linker, _ = build_linker()
    far = lambda p: "[]"   # noqa: E731 -- not marked local_only
    reading = ReportReaders(linker, interpreters={"remote": far}).read("c1", NOTE)
    assert reading["texts"][0]["read"] is False and reading["texts"][0]["reason"] == "reading_failed:ValueError"


def test_a_patients_finding_and_a_publications_manifestation_meet_by_the_same_concept():
    """A concept of the thesaurus (no HPO term) is the same for the case and for the profile: the manifestation is
    present when the case states it, doubtful when the interpreters did not agree."""
    from melampo.clinical.manifestation_state import assess

    graph = cg.CaseGraph(case_id="c2")
    card = base(words="Psoas positivo", name="positive psoas sign", what="psoas sign", interpretation="positive",
                observed_by="examination", quote="Psoas positivo.")
    add_cards(graph, "note", "Psoas positivo.", [{**card, "concept": {"concept": "MLP:0a1b2c3d4e", "code": ""},
                                                   "doubts": []}])
    wanted = cg.Manifestation("m1", "psoas sign", cg.Concept("MELAMPO", "MLP:0a1b2c3d4e"))
    assert assess(wanted, graph).state == cg.PRESENT
    doubtful = cg.CaseGraph(case_id="c3")
    add_cards(doubtful, "note", "Psoas positivo.", [{**card, "concept": {"concept": "MLP:0a1b2c3d4e", "code": ""},
                                                     "doubts": ["one_interpreter"]}])
    assert assess(wanted, doubtful).state == cg.DOUBTFUL
