"""The graph of a case (14 November 2026): branches, three levels, four states, likelihood ratios with absences."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from melampo.clinical import case_graph as cg
from melampo.clinical.case_graph_builder import (
    add_declared_diseases,
    add_laboratory,
    add_medications,
    add_report,
    build_case_graph,
)
from melampo.clinical.manifestation_state import (
    assess,
    assess_case,
    unexplained_findings,
)
from melampo.clinical.phenopacket import from_phenopacket, to_phenopacket
from melampo.data.lab_results import extract_lab_results
from melampo.memory.exam_area import ExamAreas
from melampo.reasoning.likelihood_ranking import rank

ROOT = Path(__file__).resolve().parents[1]
REPORT = ("TC addome con mdc. Colecisti distesa con parete ispessita e calcolo nell'infundibolo. "
          "Non versamento libero. Pancreas nei limiti.")
LAB = "Creatinina 1,45 mg/dL (0,50 - 1,20)\nBilirubina totale 2,8 mg/dL (0,3 - 1,2)\nLipasi 40 U/L (13 - 60)\n"


def _site(text, piece, cid, polarity="", certainty="", concept=None):
    at = text.index(piece)
    row = {"cid": cid, "mention": piece, "start": at, "end": at + len(piece), "polarity": polarity,
           "certainty": certainty}
    if concept:
        row["concept"] = concept
    return row


def _case():
    reading = {"sites": [
        _site(REPORT, "Colecisti distesa con parete ispessita", "gallbladder",
              concept={"code": "HP:0025566", "label": "Gallbladder wall thickening"}),
        _site(REPORT, "Non versamento libero", "abdomen", polarity="negated",
              concept={"code": "HP:0001541", "label": "Ascites"}),
    ]}
    graph, notes = build_case_graph(
        "case-1",
        reports=[{"document": "rep-1", "text": REPORT, "reading": reading, "title": "TC addome con mdc",
                  "date": "2026-11-10", "modality": "CT"}],
        laboratory=[{"document": "lab-1", "text": LAB, "extraction": extract_lab_results(LAB), "date": "2026-11-10"}],
        medications=[{"words": "Atorvastatina 20 mg 1 cp la sera", "name": "atorvastatina", "dose": "20",
                      "unit": "mg", "frequency": "1 cp la sera", "start": "2025-01-10"}],
        diseases=[{"words": "colecistite acuta", "status": "sospetta", "document": "rep-1"},
                  {"words": "ipertensione arteriosa", "status": "pregressa"}],
        sources={"HPO": "2026-09-01"}, taken="2026-11-14", exam_areas=ExamAreas.load())
    return graph, notes


def _manifestation(mid, words, code=None, typicality="", frequency=None, **kw):
    return cg.Manifestation(mid, words, concept=cg.Concept("HP", code, words) if code else None,
                            typicality=typicality, frequency=frequency, citations=[{"source": "test"}], **kw)


def test_the_case_enters_in_its_branches_and_the_candidate_branch_is_empty():
    graph, notes = _case()
    assert notes == []
    b = graph.branches()
    assert len(b[cg.DRUGS]) == 1 and len(b[cg.EXAMS]) == 2
    assert [d.words for d in b[cg.HYPOTHESIZED_DISEASES]] == ["colecistite acuta"]
    assert [d.words for d in b[cg.PRESENT_DISEASES]] == ["ipertensione arteriosa"]
    assert b[cg.CANDIDATE_DISEASES] == []
    assert graph.document("rep-1").areas == ["abdomen"]
    sub = graph.of_document("lab-1")
    assert len(sub["measurements"]) == 3
    creat = sub["measurements"][0]
    assert creat.provenance.words == "Creatinina 1,45 mg/dL (0,50 - 1,20)" and creat.interpretation == "above"
    for f in graph.findings:              # the words are the report's own, at their offsets
        assert REPORT[f.provenance.start:f.provenance.end] == f.words


def test_a_candidate_enters_only_from_the_diagnostic_path_with_what_proposed_it():
    graph, _ = _case()
    with pytest.raises(cg.CaseGraphError):
        graph.add_disease(cg.Disease("x", cg.CANDIDATE_DISEASES, "pancreatite", cg.PHYSICIAN))
    with pytest.raises(cg.CaseGraphError):
        graph.add_candidate(cg.Disease("x", "", "pancreatite", ""))
    graph.add_candidate(cg.Disease("x", "", "coledocolitiasi", "", citations=[{"channel": "literature"}]))
    assert [d.words for d in graph.branch(cg.CANDIDATE_DISEASES)] == ["coledocolitiasi"]
    assert graph.validate() == []
    graph.diseases[-1].origin = cg.PHYSICIAN
    assert any("candidate not found by the diagnostic path" in p for p in graph.validate())


def test_what_cannot_be_read_is_refused_never_invented():
    graph = cg.CaseGraph("c")
    _, refused = add_medications(graph, [{"name": "metformina"}])
    assert refused and graph.medications == []
    _, refused = add_declared_diseases(graph, [{"words": "diabete", "status": ""}])
    assert refused and graph.diseases == []
    graph.findings.append(cg.Finding("f", "nowhere", "", cg.Provenance("nowhere", "")))
    problems = graph.validate()
    assert any("no words" in p for p in problems)


def test_the_graph_round_trips_as_data_and_as_a_phenopacket():
    graph, _ = _case()
    data = json.loads(json.dumps(graph.as_dict()))
    assert cg.CaseGraph.from_dict(data).as_dict() == graph.as_dict()
    packet = json.loads(json.dumps(to_phenopacket(graph)))
    assert packet["metaData"]["phenopacketSchemaVersion"] == "2.0"
    excluded = [f for f in packet["phenotypicFeatures"] if f["excluded"]]
    assert excluded and excluded[0]["type"]["id"] == "HP:0001541"
    assert [i["progressStatus"] for i in packet["interpretations"]] == ["IN_PROGRESS"]
    back = from_phenopacket(packet)
    for name in ("documents", "findings", "measurements", "medications", "diseases"):
        assert [x.as_dict() for x in getattr(back, name)] == [x.as_dict() for x in getattr(graph, name)], name
    assert back.snapshot.sources == {"HPO": "2026-09-01"}


@pytest.mark.parametrize("language", ["en", "it"])
def test_twenty_cases_of_written_sentences_keep_every_word_and_place(language):
    """Step 1's criterion on the sentences of the linker's held-out sets: every mention a finding of a case,
    twenty cases, every word and offset kept through the data and through the phenopacket."""
    rows = [json.loads(x) for x in (ROOT / f"data/linking/heldout_{language}.jsonl").read_text("utf-8").splitlines()
            if x.strip()]
    cases = [rows[i::20] for i in range(20)]
    for number, group in enumerate(cases):
        text = "\n".join(r["sentence"] for r in group)
        reading, at = {"sites": []}, 0
        for r in group:
            start = at + r["start"]
            reading["sites"].append({"cid": r["target"], "mention": r["mention"], "start": start,
                                     "end": start + len(r["mention"])})
            at += len(r["sentence"]) + 1
        graph = cg.CaseGraph(f"{language}-{number}")
        add_report(graph, "d", text, reading)
        assert graph.validate() == []
        back = from_phenopacket(json.loads(json.dumps(to_phenopacket(cg.CaseGraph.from_dict(graph.as_dict())))))
        assert [f.as_dict() for f in back.findings] == [f.as_dict() for f in graph.findings]
        assert all(text[f.provenance.start:f.provenance.end] == f.words for f in back.findings)


def test_four_states_and_the_difference_between_absent_and_not_assessed():
    graph, _ = _case()
    wall = _manifestation("wall", "Gallbladder wall thickening", "HP:0025566", "hallmark",
                          verified_by=[{"kind": "imaging", "modality": "CT", "region": "abdomen"}])
    ascites = _manifestation("ascites", "Ascites", "HP:0001541", "common")
    bilirubin = _manifestation("bili", "Hyperbilirubinemia", "HP:0002904", "common",
                               verified_by=[{"kind": "laboratory", "analyte": "bilirubina totale", "direction": "above"}])
    lipase = _manifestation("lipase", "Elevated lipase", "HP:0031094", "hallmark",
                            verified_by=[{"kind": "laboratory", "analyte": "lipasi", "direction": "above"}])
    murphy = _manifestation("murphy", "Murphy sign", "HP:9999999", "hallmark")
    splenic = _manifestation("spleen", "Splenomegaly", "HP:0001744", "common", region="abdomen",
                             verified_by=[{"kind": "imaging", "modality": "CT"}])
    brain = _manifestation("brain", "Cerebral edema", "HP:0002181", "common",
                           verified_by=[{"kind": "imaging", "modality": "MR", "region": "head"}])
    got = {m.manifestation_id: assess(m, graph) for m in (wall, ascites, bilirubin, lipase, murphy, splenic, brain)}
    assert got["wall"].state == cg.PRESENT and got["wall"].basis == cg.STATED
    assert got["ascites"].state == cg.ABSENT and got["ascites"].basis == cg.STATED
    assert got["bili"].state == cg.PRESENT and got["bili"].basis == cg.MEASURED
    assert got["lipase"].state == cg.ABSENT and got["lipase"].basis == cg.MEASURED
    assert got["murphy"].state == cg.NOT_ASSESSED                      # no exam shows a clinical sign
    assert got["spleen"].state == cg.ABSENT and got["spleen"].basis == cg.SILENT   # a CT of the abdomen, silent
    assert got["brain"].state == cg.NOT_ASSESSED                       # a CT of the abdomen says nothing of the brain


def test_a_more_specific_finding_counts_and_disagreeing_facts_stay_in_doubt():
    graph, _ = _case()
    general = _manifestation("thick", "Abnormal gallbladder", "HP:0005264", "common")
    assert assess(general, graph).state == cg.NOT_ASSESSED
    below = lambda child, parent: (child, parent) == ("HP:0025566", "HP:0005264")
    assert assess(general, graph, below=below).state == cg.PRESENT
    at = REPORT.index("Pancreas nei limiti")
    graph.findings.append(cg.Finding("f-asc", "rep-1", "Pancreas nei limiti",
                                     cg.Provenance("rep-1", "Pancreas nei limiti", at, at + 19),
                                     concept=cg.Concept("HP", "HP:0001541")))
    state = assess(_manifestation("ascites", "Ascites", "HP:0001541", "common"), graph)
    assert state.state == cg.DOUBTFUL and len(state.evidence) == 2


def _profile(did, words, *manifestations):
    return cg.Disease(did, cg.HYPOTHESIZED_DISEASES, words, cg.PHYSICIAN, manifestations=list(manifestations))


def test_an_absent_hallmark_lowers_a_disease_and_an_excluding_finding_sinks_it():
    graph, _ = _case()
    wall = lambda: _manifestation("wall", "Gallbladder wall thickening", "HP:0025566", "hallmark")
    cholecystitis = _profile("chole", "acute cholecystitis", wall(),
                             _manifestation("bili", "Hyperbilirubinemia", "HP:0002904", "common",
                                            verified_by=[{"kind": "laboratory", "analyte": "bilirubina totale",
                                                          "direction": "above"}]))
    pancreatitis = _profile("pancr", "acute pancreatitis", _manifestation(
        "lipase", "Elevated lipase", "HP:0031094", "hallmark",
        verified_by=[{"kind": "laboratory", "analyte": "lipasi", "direction": "above"}]))
    ascitic = _profile("asc", "something with ascites", wall(),
                       _manifestation("ascites", "Ascites", "HP:0001541", "hallmark"))
    excluded = _profile("exc", "a disease the thickened wall excludes",
                        _manifestation("wall", "Gallbladder wall thickening", "HP:0025566", "excluding"))
    for d in (cholecystitis, pancreatitis, ascitic, excluded):
        graph.add_disease(d)
    counts = assess_case(graph)
    assert counts[cg.ABSENT] >= 2
    ranked = rank([cholecystitis, pancreatitis, ascitic, excluded])
    order = [r.disease_id for r in ranked]
    assert order[0] == "chole" and order[-1] == "exc"
    assert order.index("asc") > order.index("chole")          # the same wall, but its hallmark ascites is absent
    pan = next(r for r in ranked if r.disease_id == "pancr")
    assert pan.contributions[0].log10_lr < -0.5                # lipase within range: about a tenth
    assert math.isclose(sum(r.posterior for r in ranked), 1.0)
    assert ranked[-1].contributions[0].note == "excluding manifestation found"


def test_silence_weighs_less_than_a_stated_absence_and_not_assessed_weighs_nothing():
    graph, _ = _case()
    spleen_silent = _profile("s1", "silent", _manifestation(
        "spleen", "Splenomegaly", "HP:0001744", "hallmark", region="abdomen",
        verified_by=[{"kind": "imaging", "modality": "CT"}]))
    stated = _profile("s2", "stated", _manifestation("ascites", "Ascites", "HP:0001541", "hallmark"))
    unseen = _profile("s3", "unseen", _manifestation("murphy", "Murphy sign", "HP:9999999", "hallmark"))
    for d in (spleen_silent, stated, unseen):
        graph.add_disease(d)
    assess_case(graph)
    by = {r.disease_id: r for r in rank([spleen_silent, stated, unseen], background_frequencies={})}
    assert by["s3"].log10_odds == 0 and by["s3"].not_assessed == ["murphy"]
    assert by["s2"].log10_odds < by["s1"].log10_odds < 0


def test_the_unexplained_findings_are_what_the_path_takes_to_the_channels():
    graph, _ = _case()
    assert [f.concept.code for f in unexplained_findings(graph)] == ["HP:0025566"]
    graph.add_disease(_profile("chole", "acute cholecystitis",
                               _manifestation("wall", "Gallbladder wall thickening", "HP:0025566", "hallmark")))
    assert unexplained_findings(graph) == []


def test_a_laboratory_row_is_kept_with_its_printed_line():
    graph = cg.CaseGraph("c")
    text = "Emocromo\nHb 9,8 g/dL (12 - 16)\n"
    add_laboratory(graph, "lab", text, extract_lab_results(text))
    assert graph.measurements and graph.measurements[0].provenance.line == 2
    assert graph.measurements[0].provenance.words == "Hb 9,8 g/dL (12 - 16)"


def test_the_case_is_written_to_falkordb_and_read_back(tmp_path):
    pytest.importorskip("redislite")
    from redislite.falkordb_client import FalkorDB

    from melampo.memory.case_graph_store import load_case, read_case

    store = FalkorDB(str(tmp_path / "g.db")).select_graph("cases_test")
    graph, _ = _case()
    graph.add_disease(_profile("chole", "acute cholecystitis",
                               _manifestation("wall", "Gallbladder wall thickening", "HP:0025566", "hallmark")))
    assess_case(graph)
    assert load_case(store, graph)["findings"] == 2
    load_case(store, graph)                                     # written again: replaced, not doubled
    back = read_case(store, "case-1")
    key = lambda xs: sorted((x.as_dict() for x in xs), key=json.dumps)
    for name in ("documents", "findings", "measurements", "medications", "diseases"):
        assert key(getattr(back, name)) == key(getattr(graph, name)), name


def test_the_falkordb_writer_sends_only_parameters(monkeypatch):
    from melampo.memory.case_graph_store import load_case

    sent = []
    fake = SimpleNamespace(query=lambda q, params=None: sent.append((q, params)))
    graph, _ = _case()
    load_case(fake, graph)
    assert all("$" in q or "MATCH" in q for q, _ in sent)
    assert not any(REPORT[:20] in q for q, _ in sent)          # the words travel as parameters, never in the query


def test_the_graph_is_built_from_what_ingestion_already_holds():
    from melampo.clinical.case_graph_builder import from_ingestion

    lab_text = "Bilirubina totale 2,8 mg/dL (0,3 - 1,2)\n"
    attachment = SimpleNamespace(index=1, text=lab_text, lab=extract_lab_results(lab_text),
                                 loinc=({"analyte": "Bilirubina totale", "status": "mapped", "code": "1975-2"},))
    reading = {"texts": [{"report": "note", "read": True, "kinds": ["imaging"],
                          "sites": [_site(REPORT, "Non versamento libero", "abdomen", polarity="negated")]}]}
    graph, notes = from_ingestion("c", REPORT, SimpleNamespace(attachments=(attachment,)), reading,
                                  {"medications": [{"words": "Metformina 500 mg x2"}],
                                   "diseases": [{"words": "colecistite", "status": "sospetta"}]})
    assert notes == []
    assert graph.document("note").title == "TC addome con mdc. Colecisti distesa con parete ispessita e calcolo " \
                                           "nell'infundibolo. Non versamento libero. Pancreas nei limiti."[:200]
    assert graph.document("note").areas == ["abdomen"]
    assert graph.measurements[0].code.code == "LOINC:1975-2"
    assert graph.findings[0].polarity == cg.NEGATED
    assert graph.summary()[cg.HYPOTHESIZED_DISEASES] == 1 and graph.summary()[cg.CANDIDATE_DISEASES] == 0


def test_a_manifestation_reported_by_one_source_weighs_less_than_one_established():
    graph, _ = _case()
    reported = _manifestation("wall", "Gallbladder wall thickening", "HP:0025566", "hallmark")
    reported.evidence = "reported"
    established = _manifestation("wall", "Gallbladder wall thickening", "HP:0025566", "hallmark")
    established.evidence = "established"
    a, b = _profile("a", "a", reported), _profile("b", "b", established)
    graph.add_disease(a)
    graph.add_disease(b)
    assess_case(graph)
    by = {r.disease_id: r for r in rank([a, b], background_frequencies={})}
    assert 0 < by["a"].log10_odds < by["b"].log10_odds
    assert "reported by one source" in by["a"].contributions[0].note


def test_a_measure_against_a_verified_row_becomes_the_manifestation_present_or_absent():
    """20 November 2026: "spleen 15 cm" compared with the verified row whose deviation above is Splenomegaly."""
    from melampo.clinical.case_graph_builder import add_measures
    from melampo.clinical.manifestation_state import assess

    text = "Milza di dimensioni aumentate, diametro longitudinale 15 cm."
    piece = "diametro longitudinale 15 cm"
    statements = [{"statement": "measure", "structure": "spleen", "piece": piece, "reference": "spleen-length-us",
                   "against": "above", "clause": text, "report": "rep-1"},
                  {"statement": "measure", "structure": "portal_vein_and_splenic_vein", "piece": "calibro 16 mm",
                   "reference": "portal-vein-diameter-ct", "against": "above", "clause": "", "report": "rep-1"}]
    graph = cg.CaseGraph("case-m")
    added, notes = add_measures(graph, statements, {"rep-1": text})
    assert len(added) == 1 and "not a verified row" in notes[0]          # the row in doubt is left out
    m = added[0]
    assert (m.value, m.unit, m.interpretation) == (150.0, "mm", "above")
    assert text[m.provenance.start:m.provenance.end] == piece and graph.validate() == []
    spleno = cg.Manifestation("s", "Splenomegaly", cg.Concept("HPO", "HP:0001744"), citations=[{"id": "x"}])
    assert assess(spleno, graph).state == cg.PRESENT
    normal = cg.CaseGraph("case-n")
    add_measures(normal, [{**statements[0], "piece": "diametro longitudinale 10 cm"}],
                 {"rep-1": "Milza nei limiti, diametro longitudinale 10 cm."})
    state = assess(spleno, normal)
    assert (state.state, state.basis) == (cg.ABSENT, cg.MEASURED)
