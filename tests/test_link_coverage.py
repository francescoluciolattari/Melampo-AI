"""The second count (14 November 2026): disease -> finding links by source; counts and open ids only."""

import importlib.util
import json
from collections import Counter
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "link_coverage.py"
spec = importlib.util.spec_from_file_location("link_coverage", _SCRIPT)
lc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lc)


def test_the_links_are_counted_by_source_without_snomed_content(tmp_path, monkeypatch):
    (tmp_path / "mondo.obo").write_text("[Term]\nid: MONDO:1\nname: pneumonia\n")
    (tmp_path / "doid.obo").write_text("[Term]\nid: DOID:552\nname: pneumonia\n"
                                       "relationship: has_symptom SYMP:1 ! cough\n"
                                       "relationship: has_symptom SYMP:2 ! fever\n")

    class Client:
        def __init__(self, *a):
            pass

        def cuis(self, name, **k):
            return ["C1"] if name == "pneumonia" else []

    monkeypatch.setattr(lc.sc, "Client", Client)
    monkeypatch.setattr(lc, "umls_relations", lambda f, k, cui: Counter({
        ("SNOMEDCT_US", "has_finding_site"): 1, ("SNOMEDCT_US", "has_associated_morphology"): 1,
        ("NCI", "disease_may_have_finding"): 3, ("MSH", "mapped_to"): 1}))
    monkeypatch.setattr(lc, "monarch_phenotypes", lambda f, m: 12)
    monkeypatch.setattr(lc, "statpearls", lambda f, n: {"chapter": "NBK1", "text": "read", "imaging": True,
                                                         "imaging_mentions": 4})
    got = lc.count(lc.Fetch(pause=0), "k", tmp_path / "mondo.obo", tmp_path / "doid.obo",
                   diagnoses=("pneumonia", "acute appendicitis"))
    row = got["diagnoses"][0]
    assert row["snomed_defining"] == 2 and row["nci_finding"] == 3 and row["doid_symptoms"] == 2
    assert row["monarch_phenotypes"] == 12 and got["summary"]["with_statpearls_imaging_of_read"] == "2/2"
    assert got["summary"]["with_snomed_defining_relations"] == "1/2"
    assert "C1" not in json.dumps(got)
    assert lc.IMAGING.search("CT of the abdomen") and not lc.IMAGING.search("cytology")


OWL = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
  <owl:Class rdf:about="http://purl.obolibrary.org/obo/DOID_8469">
    <rdfs:label>influenza</rdfs:label>
    <rdfs:subClassOf><owl:Restriction>
      <owl:onProperty rdf:resource="http://purl.obolibrary.org/obo/RO_0002452"/>
      <owl:someValuesFrom rdf:resource="http://purl.obolibrary.org/obo/SYMP_0000613"/>
    </owl:Restriction></rdfs:subClassOf>
    <rdfs:subClassOf><owl:Restriction>
      <owl:onProperty rdf:resource="http://purl.obolibrary.org/obo/RO_0002452"/>
      <owl:someValuesFrom rdf:resource="http://purl.obolibrary.org/obo/SYMP_0000614"/>
    </owl:Restriction></rdfs:subClassOf>
  </owl:Class>
</rdf:RDF>
"""


def test_the_symptoms_of_the_disease_ontology_are_read_from_the_owl_file(tmp_path):
    (tmp_path / "doid.obo").write_text("[Term]\nid: DOID:8469\nname: influenza\n")
    (tmp_path / "doid.owl").write_text(OWL)
    _, without = lc.doid_symptoms(tmp_path / "doid.obo")
    assert without.get("DOID:8469", 0) == 0                     # the OBO file has no axioms: the first count's 0/50
    by_name, symptoms = lc.doid_symptoms(tmp_path / "doid.obo", tmp_path / "doid.owl")
    assert by_name["influenza"] == {"DOID:8469"} and symptoms["DOID:8469"] == 2


class _Pages:
    def __init__(self, pages):
        self.pages, self.calls = pages, 0

    def json(self, url):
        if "esearch" in url:
            return {"esearchresult": {"idlist": ["9"]}}
        return {"result": {"9": {"chapteraccessionid": "NBK1", "title": "Appendicitis"}}}

    def text(self, url):
        self.calls += 1
        return self.pages.get("printable" if "printable" in url else "plain", "")


def test_a_chapter_whose_text_was_not_read_is_never_counted_as_one_without_imaging():
    blocked = lc.statpearls(_Pages({"printable": "<html>Checking your browser before accessing</html>"}), "x")
    assert blocked["text"] == "blocked" and not blocked["imaging"]
    page = "<h2>Etiology</h2>...<h2 id='e'>Evaluation</h2><p>CT of the abdomen is the test of choice</p>"
    read = lc.statpearls(_Pages({"printable": page}), "x")
    assert read["text"] == "read" and read["imaging"]
    none = lc.statpearls(_Pages({"plain": "<h2>Etiology</h2> fever"}), "x")
    assert none["text"] == "no_evaluation_section"
