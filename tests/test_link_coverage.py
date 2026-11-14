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
    monkeypatch.setattr(lc, "statpearls", lambda f, n: {"chapter": "NBK1", "imaging": True, "imaging_mentions": 4})
    got = lc.count(lc.Fetch(pause=0), "k", tmp_path / "mondo.obo", tmp_path / "doid.obo",
                   diagnoses=("pneumonia", "acute appendicitis"))
    row = got["diagnoses"][0]
    assert row["snomed_defining"] == 2 and row["nci_finding"] == 3 and row["doid_symptoms"] == 2
    assert row["monarch_phenotypes"] == 12 and got["summary"]["with_statpearls_imaging"] == "2/2"
    assert got["summary"]["with_snomed_defining_relations"] == "1/2"
    assert "C1" not in json.dumps(got)
    assert lc.IMAGING.search("CT of the abdomen") and not lc.IMAGING.search("cytology")
