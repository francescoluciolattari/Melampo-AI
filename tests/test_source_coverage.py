"""The count of the sources (14 November 2026): no model, no SNOMED CT content written."""

import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "source_coverage.py"
spec = importlib.util.spec_from_file_location("source_coverage", _SCRIPT)
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)


class FakeClient:
    calls = 0

    def cuis(self, string, *, source_ui=""):
        return {"HP:1": ["C1"], "renal mass": ["C2"], "pneumonia": ["C3"]}.get(source_ui or string, [])

    def sources_of(self, cui):
        return {"C1": {"HPO", "SNOMEDCT_US", "NCI"}, "C2": {"SNOMEDCT_US"}, "C3": {"SNOMEDCT_US", "ICD10CM"}}[cui]

    def radlex(self, name):
        return name == "renal mass"


def test_the_count_says_which_sources_hold_a_concept_and_writes_no_snomed_content(tmp_path, monkeypatch):
    (tmp_path / "hp.obo").write_text("[Term]\nid: HP:1\nname: Renal neoplasm\n")
    (tmp_path / "mondo.obo").write_text("[Term]\nid: MONDO:1\nname: pneumonia\nxref: OMIM:1 {source=x}\n")
    (tmp_path / "p.hpoa").write_text("#x\ndatabase_id\tdisease_name\tqualifier\thpo_id\nOMIM:1\tp\t\tHP:1\n")
    labels = tmp_path / "labels.txt"
    labels.write_text("# x\nparrot-1|HP:1|large renal mass (no renal mass term)\nparrot-2|NONE|normal\n")
    monkeypatch.setattr(sc, "DIAGNOSES", ("pneumonia", "acute appendicitis"))
    got = sc.count(FakeClient(), tmp_path / "hp.obo", tmp_path / "mondo.obo", tmp_path / "p.hpoa", [labels])
    s = got["summary"]
    assert s["findings_labelled"]["SNOMED CT"] == "1/1" and s["findings_labelled"]["RadLex"] == "0/1"
    assert got["gaps"][0]["finding"] == "renal mass" and s["concepts_hpo_lacks"]["any_source"] == "1/1"
    pneumonia = got["diagnoses"][0]
    assert pneumonia["mondo"] == ["MONDO:1"] and pneumonia["hpoa_phenotypes"] == 1
    assert s["common_diagnoses"]["MONDO"] == "1/2" and s["common_diagnoses"]["with_hpo_annotations"] == "1/2"
    text = json.dumps(got)
    assert "C1" not in text and "SNOMEDCT_US" not in text       # no UMLS or SNOMED CT identifier is written
