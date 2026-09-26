"""Tests for evaluation/symptom_coverage.py and scripts/measure_symptom_coverage.py."""

import importlib.util
import json
from pathlib import Path

from melampo.evaluation.symptom_coverage import (
    ReferenceDisease,
    load_reference,
    measure,
    render_markdown,
)
from melampo.memory.symptom_sources import (
    RELATION_EXCLUDES_FINDING,
    RELATION_HAS_FINDING,
    DiseaseIndex,
    MondoDisease,
    SymptomLink,
)

ROOT = Path(__file__).resolve().parents[1]


def _index():
    return DiseaseIndex.from_mondo(
        [
            MondoDisease("MONDO:1", "flu"),
            MondoDisease("MONDO:2", "gout"),
            MondoDisease("MONDO:3", "bird flu", parents=("MONDO:1",)),
        ]
    )


def _ref(mondo_id, specialty="x"):
    return ReferenceDisease(mondo_id, mondo_id, mondo_id, "", specialty, "label")


def _link(source, mondo_id, symptom, hpo="", relation=RELATION_HAS_FINDING, tier=1):
    return SymptomLink(source, "id", "d", symptom, symptom, relation, tier, mondo_id=mondo_id, hpo_id=hpo)


def test_measure_counts_distinct_findings_per_source():
    links = [
        _link("a", "MONDO:1", "S1", "HP:1"),
        _link("a", "MONDO:1", "S1", "HP:1"),  # duplicate finding counts once
        _link("a", "MONDO:1", "S2"),
        _link("a", "MONDO:1", "S3", relation=RELATION_EXCLUDES_FINDING),
        _link("b", "MONDO:2", "S4", "HP:4"),
    ]
    result = measure([_ref("MONDO:1"), _ref("MONDO:2")], links, _index(), sources=["a", "b"], usable_threshold=2)
    per = result["per_disease"]
    assert per["MONDO:1"]["a"]["present"] == 2
    assert per["MONDO:1"]["a"]["excluded"] == 1
    assert per["MONDO:1"]["a"]["mapped_to_hpo"] == 1
    a = next(summary for summary in result["sources"] if summary["source"] == "a")
    assert a["diseases_with_links"] == 1
    assert a["diseases_usable"] == 1
    assert a["hpo_mapping_rate"] == 0.5
    assert result["uncovered_everywhere"] == []


def test_links_on_narrower_diseases_are_reported_separately():
    links = [_link("a", "MONDO:3", "S9", "HP:9")]
    result = measure([_ref("MONDO:1")], links, _index(), sources=["a"])
    cov = result["per_disease"]["MONDO:1"]["a"]
    assert cov["present"] == 0
    assert cov["via_descendants"] == 1
    assert result["sources"][0]["diseases_only_via_descendants"] == 1
    assert result["uncovered_everywhere"] == ["MONDO:1"]


def test_union_counts_only_hpo_mapped_findings():
    links = [_link("a", "MONDO:1", "S1", "HP:1"), _link("b", "MONDO:1", "S2", "HP:2"), _link("b", "MONDO:1", "S3")]
    result = measure([_ref("MONDO:1")], links, _index(), sources=["a", "b"], usable_threshold=2)
    assert result["union_hpo"] == {"covered": 1, "usable": 1}
    assert result["per_disease"]["MONDO:1"]["union_hpo"] == 2


def test_markdown_names_every_source_and_the_notes():
    result = measure([_ref("MONDO:1")], [_link("a", "MONDO:1", "S1")], _index(), sources=["a"])
    text = render_markdown(result, [_ref("MONDO:1")], notes=["NCIt not measured: no key"])
    assert "| a |" in text
    assert "NCIt not measured: no key" in text


def test_the_committed_reference_list_is_well_formed():
    reference = load_reference(ROOT / "data" / "common_diseases_reference.tsv")
    ids = [disease.mondo_id for disease in reference]
    assert len(reference) == 141
    assert len(ids) == len(set(ids))
    assert all(mondo_id.startswith("MONDO:") and len(mondo_id) == 13 for mondo_id in ids)
    assert {disease.resolution for disease in reference} <= {"label", "exact_synonym", "manual"}


def test_every_reference_id_is_current_in_the_pinned_mondo():
    mondo = ROOT / "data" / "mondo.obo"
    if not mondo.is_file():
        import pytest

        pytest.skip("data/mondo.obo not fetched (uv run dvc update data/mondo.obo.dvc)")
    from melampo.memory.symptom_sources import parse_mondo

    index = DiseaseIndex.from_mondo(parse_mondo(mondo.read_text(encoding="utf-8").splitlines()))
    reference = load_reference(ROOT / "data" / "common_diseases_reference.tsv")
    assert [d.mondo_id for d in reference if d.mondo_id not in index.diseases] == []
    assert all(index.diseases[d.mondo_id].label == d.mondo_label for d in reference)


# --------------------------------------------------------------------------
# The script, end to end on a miniature data directory
# --------------------------------------------------------------------------


def _load_script():
    spec = importlib.util.spec_from_file_location("measure_symptom_coverage", ROOT / "scripts" / "measure_symptom_coverage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_runs_offline_on_a_miniature_data_dir(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "mondo.obo").write_text(
        "[Term]\nid: MONDO:0005812\nname: influenza\nxref: DOID:8469 {source=\"MONDO:equivalentTo\"}\n"
        "xref: OMIM:100100 {source=\"MONDO:equivalentTo\"}\n",
        encoding="utf-8",
    )
    (data / "hp.obo").write_text("[Term]\nid: HP:0001945\nname: Fever\n\n[Term]\nid: HP:0012735\nname: Cough\n", encoding="utf-8")
    (data / "phenotype.hpoa").write_text(
        "#description: test\ndatabase_id\tdisease_name\tqualifier\thpo_id\treference\tevidence\tonset\tfrequency\tsex\tmodifier\taspect\tbiocuration\n"
        "OMIM:100100\tinfluenza-like\t\tHP:0012735\tPMID:1\tPCS\t\t\t\t\tP\tHPO:x\n",
        encoding="utf-8",
    )
    (data / "symp.obo").write_text("[Term]\nid: SYMP:0000613\nname: fever\n", encoding="utf-8")
    (data / "doid.owl").write_bytes(
        b'<?xml version="1.0"?><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
        b'xmlns:owl="http://www.w3.org/2002/07/owl#" xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">'
        b'<owl:Class rdf:about="http://purl.obolibrary.org/obo/DOID_8469"><rdfs:label>influenza</rdfs:label>'
        b'<rdfs:subClassOf><owl:Restriction><owl:onProperty rdf:resource="http://purl.obolibrary.org/obo/RO_0002452"/>'
        b'<owl:someValuesFrom rdf:resource="http://purl.obolibrary.org/obo/SYMP_0000613"/></owl:Restriction></rdfs:subClassOf>'
        b"</owl:Class></rdf:RDF>"
    )
    reference = tmp_path / "reference.tsv"
    reference.write_text(
        "# test\nmondo_id\tmondo_label\tquery_name\ticd10\tspecialty\tresolution\n"
        "MONDO:0005812\tinfluenza\tinfluenza\tJ11\trespiratory\tlabel\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    script = _load_script()
    out = tmp_path / "out"
    assert script.main(["--data-dir", str(data), "--reference", str(reference), "--out-dir", str(out), "--skip-ncit", "--skip-wikidata"]) == 0

    result = json.loads((out / "coverage.json").read_text(encoding="utf-8"))
    per = result["per_disease"]["MONDO:0005812"]
    assert per["hpo"]["present"] == 1 and per["hpo"]["mapped_to_hpo"] == 1
    assert per["disease_ontology"]["present"] == 1 and per["disease_ontology"]["mapped_to_hpo"] == 1
    assert per["union_hpo"] == 2
    assert "NCIt skipped" in (out / "coverage.md").read_text(encoding="utf-8")
    assert len((out / "links.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_script_reports_ncit_as_not_measured_without_a_key(monkeypatch):
    monkeypatch.delenv("UMLS_API_KEY", raising=False)
    script = _load_script()
    notes: list[str] = []
    links, _ = script.measure_ncit([], DiseaseIndex(), notes)
    assert links is None
    assert "UMLS_API_KEY not configured" in notes[0]
