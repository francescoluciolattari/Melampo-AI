"""The drug branch (15 November 2026): AIFA names to ingredients, ATC to international names, US labels; the
patient's drugs that a label names, the patient's findings a label names among its effects."""

import importlib.util
import json
from pathlib import Path

from melampo.clinical import case_graph as cg
from melampo.clinical.case_graph_builder import add_medications
from melampo.clinical.drug_branch import DrugKnowledge, drug_branch, sentences
from melampo.connectors import drug_sources as ds

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("drug_sources_script", ROOT / "scripts/drug_sources.py")
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)

AIFA = ("Codice_AIC;Denominazione;Confezione;Ragione_sociale;Stato_amministrativo;Principio_attivo;Codice_ATC\n"
        "035362018;TORVAST;20 MG COMPRESSE;VIATRIS;Autorizzata;atorvastatina calcio triidrato;C10AA05\n"
        "027353010;NORVASC;5 MG COMPRESSE;VIATRIS;Autorizzata;amlodipina besilato;C08CA01\n"
        "034501011;CLARITROMICINA DOC;500 MG;DOC;Autorizzata;claritromicina;J01FA09\n"
        "012345678;EXFORGE;5/80 MG;NOVARTIS;Autorizzata;amlodipina besilato/valsartan;C09DB01\n")


def test_aifa_columns_are_recognised_by_their_words_and_an_unknown_file_is_refused():
    products, report = ds.read_aifa(AIFA)
    assert len(products) == 4 and products[0].ingredient == "atorvastatina calcio triidrato"
    assert report["columns"]["atc"] == "Codice_ATC" and report["columns"]["name"] == "Denominazione"
    nothing, report = ds.read_aifa("a,b,c\n1,2,3\n")
    assert nothing == [] and "problem" in report
    assert ds.ingredient_core("atorvastatina calcio triidrato") == "atorvastatina"


def test_ingredients_resolve_through_their_atc_and_disagreement_stays_in_doubt():
    products, _ = ds.read_aifa(AIFA)
    wikidata = {"C10AA05": ["atorvastatin"], "C08CA01": ["amlodipine"], "J01FA09": ["clarithromycin"],
                "C09CA03": ["valsartan"]}
    ema = {"C10AA05": ["atorvastatin"], "J01FA09": ["clarithromycine"]}
    got = script.resolve(ds.ingredients_of(products), wikidata, ema)
    assert got["atorvastatina calcio triidrato"] == {"atc": ["C10AA05"], "inn": ["atorvastatin"],
                                                     "basis": "wikidata+ema"}
    assert got["amlodipina besilato"]["basis"] == "wikidata"
    assert got["claritromicina"]["basis"] == "doubt"
    assert got["amlodipina besilato/valsartan"]["inn"] == ["amlodipine", "valsartan"]
    assert got["amlodipina besilato/valsartan"]["basis"] == "stem"


def test_a_label_is_taken_only_for_the_exact_single_ingredient_and_the_most_recent():
    rows = [
        {"set_id": "a", "version": "3", "effective_time": "20250101", "openfda": {"generic_name": ["ATORVASTATIN"]},
         "drug_interactions": ["old"]},
        {"set_id": "b", "version": "9", "effective_time": "20260301", "openfda": {"generic_name": ["ATORVASTATIN"]},
         "drug_interactions": ["new"], "adverse_reactions": ["myalgia"]},
        {"set_id": "c", "effective_time": "20261001",
         "openfda": {"generic_name": ["AMLODIPINE AND ATORVASTATIN"]}, "drug_interactions": ["combo"]},
    ]
    label = ds.pick_label("atorvastatin", rows)
    assert label.set_id == "b" and label.sections["drug_interactions"] == ["new"]
    assert label.citation["licence"].startswith("CC0")
    assert ds.pick_label("simvastatin", rows) is None
    assert ds.label_query("atorvastatin") == 'openfda.generic_name.exact:"ATORVASTATIN"'


def _cache(tmp_path):
    cache = ds.DrugCache(tmp_path)
    products, _ = ds.read_aifa(AIFA)
    cache.write_json("products.json", [ds.asdict(p) for p in products])
    cache.write_json("ingredients.json", {
        "atorvastatina calcio triidrato": {"atc": ["C10AA05"], "inn": ["atorvastatin"], "basis": "wikidata+ema"},
        "amlodipina besilato": {"atc": ["C08CA01"], "inn": ["amlodipine"], "basis": "wikidata"},
        "claritromicina": {"atc": ["J01FA09"], "inn": ["clarithromycin"], "basis": "wikidata+ema"},
    })
    cache.write_json("manifest.json", {"aifa": {"taken": "2026-11-15"}})
    cache.put_label("atorvastatin", ds.DrugLabel(
        query="atorvastatin", set_id="b", version="9", effective_time="20260301",
        sections={"drug_interactions": [("Strong inhibitors of CYP3A4 increase exposure. Clarithromycin: avoid "
                                         "atorvastatin doses above 20 mg. Grapefruit juice: limit intake.")],
                  "adverse_reactions": [("The most common were nasopharyngitis and arthralgia. Rhabdomyolysis "
                                         "with acute kidney injury has been reported.")]}))
    cache.put_label("clarithromycin", ds.DrugLabel(
        query="clarithromycin", set_id="k", version="2",
        sections={"contraindications": ["Concomitant use with lovastatin or simvastatin is contraindicated."]}))
    cache.put_label("amlodipine", None)
    return tmp_path


def test_the_branch_resolves_the_drugs_and_finds_what_the_labels_say_of_the_others_and_of_the_findings(tmp_path):
    knowledge = DrugKnowledge.load(_cache(tmp_path))
    graph = cg.CaseGraph("c")
    graph.documents.append(cg.Document("lab", "laboratory"))
    graph.findings.append(cg.Finding("f1", "lab", "rabdomiolisi", cg.Provenance("lab", "rabdomiolisi", 0, 12),
                                     concept=cg.Concept("HP", "HP:0003201", "Rhabdomyolysis"), date="2026-11-10"))
    graph.findings.append(cg.Finding("f2", "lab", "febbre", cg.Provenance("lab", "febbre", 13, 19),
                                     concept=cg.Concept("HP", "HP:0001945", "Fever")))
    add_medications(graph, [
        {"words": "Torvast 20 mg 1 cp la sera", "start": "2025-01-10"},
        {"words": "claritromicina 500 mg x2", "start": "2026-11-01"},
        {"words": "Norvasc 5 mg"},
        {"words": "Integratore di magnesio"},
    ])
    report = drug_branch(graph, knowledge)
    assert report["resolved"] == 3 and len(report["unresolved"]) == 1
    assert report["without_label"] == [graph.medications[2].medication_id]
    atorva, clari = graph.medications[0], graph.medications[1]
    assert {c.code for c in atorva.codes} >= {"ATC:C10AA05", "INN:atorvastatin"}
    [x] = graph.drug_interactions                    # one sentence names the other drug; the class one is not found
    assert (x.medication, x.other) == (atorva.medication_id, clari.medication_id)
    assert x.sentence.startswith("Clarithromycin: avoid") and x.citation["set_id"] == "b"
    [e] = graph.drug_effects
    assert e.finding == "f1" and e.timing == "after_start" and "Rhabdomyolysis" in e.sentence
    assert graph.snapshot.sources["openFDA label atorvastatin"] == "b v9"
    assert graph.validate() == []
    back = cg.CaseGraph.from_dict(json.loads(json.dumps(graph.as_dict())))
    assert back.as_dict() == graph.as_dict()


def test_a_graph_of_the_first_schema_is_still_read():
    data = cg.CaseGraph("old").as_dict()
    data["schema"] = "case-graph/1"
    for k in ("drug_profiles", "drug_interactions", "drug_effects"):
        data.pop(k)
    assert cg.CaseGraph.from_dict(data).schema == cg.SCHEMA_VERSION


def test_sentences_split_where_a_label_ends_them():
    assert sentences("A is reported. B may occur (rare). 5 cases.") == ["A is reported.", "B may occur (rare).",
                                                                       "5 cases."]


def test_the_drug_branch_is_written_to_falkordb(tmp_path):
    import pytest

    pytest.importorskip("redislite")
    from redislite.falkordb_client import FalkorDB

    from melampo.memory.case_graph_store import load_case, read_case

    knowledge = DrugKnowledge.load(_cache(tmp_path / "cache"))
    graph = cg.CaseGraph("drugs")
    graph.documents.append(cg.Document("lab", "laboratory"))
    graph.findings.append(cg.Finding("f1", "lab", "rabdomiolisi", cg.Provenance("lab", "rabdomiolisi", 0, 12),
                                     concept=cg.Concept("HP", "HP:0003201", "Rhabdomyolysis")))
    add_medications(graph, [{"words": "Torvast 20 mg"}, {"words": "claritromicina 500 mg"}])
    drug_branch(graph, knowledge)
    store = FalkorDB(str(tmp_path / "g.db")).select_graph("drugs_test")
    load_case(store, graph)
    back = read_case(store, "drugs")
    assert len(back.drug_profiles) == 2 and len(back.drug_interactions) == 1 and len(back.drug_effects) == 1


def test_the_ema_table_gives_atc_to_inn_for_human_medicines(tmp_path):
    import pytest

    # openpyxl is installed by the drug-sources workflow, which reads the EMA table; the test suite may not have it
    Workbook = pytest.importorskip("openpyxl").Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(["Medicines output: medicines report"])
    ws.append([])
    ws.append(["Category", "Name of medicine", "International non-proprietary name (INN) / common name", "ATC code"])
    ws.append(["Human", "Lipitor-like", "atorvastatin", "C10AA05"])
    ws.append(["Veterinary", "Vetmed", "doxycycline", "QJ01AA02"])
    ws.append(["Human", "Combo", "amlodipine / valsartan", "C09DB01"])
    wb.save(tmp_path / "ema.xlsx")
    got, report = ds.read_ema_medicines(tmp_path / "ema.xlsx")
    assert got["C10AA05"] == ["atorvastatin"] and "QJ01AA02" not in got
    assert report["columns"]["atc"] == "ATC code"


ANAGRAFICA = ("CODICE_AIC;COD_FARMACO;COD_CONFEZIONE;DENOMINAZIONE;DESCRIZIONE;RAGIONE_SOCIALE;"
              "STATO_AMMINISTRATIVO;CODICE_ATC;PA_ASSOCIATI\n"
              "035362018;035362;018;TORVAST;20 MG COMPRESSE;VIATRIS;A;C10AA05;S\n"
              "027353010;027353;010;NORVASC;5 MG COMPRESSE;VIATRIS;A;C08CA01;S\n")
PA = ("CODICE_AIC;PRINCIPIO_ATTIVO\n035362018;ATORVASTATINA CALCIO TRIIDRATO\n027353010;AMLODIPINA BESILATO\n")


def test_the_anagrafica_without_ingredient_names_is_joined_with_the_ingredients_file():
    products, report = ds.read_aifa(ANAGRAFICA)
    assert products == [] and "problem" in report
    assert report["samples"]["ingredient_not_names"] == ["S"]           # PA_ASSOCIATI holds a flag, not names
    by_package, pa_report = ds.read_aifa_ingredients(PA)
    assert by_package["035362018"] == "ATORVASTATINA CALCIO TRIIDRATO" and "problem" not in pa_report
    products, report = ds.read_aifa(ANAGRAFICA, by_package)
    assert [p.ingredient for p in products] == ["ATORVASTATINA CALCIO TRIIDRATO", "AMLODIPINA BESILATO"]
    assert report["joined_from_ingredients_file"] == 2 and products[0].atc == "C10AA05"
