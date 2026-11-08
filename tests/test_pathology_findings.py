"""The pathologist's diagnosis given to its findings (8 November 2026): the section of each finding, the items
(TNM, grade, margins, markers) of the finding they belong to, and nothing attached by guess."""

import json

import pytest

from melampo.data.pathology_findings import INDICATION, SPECIMEN, give_the_diagnosis
from melampo.data.pathology_report import read_pathology

REPORT = ("Materiale: biopsia del colon sigmoideo.\n"
          "Notizie cliniche: sospetta neoplasia del fegato, anemia.\n"
          "Microscopia: proliferazione ghiandolare atipica infiltrante la sottomucosa.\n"
          "Immunoistochimica: CK20 positivo, CDX2 positivo.\n"
          "Diagnosi: adenocarcinoma moderatamente differenziato (G2) del colon sigmoideo, pT3 pN1a, margini indenni.\n")


def _row(text, mention, structure, nth=0):
    at = -1
    for _ in range(nth + 1):
        at = text.index(mention, at + 1)
    lo, hi = text.rfind("\n", 0, at) + 1, text.find("\n", at)
    return {"mention": mention, "structure": structure, "start": at, "end": at + len(mention),
            "clause": text[lo:hi]}


def test_the_diagnosis_takes_its_items_and_the_clinical_information_is_the_question():
    rows = [_row(REPORT, "fegato", "liver"), _row(REPORT, "colon sigmoideo", "colon", 1)]
    found, stated, report = give_the_diagnosis(rows, [], REPORT, read_pathology(REPORT).summary())
    [colon] = found
    assert colon["pathology_section"] == "diagnosis"
    kinds = {i["kind"]: i["text"] for i in colon["pathology"]["items"]}
    assert kinds["tnm"] == "pT3 pN1a" and kinds["grade"] == "G2" and kinds["margin"] == "margini indenni"
    assert {i["text"] for i in colon["pathology"]["items"] if i["kind"] == "biomarker"} == {"CK20 positivo",
                                                                                           "CDX2 positivo"}
    assert colon["pathology"]["diagnosis"].startswith("adenocarcinoma moderatamente differenziato")
    # "suspected neoplasm of the liver" in the clinical information is the question, not a finding of the liver
    assert stated == [{"mention": "fegato", "structure": "liver", "clause": rows[0]["clause"],
                       "statement": INDICATION, "section": "clinical"}]
    assert report["unplaced"] == []
    json.dumps(found)


def test_items_of_another_section_are_given_only_when_the_diagnosis_names_one_structure():
    text = ("Immunoistochimica: TTF-1 positivo.\n"
            "Diagnosi: adenocarcinoma del polmone destro pT2a; metastasi del fegato.\n")
    rows = [_row(text, "polmone destro", "lung_right"), _row(text, "fegato", "liver")]
    found, _, report = give_the_diagnosis(rows, [], text, read_pathology(text).summary())
    by = {r["structure"]: r for r in found}
    assert [i["text"] for i in by["lung_right"]["pathology"]["items"]] == ["pT2a"]   # the one before it
    assert by["liver"]["pathology"]["items"] == []
    [ttf] = report["unplaced"]
    assert ttf["text"].startswith("TTF-1") and ttf["why"] == "the_diagnosis_names_more_than_one_structure"


def test_a_diagnosis_that_names_no_structure_is_of_the_material_when_it_names_one():
    text = "Materiale: biopsia prostatica.\nDiagnosi: adenocarcinoma acinare, Gleason 3+4=7.\n"
    material = [{"cid": "prostate", "mention": "prostatica", "roles": ["procedure_site"]}]
    found, stated, report = give_the_diagnosis([], [], text, read_pathology(text).summary(), material)
    assert found == [] and stated == []
    of = report["diagnosis_of_the_specimen"]
    assert of["structure"] == "prostate" and [i["kind"] for i in of["items"]] == ["gleason"]
    # two structures in the material: nothing is given by guess
    two = material + [{"cid": "urinary_bladder", "mention": "vescica", "roles": ["procedure_site"]}]
    _, _, report = give_the_diagnosis([], [], text, read_pathology(text).summary(), two)
    assert "diagnosis_of_the_specimen" not in report
    assert report["unplaced"][0]["why"] == "the_diagnosis_names_no_structure"


def test_a_site_of_the_material_is_where_the_sample_comes_from():
    text = "Materiale: frammenti di fegato.\nDiagnosi: steatosi macrovescicolare.\n"
    rows = [_row(text, "fegato", "liver")]
    found, stated, _ = give_the_diagnosis(rows, [], text, read_pathology(text).summary())
    assert found == [] and stated[0]["statement"] == SPECIMEN


@pytest.mark.parametrize("missing", [{}, {"sections": {}}])
def test_without_a_pathology_reading_nothing_changes_but_the_section(missing):
    rows = [_row(REPORT, "colon sigmoideo", "colon", 1)]
    found, stated, _ = give_the_diagnosis(rows, [], REPORT, missing)
    assert found[0]["pathology_section"] == "header" and "pathology" not in found[0] and stated == []
