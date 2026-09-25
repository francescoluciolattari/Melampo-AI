"""Tests for data/lab_phenotypes.py -- step 4 of the document-processing plan:
out-of-range laboratory rows -> HPO phenotypes -> findings the pipeline uses.

The first block checks the curated table against the bundled ontology files
on every run: each HPO id exists, is not obsolete, still carries the label
the rule states (HPO renames terms -- "Elevated serum creatinine" became
"Elevated circulating creatinine concentration"), and is annotated to at
least one disease in phenotype.hpoa. Graph phenotype nodes are exactly
those hpoa ids labelled through hp.obo, so the last check is what makes
each label a real entry point; all 94 labels were also resolved against the
loaded verification graph while building this (94/94).
"""

import collections
from pathlib import Path

import pytest

from melampo.data.document_processing import ClinicalDocumentProcessor
from melampo.data.ingestion import ClinicalIngestionPipeline
from melampo.data.lab_phenotypes import (
    RULES,
    SPECIMEN_BLOOD,
    SPECIMEN_URINE,
    WITHHELD_FLAG_DISAGREEMENT,
    WITHHELD_NO_RULE,
    WITHHELD_NOT_OUT_OF_RANGE,
    WITHHELD_PERCENT,
    WITHHELD_QUALITATIVE_BORDERLINE,
    WITHHELD_SPECIMEN_MISMATCH,
    WITHHELD_SPECIMEN_UNKNOWN,
    WITHHELD_SUSCEPTIBILITY,
    WITHHELD_UNIT_NOT_ACCEPTED,
    map_observations_to_phenotypes,
    specimen_of,
)
from melampo.data.lab_results import extract_lab_results
from melampo.memory.concept_resolution import parse_obo

DATA = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# The table against the ontology files
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hpo_terms():
    with (DATA / "hp.obo").open(encoding="utf-8") as handle:
        return {term.term_id: term for term in parse_obo(handle)}


@pytest.fixture(scope="module")
def annotated_term_ids():
    ids = set()
    with (DATA / "phenotype.hpoa").open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(("#", "database_id")):
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) > 3 and columns[2] != "NOT":
                ids.add(columns[3])
    return ids


def _all_terms():
    return [(rule.key, direction, term) for rule in RULES for direction, term in (("high", rule.high), ("low", rule.low)) if term]


def test_every_term_exists_is_current_and_keeps_its_label(hpo_terms):
    problems = []
    for key, direction, term in _all_terms():
        found = hpo_terms.get(term.term_id)
        if found is None or found.obsolete or found.name != term.label:
            problems.append((key, direction, term.term_id, term.label, getattr(found, "name", None)))
    assert problems == []


def test_every_term_is_annotated_to_at_least_one_disease(annotated_term_ids):
    assert [(key, term.term_id) for key, _, term in _all_terms() if term.term_id not in annotated_term_ids] == []


def test_no_printed_name_belongs_to_two_rules_for_the_same_specimen():
    owners = collections.defaultdict(list)
    for rule in RULES:
        for name in rule.names:
            owners[(name, rule.specimen)].append(rule.key)
    assert {key: keys for key, keys in owners.items() if len(keys) > 1} == {}


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------


def _map(text: str):
    return map_observations_to_phenotypes(extract_lab_results(text).observations("attachment-1"))


def _labels(mapping):
    return {item.label for item in mapping.phenotypes}


def _reasons(mapping):
    return {item.analyte: item.reason for item in mapping.withheld}


REPORT = """EMOCROMO
Globuli bianchi (WBC)             11,80  H    x10^3/µL          4,00 - 10,00
Globuli rossi (RBC)                6,10       x10^6/µL          4,20 - 5,40
Emoglobina (HGB)                   9,8        g/dL              12,0 - 16,0
Neutrofili                        78,5   *    %                 40,0 - 75,0
Linfociti                          0,6        x10^3/µL          1,0 - 4,0

CHIMICA CLINICA
Creatinina                      *  1,45       mg/dL             0,50 - 1,20
eGFR (CKD-EPI)                     52         mL/min/1.73m2     > 60
Sodio                             131   H     mmol/L            135 - 145
Potassio                           4,2        mmol/L            3,5 - 5,1

COAGULAZIONE
Tempo di Quick                     45         %                 70 - 120
INR                                1,65                         0,80 - 1,20

PROTIDOGRAMMA
Albumina                          48,0        %                 55,8 - 66,1
Gamma                              2,1        g/dL              0,7 - 1,6

ESAME URINE
Leucociti                          80         /µL               0 - 25
Glucosio                          positivo                      assente
Proteine                          tracce                        assenti
Nitriti                           positivo                      negativo

ANTIBIOGRAMMA
Germe isolato: Escherichia coli
Amoxicillina                       >32        R"""


@pytest.fixture(scope="module")
def mapped():
    return _map(REPORT)


def test_out_of_range_rows_become_their_phenotypes(mapped):
    assert _labels(mapped) == {
        "Increased total leukocyte count",
        "Anemia",
        "Decreased total lymphocyte count",
        "Elevated circulating creatinine concentration",
        "Decreased glomerular filtration rate",
        "Prolonged prothrombin time",
        "Increased circulating immunoglobulin concentration",
        "Pyuria",
        "Glycosuria",
    }


def test_each_phenotype_is_traceable_to_its_line(mapped):
    creatinine = next(item for item in mapped.phenotypes if item.label == "Elevated circulating creatinine concentration")
    assert (creatinine.term_id, creatinine.direction, creatinine.sources, creatinine.analytes) == (
        "HP:0003259", "high", ("attachment-1:line-9",), ("Creatinina",),
    )


def test_urine_leukocytes_are_pyuria_never_leukocytosis(mapped):
    pyuria = next(item for item in mapped.phenotypes if item.label == "Pyuria")
    leukocytosis = next(item for item in mapped.phenotypes if item.label == "Increased total leukocyte count")
    assert pyuria.analytes == ("Leucociti",) and leukocytosis.analytes == ("Globuli bianchi (WBC)",)
    assert "Hyperglycemia" not in _labels(mapped)


def test_every_row_not_used_says_why(mapped):
    reasons = _reasons(mapped)
    assert reasons["Neutrofili"] == WITHHELD_PERCENT
    assert reasons["Albumina"] == WITHHELD_PERCENT
    assert reasons["Tempo di Quick"] == WITHHELD_UNIT_NOT_ACCEPTED
    assert reasons["Sodio"] == WITHHELD_FLAG_DISAGREEMENT
    assert reasons["Proteine"] == WITHHELD_QUALITATIVE_BORDERLINE
    assert reasons["Nitriti"] == WITHHELD_NO_RULE
    assert reasons["Potassio"] == WITHHELD_NOT_OUT_OF_RANGE
    assert reasons["Amoxicillina"] == WITHHELD_SUSCEPTIBILITY
    # A high red cell count alone is not HPO's Polycythemia (it requires red
    # cells, hemoglobin and red cell volume all above range): no blood rule.
    assert reasons["Globuli rossi (RBC)"] == WITHHELD_SPECIMEN_MISMATCH


def test_a_row_under_no_recognised_header_is_not_assumed_to_be_blood():
    mapping = _map("Leucociti   80   /µL   0 - 25\nSodio   140   mmol/L   135 - 145")
    assert mapping.phenotypes == ()
    assert _reasons(mapping)["Leucociti"] == WITHHELD_SPECIMEN_UNKNOWN


def test_the_same_phenotype_from_two_rows_is_one_finding_with_both_sources():
    mapping = _map("EMOCROMO\nEmoglobina   9,8   g/dL   12,0 - 16,0\nHb   9,6   g/dL   12,0 - 16,0")
    [anemia] = mapping.phenotypes
    assert (anemia.label, anemia.sources) == ("Anemia", ("attachment-1:line-2", "attachment-1:line-3"))


@pytest.mark.parametrize(
    ("section", "analyte", "expected"),
    [
        ("EMOCROMO", "Leucociti", SPECIMEN_BLOOD),
        ("CHIMICA CLINICA", "Glucosio", SPECIMEN_BLOOD),
        ("ESAME URINE COMPLETO", "Glucosio", SPECIMEN_URINE),
        ("URINOCOLTURA", "Leucociti", SPECIMEN_URINE),
        (None, "Glucosio urinario", SPECIMEN_URINE),
        ("CHIMICA CLINICA", "Proteinuria", SPECIMEN_URINE),
        ("LABORATORIO ANALISI CLINICHE", "Glucosio", None),
        (None, "Glucosio", None),
    ],
)
def test_specimen_comes_from_the_header_or_the_analyte_never_assumed(section, analyte, expected):
    assert specimen_of(section, analyte) == expected


# ---------------------------------------------------------------------------
# Into the pipeline's findings
# ---------------------------------------------------------------------------


def _prepare(payload, **options):
    pipeline = ClinicalIngestionPipeline(document_processor=ClinicalDocumentProcessor(), **options)
    return pipeline.prepare_payload(payload)


def test_phenotypes_are_appended_after_the_physicians_findings_without_duplicates():
    prepared = _prepare({"case_id": "c1", "findings": ["Fever", "anemia"], "attachments": [{"filename": "esami.txt", "data": REPORT.encode()}]})
    assert prepared["findings"][:2] == ["Fever", "anemia"]
    assert "Anemia" not in prepared["findings"]  # already typed, compared case-insensitively
    assert "Elevated circulating creatinine concentration" in prepared["findings"]
    assert prepared["lab_phenotypes"]["added_findings"] == prepared["findings"][2:]


def test_findings_are_created_when_the_physician_typed_none():
    prepared = _prepare({"case_id": "c1", "attachments": [{"filename": "esami.txt", "data": REPORT.encode()}]})
    assert "Anemia" in prepared["findings"]


def test_provenance_travels_with_the_case():
    pipeline = ClinicalIngestionPipeline(document_processor=ClinicalDocumentProcessor())
    case = pipeline.from_payload({"case_id": "c1", "attachments": [{"filename": "Rossi_Mario.txt", "data": REPORT.encode()}]})
    provenance = case.provenance["lab_phenotypes"]
    assert provenance["withheld_by_reason"][WITHHELD_PERCENT] == 2
    assert all("Rossi" not in str(item) for item in provenance["phenotypes"] + provenance["withheld"])


def test_it_can_be_switched_off():
    prepared = _prepare({"case_id": "c1", "attachments": [{"filename": "esami.txt", "data": REPORT.encode()}]}, derive_lab_findings=False)
    assert "findings" not in prepared and "lab_phenotypes" not in prepared


def test_caller_findings_of_an_unexpected_shape_are_left_untouched():
    prepared = _prepare({"case_id": "c1", "findings": "fever", "attachments": [{"filename": "esami.txt", "data": REPORT.encode()}]})
    assert prepared["findings"] == "fever"
    assert prepared["lab_phenotypes"]["not_merged_reason"] == "caller_findings_not_a_list"


def test_no_attachments_means_no_change_at_all():
    assert _prepare({"case_id": "c1", "findings": ["Fever"]}) == {"case_id": "c1", "findings": ["Fever"]}
