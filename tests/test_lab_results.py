"""Tests for data/lab_results.py -- step 3 of the document-processing plan:
deterministic document typing (laboratory / protein electrophoresis /
antibiogram / narrative) and laboratory + antibiogram row extraction,
wired into case attachments and CaseContext.observations.

The report fixtures follow the layouts common to Italian laboratory
information systems (pdftotext -layout columns, markdown tables, single
spaces) -- written for this module, not taken from real reports. The first
real reports remain the test that matters (see the module docstring).
"""

import json

import pytest

from melampo.data.case_attachments import CaseAttachment, process_case_attachments
from melampo.data.document_processing import ClinicalDocumentProcessor
from melampo.data.ingestion import ClinicalIngestionPipeline
from melampo.data.lab_results import (
    DOCUMENT_TYPE_ANTIBIOGRAM,
    DOCUMENT_TYPE_LABORATORY,
    DOCUMENT_TYPE_NARRATIVE,
    DOCUMENT_TYPE_PROTEIN_ELECTROPHORESIS,
    FLAG_AGREES,
    FLAG_DISAGREES,
    FLAG_LAB_DID_NOT_FLAG,
    RANGE_ABOVE,
    RANGE_BELOW,
    RANGE_NO_REFERENCE,
    RANGE_NOT_ASSESSABLE,
    RANGE_QUALITATIVE_MATCH,
    RANGE_QUALITATIVE_MISMATCH,
    RANGE_REFERENCE_NOT_PARSED,
    RANGE_WITHIN,
    extract_lab_results,
)
from melampo.types import ClinicalObservation

FULL_REPORT = """LABORATORIO ANALISI CLINICHE
Paziente: [REDACTED]                                   Data prelievo: [REDACTED_DATE]
Richiesta n. 2026/004512                                Pagina 1 di 2

ESAME                          RISULTATO     UNITA'          VALORI DI RIFERIMENTO

EMOCROMO
Globuli bianchi (WBC)             11,80  H    x10^3/µL          4,00 - 10,00
Globuli rossi (RBC)                4,85       x10^6/µL          4,20 - 5,40
Emoglobina (HGB)                  13,8        g/dL              12,0 - 16,0
Piastrine (PLT)                  250.000      /mm3              150.000 - 450.000
Neutrofili                        78,5   *    %                 40,0 - 75,0

CHIMICA CLINICA
Creatinina                      *  1,45       mg/dL             0,50 - 1,20
eGFR (CKD-EPI)                     52         mL/min/1.73m2     > 60
Colesterolo totale                245         mg/dL             < 200
Proteina C reattiva               <0,5        mg/dL             < 0,5
Ferritina                          9          ng/mL             M: 30-400  F: 13-150
Potassio                           5,1        mmol/L            3,5 - 5,1

PROTIDOGRAMMA
Albumina                          58,4        %                 55,8 - 66,1
Alfa 1                             3,9        %                 2,9 - 4,9
Alfa 2                            12,9   H    %                 7,1 - 11,8
Beta                              10,1        %                 8,4 - 13,1
Gamma                             14,7        %                 11,1 - 18,8

ESAME URINE
Colore                            giallo paglierino
Glucosio                          assente                       assente
Nitriti                           positivo                      negativo

URINOCOLTURA
Germe isolato: Escherichia coli    Carica: >100.000 UFC/mL
ANTIBIOGRAMMA
Antibiotico                        MIC        Interpretazione
Amoxicillina                       >32        R
Ciprofloxacina                     <=0,25     S
Trimetoprim/Sulfametoxazolo        >4/76      R

Valori di glicemia seriati: 95 110 102 98
Referto validato elettronicamente. Firma digitale."""


@pytest.fixture(scope="module")
def full():
    return extract_lab_results(FULL_REPORT)


def _row(extraction, analyte):
    matches = [row for row in extraction.results if row.analyte == analyte]
    assert len(matches) == 1, f"{analyte!r}: {len(matches)} rows"
    return matches[0]


# ---------------------------------------------------------------------------
# Document typing
# ---------------------------------------------------------------------------


def test_a_mixed_report_is_typed_as_every_kind_it_contains(full):
    assert full.document_types == {DOCUMENT_TYPE_LABORATORY, DOCUMENT_TYPE_PROTEIN_ELECTROPHORESIS, DOCUMENT_TYPE_ANTIBIOGRAM}


def test_albumin_alone_does_not_make_a_protein_electrophoresis():
    extraction = extract_lab_results("Albumina   4,2   g/dL   3,5 - 5,2\nGlucosio   98   mg/dL   70 - 110")
    assert extraction.document_types == {DOCUMENT_TYPE_LABORATORY}


def test_a_radiology_report_is_narrative_with_no_rows_and_no_residue():
    narrative = """TC TORACE CON MEZZO DI CONTRASTO
Quesito clinico: dispnea.
Linfonodi assenti
Diametro aortico 3.2 cm
Hb 9.8 g/dL riferita dal curante
CONCLUSIONI
Quadro nei limiti della norma. Si consiglia controllo tra 12 mesi."""
    extraction = extract_lab_results(narrative)
    assert extraction.document_types == {DOCUMENT_TYPE_NARRATIVE}
    assert extraction.results == ()
    assert extraction.unparsed_lines == ()


def test_a_single_well_formed_row_in_a_letter_is_kept_without_making_it_a_lab_report():
    letter = """LETTERA DI DIMISSIONE
Il paziente è stato ricoverato per dispnea.
Emoglobina        9,8     g/dL      12,0 - 16,0
Linfonodi assenti"""
    extraction = extract_lab_results(letter)
    assert extraction.document_types == {DOCUMENT_TYPE_NARRATIVE}
    assert [(row.analyte, row.value, row.range_status) for row in extraction.results] == [("Emoglobina", 9.8, RANGE_BELOW)]


# ---------------------------------------------------------------------------
# Rows from the full report
# ---------------------------------------------------------------------------


def test_comma_decimals_units_ranges_and_sections(full):
    row = _row(full, "Globuli bianchi (WBC)")
    assert (row.value, row.unit, row.reference.low, row.reference.high) == (11.8, "x10^3/µL", 4.0, 10.0)
    assert row.section == "EMOCROMO"
    assert row.range_status == RANGE_ABOVE and row.printed_flag == "H" and row.flag_agreement == FLAG_AGREES


def test_a_flag_printed_before_the_value_is_read(full):
    row = _row(full, "Creatinina")
    assert (row.value, row.printed_flag, row.range_status, row.flag_agreement) == (1.45, "*", RANGE_ABOVE, FLAG_AGREES)


def test_out_of_range_without_a_printed_flag_is_surfaced(full):
    egfr = _row(full, "eGFR (CKD-EPI)")
    assert (egfr.range_status, egfr.flag_agreement) == (RANGE_BELOW, FLAG_LAB_DID_NOT_FLAG)
    cholesterol = _row(full, "Colesterolo totale")
    assert (cholesterol.range_status, cholesterol.flag_agreement) == (RANGE_ABOVE, FLAG_LAB_DID_NOT_FLAG)


def test_a_value_below_detection_is_assessed_by_what_it_can_stand_for(full):
    row = _row(full, "Proteina C reattiva")
    assert (row.comparator, row.value, row.value_text, row.range_status) == ("<", 0.5, "<0,5", RANGE_WITHIN)


def test_a_value_on_an_inclusive_bound_is_within(full):
    assert _row(full, "Potassio").range_status == RANGE_WITHIN


def test_thousands_are_read_as_thousands_when_the_unit_is_a_count(full):
    row = _row(full, "Piastrine (PLT)")
    assert (row.value, row.reference.low, row.reference.high, row.range_status) == (250000.0, 150000.0, 450000.0, RANGE_WITHIN)


def test_a_sex_specific_range_is_kept_as_text_and_never_assessed(full):
    row = _row(full, "Ferritina")
    assert (row.value, row.unit, row.reference_text, row.range_status) == (9.0, "ng/mL", "M: 30-400  F: 13-150", RANGE_REFERENCE_NOT_PARSED)


def test_qualitative_rows_compare_against_a_qualitative_reference(full):
    assert _row(full, "Glucosio").range_status == RANGE_QUALITATIVE_MATCH
    nitrites = _row(full, "Nitriti")
    assert (nitrites.value, nitrites.value_text, nitrites.range_status) == (None, "positivo", RANGE_QUALITATIVE_MISMATCH)


def test_antibiogram_rows_carry_the_organism_and_mic_including_ratio_mics(full):
    rows = {row.antibiotic: row for row in full.susceptibilities}
    assert set(rows) == {"Amoxicillina", "Ciprofloxacina", "Trimetoprim/Sulfametoxazolo"}
    assert (rows["Amoxicillina"].mic_text, rows["Amoxicillina"].interpretation) == (">32", "R")
    assert (rows["Ciprofloxacina"].mic_text, rows["Ciprofloxacina"].interpretation) == ("<=0,25", "S")
    assert rows["Trimetoprim/Sulfametoxazolo"].mic_text == ">4/76"
    assert {row.organism for row in rows.values()} == {"Escherichia coli"}
    assert full.organisms == ("Escherichia coli",)


def test_column_headers_are_neither_rows_nor_residue(full):
    texts = {line for _, line in full.unparsed_lines}
    assert not any("RISULTATO" in text or "Interpretazione" in text for text in texts)
    assert not any(row.analyte.upper().startswith(("ESAME", "ANTIBIOTICO")) for row in full.results)


def test_unread_data_lines_are_returned_as_residue_never_dropped(full):
    residue = {line.split()[0] for _, line in full.unparsed_lines}
    assert residue == {"Colore", "Valori"}


def test_redacted_and_page_lines_are_not_residue(full):
    assert not any("[REDACTED" in line or "Pagina" in line for _, line in full.unparsed_lines)


# ---------------------------------------------------------------------------
# Other layouts
# ---------------------------------------------------------------------------


def test_a_markdown_table_as_nemotron_parse_outputs_it():
    markdown = """| Esame | Risultato | Unità | Valori di riferimento |
|---|---|---|---|
| Emoglobina | 10,2 | g/dL | 12,0 - 16,0 |
| Transferrina | 390 | mg/dL | 200 - 360 |"""
    extraction = extract_lab_results(markdown)
    assert [(row.analyte, row.value, row.range_status) for row in extraction.results] == [
        ("Emoglobina", 10.2, RANGE_BELOW), ("Transferrina", 390.0, RANGE_ABOVE),
    ]
    assert extraction.unparsed_lines == ()


def test_single_spaced_rows_with_digits_inside_analyte_names():
    extraction = extract_lab_results(
        "Emoglobina 10.2 g/dL 12.0-16.0 L\nAlfa 1 3.9 % 2.9-4.9\nVitamina B12 150 pg/mL 200 - 900\nCA 19-9 12 U/mL < 37"
    )
    assert [(row.analyte, row.value, row.range_status) for row in extraction.results] == [
        ("Emoglobina", 10.2, RANGE_BELOW), ("Alfa 1", 3.9, RANGE_WITHIN),
        ("Vitamina B12", 150.0, RANGE_BELOW), ("CA 19-9", 12.0, RANGE_WITHIN),
    ]


def test_an_analyte_never_ends_with_a_result_word():
    extraction = extract_lab_results("Glucosio 98 mg/dL 70 - 110\nSodio 140 mmol/L 135 - 145\nNitriti positivo negativo")
    nitrites = _row(extraction, "Nitriti")
    assert (nitrites.value_text, nitrites.range_status) == ("positivo", RANGE_QUALITATIVE_MISMATCH)


def test_qualitative_rows_outside_a_laboratory_report_are_not_results():
    assert extract_lab_results("Linfonodi assenti\nVersamento assente").results == ()


def test_a_numeric_row_needs_a_unit_or_a_range():
    extraction = extract_lab_results("EMOCROMO\nEmoglobina   13,8   g/dL   12,0 - 16,0\nSodio   140   mmol/L   135 - 145\nGlicemia 95 110 102 98")
    assert [row.analyte for row in extraction.results] == ["Emoglobina", "Sodio"]
    assert extraction.unparsed_lines == ((4, "Glicemia 95 110 102 98"),)


# ---------------------------------------------------------------------------
# Numbers and assessment
# ---------------------------------------------------------------------------


def test_an_uncorroborated_thousands_shape_is_kept_but_never_assessed():
    row = extract_lab_results("Parametro X    1.450    mg/dL    0,50 - 2,00\nSodio  140  mmol/L  135 - 145").results[0]
    assert (row.value_text, row.value, row.range_status, row.notes) == ("1.450", None, RANGE_NOT_ASSESSABLE, ("number_format_ambiguous",))


def test_a_thousands_shaped_range_corroborates_a_thousands_value():
    """No unit column at all: only the range's own thousands shape corroborates the value's."""
    row = extract_lab_results("Globuli bianchi   7.200   4.000 - 10.000\nSodio  140  mmol/L  135 - 145").results[0]
    assert (row.value, row.range_status) == (7200.0, RANGE_WITHIN)


@pytest.mark.parametrize(
    ("value", "reference", "expected"),
    [
        ("<0,5", "< 0,5", RANGE_WITHIN),
        ("<0,5", "0,3 - 1,0", RANGE_NOT_ASSESSABLE),
        (">90", "> 60", RANGE_WITHIN),
        ("60", "> 60", RANGE_BELOW),
        ("60", ">= 60", RANGE_WITHIN),
        ("200", "< 200", RANGE_ABOVE),
        ("200", "<= 200", RANGE_WITHIN),
        ("12,0", "12,0 - 16,0", RANGE_WITHIN),
        ("16,1", "12,0 - 16,0", RANGE_ABOVE),
        (">20", "12,0 - 16,0", RANGE_ABOVE),
        ("<10", "12,0 - 16,0", RANGE_BELOW),
    ],
)
def test_assessment_against_the_printed_range(value, reference, expected):
    text = f"Analita   {value}   mg/dL   {reference}\nSodio   140   mmol/L   135 - 145"
    assert extract_lab_results(text).results[0].range_status == expected


def test_a_printed_flag_that_contradicts_the_range_is_surfaced():
    extraction = extract_lab_results("Sodio      131   H   mmol/L   135 - 145\nCalcio     10,9  L   mg/dL    8,5 - 10,5")
    assert [(row.range_status, row.flag_agreement) for row in extraction.results] == [
        (RANGE_BELOW, FLAG_DISAGREES), (RANGE_ABOVE, FLAG_DISAGREES),
    ]
    assert extraction.summary()["flag_disagreement_count"] == 2


def test_a_row_without_a_range_has_no_reference_status():
    row = extract_lab_results("HbA1c   48   mmol/mol\nSodio  140  mmol/L  135 - 145").results[0]
    assert row.range_status == RANGE_NO_REFERENCE


# ---------------------------------------------------------------------------
# Wiring: attachments, observations, CaseContext
# ---------------------------------------------------------------------------


def test_observations_are_traceable_to_attachment_and_line_never_to_a_filename():
    bundle = process_case_attachments(
        [CaseAttachment(filename="Rossi_Mario_esami.txt", data=FULL_REPORT.encode())], processor=ClinicalDocumentProcessor()
    )
    observations = bundle.observations()
    creatinine = next(item for item in observations if item.code == "Creatinina")
    assert creatinine.source == "attachment-1:line-15"
    assert (creatinine.value, creatinine.unit, creatinine.interpretation) == (1.45, "mg/dL", RANGE_ABOVE)
    assert creatinine.reference_range["low"] == 0.5 and creatinine.reference_range["high"] == 1.2
    assert creatinine.details["code_system"] == "unmapped_local_name"
    assert all("Rossi" not in (item.source or "") for item in observations)


def test_the_attachment_summary_carries_counts_only():
    bundle = process_case_attachments(
        [CaseAttachment(filename="Rossi_Mario_esami.txt", data=FULL_REPORT.encode())], processor=ClinicalDocumentProcessor()
    )
    laboratory = bundle.summary()[0]["laboratory"]
    assert laboratory["document_types"] == sorted({DOCUMENT_TYPE_LABORATORY, DOCUMENT_TYPE_PROTEIN_ELECTROPHORESIS, DOCUMENT_TYPE_ANTIBIOGRAM})
    assert laboratory["susceptibility_count"] == 3
    serialised = json.dumps(bundle.summary())
    assert "Creatinina" not in serialised and "Rossi" not in serialised


def test_case_context_receives_payload_observations_first_then_attachment_rows():
    pipeline = ClinicalIngestionPipeline(document_processor=ClinicalDocumentProcessor())
    case = pipeline.from_payload({
        "case_id": "c1",
        "observations": [{"code": "peso", "value": 70, "unit": "kg"}],
        "attachments": [{"filename": "esami.txt", "data": FULL_REPORT.encode()}],
    })
    assert case.observations[0] == ClinicalObservation(code="peso", value=70, unit="kg")
    bundle_count = len(extract_lab_results(FULL_REPORT).results) + len(extract_lab_results(FULL_REPORT).susceptibilities)
    assert len(case.observations) == 1 + bundle_count
    kinds = {item.details.get("kind") for item in case.observations[1:]}
    assert kinds == {"laboratory_result", "antimicrobial_susceptibility"}


def test_payload_observations_keep_working_with_the_new_optional_fields():
    observation = ClinicalObservation(code="x", value=1)
    assert (observation.reference_range, observation.interpretation, observation.details) == (None, None, {})
