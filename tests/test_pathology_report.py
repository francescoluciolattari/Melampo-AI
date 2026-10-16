"""The pathology report read deterministically (16 October 2026): sections, systems, margins, nodes, markers."""

from melampo.data.pathology_report import read_pathology

IT_BREAST = """ESAME ISTOLOGICO
Materiale: quadrantectomia mammella sinistra con linfonodo sentinella.
Notizie cliniche: nodulo QSE sinistro, BI-RADS 5.
Macroscopia: frammento fibroadiposo di 6 x 4 x 3 cm; neoformazione di dimensioni massime 2,1 cm.
Microscopia: carcinoma duttale infiltrante (NST), grado istologico 2 (3+2+2) sec. Nottingham.
Invasione linfovascolare: assente. Invasione perineurale non evidente.
Margini di resezione indenni, distanza minima dal margine 4 mm.
Linfonodo sentinella: 1/2 linfonodi metastatici (macrometastasi).
Immunoistochimica:
RE 95% (intensità forte)
RP 30%
HER2 score 2+ (equivoco), FISH: non amplificato
Ki-67 25%
Diagnosi: Carcinoma duttale infiltrante della mammella sinistra, G2, pT2 pN1a(sn), margini indenni (AJCC 8a ed.).
"""
EN_PROSTATE = """SURGICAL PATHOLOGY REPORT
Specimen: radical prostatectomy.
Microscopic description: acinar adenocarcinoma, Gleason score 4+3=7, Grade Group 3.
Perineural invasion: present. Lymphovascular invasion: not identified.
Margins: positive at the right apex.
Lymph nodes: 0 of 12 lymph nodes involved.
Final diagnosis: Prostatic acinar adenocarcinoma, Gleason 4+3=7 (ISUP 3), pT3a pN0, R1.
"""
IT_COLON = """ANATOMIA PATOLOGICA
Materiale: emicolectomia destra.
Microscopia: adenocarcinoma moderatamente differenziato (G2) infiltrante la sottosierosa.
Immunoistochimica:
MLH1 conservata, PMS2 conservata, MSH2 conservata, MSH6 persa
CDX2 positivo
CK20 positivo, CK7 negativo
BRAF V600E: wild type
Diagnosi: adenocarcinoma del colon destro, pT3 pN2a (5/18 linfonodi), margini liberi, R0.
"""
IT_LIVER = """ESAME ISTOLOGICO
Materiale: biopsia epatica.
Microscopia: infiltrato linfocitario portale con attività di interfaccia.
Diagnosi: epatite cronica attiva, grading 6, staging 2 (Ishak); METAVIR A2F2.
"""
EN_LUNG = """PATHOLOGY
Specimen: CT-guided core biopsy, right upper lobe.
Immunohistochemistry: TTF-1 positive, p40 negative. PD-L1 (22C3) TPS 60%.
Molecular: EGFR not mutated; ALK negative.
Diagnosis: Adenocarcinoma of the lung, solid predominant.
"""


def items(text, kind):
    return [i.value for i in read_pathology(text).items if i.kind == kind]


def test_sections_and_diagnosis():
    r = read_pathology(IT_BREAST)
    assert {"material", "clinical", "macroscopy", "microscopy", "immunohistochemistry", "diagnosis"} <= set(r.sections)
    assert r.diagnosis.startswith("Carcinoma duttale infiltrante della mammella sinistra")
    lo, hi = r.diagnosis_span
    assert IT_BREAST[lo:hi] == r.diagnosis


def test_breast_report():
    assert items(IT_BREAST, "tnm") == [{"T": "pT2", "N": "pN1a(sn)"}]
    assert items(IT_BREAST, "tnm_edition") == [{"edition": "8"}]
    assert items(IT_BREAST, "nottingham") == [{"value": "2(3+2+2)"}]
    assert {"status": "free", "as_printed": "indenni", "distance": "4", "unit": "mm"} in items(IT_BREAST, "margin")
    assert items(IT_BREAST, "lymph_nodes") == [{"positive": "1", "examined": "2"}]
    assert [v["status"] for v in items(IT_BREAST, "invasion")] == ["absent", "absent"]
    markers = {v["marker"]: v for v in items(IT_BREAST, "biomarker")}
    assert markers["ER"]["percent"] == "95" and markers["PgR"]["percent"] == "30" and markers["Ki-67"]["percent"] == "25"
    assert markers["HER2"]["score"] == "2+" and markers["HER2"]["status"] == "equivocal"
    assert markers["HER2"]["ish"] == "not_amplified"
    assert read_pathology(IT_BREAST).residue == []


def test_prostate_report_in_english():
    assert {"value": "4+3=7"} in items(EN_PROSTATE, "gleason")
    assert {"value": "3"} in items(EN_PROSTATE, "grade_group")
    assert items(EN_PROSTATE, "tnm") == [{"T": "pT3a", "N": "pN0"}]
    assert items(EN_PROSTATE, "residual_tumour") == [{"value": "R1"}]
    assert {v["type"]: v["status"] for v in items(EN_PROSTATE, "invasion")} == {"perineural": "present",
                                                                                "lymphovascular": "absent"}
    assert items(EN_PROSTATE, "lymph_nodes") == [{"positive": "0", "examined": "12"}]
    assert {"status": "involved", "as_printed": "positive"} in items(EN_PROSTATE, "margin")


def test_mismatch_repair_proteins_and_molecular_tests():
    markers = {v["marker"]: v.get("status") for v in items(IT_COLON, "biomarker")}
    assert markers["MLH1"] == markers["PMS2"] == markers["MSH2"] == "retained" and markers["MSH6"] == "lost_or_absent"
    assert markers["CK7"] == "negative" and markers["CK20"] == "positive"
    assert items(IT_COLON, "molecular") == [{"gene": "BRAF", "status": "wild_type", "variant_tested": "V600E"}]
    assert items(IT_COLON, "lymph_nodes") == [{"positive": "5", "examined": "18"}]
    lung = {v["marker"]: v for v in items(EN_LUNG, "biomarker")}
    assert lung["PD-L1"]["index"] == "TPS" and lung["PD-L1"]["percent"] == "60" and lung["p40"]["status"] == "negative"
    assert {"gene": "EGFR", "status": "wild_type"} in items(EN_LUNG, "molecular")


def test_chronic_hepatitis_systems():
    assert items(IT_LIVER, "ishak") == [{"grading": "6", "staging": "2"}]
    assert items(IT_LIVER, "metavir") == [{"value": "A2F2"}]


def test_what_is_not_a_tnm_category_or_a_marker():
    text = ("Esame RM: sequenze T1 e T2 pesate; crollo di T11.\nDiagnosi: nessuna alterazione. "
            "Il re ha visitato l'ospedale.")
    assert items(text, "tnm") == [] and items(text, "biomarker") == []
    assert items("Diagnosi: carcinoma, T2 N0 M0.", "tnm") == [{"T": "T2", "N": "N0", "M": "M0"}]


def test_an_unread_line_of_a_structured_section_is_residue():
    text = "Immunoistochimica:\nKi-67 30%\nMarcatore XYZ123 intensamente 3 su 4\nDiagnosi: carcinoma."
    assert read_pathology(text).residue == [(3, "Marcatore XYZ123 intensamente 3 su 4")]


def test_an_attachment_recognised_as_pathology_carries_its_reading():
    from melampo.data.case_attachments import CaseAttachment, process_case_attachments
    from melampo.data.document_processing import ClinicalDocumentProcessor

    item = CaseAttachment.from_payload_item({"data": IT_BREAST.encode(), "filename": "istologico.txt"})
    attachment = process_case_attachments([item], ClinicalDocumentProcessor()).attachments[0]
    summary = attachment.summary()
    assert "pathology" in summary["routing"]["interpreters"]
    assert {i["kind"] for i in attachment.pathology["items"]} >= {"tnm", "margin", "biomarker", "lymph_nodes"}
    assert set(summary["pathology"]["items_by_kind"]) >= {"tnm", "margin"} and "Carcinoma" not in str(summary)
