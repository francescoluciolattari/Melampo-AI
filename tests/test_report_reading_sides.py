"""What the reader before gave its findings, given to the cards of the one procedure (27 November 2026): the row
of the reference library a measure is compared with, the pathologist's diagnosis with its TNM, grade and margins,
and the image side when the report is paired with its exam. The interpreters are stand-ins on this machine."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from melampo.clinical import case_graph as cg
from melampo.clinical.case_graph_builder import add_cards, add_measures
from melampo.data.report_reading import ReportReaders, build_linker
from melampo.memory.reference_values import ReferenceValues
from melampo.reading.case_sides import (
    compare_with_references,
    image_of_cards,
    technique_of,
)


@pytest.fixture(scope="module")
def linker():
    return build_linker()[0]


def base(**kw):
    out = {"words": "", "name": "", "category": "manifestation", "specific": True, "what": "", "interpretation":
           "present", "value": None, "threshold": None, "site": "", "side": "", "observed_by": "imaging",
           "time": "", "nature": "pathological", "part": "findings", "state": "present", "temporality": "current", "subject": "patient",
           "quote": ""}
    return {**out, **kw}


def interpreter(cards, row="1"):
    """A stand-in interpreter: its cards, the other's findings the same, and the row it chooses for a measure."""
    def chat(prompt):
        if prompt.startswith("You read"):
            return json.dumps(cards)
        if prompt.startswith("Measures read"):
            chat.references.append(prompt)
            return f'[{{"item": 1, "row": {row}}}]'
        return "[]"
    chat.local_only = True
    chat.references = []
    return chat


ECO = "Ecografia dell'addome superiore. Milza di 15 cm di diametro longitudinale. Fegato nei limiti."
SPLEEN = base(words="Milza di 15 cm", name="enlarged spleen", what="spleen length", interpretation="increased",
              value={"number": 15, "unit": "cm"}, site="milza", quote="Milza di 15 cm di diametro longitudinale.")


def test_a_measure_both_interpreters_compare_with_one_row_becomes_a_measurement_of_the_library(linker):
    """ "Milza 15 cm" on ultrasound: the row of the library (upper limit 120 mm) the interpreters choose; the
    comparison is made here with the sentence's number, and the manifestation splenomegaly is present."""
    from melampo.clinical.manifestation_state import assess

    a, b = interpreter([SPLEEN]), interpreter([SPLEEN])
    readers = ReportReaders(linker, interpreters={"a": a, "b": b}, references=ReferenceValues.load())
    reading = readers.read("c1", ECO)
    [card] = reading["texts"][0]["cards"]
    assert card["site_code"] == "spleen" and "spleen-length-us" in a.references[0]
    assert card["reference"]["row"] == "spleen-length-us" and card["reference"]["value"] == 150.0
    assert card["reference"]["against"] == "above"
    [measure] = [s for s in reading["statements"] if s["statement"] == "measure"]
    graph = cg.CaseGraph(case_id="c1")
    add_cards(graph, "note", ECO, reading["texts"][0]["cards"])
    added, notes = add_measures(graph, reading["statements"], {"note": ECO})
    assert notes == [] and added[0].interpretation == "above" and added[0].unit == "mm" and added[0].value == 150.0
    assert measure["piece"] == "15 cm" and ECO[added[0].provenance.start:].startswith("15 cm")
    wanted = cg.Manifestation("m1", "splenomegaly", cg.Concept("HPO", "HP:0001744"))
    assert assess(wanted, graph).state == cg.PRESENT
    json.dumps(reading)


def test_interpreters_that_choose_different_rows_leave_the_measure_without_one_and_say_so(linker):
    readers = ReportReaders(linker, interpreters={"a": interpreter([SPLEEN], "1"), "b": interpreter([SPLEEN], "null")},
                            references=ReferenceValues.load())
    reading = readers.read("c1", ECO)
    [card] = reading["texts"][0]["cards"]
    assert card["reference"]["row"] is None and card["reference"]["doubt"] == "reference_chosen_differently"
    assert not [s for s in reading["statements"] if s["statement"] == "measure"]
    assert any(p["reason"] == "doubt:reference_chosen_differently" for p in reading["to_a_person"])


def test_a_number_the_sentence_does_not_write_is_never_compared():
    """The card says 16 cm, the sentence 15 cm: the measure is not the sentence's, and no row is asked."""
    card = {**SPLEEN, "value": {"number": 16, "unit": "cm"}, "site_code": "spleen", "doubts": []}
    asked = []
    out = compare_with_references([card], ECO, ReferenceValues.load(), lambda r, q: asked.append(q) or "[]", ["a"])
    assert out == ([], []) and asked == [] and "reference" not in card
    one = {**SPLEEN, "site_code": "spleen", "doubts": ["one_interpreter"]}        # read by one interpreter alone
    assert compare_with_references([one], ECO, ReferenceValues.load(), lambda r, q: "[]", ["a"]) == ([], [])


def test_the_technique_is_the_exams():
    assert technique_of(ECO) == "US" and technique_of("TC del torace con mdc.") == "CT"
    assert technique_of("RM encefalo.") == "MR" and technique_of("Visita di controllo.") == ""


PATHOLOGY = ("Materiale: biopsia del colon.\n"
             "Notizie cliniche: sospetta neoplasia del fegato, anemia.\n"
             "Diagnosi: adenocarcinoma moderatamente differenziato (G2) del colon, pT3 pN1a, margini indenni.\n")
PATHOLOGY_CARDS = [
    base(words="sospetta neoplasia del fegato", name="liver neoplasm", what="neoplasm", site="fegato",
         state="uncertain", observed_by="history", part="indication", quote="sospetta neoplasia del fegato, anemia."),
    base(words="anemia", name="anemia", what="anemia", observed_by="history", part="indication",
         quote="sospetta neoplasia del fegato, anemia."),
    base(words="adenocarcinoma moderatamente differenziato (G2) del colon", name="colon adenocarcinoma",
         what="adenocarcinoma", site="colon", observed_by="histopathology", part="impression",
         quote="adenocarcinoma moderatamente differenziato (G2) del colon, pT3 pN1a, margini indenni."),
]


def test_the_pathologists_diagnosis_takes_its_tnm_grade_and_margins_and_the_clinical_information_is_read_as_any_text(
        linker):
    attachment = SimpleNamespace(index=1, text=PATHOLOGY, pathology={},
                                 routing={"kinds": ["pathology_report"], "interpreters": ["pathology"]})
    bundle = SimpleNamespace(attachments=[attachment], imaging_studies=list)
    readers = ReportReaders(linker, interpreters={"a": interpreter(PATHOLOGY_CARDS), "b": interpreter(PATHOLOGY_CARDS)})
    reading = readers.read("c1", "", bundle)
    by = {c["name"]: c for c in reading["texts"][0]["cards"]}
    colon = by["colon adenocarcinoma"]
    assert colon["pathology_section"] == "diagnosis"
    items = {i["kind"]: i["text"] for i in colon["pathology"]["items"]}
    assert items["tnm"] == "pT3 pN1a" and items["grade"] == "G2" and items["margin"] == "margini indenni"
    # the same interpreter as every text: the suspicion of the clinical information is the question, a hypothesis;
    # the anemia it states is the patient's
    assert [s["name"] for s in reading["statements"] if s["statement"] == "indication"] == ["liver neoplasm"]
    assert sorted(f["name"] for f in reading["findings"]) == ["anemia", "colon adenocarcinoma"]
    assert by["anemia"]["pathology_section"] == "clinical" and "pathology" not in by["anemia"]
    graph = cg.CaseGraph(case_id="c1")
    found, _ = add_cards(graph, "attachment-1", PATHOLOGY, reading["texts"][0]["cards"], kind="pathology")
    colon = next(f for f in found if f.details["name"] == "colon adenocarcinoma")
    assert colon.site == "colon" and colon.details["pathology"]["items"]
    assert [d.words for d in graph.diseases if d.branch == cg.HYPOTHESIZED_DISEASES] == [
        "sospetta neoplasia del fegato"]
    json.dumps(reading)


REPORT = ("TC del torace con mdc. Nel lobo superiore destro nodulo solido di 12 mm a margini spiculati. "
          "Fegato nei limiti. Non versamento pleurico.")
NODULE = base(words="nodulo solido di 12 mm", name="solid pulmonary nodule", what="nodule",
              value={"number": 12, "unit": "mm"}, site="lobo superiore destro",
              quote="Nel lobo superiore destro nodulo solido di 12 mm a margini spiculati.")


class _Atlas:
    def all_structures(self, volume, affine, modality="CT"):
        return {"lung_upper_lobe_right": np.ones(volume.shape, bool)}


def test_a_card_of_the_report_paired_with_its_exam_is_looked_for_in_the_image():
    from melampo.imaging.segmenters import MaskSegmenter

    shape, spacing = (20, 128, 128), 0.7
    z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    lesion = ((y - 64) ** 2 + (x - 64) ** 2 <= (6.0 / spacing) ** 2) & (abs(z - 10) <= 1)
    affine = np.diag([spacing, spacing, 2.5, 1.0])
    volume = {"array": np.zeros(shape, np.float32) - 800, "affine": affine, "modality": "CT", "contrast": "unknown"}
    cards = [{**NODULE, "site_code": "lung_upper_lobe_right", "doubts": []},
             {**NODULE, "name": "pleural effusion", "state": "absent", "site_code": "pleura", "doubts": []}]
    looked = image_of_cards(cards, volume, MaskSegmenter({NODULE["quote"]: lesion}, "test"), _Atlas())
    assert looked == 1 and "image" not in cards[1]                      # an absent finding is not looked for
    image = cards[0]["image"]
    assert image["status"] == "agree" and image["provider"] == "given_mask" and image["size_check"] == "consistent"
    assert abs(image["measures"]["longest_axial_diameter_mm"] - 12.0) < 1.5
    json.dumps(image)


def test_without_an_exam_each_card_says_why_there_is_no_image(linker):
    readers = ReportReaders(linker, interpreters={"a": interpreter([NODULE]), "b": interpreter([NODULE])})
    reading = readers.read("c1", REPORT)
    [card] = reading["texts"][0]["cards"]
    assert card["image"] == {"status": "no_image", "notes": ["no_volume_in_the_attachments"]}
    assert reading["texts"][0]["image"] == {"looked_for": 0, "why_not": "no_volume_in_the_attachments"}


def test_the_exam_paired_with_its_report_gives_the_card_its_image_in_the_ingestion(linker):
    """The same path as the reader before: pydicom's CT sample paired with the one imaging report."""
    pytest.importorskip("pydicom")
    from test_report_reading import _series_payload

    from melampo.data.dicom_volume import assess_series
    from melampo.data.document_processing import ClinicalDocumentProcessor
    from melampo.data.ingestion import ClinicalIngestionPipeline
    from melampo.imaging.segmenters import MaskSegmenter

    probe = ClinicalIngestionPipeline(document_processor=ClinicalDocumentProcessor(), read_reports=False)
    study = probe.from_payload(_series_payload(REPORT)).imaging[0]
    assessment = assess_series(study.dicom_instances)
    shape = (assessment.slice_count, assessment.rows, assessment.columns)
    z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    lesion = ((y - 64) ** 2 + (x - 64) ** 2 <= (6.0 / assessment.pixel_spacing[0]) ** 2) & (abs(z - 10) <= 1)
    readers = ReportReaders(linker, MaskSegmenter({NODULE["quote"]: lesion}, "test"), _Atlas(),
                            interpreters={"a": interpreter([NODULE]), "b": interpreter([NODULE])})
    ingestion = ClinicalIngestionPipeline(document_processor=ClinicalDocumentProcessor(), report_readers=readers)
    reading = ingestion.from_payload(_series_payload(REPORT)).provenance["report_reading"]
    [finding] = reading["findings"]
    assert finding["image"]["status"] == "agree" and finding["image"]["size_check"] == "consistent"
    assert reading["texts"][0]["image"] == {"looked_for": 1, "why_not": ""}
    json.dumps(reading)


# -- the router, the part of the document (27 November 2026) ------------------------------------------------

CT = ("QUESITO: sospetta appendicite acuta, dolore in fossa iliaca destra.\n"
      "REPERTO: appendice ispessita (diametro 11 mm) con iperdensità del grasso periappendicolare.\n")
QUESTION = base(words="sospetta appendicite acuta", name="acute appendicitis", what="appendicitis",
                state="uncertain", observed_by="history", part="indication",
                quote="QUESITO: sospetta appendicite acuta, dolore in fossa iliaca destra.")
PAIN = base(words="dolore in fossa iliaca destra", name="right lower quadrant pain", what="abdominal pain",
            observed_by="history", part="indication", site="fossa iliaca destra",
            quote="QUESITO: sospetta appendicite acuta, dolore in fossa iliaca destra.")
APPENDIX = base(words="appendice ispessita (diametro 11 mm)", name="thickened appendix", what="appendix diameter",
                interpretation="increased", value={"number": 11, "unit": "mm"}, site="appendice", part="findings",
                quote="REPERTO: appendice ispessita (diametro 11 mm) con iperdensità del grasso periappendicolare.")


def test_what_the_requester_suspects_is_the_question_a_hypothesis_never_a_current_finding(linker):
    """claims4 (5 November 2026): the referral's sentences were attached as current findings. The interpreters read
    the part on the whole document; the suspicion of the question is a hypothesized disease, the symptom the
    requester states stays the patient's, not looked for in the image; the findings are the exam's."""
    readers = ReportReaders(linker, interpreters={"a": interpreter([QUESTION, PAIN, APPENDIX]),
                                                  "b": interpreter([QUESTION, PAIN, APPENDIX])})
    reading = readers.read("c1", CT)
    assert [s["name"] for s in reading["statements"] if s["statement"] == "indication"] == ["acute appendicitis"]
    assert {f["name"]: f["part"] for f in reading["findings"]} == {"right lower quadrant pain": "indication",
                                                                    "thickened appendix": "findings"}
    by = {c["name"]: c for c in reading["texts"][0]["cards"]}
    assert by["thickened appendix"]["part_heading"] == "findings" and by["acute appendicitis"]["part_heading"] == \
        "indication" and reading["texts"][0]["parts_differ_from_headings"] == 0
    graph = cg.CaseGraph(case_id="c1")
    found, _ = add_cards(graph, "note", CT, reading["texts"][0]["cards"])
    assert {f.details["name"] for f in found} == {"right lower quadrant pain", "thickened appendix"}
    [hypothesis] = [d for d in graph.diseases if d.branch == cg.HYPOTHESIZED_DISEASES]
    assert hypothesis.words == "sospetta appendicite acuta" and hypothesis.status_words == "indication"
    assert CT[hypothesis.provenance.start:].startswith("sospetta appendicite acuta")


def test_a_part_the_heading_contradicts_is_a_doubt_kept_in_the_path(linker):
    """Both interpreters read the referral's suspicion as a finding of the exam; the heading says QUESITO: the doubt
    stays with the card, which goes to a person and enters the path stated with uncertainty."""
    wrong = {**APPENDIX, "words": "sospetta appendicite acuta", "name": "acute appendicitis", "state": "present",
             "quote": QUESTION["quote"], "value": None, "site": ""}
    readers = ReportReaders(linker, interpreters={"a": interpreter([wrong]), "b": interpreter([wrong])})
    reading = readers.read("c1", CT)
    [card] = reading["texts"][0]["cards"]
    assert card["part"] == "findings" and card["part_heading"] == "indication"
    assert "part_differs_from_heading" in card["doubts"]
    assert any("part_differs_from_heading" in p["reason"] for p in reading["to_a_person"])
    [finding], _ = add_cards(cg.CaseGraph(case_id="c1"), "note", CT, [card])
    assert finding.certainty == "uncertain"


LAB = "Emocromo\nEmoglobina 10.2 g/dL (13.5-17.5)\nCreatinina 1.9 mg/dL (0.7-1.2)\n"


def test_a_document_the_router_gives_only_to_the_laboratory_extractor_is_not_read_again(linker):
    from melampo.data.lab_results import extract_lab_results

    calls = []
    def chat(prompt):
        calls.append(prompt)
        return "[]"
    chat.local_only = True
    lab = SimpleNamespace(index=1, text=LAB, routing={"kinds": ["laboratory"], "interpreters": ["lab_table"]},
                          lab=extract_lab_results(LAB))
    reading = ReportReaders(linker, interpreters={"a": chat}).read("c1", "", SimpleNamespace(
        attachments=[lab], imaging_studies=list))
    assert reading["texts"] == [{"report": "attachment-1", "kinds": ["laboratory"], "read": False,
                                 "reason": "read_by:lab_table"}] and calls == []


def test_a_row_of_a_letter_the_laboratory_extractor_read_enters_the_case_once(linker):
    from melampo.data.lab_results import extract_lab_results

    letter = "Lettera di dimissione. Paziente con anemia nota.\nEmoglobina 10.2 g/dL (13.5-17.5)\n"
    anemia = base(words="anemia", name="anemia", what="anemia", observed_by="history", part="history",
                  temporality="past", quote="Paziente con anemia nota.")
    hb = base(words="Emoglobina 10.2 g/dL", name="decreased hemoglobin", what="hemoglobin", category="test",
              interpretation="decreased", value={"number": 10.2, "unit": "g/dL"}, observed_by="laboratory",
              part="other", quote="Emoglobina 10.2 g/dL (13.5-17.5)")
    attachment = SimpleNamespace(index=1, text=letter, lab=extract_lab_results(letter), routing={
        "kinds": ["clinical_note"], "interpreters": ["lab_table", "narrative_reader"]})
    reading = ReportReaders(linker, interpreters={"a": interpreter([anemia, hb]), "b": interpreter([anemia, hb])}) \
        .read("c1", "", SimpleNamespace(attachments=[attachment], imaging_studies=list))
    by = {c["name"]: c for c in reading["texts"][0]["cards"]}
    assert by["decreased hemoglobin"]["read_by"] == "lab_table" and "read_by" not in by["anemia"]
    assert reading["texts"][0]["read_by_lab_table"] == 1


def test_a_document_of_no_recognised_kind_goes_to_a_person_and_is_not_read_into_the_case(linker):
    stray = SimpleNamespace(index=2, text="Ricevuta di pagamento n. 123, importo 40 euro.", routing={
        "kinds": ["clinical_note"], "interpreters": ["narrative_reader", "lab_table"], "model_kinds": ["other"],
        "notes": ["no_kind_recognised_read_as_narrative"]})
    reading = ReportReaders(linker, interpreters={"a": interpreter([])}).read("c1", "", SimpleNamespace(
        attachments=[stray], imaging_studies=list))
    assert reading["texts"][0]["reason"] == "kind_not_recognised"
    assert reading["to_a_person"][0]["reason"] == "document_set_aside:kind_not_recognised"
    [question] = reading["questions"]
    assert question["kind"] == "document:kind_not_recognised" and {o["code"] for o in question["options"]} >= {
        "this_patient", "another_patient", "not_medical", "not_pertinent"}
