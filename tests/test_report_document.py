"""The report read as a document (6 November 2026): its sections, its conclusion, the sentence before, as a
radiologist reads them; each contradiction sends the sentence to a person and none sets a finding aside."""

import json

from melampo.evaluation.llm_reading import claims_prompt, read_claims
from melampo.memory.finding_owner import HERE, UNSURE, FindingOwner, report_context
from melampo.memory.report_sections import (
    FINDINGS,
    IMPRESSION,
    INDICATION,
    TECHNIQUE,
    impression_of,
    kind_of,
    section_at,
    sections,
)

REPORT = ("CT of the abdomen. Referral data: signs of heart failure decompensation. Technique: helical CT. "
          "Findings: The celiac trunk is patent. There is a narrowing at its origin from the aorta. "
          "Liver: normal size. Conclusion: stenosis of the celiac trunk.")


def _answer(**kw):
    base = {"marked": kw.pop("marked", ""), "same": "yes", "observed": [], "normal": [], "measure": [], "past": [],
            "certainty": [], "alternatives": [], "device": [], "not_assessable": [], "indication": [],
            "procedure": [], "conclusion": "none", "conclusion_of": ""}
    base.update(kw)
    return base


_ORGANS = """[Term]
id: UBERON:0002113
name: kidney
subset: organ_slim

[Term]
id: UBERON:0004538
name: left kidney
is_a: UBERON:0002113 ! kidney

[Term]
id: UBERON:0002106
name: spleen
subset: organ_slim
"""


def _owner(*answers, references=None):
    from melampo.memory import hpo_candidates as hc
    from melampo.memory.finding_owner import Anatomy

    def chat_of(a):
        return lambda prompt: json.dumps(a)

    # the reader asks whether the words recall another organ (7 November 2026): it needs UBERON's organs
    anatomy = Anatomy(hc.Anatomy.from_obo(_ORGANS.splitlines()))
    return FindingOwner(None, {f"m{i}": chat_of(a) for i, a in enumerate(answers)}, anatomy, question="claims",
                        references=references)


def test_the_kinds_of_section_follow_the_loinc_sections_of_a_report():
    kinds = [s.kind for s in sections(REPORT)]
    assert kinds == [INDICATION, TECHNIQUE, FINDINGS, IMPRESSION]       # "Liver:" is an organ, not a section
    assert section_at(REPORT, REPORT.index("signs of")).kind == INDICATION
    assert section_at(REPORT, REPORT.index("Liver")).kind == FINDINGS
    assert impression_of(REPORT) == "stenosis of the celiac trunk."
    assert kind_of("Clinical data from the referral") == INDICATION and kind_of("Quesito clinico") == INDICATION
    assert kind_of("Findings and assessment") == FINDINGS and kind_of("Bladder") == ""
    assert sections("Ao: 33 mm; LA: 38 mm") == []                          # a table of measures


def test_the_context_gives_the_section_the_conclusion_and_the_sentence_before():
    c = report_context(REPORT, REPORT.index("There is"))
    assert c["section_kind"] == FINDINGS and c["conclusion"] == "stenosis of the celiac trunk."
    assert c["previous"] == "Findings: The celiac trunk is patent."
    assert report_context(REPORT, REPORT.index("stenosis of the"))["conclusion"] == ""   # the conclusion itself


def test_the_question_shows_the_document_and_asks_the_new_kinds():
    p = " ".join(claims_prompt("signs of heart failure decompensation", 9, 14, "heart", section="Referral data",
                               section_kind=INDICATION, conclusion="cardiomegaly", previous="CT of the chest.").split())
    assert "a section of indication" in p and "Previous sentence (context only)" in p
    assert "Conclusion of the report (context only)" in p
    assert '"indication": []' in p and '"procedure": []' in p and '"refers_to": ""' in p
    assert "the size of a lesion in it belongs to that lesion" in p and "an implanted valve" in p


def test_a_finding_read_in_the_referral_goes_to_a_person_and_an_indication_is_kept_as_such():
    s = "signs of heart failure decompensation"
    ctx = {"section_kind": INDICATION}
    finding = _answer(marked="heart", observed=[{"words": "signs of heart failure decompensation", "of": "it"}])
    got = _owner(finding, finding)("heart", s, s.index("heart"), "en", "heart", ctx)
    assert got.verdict == UNSURE and "the_finding_is_in_the_indication" in got.reader
    asked = _answer(marked="heart", indication=["signs of heart failure decompensation"])
    got = _owner(asked, asked)("heart", s, s.index("heart"), "en", "heart", ctx)
    assert got.verdict == HERE and got.claims["indication"] == ["signs of heart failure decompensation"]


def test_its_and_the_conclusion_are_checked_against_the_anatomy():
    s = "Its upper pole is in contact with the spleen"
    previous = "The left kidney is displaced."
    said = _answer(marked="spleen", observed=[{"words": "Its upper pole is in contact", "of": "it", "relation": "",
                                               "refers_to": "The left kidney"}])
    read = read_claims(json.dumps(said), s, "spleen", previous=previous)
    assert read["observed"][0]["refers_to"] == "The left kidney"
    assert read_claims(json.dumps(said), s, "spleen")["observed"][0]["refers_to"] == ""   # not copied: not read
    got = _owner(said, said)("spleen", s, s.index("spleen"), "en", "spleen", {"previous": previous})
    assert got.verdict == UNSURE and "its_refers_to_another_structure" in got.reader
    # a part of the same organ, or words without a pronoun, are not another structure
    part = _answer(marked="spleen", observed=[{"words": "Its upper pole is in contact", "of": "it", "relation": "",
                                               "refers_to": "the spleen"}])
    assert "its_refers_to_another_structure" not in _owner(part, part)(
        "spleen", s, s.index("spleen"), "en", "spleen", {"previous": "The spleen is enlarged."}).reader
    s = "There is a cyst near the spleen"
    told = _answer(marked="spleen", observed=[{"words": "a cyst", "of": "it"}], conclusion="other",
                   conclusion_of="left kidney")
    got = _owner(told, told)("spleen", s, s.index("spleen"), "en", "spleen",
                             {"conclusion": "simple cyst of the left kidney."})
    assert got.verdict == UNSURE and "the_conclusion_gives_it_to_another_structure" in got.reader
    same = _answer(marked="spleen", observed=[{"words": "a cyst", "of": "it"}], conclusion="same")
    assert _owner(same, same)("spleen", s, s.index("spleen"), "en", "spleen",
                              {"conclusion": "splenic cyst."}).verdict == HERE


def test_the_size_of_a_lesion_is_not_compared_with_the_organ():
    s = "A tumor of the right kidney measuring 10.8 cm"
    said = _answer(marked="right kidney", observed=[{"words": "A tumor of the right kidney measuring 10.8 cm",
                                                     "of": "it"}],
                   measure=[{"words": "measuring 10.8 cm", "reference": "kidney-length-male", "against": "within"}])
    from melampo.memory.reference_values import ReferenceValues

    got = _owner(said, said, references=ReferenceValues.load())("right kidney", s, s.index("right kidney"), "en",
                                                                "kidney_right", {})
    assert got.verdict == UNSURE and "the_measure_is_of_what_is_observed" in got.reader


def test_a_section_before_the_findings_is_certain_only_as_far_as_its_heading_reaches():
    # 7 November 2026, PARROT: the referral given in one line with the findings ("Referral data: pain in the liver
    # region. The liver is not enlarged …") was read as reaching the end of the report
    one_line = "US of the abdomen. Referral data: pain in the liver region. The liver is not enlarged."
    assert section_at(one_line, one_line.index("pain")).kind == INDICATION
    assert section_at(one_line, one_line.index("liver is")).kind == ""          # the models read it
    lines = "Data from the referral: Consciousness disturbances. Please rule out infection.\nSigns of atrophy."
    assert section_at(lines, lines.index("Please")).kind == INDICATION         # to its line break
    assert section_at(lines, lines.index("Signs")).kind == ""
    exam = "Data on the referral: altered mental status. Chest X-ray AP, in a lying position: Asymmetry."
    assert section_at(exam, exam.index("Asymmetry")).kind == ""               # a heading of no section kind
    assert section_at(REPORT, REPORT.index("narrowing")).kind == FINDINGS
    assert section_at(REPORT, REPORT.index("stenosis")).kind == IMPRESSION
