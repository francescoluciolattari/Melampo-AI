"""The state of a report: sections, header, language, region; and what the linker does with it."""

import json
from pathlib import Path

import pytest

from melampo.evaluation import gold_set as gs
from melampo.memory import anatomy_linker as al
from melampo.memory.report_state import (
    CLINICAL,
    CONCLUSIONS,
    FINDINGS,
    TECHNIQUE,
    ReportState,
)

_DATA = Path(__file__).resolve().parents[1] / "data" / "linking"


@pytest.fixture(scope="module")
def linker():
    lexicon = al.Lexicon.from_json(
        json.loads((_DATA / "anatomy_lexicon.json").read_text("utf-8"))
    )
    return al.AnatomyLinker(lexicon, [], {})


def _sections(text):
    state = ReportState.parse(text)
    return [(s.kind, text[s.start : s.end].strip()) for s in state.sentences]


def test_a_labelled_header_is_not_part_of_the_finding_sentence():
    text = "Quesito clinico: sospetta colelitiasi. La colecisti è regolare. Conclusioni: Nulla di rilevante."
    assert _sections(text) == [
        (CLINICAL, "sospetta colelitiasi."),
        (FINDINGS, "La colecisti è regolare."),
        (CONCLUSIONS, "Nulla di rilevante."),
    ]
    state = ReportState.parse(text)
    at = text.index("colecisti")
    assert state.sentence_at(at, at + 9) == "La colecisti è regolare."
    assert state.section_at(at) == FINDINGS
    assert state.section_at(text.index("Quesito")) == "label"


def test_a_clinical_question_that_runs_into_the_findings_is_split():
    # real PARROT pattern: no full stop between the question and the first finding
    text = "Quesito clinico: controllo in paziente con sospetta colelitiasi Il fegato è nei limiti."
    kinds = _sections(text)
    assert kinds[0][0] == CLINICAL and "fegato" not in kinds[0][1]
    assert kinds[1] == (FINDINGS, "Il fegato è nei limiti.")


def test_what_follows_a_labelled_technique_sentence_is_the_report():
    text = "Dati tecnici: esame eseguito con mdc. Referto Fegato regolare. La milza è nei limiti."
    assert _sections(text) == [
        (TECHNIQUE, "esame eseguito con mdc."),
        (FINDINGS, "Fegato regolare."),
        (FINDINGS, "La milza è nei limiti."),
    ]


def test_an_unlabelled_technique_sentence_and_a_title_are_header_not_findings():
    text = "RM RACHIDE LOMBOSACRALE Indicazione e note anamnestiche: trauma. Sequenze: sagittali TSE T1, T2. Frattura di L1."
    kinds = [k for k, _ in _sections(text)]
    assert kinds[0] == TECHNIQUE  # the exam name
    assert CLINICAL in kinds and kinds.count(FINDINGS) == 1
    assert ReportState.parse(text).spine_scope


def test_the_anonymisation_mask_is_not_an_exam_title():
    text = "XXXX XXXX The heart is normal."
    assert _sections(text)[0][0] == FINDINGS


def test_a_plain_report_has_findings_only():
    text = "Normal cardiac contour. Right sided pleural effusion."
    state = ReportState.parse(text)
    assert [s.kind for s in state.sentences] == [FINDINGS, FINDINGS]
    assert not state.labelled and not state.spine_scope


def test_language_and_modality_come_from_the_header():
    text = "Dati tecnici: RM con sequenze T2. Referto Fegato nei limiti."
    state = ReportState.parse(text)
    assert state.language == "it"
    assert state.modalities == {"mri"}


def test_a_sentence_boundary_is_not_an_abbreviation():
    text = "Lobo sup. dx regolare. Milza nei limiti."
    assert [t for _, t in _sections(text)] == [
        "Lobo sup. dx regolare.",
        "Milza nei limiti.",
    ]


# --- the spine as the region the whole report is about ---------------------------------------


def test_spine_scope_from_the_sentences_when_there_is_no_header():
    text = "Accentuazione della lordosi lombare. Anterolistesi di L4 su L5, grado 2."
    assert ReportState.parse(text).spine_scope


def test_another_region_anywhere_in_the_report_cancels_the_spine_scope():
    text = "Accentuazione della lordosi lombare. Anterolistesi di L4 su L5. Fegato nei limiti."
    assert not ReportState.parse(text).spine_scope


def test_one_spine_word_is_not_a_spine_report():
    assert not ReportState.parse("Frattura recente di L4.").spine_scope


# --- what the linker does with the state -------------------------------------------------------

SPINE = "Accentuazione della lordosi lombare in clinostatismo. {} Segni di artrosi interapofisaria."


def _link(linker, sentence, mention="L4", template=SPINE):
    text = template.format(sentence)
    at = text.index(mention, text.index(sentence))
    state = ReportState.parse(text)
    return linker.link(
        mention, state.sentence_at(at, at + len(mention)), report=state, at=at
    )


def test_a_level_code_in_a_spine_report_is_a_vertebra(linker):
    result = _link(linker, "Anterolistesi di primo grado di L4 su L5.")
    assert (result.status, result.cid) == (al.ACCEPTED, "vertebrae_L4")
    assert result.reason == "level_code_from_the_report_state"


def test_the_same_sentence_alone_still_abstains(linker):
    result = linker.link("L4", "Anterolistesi di primo grado di L4 su L5.")
    assert result.status == al.ABSTAINED
    assert result.reason == "level_code_without_evidence"


@pytest.mark.parametrize(
    "sentence",
    [
        "Protrusione discale L4-L5.",  # a disc
        "Compressione della radice L5.",  # a root
        "Riduzione degli spazi intersomatici del tratto L3-L5.",  # the spaces between vertebrae
        "Iperintensità di segnale in T2 a livello di L4.",  # MRI signal
        "Tumore stadio T4 con infiltrazione.",  # a T stage
    ],
)
def test_the_state_never_turns_a_disc_root_space_signal_or_stage_into_a_vertebra(
    linker, sentence
):
    mention = "T4" if "stadio" in sentence else "L4"
    if mention not in sentence:
        mention = "L5" if "L5" in sentence else "L3"
    result = _link(linker, sentence, mention)
    assert result.reason != "level_code_from_the_report_state"


def test_t1_and_t2_stay_sequences_even_in_a_spine_report(linker):
    result = _link(linker, "Alterazione in T2 del midollo.", "T2")
    assert result.status == al.ABSTAINED


def test_a_level_code_in_the_header_is_not_supported_by_the_report_state(linker):
    text = "RM RACHIDE LOMBOSACRALE Indicazione: dolore a livello di L4. Accentuazione della lordosi lombare."
    state = ReportState.parse(text)
    at = text.index("L4")
    assert state.section_at(at) == CLINICAL
    assert not al.level_from_report("@L4", "dolore a livello di L4", state, at)


def test_without_a_report_nothing_changes(linker):
    assert not al.level_from_report("@L4", "Anterolistesi di L4 su L5.", None, None)


def test_the_header_becomes_the_context_of_an_ambiguous_form():
    toy = al.SenseInventory.from_json(
        {
            "forms": {
                "xyz": {
                    "organ": {
                        "anatomical": True,
                        "languages": ["en"],
                        "strong": ["biopsy"],
                        "weak": ["tissue"],
                    },
                    "unit": {"anatomical": False, "strong": ["kilo"]},
                }
            }
        }
    )
    lexicon = al.Lexicon.from_json({"classes": {}})
    linker = al.AnatomyLinker(lexicon, [], {}, senses=toy)
    text = "Clinical history: biopsy of tissue. The xyz is normal."
    state = ReportState.parse(text)
    at = text.index("xyz")
    sentence = state.sentence_at(at, at + 3)
    alone = linker.link("xyz", sentence)
    with_state = linker.link("xyz", sentence, report=state, at=at)

    def senses(result):
        return next(e for e in result.trace if e.stream == "senses")

    assert senses(alone).verdict == al.VETO  # silence is not evidence
    assert senses(with_state).verdict == al.SUPPORT  # the header is, at half weight


# --- the sampler uses it ----------------------------------------------------------------------


def test_sample_items_carry_a_clean_sentence_and_the_section():
    lexicon = al.Lexicon.from_json(
        json.loads((_DATA / "anatomy_lexicon.json").read_text("utf-8"))
    )
    reports = [
        {
            "report_id": "R",
            "text": "Quesito clinico: sospetta colelitiasi Il fegato è nei limiti.",
        }
    ]
    items = gs.sample_items(reports, lexicon)
    assert [i["sentence"] for i in items] == ["Il fegato è nei limiti."]
    assert items[0]["section"] == FINDINGS
