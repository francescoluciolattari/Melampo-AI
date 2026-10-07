import json
from pathlib import Path

import pytest

from melampo.memory import anatomy_linker as al
from melampo.memory import exam_frame as ef
from melampo.memory.report_state import ReportState

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def frames():
    return ef.ExamFrames.load()


@pytest.fixture(scope="module")
def lexicon():
    return al.Lexicon.from_json(
        json.loads((ROOT / "data/linking/anatomy_lexicon.json").read_text("utf-8"))
    )


@pytest.mark.parametrize(
    ("sentence", "frame"),
    [
        (
            "Thyroid, parathyroid, and vitamin D assay were within normal limits.",
            ef.LABORATORY,
        ),
        (
            "Hemoglobin 11.4 g/dl, creatinine normal, serum bilirubin raised.",
            ef.LABORATORY,
        ),
        (
            "Esami ematochimici: emoglobina 11 g/dL, creatinina nei limiti.",
            ef.LABORATORY,
        ),
        (
            "Her heart rate was 144 beats per minute and blood pressure 90/60 mmHg.",
            ef.VITAL_SIGNS,
        ),
        ("CT scan of the neck showed an enlarged thyroid.", ef.IMAGING),
        (
            "Ultrasound of the abdomen: liver normal. Serum bilirubin 1.2 mg/dL.",
            ef.IMAGING,
        ),
        ("The thyroid gland was normally located in the anterior neck.", ef.UNKNOWN),
        ("Fegato di dimensioni regolari.", ef.UNKNOWN),
        # one word of a frame is not a frame
        (
            "Neurogenic bladder was diagnosed on laboratory and clinical information.",
            ef.UNKNOWN,
        ),
        ("There was profound bradycardia.", ef.UNKNOWN),
        ("She had shortness of breath and bladder irritation symptoms.", ef.UNKNOWN),
    ],
)
def test_the_frame_of_a_sentence(frames, sentence, frame):
    assert frames.of(sentence).frame == frame


def test_without_data_every_sentence_is_unknown():
    assert (
        ef.ExamFrames.empty().of("Hemoglobin 11 g/dl, serum creatinine").frame
        == ef.UNKNOWN
    )


def test_the_frame_cues_name_no_structure(frames, lexicon):
    names = {
        w
        for entry in lexicon.classes.values()
        for name in entry["it"] + entry["en"]
        for w in name.lower().split()
    }
    cues = set().union(*frames.strong.values(), *frames.weak.values())
    # a cue that is also a word of an anatomical name would make the frame depend on a structure
    assert not (cues & names), sorted(cues & names)


def test_a_structure_named_in_a_laboratory_sentence_is_not_linked(lexicon):
    linker = al.AnatomyLinker(lexicon, [], {})
    result = linker.link(
        "thyroid",
        "Thyroid, parathyroid, and vitamin D assay were within normal limits.",
    )
    assert result.status == al.ABSTAINED
    assert result.reason == "frame_is_a_measurement:laboratory"
    assert any(e.stream == "frame" for e in result.trace)
    assert (
        linker.link("thyroid", "The thyroid gland was normally located.").cid
        == "thyroid_gland"
    )


def test_without_the_frames_the_same_sentence_was_linked(lexicon):
    linker = al.AnatomyLinker(lexicon, [], {}, frames=ef.ExamFrames.empty())
    sentence = "Thyroid, parathyroid, and vitamin D assay were within normal limits."
    assert linker.link("thyroid", sentence).status == al.ACCEPTED


def test_the_findings_of_an_imaging_report_are_about_the_images(lexicon):
    text = (
        "RM RACHIDE LOMBOSACRALE\nTecnica: sequenze T1 e T2.\n"
        "Referto: Fegato regolare, creatinina 1.1 mg/dL, emoglobina 12 g/dL, siero nei limiti."
    )
    state = ReportState.parse(text)
    at = text.index("Fegato")
    sentence = state.sentence_at(at, at + 6)
    linker = al.AnatomyLinker(lexicon, [], {})
    assert linker.link("Fegato", sentence).reason == "frame_is_a_measurement:laboratory"
    assert linker.link("Fegato", sentence, report=state, at=at).status == al.ACCEPTED
    clinical = "RM RACHIDE LOMBOSACRALE\nQuesito clinico: emoglobina 12 g/dL, siero e fegato in studio.\nReferto: Nulla."
    state = ReportState.parse(clinical)
    at = clinical.index("fegato")
    sentence = state.sentence_at(at, at + 6)
    assert linker.link("fegato", sentence, report=state, at=at).status == al.ABSTAINED
