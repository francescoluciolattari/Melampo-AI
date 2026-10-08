"""Rules found on corpora labelled by others (CRAFT, MedMentions, 8 October 2026)."""

import json
from pathlib import Path

import pytest

from melampo.memory import anatomy_linker as al
from melampo.memory import anatomy_parts as ap
from melampo.memory.anatomy_graph import AnatomyGraph
from melampo.memory.report_state import ReportState

DATA = Path(__file__).resolve().parents[1] / "data" / "linking"
UBERON = DATA / "uberon-basic.obo"


@pytest.fixture(scope="module")
def lexicon():
    return al.Lexicon.from_json(
        json.loads((DATA / "anatomy_lexicon.json").read_text("utf-8"))
    )


@pytest.fixture(scope="module")
def parts(lexicon):
    return ap.PartTable.from_json(
        json.loads((DATA / "anatomy_parts.json").read_text("utf-8")), lexicon
    )


def test_an_italian_only_name_in_an_english_sentence_is_not_the_structure(
    lexicon, parts
):
    linker = al.AnatomyLinker(lexicon, [], {}, parts=parts)
    result = linker.link("sigma", "The sigma factor binds the promoter in these cells.")
    assert (result.status, result.reason) == (al.ABSTAINED, "name_of_another_language")
    assert linker.link("sigma", "Diverticolosi del sigma e del colon.").cid == "colon"


@pytest.mark.skipif(not UBERON.exists(), reason="UBERON file missing")
@pytest.mark.parametrize(
    "mention,sentence,stops",
    [
        (
            "heart",
            "Cells of the secondary heart field migrate to the outflow tract.",
            True,
        ),
        ("brain", "No sterol crosses the blood-brain barrier.", True),
        ("left hip", "Left hip joint effusion.", True),
        ("brain", "MRI of the renal and brain arteries confirmed the diagnosis.", True),
        ("liver", "The liver parenchyma is homogeneous.", False),  # a part: kept
        ("aorta", "Thoracic aorta calcified.", False),
        ("left lower lobe", "A calcified granuloma in the left lower lobe.", False),
    ],
)
def test_a_word_inside_a_longer_anatomical_name_is_not_the_structure(
    lexicon, parts, mention, sentence, stops
):
    with open(UBERON, encoding="utf-8") as handle:
        terms = al.load_obo_terms(handle)
    with open(UBERON, encoding="utf-8") as handle:
        graph = AnatomyGraph.from_obo(handle)
    pool, equivalent = al.build_pool(lexicon, terms)
    linker = al.AnatomyLinker(lexicon, pool, equivalent, parts=parts, graph=graph)
    result = linker.link(mention, sentence)
    assert result.reason.startswith("mention_is_inside_a_longer_name") is stops


def _abbreviation(lexicon, text, document_type):
    linker = al.AnatomyLinker(lexicon, [], {}, document_type=document_type)
    state = ReportState.parse(text)
    at = text.index("SVC")
    return linker.link("SVC", state.sentence_at(at, at + 3), report=state, at=at)


def test_an_abbreviation_outside_a_radiology_text_needs_signs_of_imaging(lexicon):
    assert (
        _abbreviation(
            lexicon, "SVC grown in FBS showed more adipocytes.", "literature"
        ).reason
        == "abbreviation_outside_an_imaging_context"
    )
    assert (
        _abbreviation(
            lexicon,
            "CT showed thrombosis of the SVC and of the brachiocephalic vein.",
            "literature",
        ).status
        == al.ACCEPTED
    )


def test_in_a_radiology_report_an_abbreviation_is_radiology(lexicon):
    result = _abbreviation(lexicon, "Right catheter tip upper SVC.", "radiology_report")
    assert (result.status, result.cid) == (al.ACCEPTED, "superior_vena_cava")


@pytest.mark.parametrize(
    "mention,sentence,stops",
    [
        ("Heart", "The American Heart Association recommends daily activity.", True),
        ("Heart", "Data from the Dallas Heart Study were analysed.", True),
        (
            "Prostate",
            "Evaluation of the Prostate Imaging Reporting and Data System.",
            True,
        ),
        ("Liver", "Guidelines of the International Liver Transplant Society.", True),
        ("Liver", "Liver: normal size and echotexture.", False),
        ("Heart", "Heart size is normal.", False),
        ("Fegato", "Fegato di dimensioni regolari.", False),
    ],
)
def test_a_word_of_a_proper_name_is_not_the_structure(
    lexicon, mention, sentence, stops
):
    result = al.AnatomyLinker(lexicon, [], {}).link(mention, sentence)
    assert result.reason.startswith("mention_is_a_word_of_a_proper_name") is stops


@pytest.mark.parametrize(
    "mention,sentence",
    [
        ("liver", "LKB1 (liver kinase B1) was inhibited."),
        ("pancreas", "The artificial pancreas may be marketed within a year."),
        ("heart", "Virtual heart models were proposed for validation."),
        ("brain", "Canine brain phantoms were fabricated from skulls."),
        ("liver", "Serum liver tests were normal."),
    ],
)
def test_the_name_of_another_thing_made_from_an_organ_name_is_not_the_organ(
    lexicon, mention, sentence
):
    result = al.AnatomyLinker(lexicon, [], {}).link(mention, sentence)
    assert result.status == al.ABSTAINED
    assert result.reason.startswith("attribute_head_names_a_measurement")


def test_heart_size_and_prostate_volume_stay_findings_on_the_structure(lexicon):
    linker = al.AnatomyLinker(lexicon, [], {})
    assert linker.link("Heart", "Heart size is normal.").cid == "heart"
    assert linker.link("prostate", "The prostate volume is 45 mL.").cid == "prostate"


def test_the_language_alone_cannot_make_a_sense(lexicon):
    linker = al.AnatomyLinker(lexicon, [], {})
    result = linker.link(
        "axis", "Distances oriented along the medio-lateral axis of the hand."
    )
    assert result.status == al.ABSTAINED and result.stage == "senses"


@pytest.mark.parametrize(
    "sentence,accepted",
    [
        ("The Cancer Genome Atlas analysis indicates poor survival.", False),
        ("Images were registered with multi-atlas segmentation.", False),
        ("Fracture of the atlas and of the dens.", True),
        ("C1-C2 fusion with atlas laminar hooks.", True),
    ],
)
def test_atlas_is_the_first_vertebra_only_with_signs_of_the_spine(
    lexicon, sentence, accepted
):
    result = al.AnatomyLinker(lexicon, [], {}).link("atlas", sentence)
    assert (result.status == al.ACCEPTED) is accepted
    if accepted:
        assert result.cid == "vertebrae_C1"


def test_the_neighbours_are_read_at_this_occurrence_and_as_a_whole_word(lexicon):
    linker = al.AnatomyLinker(lexicon, [], {})
    # "thyroid" is not the thyroid of "Hypothyroidism": the head next to the real word counts
    result = linker.link(
        "thyroid", "Hypothyroidism and raised thyroid antibody levels."
    )
    assert result.reason == "attribute_head_names_a_measurement:antibody"
    # two occurrences: the one at the given position is read
    text = "Brain volume correlates with intrinsic brain activity."
    state = ReportState.parse(text)
    second = text.index("brain activity")
    first = linker.link("Brain", state.sentence_at(0, 5), report=state, at=0)
    other = linker.link(
        "brain", state.sentence_at(second, second + 5), report=state, at=second
    )
    assert first.cid == "brain"
    assert other.reason == "attribute_head_names_a_measurement:activity"


def test_a_measurement_shared_by_two_coordinated_organs(lexicon):
    linker = al.AnatomyLinker(lexicon, [], {})
    result = linker.link("Liver", "Liver and renal function tests were normal.")
    assert result.reason == "attribute_head_names_a_measurement:function"
    # two structures, not a shared measurement
    result = linker.link(
        "thyroid", "Resection of the thyroid and central lymph node compartment."
    )
    assert result.status == al.ACCEPTED
