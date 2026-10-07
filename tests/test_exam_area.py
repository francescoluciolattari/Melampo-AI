import pytest

from melampo.memory import anatomy_linker as al
from melampo.memory.exam_area import ADJACENT, EXPECTED, OUTSIDE, UNKNOWN, ExamAreas
from melampo.memory.report_state import ReportState


@pytest.fixture(scope="module")
def areas():
    return ExamAreas.load()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("RM pelvi: anca sinistra con regolare segnale.", {"pelvis"}),
        ("TC torace-addome con mdc", {"thorax", "abdomen"}),
        (
            "CT of the chest, abdomen and pelvis was performed.",
            {"thorax", "abdomen", "pelvis"},
        ),
        ("Brain MRI showed a lesion.", {"head"}),
        ("A contrast-enhanced abdominal CT revealed a mass.", {"abdomen"}),
        ("RX TORACE IN 2 PROIEZIONI", {"thorax"}),
        ("MRI of the cervical spine", {"spine"}),
        ("RM ginocchio sinistro", {"lower_limb"}),
        ("Ecografia tiroidea: nodulo.", {"neck"}),
        ("PET-CT showed uptake.", {"whole_body"}),
        # the area is where the exam is named, nowhere else
        ("MRI showed a lesion of the femur.", set()),
        ("Pregressa frattura del femore.", set()),
    ],
)
def test_the_area_is_read_from_the_name_of_the_exam(areas, text, expected):
    assert areas.of(text) == frozenset(expected)


@pytest.mark.parametrize(
    ("region", "exam", "relation"),
    [
        ("pelvis", {"pelvis"}, EXPECTED),
        ("abdomen", {"thorax"}, ADJACENT),  # a chest CT shows the upper abdomen
        ("lower_limb", {"pelvis"}, ADJACENT),  # a pelvic CT shows the femoral heads
        ("lower_limb", {"thorax"}, OUTSIDE),
        ("head", {"whole_body"}, EXPECTED),
        ("thorax", set(), UNKNOWN),
    ],
)
def test_how_a_structure_stands_to_the_area(areas, region, exam, relation):
    assert areas.relation(region, frozenset(exam)) == relation


def test_the_report_area_comes_from_title_and_technique_not_from_the_history():
    state = ReportState.parse(
        "RM PELVI\nQuesito clinico: dolore. Pregressa frattura del femore.\n"
        "Tecnica: sequenze T1 e T2.\nReferto: anca sinistra con regolare segnale."
    )
    assert state.areas == frozenset({"pelvis"})


# --- the linker: expectation, prediction error, conflict-triggered reading -------------------------


@pytest.fixture(scope="module")
def lexicon():
    import json
    from pathlib import Path

    data = Path(__file__).resolve().parents[1] / "data" / "linking"
    return al.Lexicon.from_json(
        json.loads((data / "anatomy_lexicon.json").read_text("utf-8"))
    )


def _linker(lexicon, chats=None, ambiguous=frozenset()):
    pool, equivalent = al.build_pool(lexicon, [])
    return al.AnatomyLinker(
        lexicon, pool, equivalent, chats=chats or {}, ambiguous=ambiguous
    )


def test_a_structure_of_the_area_is_supported_by_it(lexicon):
    result = _linker(lexicon).link("milza", "TC addome: milza di dimensioni regolari.")
    assert result.status == al.ACCEPTED
    assert "area" in result.support and "name" in result.support
    assert result.conflicts == ()


def test_an_unambiguous_name_far_from_the_area_is_kept_with_the_conflict(lexicon):
    """An incidental or historical mention outside the area is normal radiology: never a veto."""
    result = _linker(lexicon).link(
        "femore sinistro", "TC torace: esiti di frattura del femore sinistro."
    )
    assert (result.status, result.cid) == (al.ACCEPTED, "femur_left")
    assert "outside_the_exam_area" in result.conflicts
    assert result.convergence == len(result.support) - len(result.conflicts)


def test_an_ambiguous_form_far_from_the_area_with_no_model_abstains(lexicon):
    key = al.normalise("milza")
    result = _linker(lexicon, ambiguous=frozenset({key})).link(
        "milza", "RM encefalo: milza nei limiti."
    )
    assert result.status == al.ABSTAINED
    assert result.reason.startswith("ambiguous_form_outside_the_exam_area")


def test_an_ambiguous_form_far_from_the_area_is_read_by_the_models_even_without_verify(
    lexicon,
):
    asked = []

    def yes(prompt):
        asked.append(prompt)
        return "YES"

    key = al.normalise("milza")
    linker = _linker(lexicon, chats={"a": yes, "b": yes}, ambiguous=frozenset({key}))
    assert linker.verify is False
    result = linker.link("milza", "RM encefalo: milza nei limiti.")
    assert result.status == al.ACCEPTED and len(asked) == 2
    assert "models" in result.support
    assert result.conflicts == ("outside_the_exam_area",)


def test_a_measurement_keeps_the_structure_it_measures_without_linking_it(lexicon):
    result = _linker(lexicon).link("fegato", "Funzione del fegato nella norma.")
    assert result.status == al.ABSTAINED and result.cid is None
    assert (result.role, result.about) == ("inherent_location", "liver")


def test_a_procedure_on_the_structure_links_it_as_procedure_site(lexicon):
    result = _linker(lexicon).link("fegato", "Biopsia del fegato: epatite cronica.")
    assert (result.status, result.cid, result.role) == (
        al.ACCEPTED,
        "liver",
        "procedure_site",
    )


@pytest.mark.parametrize(
    ("reply", "verdict"),
    [
        ("1", "YES"),
        ("Answer: 1", "YES"),
        ("3", "NO"),
        ("4.", "NO"),
        ("5", "UNSURE"),
        ("", "UNSURE"),
        ("1 or 2", "UNSURE"),
    ],
)
def test_the_choice_reply_is_read_as_a_verdict(reply, verdict):
    assert al._choice_verdict(reply) == verdict


def test_the_choice_check_offers_balanced_options_and_keeps_only_option_one(lexicon):
    asked = []

    def measurement(prompt):
        asked.append(prompt)
        return "3"

    key = al.normalise("milza")
    linker = al.AnatomyLinker(
        lexicon,
        *al.build_pool(lexicon, []),
        chats={"a": measurement, "b": lambda p: "1"},
        ambiguous=frozenset({key}),
        verify=True,
        verify_style="choice",
    )
    result = linker.link("milza", "Milza nei limiti.")
    assert result.status == al.ABSTAINED and result.reason == "context_check_failed:no"
    assert "5. the sentence does not settle it" in asked[0]
    assert "YES" not in asked[0]
