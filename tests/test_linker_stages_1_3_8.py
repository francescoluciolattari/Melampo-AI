"""Stage 1 (side of the exam, structures already named), stage 3 (roots), stage 8 (conflict policy)."""

import json
from pathlib import Path

import pytest

from melampo.memory import anatomy_linker as al
from melampo.memory.blind_reader import Reading
from melampo.memory.exam_area import ExamAreas
from melampo.memory.morphology import Morphology
from melampo.memory.report_state import ReportState

DATA = Path(__file__).resolve().parents[1] / "data" / "linking"


@pytest.fixture(scope="module")
def linker():
    """The exam side is taken (inherit_exam_side=True) to test the mechanism."""
    lexicon = al.Lexicon.from_json(
        json.loads((DATA / "anatomy_lexicon.json").read_text("utf-8"))
    )
    pool, equivalent = al.build_pool(lexicon, [])
    return al.AnatomyLinker(lexicon, pool, equivalent, inherit_exam_side=True)


def _link(linker, text, mention, **kwargs):
    state = ReportState.parse(text)
    at = text.index(mention)
    return linker.link(
        mention, state.sentence_at(at, at + len(mention)), report=state, at=at, **kwargs
    )


@pytest.mark.parametrize(
    "text,sides",
    [
        ("RM ginocchio destro: versamento.", {"right"}),
        ("MRI of the left knee", {"left"}),
        ("RM del ginocchio sn", {"left"}),
        ("Rx spalle bilaterale", {"both"}),
        ("TC torace", set()),
        (
            "CT chest. Left lung nodule",
            set(),
        ),  # the side of a finding is not the exam's
    ],
)
def test_the_side_is_read_from_the_exam_name_only(text, sides):
    assert ExamAreas.load().sides_of(text) == frozenset(sides)


def test_the_exam_name_can_be_blanked_out():
    rest = ExamAreas.load().without_exam_names(
        "RM ginocchio destro: frattura del femore."
    )
    assert "destro" not in rest and "femore" in rest


def test_a_paired_structure_without_a_side_takes_the_side_of_the_exam(linker):
    result = _link(
        linker, "RM ginocchio destro: frattura del femore distale.", "femore"
    )
    assert (result.status, result.cid) == (al.ACCEPTED, "femur_right")
    assert result.reason == "side_from_the_exam:right"
    assert "exam_side" not in result.support  # the side cannot confirm itself


@pytest.mark.parametrize(
    "text,mention",
    [
        (
            "RM ginocchio destro: femore sinistro non valutabile, femore normale.",
            "femore normale",
        ),
        ("Rx spalle bilaterale: frattura della clavicola.", "clavicola"),
        ("TC torace: rene nei limiti.", "rene"),
        (
            "RM ginocchio destro: rispetto al controlaterale il femore è normale.",
            "femore",
        ),
    ],
)
def test_no_side_is_taken_when_the_sentence_or_the_exam_does_not_allow_it(
    linker, text, mention
):
    result = _link(linker, text, mention)
    assert result.reason != "side_from_the_exam:right"
    assert not (result.cid or "").endswith("_right")


def test_the_other_side_in_an_exam_of_one_side_goes_to_review(linker):
    result = _link(
        linker, "RM ginocchio destro: il femore sinistro è normale.", "femore sinistro"
    )
    assert result.status == al.ABSTAINED
    assert result.reason == "side_contradicts_the_exam:right"


def test_the_other_side_named_as_a_comparison_is_a_conflict_not_a_stop(linker):
    result = _link(
        linker, "RM ginocchio destro: rispetto al femore sinistro.", "femore sinistro"
    )
    assert (result.status, result.cid) == (al.ACCEPTED, "femur_left")
    assert "side_differs_from_the_exam" in result.conflicts


def test_the_same_side_as_the_exam_supports(linker):
    result = _link(
        linker,
        "Rx spalla sinistra: clavicola sinistra fratturata.",
        "clavicola sinistra",
    )
    assert "exam_side" in result.support


def test_a_structure_named_in_another_sentence_supports(linker):
    text = "TC addome. Il fegato è aumentato di volume. Nel fegato una lesione di 2 cm."
    state = ReportState.parse(text)
    at = text.rindex("fegato")
    result = linker.link("fegato", state.sentence_at(at, at + 6), report=state, at=at)
    assert "discourse" in result.support


@pytest.mark.parametrize(
    "mention,expected",
    [
        ("epatomegalia", ("liver",)),
        ("hepatic", ("liver",)),
        ("nefrolitiasi", ("kidney",)),
        ("splenomegaly", ("spleen",)),
        ("cardinale", ()),
        ("gastrocnemio", ()),
        ("cardias", ()),
        ("mieloma", ()),
    ],
)
def test_roots_propose_the_structure(mention, expected):
    assert Morphology.load().proposes(mention) == expected


def test_roots_never_accept_but_are_proposals_on_an_abstention(linker):
    result = linker.link("hepatic", "Hepatic steatosis.")
    assert result.status == al.ABSTAINED
    assert result.proposals == ("liver",)


def test_a_root_of_the_linked_structure_is_one_more_mechanism(linker):
    result = linker.link("lobo epatico destro", "Lesione del lobo epatico destro.")
    if result.status == al.ACCEPTED:
        assert "morphology" in result.support


class _Against:
    def read(self, mention, linked):
        return Reading("against", "stub", "x", "x", 0.9, 0.5)


def test_the_conflict_monitor_sends_a_weakly_supported_contradicted_link_to_review(
    linker,
):
    from dataclasses import replace

    reviewing = replace(linker, blind=_Against(), conflict_policy="review")
    result = reviewing.link("fegato", "Il fegato è nei limiti.")
    assert result.status == al.ABSTAINED
    assert result.reason == "streams_disagree:blind_reader_disagrees"
    assert result.options == ["liver"]
    recording = replace(linker, blind=_Against())
    assert recording.link("fegato", "Il fegato è nei limiti.").status == al.ACCEPTED


def test_by_default_the_side_of_the_exam_is_only_a_proposal(linker):
    from dataclasses import replace

    result = _link(
        replace(linker, inherit_exam_side=False),
        "RM ginocchio destro: frattura del femore distale.",
        "femore",
    )
    assert result.status == al.ABSTAINED
    assert result.proposals == ("femur_right",)
