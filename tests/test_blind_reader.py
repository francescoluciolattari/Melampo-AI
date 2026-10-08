"""The blind reader: a second reading of the mention with no model, no network and no sight of
the first answer. It supports, disagrees or stays silent."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from melampo.memory import anatomy_linker as al
from melampo.memory.anatomy_graph import AnatomyGraph
from melampo.memory.blind_reader import AGAINST, SILENT, SUPPORT, BlindReader, Reading

DATA = Path(__file__).resolve().parents[1] / "data" / "linking"
UBERON = DATA / "uberon-basic.obo"

pytestmark = pytest.mark.skipif(not UBERON.exists(), reason="UBERON file missing")


@pytest.fixture(scope="module")
def world():
    lexicon = al.Lexicon.from_json(
        json.loads((DATA / "anatomy_lexicon.json").read_text("utf-8"))
    )
    with open(UBERON, encoding="utf-8") as handle:
        terms = al.load_obo_terms(handle)
    with open(UBERON, encoding="utf-8") as handle:
        graph = AnatomyGraph.from_obo(handle)
    pool, equivalent = al.build_pool(lexicon, terms)
    parts = json.loads((DATA / "anatomy_parts.json").read_text("utf-8"))
    reader = BlindReader.from_sources(lexicon, terms, equivalent, graph, parts)
    return SimpleNamespace(
        lexicon=lexicon, pool=pool, equivalent=equivalent, graph=graph, reader=reader
    )


@pytest.mark.parametrize(
    "mention,linked",
    [
        ("liver", "liver"),
        ("fegato", "liver"),
        ("left ventricle", "heart"),  # a part: its ancestors reach the linked class
        (
            "lingula",
            "lung_upper_lobe_left",
        ),  # the curated part table, not the cerebellum
    ],
)
def test_it_supports_the_same_structure_or_a_part(world, mention, linked):
    assert world.reader.read(mention, linked).verdict == SUPPORT


@pytest.mark.parametrize(
    "mention,linked",
    [("kidney", "liver"), ("pulmonary vein", "lung_right")],
)
def test_it_disagrees_when_it_confidently_reads_another_structure(
    world, mention, linked
):
    reading = world.reader.read(mention, linked)
    assert reading.verdict == AGAINST
    assert reading.reason.startswith("reads_as:")


@pytest.mark.parametrize(
    "mention", ["12th thoracic vertebra", "L4", "xyzq", "frequenza", "ab"]
)
def test_it_stays_silent_on_digits_codes_and_unknown_words(world, mention):
    assert world.reader.read(mention, "liver").verdict == SILENT


def test_it_never_sees_the_linkers_answer_in_what_it_derives(world):
    # derive() takes the mention alone: the same ranking whatever the linker chose
    assert world.reader.derive("kidney")[0][0].startswith("kidney")


def test_names_that_need_context_are_not_names_for_a_reader_that_sees_only_the_mention(
    world,
):
    own = {cid: set() for cid in world.lexicon.classes}
    for cid, text in world.reader._names:
        if cid in own:
            own[cid].add(text)
    checked = 0
    for cid, entry in world.lexicon.classes.items():
        for name in entry.get("requires_context", {}):
            text = " ".join(name.lower().split())
            plain = {
                " ".join(n.lower().split())
                for n in [*entry["it"], *entry["en"]]
                if n not in entry["requires_context"]
            }
            if text not in plain:
                checked += 1
                assert text not in own[cid]
    assert checked


class _Fixed:
    """A reader that always gives the same reading, to test the linker's use of it."""

    def __init__(self, verdict, label="x"):
        self.verdict, self.label = verdict, label

    def read(self, mention, linked):
        return Reading(self.verdict, "stub", "best", self.label, 0.9, 0.5)


def _linker(world, blind, veto):
    return al.AnatomyLinker(
        world.lexicon,
        world.pool,
        world.equivalent,
        graph=world.graph,
        blind=blind,
        blind_veto=veto,
    )


def _trace(result, stream):
    return [e for e in result.trace if e.stream == stream]


def test_the_linker_records_the_blind_reading_and_counts_it_as_a_mechanism(world):
    result = _linker(world, _Fixed("support"), False).link(
        "fegato", "Il fegato è nei limiti."
    )
    assert result.status == al.ACCEPTED
    assert _trace(result, "blind")[0].verdict == al.SUPPORT
    assert "blind" in result.support


def test_without_the_veto_a_disagreement_is_a_conflict_not_a_stop(world):
    result = _linker(world, _Fixed("against"), False).link(
        "fegato", "Il fegato è nei limiti."
    )
    assert result.status == al.ACCEPTED
    assert "blind_reader_disagrees" in result.conflicts


def test_with_the_veto_a_disagreement_abstains_with_the_reason(world):
    result = _linker(world, _Fixed("against", "rene"), True).link(
        "fegato", "Il fegato è nei limiti."
    )
    assert result.status == al.ABSTAINED
    assert result.reason.startswith("blind_reader_disagrees")


def test_a_silent_reader_changes_nothing(world):
    plain = al.AnatomyLinker(
        world.lexicon, world.pool, world.equivalent, graph=world.graph
    ).link("fegato", "Il fegato è nei limiti.")
    silent = _linker(world, _Fixed("silent"), True).link(
        "fegato", "Il fegato è nei limiti."
    )
    assert (silent.status, silent.cid) == (plain.status, plain.cid)


def test_the_real_reader_does_not_stop_what_the_lexicon_links(world):
    linker = _linker(world, world.reader, True)
    for mention, sentence in [
        ("fegato", "Il fegato è nei limiti."),
        ("left kidney", "The left kidney is normal."),
    ]:
        assert linker.link(mention, sentence).status == al.ACCEPTED


@pytest.mark.parametrize(
    "mention,linked",
    [
        ("rene sinistro", "kidney_right"),
        ("12th rib left", "rib_right_12"),
        ("right lung", "lung_upper_lobe_left"),
    ],
)
def test_it_reads_the_side_again_from_the_words(world, mention, linked):
    reading = world.reader.read(mention, linked)
    assert reading.verdict == AGAINST and reading.reason.startswith("side_word_says:")


def test_a_reading_that_adds_a_head_word_is_not_a_disagreement(world):
    assert world.reader.read("Thoracic aortic", "aorta").verdict != AGAINST
