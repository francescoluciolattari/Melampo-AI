"""The construction-integration reader: space, predication, memories, integration."""

import numpy as np
import pytest

pytest.importorskip("scipy")

from melampo.memory import chunk_lattice as cl  # noqa: E402
from melampo.memory import ci_reader as ci  # noqa: E402

TEXTS = (
    ["the heart rate and the heart rhythm were measured by the cardiac monitor in the clinic"] * 12
    + ["the surgeon removed the liver during the transplant operation and the liver graft was placed"] * 12
    + ["maroilles cheese rind and the heart of the cheese were sampled for bacteria in the dairy"] * 12
)


@pytest.fixture(scope="module")
def space():
    return ci.SemanticSpace.build(TEXTS, dim=8, min_count=2, chunk=12)


def lattice():
    memory = cl.BlockMemory.from_json({
        "heads": {"rate": ["property", 0.9], "weight": ["property", 0.9], "heart": ["anatomy", 1.0],
                  "liver": ["anatomy", 1.0], "donation": ["procedure", 0.9], "cheese": ["food", 0.9]},
        "names": {},
    })
    return cl.ChunkLattice(memory)


def test_space_roundtrip_and_unit_vectors(space, tmp_path):
    path = tmp_path / "s.npz"
    space.save(path)
    again = ci.SemanticSpace.load(path)
    assert again.words == space.words and np.allclose(again.vectors, space.vectors)
    assert abs(float(np.linalg.norm(space.vec("heart"))) - 1.0) < 1e-5
    assert space.vec("unknownword") is None


def test_words_of_one_context_are_closer_than_words_of_another(space):
    near = float(space.vec("rhythm") @ space.vec("heart"))
    far = float(space.vec("rhythm") @ space.vec("transplant"))
    assert near > far


def test_predication_is_the_predicate_adjusted_by_the_argument(space):
    plain = space.vec("rate")
    predicated = space.predicate("rate", ["heart"], m=10, k=2)
    assert abs(float(np.linalg.norm(predicated)) - 1.0) < 1e-5
    assert float(predicated @ space.vec("heart")) > float(plain @ space.vec("heart"))
    assert space.predicate("nonexistent", ["heart"]) is None


def test_reading_of_kind():
    assert ci.reading_of_kind("anatomy") == ci.STRUCTURE
    assert ci.reading_of_kind("procedure") == "procedure_site"
    assert ci.reading_of_kind("property") == "inherent_location"
    assert ci.reading_of_kind("food") == "not_a_body_site"
    assert ci.reading_of_kind("conceptual") is None


def test_cues_are_the_same_on_both_sides():
    lat = lattice()
    text = "The heart rate was normal."
    start = text.index("heart")
    cues = ci.make_cues(lat, text, start, start + 5)
    assert (cues.word, cues.nxt, cues.head, cues.previous) == ("heart", "rate", "rate", "")
    assert cues.head_kind == "property" and cues.anatomy
    levels = [k[0] for k in cues.keys()]
    assert levels[0] == "word+next" and levels[-1] == "word" and "anatomy+head-kind" in levels


def test_traces_leave_out_the_document_being_read():
    lat = lattice()
    memory = ci.TraceMemory()
    text = "The heart rate was normal."
    start = text.index("heart")
    memory.add("d1", text, start, start + 10, "inherent_location", lat)
    memory.add("d2", text, start, start + 10, "inherent_location", lat)
    cues = ci.make_cues(lat, text, start, start + 5)
    own_excluded = memory.recall(cues, "d1")
    assert own_excluded and own_excluded[0][2]["inherent_location"] == 1
    assert memory.recall(cues, "d3")[0][2]["inherent_location"] == 2
    only = ci.TraceMemory()
    only.add("d1", text, start, start + 10, "inherent_location", lat)
    assert only.recall(cues, "d1") == []


def test_integration_lets_the_evidence_that_fits_win_and_stops():
    reader = ci.CIReader(lattice())
    nodes = [("block", 1.0, {"procedure_site": 1.0}), ("head", 0.8, {"procedure_site": 1.0}),
             ("weak", 0.3, {ci.STRUCTURE: 1.0})]
    act, cycles = reader.integrate(nodes)
    assert act["procedure_site"] > act[ci.STRUCTURE] and cycles < ci.MAX_CYCLES
    # no evidence at all: nothing is activated, nothing is invented
    empty, _ = reader.integrate([])
    assert not any(empty.values())


def test_the_whole_reader_reads_a_measure_and_a_plain_structure(space):
    protos = ci.prototypes(space, [("anatomy", "heart liver organ"), ("property", "rate rhythm measured")])
    reader = ci.CIReader(lattice(), space, protos, None)
    plain = reader.read("heart", "The heart was normal in size.", 4)
    measure = reader.read("heart", "The heart rate was normal.", 4)
    assert plain.top == ci.STRUCTURE
    assert measure.shares[ci.STRUCTURE] < plain.shares[ci.STRUCTURE]
    assert measure.nodes and sum(measure.shares.values()) == pytest.approx(1.0)
