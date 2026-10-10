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


def test_sentence_around_returns_the_sentence_of_the_mention():
    text = "The scan was clear. The heart of the cheese was sampled. Nothing else was seen."
    i = text.index("heart")
    assert ci.sentence_around(text, i, i + 5) == "The heart of the cheese was sampled."
    assert ci.sentence_around("one sentence only", 4, 12) == "one sentence only"


def test_ambito_is_the_kind_of_the_other_things_named():
    memory = lattice().memory
    assert ci.ambito(memory, "the cheese and the cheese rind")[0] == "food"
    assert ci.ambito(memory, "the heart and the liver") == ("none", "none")  # structures are not the ambito


def test_ambito_of_a_spread_sentence_is_not_an_arbitrary_kind():
    memory = cl.BlockMemory.from_json({
        "heads": {"rate": ["property", 0.9], "cheese": ["food", 0.9], "donation": ["procedure", 0.9],
                  "sample": ["conceptual", 0.9], "agar": ["chemical", 0.9]},
        "names": {},
    })
    # five kinds, one each: none reaches the share, the answer is the two first in alphabetical order,
    # and the catch-all kind is never part of it
    key, top = ci.ambito(memory, "sample cheese agar donation rate")
    assert key == "chemical+food" and "conceptual" not in key
    assert ci.ambito(memory, "sample sample") == ("none", "none")


def test_the_ambito_node_counts_only_the_excess_over_the_base_rate():
    lat = lattice()
    traces = ci.TraceMemory()
    def add(doc, text, role):
        i = text.index("heart")
        traces.add(doc, text, i, i + 5, role, lat)
    for n in range(8):
        add(f"d{n}", "the heart of the cheese was sampled", "not_a_body_site" if n < 2 else "structure")
    for n in range(8, 40):
        add(f"d{n}", "the heart rate was measured", "structure")
    base = traces.base_rate(None)
    assert abs(sum(base.values()) - 1.0) < 1e-9 and base["structure"] > 0.9
    reader = ci.CIReader(lat, None, None, traces)
    nodes = reader.construct("heart", "the heart of the cheese was sampled", 4, doc="other")
    dom = [n for n in nodes if n[0].startswith("domain:")]
    assert dom, "the ambito recalls traces"
    for _, _, links in dom:
        assert links.get("structure", 0.0) == 0.0  # the usual reading adds nothing
        assert links.get("not_a_body_site", 0.0) > 0.0  # what the ambito changes does
