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


def test_resources_roundtrip_and_the_linker_reads_in_trace_only(space, tmp_path):
    from melampo.memory import anatomy_linker as al

    lat = lattice()
    traces = ci.TraceMemory(space)
    text = "the heart rate was measured"
    i = text.index("heart")
    for n in range(3):
        traces.add(f"d{n}", text, i, i + 10, "inherent_location", lat)
    protos = {"structure": space.vec("heart"), "inherent_location": space.vec("rate")}
    ci.save_resources(tmp_path, space, protos, traces)
    reader = ci.load_reader(tmp_path, lat)
    assert reader.traces.size == 3 and set(reader.protos) == set(protos)
    again = reader.read("heart", text, i, None)
    first = ci.CIReader(lat, space, protos, traces).read("heart", text, i, None)
    assert again.decision == first.decision

    # the linker keeps its decision whatever the reader says; the reading is only written down
    import json
    from pathlib import Path

    from melampo.memory import anatomy_parts as ap

    data = Path(__file__).resolve().parents[1] / "data" / "linking"
    lexicon = al.Lexicon.from_json(json.loads((data / "anatomy_lexicon.json").read_text("utf-8")))
    parts = ap.PartTable.from_json(json.loads((data / "anatomy_parts.json").read_text("utf-8")), lexicon)
    sentence = "The heart rate was normal."
    plain = al.AnatomyLinker(lexicon, [], {}, parts=parts).link("heart", sentence)
    with_reader = al.AnatomyLinker(lexicon, [], {}, parts=parts, ci_reader=reader).link("heart", sentence)
    assert (plain.status, plain.cid, plain.reason) == (with_reader.status, with_reader.cid, with_reader.reason)
    assert plain.reading == "" and with_reader.reading.count(":") == 2


class _StubReader:
    """A reader that says what it is told: the linker's use of the reader is what is tested."""

    def __init__(self, decision, lattice=None):
        self.decision, self.lattice = decision, lattice

    def document(self, text):
        return None, ""

    def discourse(self, text):
        return None

    def read(self, mention, sentence, start, doc, vector, key, discourse=None, doc_at=None):
        return ci.CIReading(self.decision, self.decision, 0.6, {self.decision: 0.8}, 3)


def _linker(decision, mode, **extra):
    import json
    from pathlib import Path

    from melampo.memory import anatomy_linker as al
    from melampo.memory import anatomy_parts as ap

    data = Path(__file__).resolve().parents[1] / "data" / "linking"
    lexicon = al.Lexicon.from_json(json.loads((data / "anatomy_lexicon.json").read_text("utf-8")))
    parts = ap.PartTable.from_json(json.loads((data / "anatomy_parts.json").read_text("utf-8")), lexicon)
    return al.AnatomyLinker(lexicon, [], {}, parts=parts, ci_reader=_StubReader(decision, **extra),
                            ci_reader_mode=mode)


SENTENCE = "The heart is large."


def test_the_reader_mode_is_checked():
    import pytest

    with pytest.raises(ValueError):
        _linker("structure", "loud")


def test_trace_mode_records_nothing_against_the_link():
    r = _linker("not_a_body_site", "trace").link("heart", SENTENCE)
    plain = _linker("structure", "trace").link("heart", SENTENCE)
    assert (r.status, r.cid) == (plain.status, plain.cid)
    assert "reader_reads_a_non_site" not in r.conflicts


def test_record_mode_writes_the_conflict_and_keeps_the_link():
    from melampo.memory import anatomy_linker as al

    r = _linker("not_a_body_site", "record").link("heart", SENTENCE)
    assert r.status == al.ACCEPTED and "reader_reads_a_non_site" in r.conflicts


def test_review_mode_sends_a_weakly_supported_link_to_review_and_never_changes_one():
    from melampo.memory import anatomy_linker as al

    plain = _linker("structure", "review").link("heart", SENTENCE)
    held = _linker("not_a_body_site", "review").link("heart", SENTENCE)
    assert plain.status == al.ACCEPTED
    assert held.status == al.ABSTAINED and held.reason == "streams_disagree:reader_reads_a_non_site"
    assert held.options == [plain.cid]  # the link it would have made is offered to the reviewer, not changed
    # a role reading is not the reader's to give: it never routes
    assert _linker("procedure_site", "review").link("heart", SENTENCE).status == al.ACCEPTED


def test_the_reader_is_not_used_where_another_method_decides():
    from melampo.memory import chunk_lattice as cl

    memory = cl.BlockMemory.from_json({"heads": {"donation": ["procedure", 1.0, 0]}, "names": {}})
    lat = cl.ChunkLattice(memory)
    linker = _linker("not_a_body_site", "review", lattice=lat)
    # "liver donation": the lattice reads a procedure site -> its method, not the reader's
    got = linker.read("liver", "Liver donation was done.")
    reader = next(e for e in got if e.stream == "reader")
    assert reader.verdict == "silent" and "other_method=lattice" in reader.reason
    # a procedure neighbour is the role rule's
    got = linker.read("liver", "Liver biopsy was done.")
    assert next(e for e in got if e.stream == "reader").reason.endswith("other_method=role")
    # alone in a sentence nothing else speaks: the reader reads against
    got = linker.read("heart", SENTENCE)
    assert next(e for e in got if e.stream == "reader").verdict == "against"


# -- what a person does that the reader did not (10 October 2026, after the diagnosis) ---------------------


def _procedure_lattice():
    memory = cl.BlockMemory.from_json({
        "heads": {"placement": ["procedure", 0.95], "filter": ["device", 0.9], "heart": ["anatomy", 1.0],
                  "cheese": ["food", 1.0], "bulb": ["device", 1.0], "layer": ["anatomy", 1.0],
                  "rind": ["anatomy", 0.9]},
        "names": {},
    })
    return cl.ChunkLattice(memory)


def test_an_abbreviation_in_brackets_does_not_break_the_name():
    lat = _procedure_lattice()
    plain = lat.read("inferior vena cava", "evaluate inferior vena cava filter placement in the office")
    through = lat.read("inferior vena cava", "evaluate inferior vena cava (IVC) filter placement in the office")
    assert plain.outcome == "role"  # the name before the device and the procedure is not the structure
    assert (through.outcome, through.role, through.block) == (plain.outcome, plain.role, plain.block)
    # the offset of the mention is kept when a bracket before it is removed
    again = lat.read("vena cava", "the IVC (inferior vena cava, IVC) filter (IVC) placement", 8)
    assert again.outcome != "unread"


def test_only_a_defined_abbreviation_is_seen_through():
    from melampo.memory.chunk_lattice import see_through_abbreviations as see

    assert see("the kidney (KD) lesion", 4, "kidney")[0] == "the kidney lesion"
    assert see("the kidney (left) lesion", 4, "kidney")[0] == "the kidney (left) lesion"
    assert see("the kidney (n=24) lesion", 4, "kidney")[0] == "the kidney (n=24) lesion"
    assert see("the kidney (IL-6) lesion", 4, "kidney")[0] == "the kidney lesion"
    # a bracket before the mention shifts its offset
    text, start = see("the aorta (AO) and the kidney lesion", 23, "kidney")
    assert text == "the aorta and the kidney lesion" and text[start : start + 6] == "kidney"
    # the mention itself, written as an abbreviation, stays
    assert see("the vena cava (IVC) filter", 18, "IVC")[0] == "the vena cava (IVC) filter"


def test_a_rule_that_changes_the_sense_enters_the_network_and_overrules_the_default(space):
    protos = ci.prototypes(space, [("anatomy", "heart liver organ"), ("food", "cheese rind dairy")])
    reader = ci.CIReader(_procedure_lattice(), space, protos, None)
    sentence = "Samples of the rind and heart of Maroilles cheese were used."
    start = sentence.index("heart")
    nodes = reader.construct("heart", sentence, start)
    rule = [n for n in nodes if n[0].startswith("rule:")]
    assert rule and rule[0][2]["not_a_body_site"] > 0 and rule[0][2][ci.STRUCTURE] < 0
    assert reader.read("heart", sentence, start).decision == "not_a_body_site"
    # a role the structure takes ("appearance of the heart") is not a sense of the word: no rule node
    nodes = reader.construct("heart", "The size of the heart was normal.", 12)
    assert not [n for n in nodes if n[0].startswith("rule:")]


def test_a_sense_settled_in_the_text_is_kept_for_its_other_occurrences(space):
    protos = ci.prototypes(space, [("anatomy", "heart liver organ"), ("food", "cheese rind dairy")])
    reader = ci.CIReader(_procedure_lattice(), space, protos, None)
    text = ("Samples of the rind and heart of Maroilles cheese were used. "
            "Strains were found on the rind and in the heart.")
    settled = reader.discourse(text)
    assert len(settled) == 1 and "heart" in settled.seen
    second = text.index("in the heart") + 7
    sentence = "Strains were found on the rind and in the heart."
    got = reader.read("heart", sentence, sentence.index("heart"), None, None, "", settled, second)
    assert got.decision == "not_a_body_site" and any(n.startswith("discourse:heart") for n in got.nodes)
    # without the text's memory the same sentence is a plain structure
    alone = reader.read("heart", sentence, sentence.index("heart"))
    assert alone.decision != "not_a_body_site"
    # the occurrence that settled the sense does not support itself
    first = text.index("heart")
    assert settled.node("heart", first) is None and settled.node("heart", second)


def test_a_text_settles_a_sense_only_through_food_and_never_through_a_role(space):
    protos = ci.prototypes(space, [("anatomy", "heart liver organ"), ("food", "cheese rind dairy")])
    reader = ci.CIReader(_procedure_lattice(), space, protos, None)
    # a device head ("bulb") is too noisy a lexicon entry to settle a sense for the whole text
    assert len(reader.discourse("The outer layer of the olfactory bulb was thin. The layer was stained.")) == 0
    # a role in a compound is local to its phrase
    assert len(reader.discourse("The heart donation was done. The heart was large.")) == 0
