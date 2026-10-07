import json
import threading
from pathlib import Path

import pytest

from melampo.memory import anatomy_linker as al
from melampo.memory.anatomy_graph import AnatomyGraph, Lift

DATA = Path(__file__).resolve().parent.parent / "data" / "linking"
UBERON = DATA / "uberon-basic.obo"

OBO = """format-version: 1.2

[Term]
id: UBERON:0000464
name: anatomical space

[Term]
id: UBERON:0001981
name: blood vessel

[Term]
id: U:organ
name: organ

[Term]
id: U:kidney
name: kidney
is_a: U:organ

[Term]
id: U:lkidney
name: left kidney
is_a: U:kidney

[Term]
id: U:rkidney
name: right kidney
is_a: U:kidney
relationship: mutually_spatially_disjoint_with U:lkidney

[Term]
id: U:pelvis
name: renal pelvis
relationship: part_of U:kidney

[Term]
id: U:colon
name: colon
is_a: U:organ

[Term]
id: U:subdiv
name: subdivision of colon
relationship: part_of U:colon

[Term]
id: U:sigmoid
name: sigmoid colon
is_a: U:subdiv

[Term]
id: U:mucosa
name: mucosa of sigmoid colon
relationship: part_of U:sigmoid

[Term]
id: U:liver
name: liver
is_a: U:organ

[Term]
id: U:hepvein
name: hepatic vein
is_a: UBERON:0001981
relationship: part_of U:liver

[Term]
id: U:skull
name: skull
is_a: U:organ

[Term]
id: U:sinus
name: frontal sinus
is_a: UBERON:0000464
relationship: part_of U:skull

[Term]
id: U:cartilage
name: costal cartilage

[Term]
id: U:xiphoid
name: xiphoid cartilage
is_a: U:cartilage

[Term]
id: U:junction
name: hepatocolic junction
relationship: part_of U:liver
relationship: part_of U:colon

[Term]
id: U:old
name: obsolete thing
is_a: U:colon
is_obsolete: true
"""

ANCHORS = {
    "anchors": {
        "kidney_left": {
            "uberon": {"U:lkidney": "left kidney"},
            "family": {"U:kidney": "kidney"},
        },
        "kidney_right": {
            "uberon": {"U:rkidney": "right kidney"},
            "family": {"U:kidney": "kidney"},
        },
        "colon": {"uberon": {"U:colon": "colon"}},
        "liver": {"uberon": {"U:liver": "liver"}},
        "skull": {"uberon": {"U:skull": "skull"}},
        "costal_cartilages": {"uberon": {"U:cartilage": "costal cartilage"}},
    }
}


@pytest.fixture(scope="module")
def graph():
    return AnatomyGraph.from_obo(OBO.splitlines(True), ANCHORS)


def test_a_finer_term_climbs_to_its_class_and_says_how(graph):
    lifted = graph.lift("U:mucosa")
    assert lifted == Lift(
        "colon", "part_of", 3, ("U:mucosa", "U:sigmoid", "U:subdiv", "U:colon")
    )


def test_max_depth_is_respected(graph):
    assert graph.lift("U:mucosa", max_depth=2) is None


def test_a_part_of_a_paired_organ_needs_the_side(graph):
    assert graph.lift("U:pelvis") == "part_of_a_paired_structure_without_a_side"
    assert graph.lift("U:pelvis", ["<sn>"]).cid == "kidney_left"
    assert graph.lift("U:pelvis", ["<dx>"]).cid == "kidney_right"
    assert graph.lift("U:pelvis", ["<dx>", "<sn>"]) == "mention_names_both_sides"


def test_the_unsided_term_plus_the_stated_side_is_the_class(graph):
    assert graph.lift("U:kidney", ["<dx>"]) == Lift(
        "kidney_right", "equal", 0, ("U:kidney",)
    )


def test_an_anchor_on_the_other_side_is_refused(graph):
    assert graph.lift("U:rkidney", ["<sn>"]) == "the_class_is_on_the_other_side"


def test_spaces_vessels_and_kinds_are_not_lifted(graph):
    assert graph.lift("U:sinus") == "anatomical_space_is_not_inside_the_class"
    assert graph.lift("U:hepvein") == "blood_vessel_is_not_inside_the_class"
    assert graph.lift("U:xiphoid") == "a_kind_of_the_class_is_not_the_class"


def test_two_classes_at_the_same_depth_is_a_refusal(graph):
    assert graph.lift("U:junction") == "graph_parents_disagree"


def test_two_classes_at_different_depths_is_a_refusal():
    """UBERON puts the duodenum inside the small intestine; the segmentation keeps them apart."""
    obo = (
        "[Term]\nid: U:si\nname: small intestine\n\n"
        "[Term]\nid: U:duo\nname: duodenum\nrelationship: part_of U:si\n\n"
        "[Term]\nid: U:mucosa\nname: duodenal mucosa\nrelationship: part_of U:duo\n\n"
        "[Term]\nid: U:crypt\nname: crypt of duodenum\nrelationship: part_of U:si\n"
        "relationship: part_of U:mucosa\n\n"
        "[Term]\nid: U:sicrypt\nname: crypt of small intestine\nrelationship: part_of U:si\n"
    )
    anchors = {
        "anchors": {
            "small_bowel": {"uberon": {"U:si": "small intestine"}},
            "duodenum": {"uberon": {"U:duo": "duodenum"}},
        }
    }
    g = AnatomyGraph.from_obo(obo.splitlines(True), anchors)
    assert g.lift("U:crypt") == "graph_reaches_two_classes"
    # The mucosa reaches the duodenum first and the small intestine one step later: still two.
    assert g.lift("U:mucosa") == "graph_reaches_two_classes"


def test_obsolete_terms_are_dropped(graph):
    assert "U:old" not in graph.names and "U:old" not in graph.parents


def test_neighbours(graph):
    assert graph.related("kidney_left", "kidney_right") == "the_other_side"
    assert graph.related("U:sigmoid", "U:subdiv") == "parent_and_child"
    assert graph.related("U:lkidney", "U:rkidney") == "declared_disjoint"
    assert graph.related("U:mucosa", "U:liver") is None
    assert graph.related("colon", "colon") is None


def test_sisters_under_a_category_are_not_neighbours():
    many = "".join(
        f"\n[Term]\nid: U:c{i}\nname: child {i}\nis_a: U:cat\n" for i in range(60)
    )
    g = AnatomyGraph.from_obo(
        ("[Term]\nid: U:cat\nname: category\n" + many).splitlines(True),
        {"anchors": {}},
    )
    assert g.related("U:c1", "U:c2") is None


def test_anchor_errors_fail_loudly():
    twice = {
        "anchors": {
            "colon": {"uberon": {"U:colon": "colon"}},
            "liver": {"uberon": {"U:colon": "colon"}},
        }
    }
    with pytest.raises(ValueError, match="anchored to"):
        AnatomyGraph.from_obo(OBO.splitlines(True), twice)
    unsided = {"anchors": {"colon": {"family": {"U:organ": "organ"}}}}
    with pytest.raises(ValueError, match="unsided class"):
        AnatomyGraph.from_obo(OBO.splitlines(True), unsided)


def test_the_shipped_anchors_are_consistent_with_the_lexicon():
    anchors = json.loads((DATA / "anatomy_graph_anchors.json").read_text("utf-8"))
    classes = json.loads((DATA / "anatomy_lexicon.json").read_text("utf-8"))["classes"]
    seen: dict[str, str] = {}
    for cid, entry in anchors["anchors"].items():
        assert cid in classes
        for node in entry.get("uberon", {}):
            assert node not in seen, (node, seen.get(node), cid)
            seen[node] = cid
        if entry.get("family"):
            assert cid.endswith(("_left", "_right")) or cid.startswith(
                ("rib_left", "rib_right")
            )


@pytest.mark.skipif(
    not UBERON.exists(), reason="uberon-basic.obo is downloaded by the bench workflow"
)
def test_the_shipped_anchors_match_the_pinned_uberon_and_known_lifts():
    with open(UBERON, encoding="utf-8") as handle:
        g = AnatomyGraph.from_obo(handle)
    anchors = json.loads((DATA / "anatomy_graph_anchors.json").read_text("utf-8"))
    for entry in anchors["anchors"].values():
        for node, name in {
            **entry.get("uberon", {}),
            **entry.get("family", {}),
        }.items():
            assert g.names[node] == name
    by_name = {v: k for k, v in g.names.items()}
    assert g.lift(by_name["sigmoid colon"]).cid == "colon"
    assert (
        g.lift(by_name["neck of femur"]) == "part_of_a_paired_structure_without_a_side"
    )
    assert g.lift(by_name["neck of femur"], ["<dx>"]).cid == "femur_right"
    assert not isinstance(g.lift(by_name["falciform ligament"]), Lift)


# --- the linker -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lexicon():
    return al.Lexicon.from_json(
        json.loads((DATA / "anatomy_lexicon.json").read_text("utf-8"))
    )


def _models(answer="1"):
    return {"a": lambda p: answer, "b": lambda p: answer}


def _sigmoid_linker(lexicon, graph, **extra):
    terms = [{"id": "U:sigmoid", "name": "sigmoid colon", "synonyms": []}]
    pool, equivalent = al.build_pool(lexicon, terms)
    return al.AnatomyLinker(
        lexicon,
        pool,
        equivalent,
        retriever=lambda m, s, n: ["U:sigmoid"],
        chats=_models(),
        translate=False,
        graph=graph,
        **extra,
    )


def test_by_default_the_parent_is_a_proposal_not_a_link(lexicon, graph):
    result = _sigmoid_linker(lexicon, graph).link(
        "sigmoid colon", "Diverticula of the sigmoid colon."
    )
    assert (result.status, result.cid, result.fallback) == (
        al.ACCEPTED,
        "U:sigmoid",
        "colon",
    )
    assert (
        result.trace[-1].stream == "graph" and result.trace[-1].options[-1] == "U:colon"
    )


def test_when_enabled_the_finer_term_becomes_the_class_with_the_relation(
    lexicon, graph
):
    result = _sigmoid_linker(lexicon, graph, accept_parent_fallback=True).link(
        "sigmoid colon", "Diverticula of the sigmoid colon."
    )
    assert (result.status, result.cid, result.relation, result.part) == (
        al.ACCEPTED,
        "colon",
        "part_of",
        "sigmoid colon",
    )


def test_without_the_graph_nothing_changes(lexicon):
    result = _sigmoid_linker(lexicon, None).link(
        "sigmoid colon", "Diverticula of the sigmoid colon."
    )
    assert (result.cid, result.fallback) == ("U:sigmoid", "")


def test_a_refused_lift_keeps_the_link_as_it_was_and_says_why(lexicon, graph):
    terms = [{"id": "U:sinus", "name": "frontal sinus", "synonyms": []}]
    pool, equivalent = al.build_pool(lexicon, terms)
    linker = al.AnatomyLinker(
        lexicon,
        pool,
        equivalent,
        retriever=lambda m, s, n: ["U:sinus"],
        chats=_models(),
        translate=False,
        graph=graph,
    )
    result = linker.link("frontal sinus", "Mucosal thickening in the frontal sinus.")
    assert result.cid == "U:sinus"
    assert result.trace[-1].reason == "anatomical_space_is_not_inside_the_class"


def test_a_choice_with_a_neighbour_among_the_options_is_not_trusted(lexicon, graph):
    pool, equivalent = al.build_pool(lexicon)
    linker = al.AnatomyLinker(lexicon, pool, equivalent, graph=graph)
    chosen = al.LinkResult(
        al.ACCEPTED,
        "kidney_left",
        "deliberation",
        "models_agree_and_checks_pass",
        ["kidney_left", "kidney_right"],
        {"a": "kidney_left", "b": "kidney_left"},
    )
    result, evidence = linker._converge_on_graph(chosen, "rene")
    assert result.status == al.ABSTAINED
    assert result.reason == "neighbour_passes_the_same_checks:the_other_side"
    assert evidence.verdict == al.VETO


# --- T0: streams, trace, parallel models -------------------------------------------------------


def test_every_cheap_stream_is_read_even_when_one_vetoes(lexicon):
    pool, equivalent = al.build_pool(lexicon)
    linker = al.AnatomyLinker(lexicon, pool, equivalent)
    result = linker.link("GB", "Hb 12,3 g/dL; GB 5040/mmc.")
    assert result.status == al.ABSTAINED and result.stage == "senses"
    streams = {(e.stream, e.verdict, e.cid) for e in result.trace}
    # The conflict is visible: the lexicon alone would have said gallbladder.
    assert ("senses", al.VETO, "gallbladder") in streams
    assert ("lexicon", al.SUPPORT, "gallbladder") in streams


def test_the_two_models_are_asked_at_the_same_time(lexicon):
    barrier = threading.Barrier(2, timeout=5)

    def model(prompt):
        barrier.wait()  # raises BrokenBarrierError if the calls were sequential
        return "1"

    terms = [{"id": "U:sigmoid", "name": "sigmoid colon", "synonyms": []}]
    pool, equivalent = al.build_pool(lexicon, terms)
    linker = al.AnatomyLinker(
        lexicon,
        pool,
        equivalent,
        retriever=lambda m, s, n: ["U:sigmoid"],
        chats={"a": model, "b": model},
        translate=False,
    )
    assert linker.link("sigmoid colon", "Sigmoid colon diverticula.").cid == "U:sigmoid"
