"""Offline tests of the phrase-probe experiment (no model, no network): a tiny NCIt-like ontology."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("numpy")

import phrase_probe as pp  # noqa: E402

OBO = """format-version: 1.2

[Term]
id: NCIT:C12219
name: Anatomic Structure, System, or Substance

[Term]
id: NCIT:C1
name: Heart
is_a: NCIT:C12219

[Term]
id: NCIT:C1949
name: Food or Food Product

[Term]
id: NCIT:C2
name: Cheese
is_a: NCIT:C1949

[Term]
id: NCIT:C97325
name: Manufactured Object

[Term]
id: NCIT:C3
name: Vena Cava Filter
is_a: NCIT:C97325

[Term]
id: NCIT:C4
name: Blood Filter
is_a: NCIT:C97325

[Term]
id: NCIT:C5
name: Air Filter
is_a: NCIT:C97325

[Term]
id: NCIT:C17021
name: Protein

[Term]
id: NCIT:C6
name: Interleukin
is_a: NCIT:C17021

[Term]
id: NCIT:C28428
name: Retired Concept

[Term]
id: NCIT:C7
name: Transfusion
is_a: NCIT:C28428
"""


@pytest.fixture(scope="module")
def kinds():
    return pp.Kinds(list(pp.read_obo(OBO.splitlines(True))))


def case(mention, sentence, label=0, corpus="medmentions", doc="d1"):
    return {"mention": mention, "sentence": sentence, "label": label, "corpus": corpus, "doc": doc,
            "labels": [], "status": "accepted", "conflicts": [], "convergence": 3}


def test_kinds_come_from_the_ontology_and_skip_retired_classes(kinds):
    assert kinds.word_kind("cheese") == ("food", 1.0)
    assert kinds.word_kind("filter")[0] == "device"  # three names end with "filter"
    assert kinds.word_kind("transfusion") == (None, 0.0)  # retired class
    assert kinds.word_kind("heart") == ("anatomy", 1.0)


def test_head_reads_the_phrase_after_dropping_nominalisations(kinds):
    c = case("inferior vena cava", "Office-based inferior vena cava filter placement is safe.")
    pp.arm_head(c, kinds)
    assert c["head_word"] == "filter" and c["head_kind"] == "device" and c["head_other"] == 1.0


def test_object_and_one_sense_per_discourse(kinds):
    first = case("heart", "Samples from rind and heart of Maroilles cheese were used.", 1)
    later = case("heart", "38 on the rind and 67 in the heart were identified.", 1)
    other_doc = case("heart", "38 on the rind and 67 in the heart were identified.", 0, doc="d2")
    cases = [first, later, other_doc]
    for c in cases:
        pp.arm_object(c, kinds)
    assert first["object"] == 1.0 and first["object_kind"] == "food"
    assert later["object"] == 0.0
    pp.arm_discourse(cases, cases, kinds)
    assert [c["discourse"] for c in cases] == [1.0, 1.0, 0.0]


def test_a_typo_is_a_recorded_hypothesis_that_can_change_the_head(kinds):
    c = case("heart", "Higher levels of heart interlukine-6 were observed.", 1)
    pp.arm_head(c, kinds)
    pp.arm_typo(c, kinds, pp.Speller(kinds.vocabulary))
    assert c["typo_hypothesis"]["written"] == "interlukine-6"
    assert c["typo_hypothesis"]["read_as"] == "interleukin"
    assert c["typo"] > 0


def test_frequent_words_of_the_corpus_are_not_typos(kinds):
    speller = pp.Speller(kinds.vocabulary | {"underwent"})
    assert speller.correct("underwent") is None
    assert speller.correct("interlukine") == "interleukin"


def test_damerau_counts_a_transposition_as_one():
    assert pp.damerau("filetr", "filter", 2) == 1
    assert pp.damerau("abc", "xyz", 1) == 2


def test_word_groups_stay_in_the_clause():
    groups = pp.word_groups(case("spleen", "Anti-CD45RB and donor-specific spleen cells transfusion, given once"))
    assert "donor-specific spleen cells transfusion" in groups
    assert all("given" not in g for g in groups)


def test_gliner_reads_the_longest_covering_span():
    c = case("inferior vena cava", "inferior vena cava filter placement")
    entities = [
        {"start": 0, "end": 18, "label": "anatomical structure", "score": 0.9},
        {"start": 0, "end": 25, "label": "medical device or object", "score": 0.7},
    ]
    pp.apply_gliner(c, 0, 18, entities)
    assert c["gliner_kind"] == "device" and c["gliner_other"] == 0.7


def test_llm_answers_are_counted_per_model_and_lost_answers_are_not_votes():
    c = case("heart", "heart of Maroilles cheese")
    pp.apply_llm(c, {
        "a": {"phrase": "heart of Maroilles cheese", "type": "food", "body_site": "no"},
        "b": {"error": "HTTP 429"},
    })
    assert c["llm_other"] == 1.0 and c["llm_not_site"] == 1.0 and c["llm_kind"] == "food"
    d = case("heart", "heart")
    pp.apply_llm(d, {"a": {"error": "x"}})
    assert d["llm_lost"] == 1.0 and "llm_other" not in d


def test_report_has_gate_and_role_and_renders(kinds):
    class Args:
        llm = "none"
        llm_sample = 10
        workers = 1

    cases = [
        case("heart", "Samples from rind and heart of Maroilles cheese were used.", 1),
        case("liver", "The liver is enlarged.", 0, doc="d2"),
    ]
    cases[0]["labels"] = ["C1254362|T082"]
    texts = {("medmentions", "d1"): cases[0]["sentence"], ("medmentions", "d2"): cases[1]["sentence"]}
    report = pp.run(Args(), rows=cases, texts=texts, cases=cases, kinds=kinds)
    assert report["gate"]["uncertain_errors"] == 0  # same profile as a right link
    assert report["roles"]["object_kind"] == {"read": 1, "same_role_as_gold": 1}
    assert report["features"]["discourse"]["auc"] == 1.0
    text = pp.markdown(report)
    assert "Gate: only when uncertain?" in text and "heart" in text
