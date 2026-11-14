"""The concept of what is seen, in the same reading (14 November 2026): the readers choose the HPO term by letter
in the claims answer; the choices of the two are integrated and kept only if the words state the term; the words
stay beside it."""

import json

import pytest

from melampo.evaluation import llm_reading as lr
from melampo.memory import finding_owner as fo
from melampo.memory import hpo_candidates as hc

UBERON = """[Term]
id: UBERON:0002048
name: lung
subset: organ_slim
property_value: has_relational_adjective "pulmonary" xsd:string
"""
HPO = """[Term]
id: HP:0000118
name: Phenotypic abnormality

[Term]
id: HP:0002088
name: Abnormal lung morphology
is_a: HP:0000118 ! Phenotypic abnormality

[Term]
id: HP:0033560
name: Pulmonary nodule
is_a: HP:0002088 ! Abnormal lung morphology

[Term]
id: HP:0033558
name: Solid pulmonary nodule
is_a: HP:0033560 ! Pulmonary nodule

[Term]
id: HP:0033561
name: Calcified pulmonary nodule
is_a: HP:0033560 ! Pulmonary nodule
"""


@pytest.fixture
def phenotypes(monkeypatch):
    monkeypatch.setattr(hc, "site_anchors", lambda cid, path=None: ["UBERON:0002048"])
    return hc.Phenotypes.from_obo(HPO.splitlines(), hc.Anatomy.from_obo(UBERON.splitlines()))


def test_the_terms_are_offered_by_letter_and_read_back_to_their_ids():
    concepts = [("HP:0033558", "Solid pulmonary nodule"), ("HP:0033560", "Pulmonary nodule")]
    prompt = lr.claims_prompt("A solid nodule of 12 mm.", 8, 14, "lung", concepts=concepts)
    assert "A. Solid pulmonary nodule" in prompt and '"concept": "<letter>|none"' in prompt
    assert "never replaces them" in prompt
    # without terms the question is the one measured: unchanged
    assert "concept" not in lr.claims_prompt("A solid nodule of 12 mm.", 8, 14, "lung")
    answer = json.dumps({"marked": "nodule", "same": "yes", "observed": [
        {"words": "solid nodule", "differs": "yes", "of": "it", "concept": "A"}]})
    got = lr.read_claims(answer, "A solid nodule of 12 mm.", "nodule", concepts=concepts)
    assert got["observed"][0]["concept"] == "HP:0033558" and got["observed"][0]["words"] == "solid nodule"
    none = answer.replace('"A"', '"none"')
    assert lr.read_claims(none, "A solid nodule of 12 mm.", "nodule", concepts=concepts)["observed"][0]["concept"] == ""


def _owner(phenotypes):
    return fo.FindingOwner(None, {}, phenotypes=phenotypes, question="claims")


def _valid(*terms):
    return {f"m{i}": {"valid": True, "observed": [{"words": "w", "of": "it", "concept": t}]} for i, t in
            enumerate(terms)}


def test_the_readers_choices_are_integrated_and_the_words_decide(phenotypes):
    owner = _owner(phenotypes)
    claims = {"findings": [{"words": "solid nodule", "of": "it"}]}
    got = owner._concept(_valid("HP:0033558", "HP:0033558"), claims, "lung_upper_lobe_right")
    assert got["term"] == "HP:0033558" and got["by"] == "readers_and_words" and got["words"] == "solid nodule"
    # two different terms: the one both are
    got = owner._concept(_valid("HP:0033558", "HP:0033561"), claims, "lung_upper_lobe_right")
    assert got["term"] == "HP:0033560"
    # a term the words do not state is not kept: "calcified" is not written, the words state a nodule
    got = owner._concept(_valid("HP:0033561", "HP:0033561"), {"findings": [{"words": "nodule", "of": "it"}]},
                         "lung_upper_lobe_right")
    assert got["term"] == "HP:0033560" and got["by"] == "the_words_stated_a_more_general_term"
    # a reader that gives no term: no concept, and why
    got = owner._concept(_valid("HP:0033558", ""), claims, "lung_upper_lobe_right")
    assert got["term"] == "" and got["why"] == "a_reader_gave_no_term"
    # without HPO: nothing is chosen, nothing is offered
    assert _owner(None)._concept(_valid("HP:0033558"), claims, "x") == {}
    assert _owner(None)._concept_options("a nodule", "x") == []
    options = owner._concept_options("A solid nodule of 12 mm in the right upper lobe", "lung_upper_lobe_right")
    assert ("HP:0033558", "Solid pulmonary nodule") in options


def test_a_finding_read_at_its_structure_carries_its_concept_beside_its_words(phenotypes):
    clause = "In the right upper lobe a solid nodule of 12 mm is seen."

    def chat(prompt):
        letters = {line[3:]: line[0] for line in prompt.splitlines() if line[1:3] == ". "}
        return json.dumps({"marked": "right upper lobe", "same": "yes", "observed": [
            {"words": "a solid nodule", "aspect": "lesion", "usual": "no nodule", "differs": "yes", "of": "it",
             "relation": "In", "concept": letters.get("Solid pulmonary nodule", "none")}],
            "measure": [{"of": "what_is_seen", "words": "12 mm"}], "conclusion": "none"})

    owner = fo.FindingOwner(None, {"a": chat, "b": chat}, phenotypes=phenotypes, question="claims")
    got = owner("right upper lobe", clause, clause.index("right upper lobe"), "en", "lung_upper_lobe_right")
    assert got.verdict == fo.HERE
    assert got.concept["term"] == "HP:0033558" and got.concept["words"] == "a solid nodule"
    assert got.as_dict()["concept"]["name"] == "Solid pulmonary nodule"
