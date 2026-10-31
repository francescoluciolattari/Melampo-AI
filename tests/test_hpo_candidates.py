"""The HPO candidates of a site and the deterministic choice (31 October 2026), on a small synthetic HPO and UBERON
with the real shapes: a term with a logical definition, terms named after the organ's adjective, descendants."""

import json
import sys
from pathlib import Path

from melampo.memory import hpo_candidates as hc

UBERON = """format-version: 1.2

[Term]
id: UBERON:0002048
name: lung
subset: organ_slim
synonym: "pulmo" EXACT []
property_value: has_relational_adjective "pulmonary" xsd:string

[Term]
id: UBERON:0008948
name: upper lobe of lung
relationship: part_of UBERON:0002048 ! lung

[Term]
id: UBERON:0002170
name: upper lobe of right lung
is_a: UBERON:0008948 ! upper lobe of lung

[Term]
id: UBERON:0002107
name: liver
subset: organ_slim
property_value: has_relational_adjective "hepatic" xsd:string
"""

HPO = """format-version: 1.2

[Term]
id: HP:0000118
name: Phenotypic abnormality

[Term]
id: HP:0002088
name: Abnormal lung morphology
is_a: HP:0000118 ! Phenotypic abnormality
intersection_of: PATO:0000051 ! morphology
intersection_of: inheres_in UBERON:0002048 ! lung

[Term]
id: HP:0031983
name: Abnormal pulmonary thoracic imaging finding
is_a: HP:0002088 ! Abnormal lung morphology

[Term]
id: HP:0033608
name: Pulmonary nodule
synonym: "Spot on the lung" EXACT layperson []
is_a: HP:0031983 ! Abnormal pulmonary thoracic imaging finding

[Term]
id: HP:0033609
name: Solid pulmonary nodule
is_a: HP:0033608 ! Pulmonary nodule

[Term]
id: HP:0033612
name: Pure ground-glass pulmonary nodule
is_a: HP:0033608 ! Pulmonary nodule

[Term]
id: HP:0001407
name: Hepatic cysts
is_a: HP:0000118 ! Phenotypic abnormality

[Term]
id: HP:0002240
name: Hepatomegaly
synonym: "Enlarged liver" EXACT layperson []
is_a: HP:0000118 ! Phenotypic abnormality

[Term]
id: HP:9999999
name: Obsolete pulmonary thing
is_obsolete: true
"""


def _phenotypes():
    return hc.Phenotypes.from_obo(HPO.splitlines(), hc.Anatomy.from_obo(UBERON.splitlines()))


LOBE = ["UBERON:0002170", "UBERON:0008948"]


def test_the_site_climbs_to_its_organ_and_its_adjective():
    p = _phenotypes()
    classes = p.anatomy.site_classes(LOBE)
    assert "UBERON:0002048" in classes
    assert ("pulmonary",) in p.site_words(LOBE)


def test_candidates_by_definition_name_and_descendant():
    p = _phenotypes()
    c = p.candidates(LOBE)
    assert "definition" in c["HP:0002088"] and "name" in c["HP:0033608"]
    assert "HP:0033612" in c and "HP:0001407" not in c and "HP:9999999" not in c
    assert p.general(c) == "HP:0002088"


def test_the_deterministic_choice_is_the_most_specific_term_the_clause_names():
    p = _phenotypes()
    assert p.deterministic(LOBE, "solid nodule of 12 mm in the right upper lobe") == ("HP:0033609", "named")
    assert p.deterministic(LOBE, "opacity in the right upper lobe")[0] is None
    assert p.deterministic(["UBERON:0002107"], "the liver is enlarged")[0] is None   # layperson: ranking only


def test_the_shortlist_keeps_the_general_terms_and_ranks_by_shared_words():
    p = _phenotypes()
    short = p.shortlist(LOBE, "ground-glass nodule", 2)
    assert short[0] == "HP:0033612" and "HP:0002088" in short
    liver = p.shortlist(["UBERON:0002107"], "The liver is enlarged", 5)
    assert liver[0] == "HP:0002240"                                   # through its layperson synonym


def test_the_probe_scores_and_the_model_chooses_among_real_terms(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import hpo_probe as hp

    p = _phenotypes()
    assert hp.outcome(p, "HP:0033608", "HP:0033609") == "too_general"
    assert hp.outcome(p, "HP:0033609", "HP:0033608") == "too_specific"
    assert hp.outcome(p, hp.NONE, "HP:0033608") == "none_for_term"
    assert hp.outcome(p, None, "HP:0033608") == "abstained"
    assert hp.parse_letter("B.", 3) == 1 and hp.parse_letter("D", 3) == 3 and hp.parse_letter("??", 3) is None
    prompt = hp.choice_prompt("nodulo solido", "lung", [("HP:0033609", "Solid pulmonary nodule")], "it")
    assert "A. Solid pulmonary nodule" in prompt and "B. none of these" in prompt and "Italian" in prompt
    it = {"HP:0033608": ("Nodulo polmonare", "OFFICIAL")}
    assert hp.deterministic_it(p, LOBE, "Nel lobo superiore destro nodulo polmonare di 9 mm", it) == \
        ("HP:0033608", "named")
    (tmp_path / "hp.obo").write_text(HPO)
    (tmp_path / "uberon.obo").write_text(UBERON)
    (tmp_path / "items.jsonl").write_text(json.dumps(
        {"item_id": "x-en", "pair": "x", "language": "en", "structure": "lung_upper_lobe_right",
         "mention": "right upper lobe", "clause": "Solid nodule of 12 mm in the right upper lobe"}) + "\n")
    (tmp_path / "labels.txt").write_text("x|HP:0033609|solid nodule\n")
    args = type("A", (), dict(hpo=tmp_path / "hp.obo", uberon=tmp_path / "uberon.obo", italian=None,
                              items=tmp_path / "items.jsonl", labels=tmp_path / "labels.txt", limit=0, models="",
                              shortlist=5, workers=1))()
    report = hp.run_probe(args)
    row = report["rows"][0]
    assert row["A"]["outcome"] == "exact" and row["gold_in_candidates"] and row["gold_in_shortlist"]
