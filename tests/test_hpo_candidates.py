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


def test_a_term_is_kept_only_where_the_sentence_states_it():
    """"Stated, not inferred" (31 October 2026): a choice the sentence does not state climbs to the nearest
    ancestor it states; a name that says only "abnormal, here" is what any abnormal finding at the site states."""
    p = _phenotypes()
    assert p.support("HP:0033609", LOBE, "solid nodule of 12 mm") == {"name"}
    assert p.stated("HP:0033609", LOBE, "nodule of 12 mm in the right upper lobe") == ("HP:0033608", 1)
    assert p.stated("HP:0033612", LOBE, "opacity in the right upper lobe")[0] == "HP:0002088"
    assert "site_only" in p.support("HP:0002088", LOBE, "anything")
    assert p.stated("HP:0002240", ["UBERON:0002107"], "the liver is enlarged") == ("HP:0002240", 0)


def test_the_quality_of_the_logical_definition_is_evidence_too():
    pato = '[Term]\nid: PATO:0000586\nname: increased size\nsynonym: "enlarged" RELATED []\n'
    hpo = HPO + ('\n[Term]\nid: HP:0001744\nname: Splenomegaly\nis_a: HP:0000118 ! Phenotypic abnormality\n'
                 'intersection_of: PATO:0000586 ! increased size\nintersection_of: inheres_in UBERON:0002106 ! spleen\n')
    uberon = UBERON + '\n[Term]\nid: UBERON:0002106\nname: spleen\nsubset: organ_slim\n'
    p = hc.Phenotypes.from_obo(hpo.splitlines(), hc.Anatomy.from_obo(uberon.splitlines()))
    p.qualities = hc.read_qualities(pato.splitlines())
    assert p.support("HP:0001744", ["UBERON:0002106"], "The spleen is enlarged (14 cm)") == {"quality"}
    assert p.support("HP:0001744", ["UBERON:0002106"], "The spleen is normal") == set()


def test_equivalent_terms_count_as_one_and_the_common_ancestor():
    hpo = HPO + ('\n[Term]\nid: HP:0000001\nname: Rib fracture\nis_a: HP:0000118 ! Phenotypic abnormality\n'
                 '\n[Term]\nid: HP:0000002\nname: Fractured rib\nis_a: HP:0000118 ! Phenotypic abnormality\n')
    p = hc.Phenotypes.from_obo(hpo.splitlines(), hc.Anatomy.from_obo(UBERON.splitlines()))
    assert p.equivalent("HP:0000001", "HP:0000002") and not p.equivalent("HP:0033609", "HP:0033612")
    assert p.common_ancestor(["HP:0033609", "HP:0033612"]) == "HP:0033608"
    assert p.common_ancestor(["HP:0033609"]) == "HP:0033609"


def test_the_descent_goes_down_the_site_branches_and_the_integration_checks_it():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import hpo_probe as hp

    p = _phenotypes()
    assert "HP:0031983" in p.descent_roots(LOBE) or "HP:0002088" in p.descent_roots(LOBE)
    wanted = ["Abnormal lung morphology", "Abnormal pulmonary thoracic imaging finding", "Pulmonary nodule",
              "Solid pulmonary nodule"]

    def chat(prompt):          # a reader that goes down to the solid nodule, then stops
        options = [line for line in prompt.splitlines() if line[:2] in {f"{x}." for x in hp.LETTERS}]
        for line in options:
            if line[3:] in wanted and f'"{line[3:]}"' not in prompt:
                return line[0]
        return options[-1][0]

    term, path = hp.descend(p, chat, LOBE, "Solid nodule of 12 mm", "Solid nodule of 12 mm", "lung", "en")
    assert term == "HP:0033609" and path[-1] == "HP:0033609"
    row = {"B:m": {"answer": "HP:0033612"}, "D:m": {"answer": "HP:0033609"},
           "_english": {"m": "Solid nodule of 12 mm"}}
    hp.integrate(p, row, LOBE, ["m"], "HP:0033608")
    assert row["I"]["answer"] == "HP:0033608" and row["I"]["outcome"] == "exact"   # the common term, stated
    row = {"B:m": {"answer": hp.NONE}, "D:m": {"answer": "HP:0033609"}, "_english": {"m": "x"}}
    hp.integrate(p, row, LOBE, ["m"], "HP:0033608")
    assert row["I"]["answer"] is None and row["I"]["why"] == "none_against_a_term"


def test_a_hedged_term_is_not_stated():
    """After the third run: the radiologist's hypothesis ("suggestive of", "consistent with") is not what the
    sentence states, even when its words are there."""
    assert hc.unhedged("A 5 mm formation, suggestive of a cholesterol polyp") == "A 5 mm formation, "
    assert hc.unhedged("Solid nodule of 12 mm") == "Solid nodule of 12 mm"
    p = _phenotypes()
    text = hc.unhedged("Opacity of 12 mm, likely a solid nodule")
    assert p.stated("HP:0033609", LOBE, text)[0] == "HP:0002088"
