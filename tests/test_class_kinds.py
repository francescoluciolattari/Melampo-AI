"""What a class of the atlas is, read from UBERON and HPO, the same for every class (11 November 2026)."""

import json

from melampo.memory import class_kinds as ck

HPO = """format-version: 1.2

[Term]
id: HP:0000118
name: Phenotypic abnormality

[Term]
id: HP:0000077
name: Abnormality of the kidney
is_a: HP:0000118 ! Phenotypic abnormality

[Term]
id: HP:0012210
name: Abnormal renal morphology
is_a: HP:0000077 ! Abnormality of the kidney
intersection_of: PATO:0000051 ! morphology
intersection_of: inheres_in_part_of UBERON:0002113 ! kidney

[Term]
id: HP:0000107
name: Renal cyst
synonym: "Kidney cyst" EXACT layperson []
is_a: HP:0012210 ! Abnormal renal morphology
"""


def test_a_class_named_as_a_phenotype_is_a_finding_of_the_structure_hpo_says(tmp_path):
    hpo = tmp_path / "hp.obo"
    hpo.write_text(HPO)
    ck._CACHE.clear()
    kinds = ck.class_kinds(hpo)
    assert kinds["kidney_cyst_left"]["kind"] == "finding" and kinds["kidney_cyst_left"]["site"] == "kidney_left"
    assert kinds["kidney_cyst_right"]["site"] == "kidney_right"
    assert kinds["kidney_cyst_left"]["phenotype"] == "HP:0000107"
    # a class with no UBERON term that is no abnormality stays a structure (a segment of the liver)
    assert kinds["liver_segment_2"]["kind"] == "structure"
    # a class anchored to UBERON is a structure, whatever its name
    assert kinds["kidney_left"]["kind"] == "structure"
    assert ck.as_site("kidney_cyst_left", hpo)["cid"] == "kidney_left"
    assert ck.as_site("spleen", hpo) == {"cid": "spleen"}


def test_without_hpo_no_class_is_turned_into_a_finding(tmp_path, monkeypatch):
    ck._CACHE.clear()
    monkeypatch.setattr(ck, "hpo_path", lambda: None)
    kinds = ck.class_kinds()
    assert kinds["kidney_cyst_left"]["kind"] == "structure" and "not read" in kinds["kidney_cyst_left"]["why"]
    ck._CACHE.clear()


def test_a_name_never_crosses_a_parenthesis(monkeypatch):
    from melampo.data.report_reading import build_linker
    from melampo.evaluation.gold_set import propose_mentions

    linker = build_linker()[0]
    text = "A hypodense lesion was observed in the upper pole of the left kidney (cyst?)"
    monkeypatch.delenv("MELAMPO_FINDER", raising=False)
    assert [m["mention"] for m in propose_mentions(text, linker.lexicon, linker.parts)] == ["left kidney"]
    # the samples labelled before 12 November 2026 are drawn as they were
    monkeypatch.setenv("MELAMPO_FINDER", "2026-11-11")
    assert [m["mention"] for m in propose_mentions(text, linker.lexicon, linker.parts)] == ["left kidney (cyst"]


def test_an_item_that_is_not_the_item_labelled_is_known():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import owner_probe as op

    item = {"clause": "A cyst in the left kidney", "mention": "left kidney", "structure": "kidney_left"}
    assert op.fingerprint(item) != op.fingerprint({**item, "structure": "kidney_cyst_left"})
    known = op.fingerprints("ctrate5")
    assert len(known) == 300 and all(len(v) == 16 for v in known.values())
    json.dumps(known)
