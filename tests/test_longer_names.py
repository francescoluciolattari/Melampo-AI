"""The known name of another kind of thing that contains a structure's name (8-9 October 2026)."""

import importlib.util
import json
from pathlib import Path

import pytest

from melampo.memory import anatomy_linker as al
from melampo.memory import anatomy_parts as ap
from melampo.memory.anatomy_graph import AnatomyGraph
from melampo.memory.longer_names import LongerNames

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "linking"
UBERON = DATA / "uberon-basic.obo"

NAMES = LongerNames.from_json(
    {
        "names": {
            "international prostate symptom score": "assessment_tool",
            "liver fatty acid binding protein": "protein",
            "vena cava filter": "chemical",
        }
    }
)


@pytest.mark.parametrize(
    "mention,sentence,expected",
    [
        (
            "prostate",
            "Mean International Prostate Symptom Score was lower.",
            ("international prostate symptom score", "assessment_tool"),
        ),
        (
            "Liver",
            "Liver Fatty Acid Binding Protein Deficiency Provokes Stress.",
            ("liver fatty acid binding protein", "protein"),
        ),
        # the hyphen is a word boundary, as in the names of the ontologies
        (
            "liver",
            "decreased liver fatty-acid-binding protein levels",
            ("liver fatty acid binding protein", "protein"),
        ),
        # the words must stand in the sentence in the order of the name, all of them
        ("prostate", "The prostate symptom score was lower.", ("", "")),
        ("liver", "The liver binding protein was seen.", ("", "")),
        # a mention that is no word of any listed name
        ("prostate", "Prostate cancer and the symptom score.", ("", "")),
        ("heart", "", ("", "")),
    ],
)
def test_the_listed_name_is_found_only_when_the_sentence_writes_it(
    mention, sentence, expected
):
    assert NAMES.containing(mention, sentence) == expected


def test_an_empty_list_decides_nothing():
    assert LongerNames().containing("prostate", "International Prostate Symptom Score") == (
        "",
        "",
    )


def test_the_second_occurrence_is_judged_on_its_own_words():
    sentence = "The prostate was normal. International Prostate Symptom Score was 8."
    start = sentence.index("Prostate")
    assert NAMES.containing("Prostate", sentence, start)[1] == "assessment_tool"
    assert NAMES.containing("prostate", sentence, sentence.index("prostate"))[1] == ""


def test_the_shipped_file_has_provenance_and_the_four_kinds():
    data = json.loads((DATA / "longer_names.json").read_text("utf-8"))
    assert data["sources"]["ncit"]["data_version"]
    assert len(data["sources"]["ncit"]["sha256"]) == 64
    assert set(data["names"].values()) <= {"protein", "gene", "chemical", "assessment_tool"}
    assert len(data["names"]) > 1000
    # names of anatomy, disease and procedure are not in it
    for name in ("prostate cancer", "mouse brain", "liver biopsy", "liver transplantation"):
        assert name not in data["names"]


# -- the builder ---------------------------------------------------------------------------------


def _builder():
    spec = importlib.util.spec_from_file_location(
        "build_longer_names", ROOT / "scripts" / "build_longer_names.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_OBO = """format-version: 1.2
data-version: releases/test

[Term]
id: NCIT:C17021
name: Protein

[Term]
id: NCIT:C20993
name: Research or Clinical Assessment Tool

[Term]
id: NCIT:C20994
name: Clinical or Research Assessment Question
is_a: NCIT:C20993

[Term]
id: NCIT:C1
name: Liver Fatty Acid Binding Protein
synonym: "L-FABP liver type" EXACT []
is_a: NCIT:C17021

[Term]
id: NCIT:C2
name: International Prostate Symptom Score
is_a: NCIT:C20993

[Term]
id: NCIT:C3
name: Have Shoulder or Arm Pain
is_a: NCIT:C20994

[Term]
id: NCIT:C4
name: Age at Diagnosis of Congenital Heart Disease
is_a: NCIT:C20993

[Term]
id: NCIT:C5
name: CDISC Questionnaire Liver Disease Score Terminology
is_a: NCIT:C20993

[Term]
id: NCIT:C6
name: Obsolete Liver Protein
is_obsolete: true
is_a: NCIT:C17021

[Term]
id: NCIT:C7
name: Liver Cancer
"""


def test_the_builder_keeps_instruments_and_proteins_and_drops_the_rest(tmp_path):
    builder = _builder()
    obo = tmp_path / "ncit.obo"
    obo.write_text(_OBO, "utf-8")
    runs = {("liver",), ("prostate",), ("shoulder",), ("heart",)}
    names, counts = builder.build([("ncit", builder.ncit_names(obo, runs))], runs)
    assert names == {
        "liver fatty acid binding protein": "protein",
        "international prostate symptom score": "assessment_tool",
        "l fabp liver type": "protein",
    }
    assert counts["ncit:protein"] == 2


def test_the_builder_strips_the_organism_of_a_protein_ontology_name(tmp_path):
    builder = _builder()
    obo = tmp_path / "pr.obo"
    obo.write_text(
        "format-version: 1.2\n\n[Term]\nid: PR:1\nname: liver fatty acid-binding protein (mouse)\n",
        "utf-8",
    )
    runs = {("liver",)}
    names, _ = builder.build([("pr", builder.pr_names(obo, runs))], runs)
    assert names == {"liver fatty acid binding protein": "protein"}


# -- the linker ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def linker():
    lexicon = al.Lexicon.from_json(
        json.loads((DATA / "anatomy_lexicon.json").read_text("utf-8"))
    )
    parts = ap.PartTable.from_json(
        json.loads((DATA / "anatomy_parts.json").read_text("utf-8")), lexicon
    )
    with open(UBERON, encoding="utf-8") as handle:
        terms = al.load_obo_terms(handle)
    with open(UBERON, encoding="utf-8") as handle:
        graph = AnatomyGraph.from_obo(handle)
    pool, equivalent = al.build_pool(lexicon, terms)
    return al.AnatomyLinker(lexicon, pool, equivalent, parts=parts, graph=graph)


@pytest.mark.skipif(not UBERON.exists(), reason="UBERON file missing")
@pytest.mark.parametrize(
    "mention,sentence,kind",
    [
        (
            "prostate",
            "Mean international prostate symptom score in patients with prostatitis was lower.",
            "assessment_tool",
        ),
        (
            "Liver",
            "Liver Fatty Acid Binding Protein Deficiency Provokes Oxidative Stress.",
            "protein",
        ),
    ],
)
def test_the_linker_abstains_inside_the_name_of_another_thing(
    linker, mention, sentence, kind
):
    result = linker.link(mention, sentence)
    assert result.status == al.ABSTAINED
    assert result.reason.startswith(f"mention_is_inside_the_name_of_another_thing:{kind}:")


@pytest.mark.skipif(not UBERON.exists(), reason="UBERON file missing")
@pytest.mark.parametrize(
    "mention,sentence,cid",
    [
        ("prostate", "Prostate cancer was found on biopsy.", "prostate"),
        ("liver", "Liver gene expression was measured in mice.", "liver"),
        ("liver", "The liver is enlarged with a 3 cm lesion.", "liver"),
        ("brain", "The mouse brain was fixed.", "brain"),
    ],
)
def test_the_linker_keeps_the_link_for_names_of_anatomy_and_disease(
    linker, mention, sentence, cid
):
    result = linker.link(mention, sentence)
    assert (result.status, result.cid) == (al.ACCEPTED, cid)


def test_a_name_is_written_in_one_run_not_across_a_list():
    names = LongerNames(
        {"brain protein 3": "protein", "liver fatty acid binding protein": "protein"}
    )
    listing = "Tissues: 1, heart; 2, brain; 3, spleen; 4, lung."
    assert names.containing("brain", listing, listing.index("brain"))[1] == ""
    text = "Brain protein 3 was lower."
    assert names.containing("Brain", text, 0)[1] == "protein"
    listed = "1, brain; protein 3 was lower."
    assert names.containing("brain", listed, 3)[1] == ""
    sentence = "Liver fatty acid-binding protein was measured."
    assert names.containing("Liver", sentence, 0)[1] == "protein"
    broken = "Liver fatty acid, binding protein was measured."
    assert names.containing("Liver", broken, 0)[1] == ""


def test_a_name_that_is_a_structure_and_a_number_is_not_enough():
    names = LongerNames({"rib 1": "protein", "brain gst": "protein"})
    sentence = "Fracture of rib 1 on the right."
    assert names.containing("rib", sentence, sentence.index("rib"))[1] == ""
    assert names.containing("brain", "brain GST was measured", 0)[1] == "protein"
