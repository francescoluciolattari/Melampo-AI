"""A structure's name inside a longer name that is not the structure's (8 October 2026).

Rules read from CRAFT and MedMentions errors: the hyphen compound, the abbreviation the text
defines, the material of a graft, the longer names of the ontology (synonyms) and the evidence of a
developing organism. Each one is the same for every structure.
"""

import json
from pathlib import Path

import pytest

from melampo.memory import anatomy_linker as al
from melampo.memory import anatomy_parts as ap
from melampo.memory.anatomy_graph import AnatomyGraph
from melampo.memory.compounds import (
    defined_abbreviation,
    defined_short_form,
    definitions,
    hyphen_compound,
)
from melampo.memory.development import DevelopmentalFrame
from melampo.memory.word_senses import SenseInventory

DATA = Path(__file__).resolve().parents[1] / "data" / "linking"
UBERON = DATA / "uberon-basic.obo"


HEADS = SenseInventory.load().non_site_heads


# -- hyphen compound ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mention,sentence,joined",
    [
        ("brain", "The role of the gut-brain axis in addiction.", "gut"),
        ("liver", "The gut-liver axis model explains this.", "gut"),
        ("brain", "N-terminal pro-brain natriuretic peptide.", "pro"),
        ("heart", "A heart-lung machine was used.", "lung"),
        # where / which side / how many: the structure is still the structure
        ("liver", "An intra-hepatic liver lesion.", ""),
        ("kidney", "A right-sided kidney.", ""),
        ("liver", "Lesion of C5-C6 and the liver.", ""),
        ("liver", "Normal liver - no focal lesion.", ""),  # a spaced dash is punctuation
    ],
)
def test_hyphen_compound(mention, sentence, joined):
    assert hyphen_compound(mention, sentence) == joined


# -- abbreviation defined in the text (Schwartz and Hearst) --------------------------------------


def test_the_long_form_is_found_from_the_letters_of_the_abbreviation():
    found = {d.short: d.long for d in definitions(
        "levels of liver fatty acid binding protein (L-FABP) and its target"
    )}
    assert found == {"L-FABP": "liver fatty acid binding protein"}
    found = {d.short: d.long for d in definitions(
        "and N-terminal pro-brain natriuretic peptide [NT-proBNP], cardiac troponins"
    )}
    assert found["NT-proBNP"] == "N-terminal pro-brain natriuretic peptide"


def test_a_bracket_that_is_not_an_abbreviation_defines_nothing():
    assert definitions("The liver (see Figure 2) is normal.") == []
    assert definitions("Dose 5 mg (twice daily).") == []


@pytest.mark.parametrize(
    "mention,sentence,short",
    [
        (
            "spinal cord",
            "TWIK-related spinal cord K(+) (TRESK) channel are members of the family.",
            "TRESK",
        ),
        (
            "liver",
            "decreased the expression of liver fatty acid binding protein (L-FABP) and",
            "L-FABP",
        ),
        (
            "spleen",
            "Donor-specific spleen cell transfusion (DST) alone failed.",
            "DST",
        ),
        (
            "brainstem",
            "recording of auditory brainstem response (ABR) and distortion product",
            "ABR",
        ),
        # the long form is the mention itself: the abbreviation of the structure
        ("inferior vena cava", "evaluate inferior vena cava (IVC) filter placement", ""),
        # the long form is a disease, a library, a locus: the head does not make it another thing
        ("heart", "For instance, congenital heart disease (CHD) typically consists of", ""),
        ("prostate", "benign prostatic hyperplasia of the prostate (BPH) and cancer", ""),
        ("brain", "part of the Mouse Brain Library (MBL).", ""),
        ("left kidney", "the left kidney (LK) was normal", ""),
        ("liver", "The liver (see Figure 2) is normal.", ""),
    ],
)
def test_a_part_of_a_defined_long_form_is_not_the_structure(mention, sentence, short):
    assert defined_abbreviation(mention, sentence, heads=HEADS) == short


def test_the_abbreviation_itself_has_the_meaning_the_text_gives_it():
    sentence = 'and the " drug effects on the nervous system" (DENS) scale.'
    assert (
        defined_short_form("DENS", sentence, heads=HEADS)
        == "drug effects on the nervous system"
    )
    assert defined_short_form("DENS", "The DENS is intact.", heads=HEADS) == ""
    # an abbreviation of a structure keeps its meaning
    assert (
        defined_short_form("CHD", "congenital heart disease (CHD) is common", heads=HEADS)
        == ""
    )


# -- material of a graft -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "mention,sentence,qualifier",
    [
        ("brain", "between homologous brain regions", ""),  # not a graft: the phrase goes on
        (
            "costal cartilage",
            "Autologous vs Irradiated Homologous Costal Cartilage as Graft Material",
            "homologous",
        ),
        ("bone", "an autologous bone graft", "autologous"),
        ("spleen", "donor-specific spleen cell transfusion", "donor specific"),
        # a site, not a material
        ("liver", "the transplanted liver", ""),
        ("lung", "the irradiated lung", ""),
        ("liver", "the liver of the donor", ""),
    ],
)
def test_material_qualifier(mention, sentence, qualifier):
    assert SenseInventory.load().material_qualifier(mention, sentence) == qualifier


# -- the linker --------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def linker():
    lexicon = al.Lexicon.from_json(
        json.loads((DATA / "anatomy_lexicon.json").read_text("utf-8"))
    )
    parts = ap.PartTable.from_json(
        json.loads((DATA / "anatomy_parts.json").read_text("utf-8")), lexicon
    )
    if not UBERON.exists():
        return al.AnatomyLinker(lexicon, [], {}, parts=parts)
    with open(UBERON, encoding="utf-8") as handle:
        terms = al.load_obo_terms(handle)
    with open(UBERON, encoding="utf-8") as handle:
        graph = AnatomyGraph.from_obo(handle)
    pool, equivalent = al.build_pool(lexicon, terms)
    return al.AnatomyLinker(lexicon, pool, equivalent, parts=parts, graph=graph)


@pytest.mark.parametrize(
    "mention,sentence,reason",
    [
        (
            "brain",
            "Targeting the gut-brain axis in drug addiction.",
            "mention_is_joined_by_a_hyphen_to:gut:axis",
        ),
        (
            "liver",
            "Decreased liver fatty acid binding protein (L-FABP) in the cells.",
            "mention_is_a_word_of_the_name_defined_as:L-FABP",
        ),
        (
            "costal cartilage",
            "Autologous costal cartilage was used in rhinoplasty.",
            "tissue_taken_as_graft_material:autologous",
        ),
        (
            "DENS",
            'and the " drug effects on the nervous system" (DENS) scale.',
            "abbreviation_defined_in_the_text_as:drug effects on the nervous system",
        ),
    ],
)
def test_the_linker_abstains_with_the_reason(linker, mention, sentence, reason):
    result = linker.link(mention, sentence)
    assert (result.status, result.reason) == (al.ABSTAINED, reason)


@pytest.mark.parametrize(
    "mention,sentence,cid",
    [
        ("liver", "Fetal-liver-derived macrophages were cultured.", None),
        ("liver", "Sections at head-neck, neck-liver and liver-kidney levels.", None),
        ("heart", "Congenital heart disease (CHD) is common in children.", "heart"),
        ("liver", "The liver is enlarged with a 3 cm lesion.", "liver"),
        ("liver", "Intra-hepatic bile ducts are not dilated; the liver is normal.", "liver"),
        ("inferior vena cava", "The inferior vena cava (IVC) is patent.", "inferior_vena_cava"),
        ("kidney", "A cyst in the right-sided kidney.", None),
    ],
)
def test_the_linker_keeps_the_ordinary_link(linker, mention, sentence, cid):
    result = linker.link(mention, sentence)
    if cid:
        assert (result.status, result.cid) == (al.ACCEPTED, cid)
    else:
        # none of the new checks may be what stops it
        assert not result.reason.startswith(
            (
                "mention_is_joined_by_a_hyphen_to",
                "mention_is_a_word_of_the_name_defined_as",
                "abbreviation_defined_in_the_text_as",
                "tissue_taken_as_graft_material",
            )
        )


@pytest.mark.skipif(not UBERON.exists(), reason="UBERON file missing")
@pytest.mark.parametrize(
    "sentence",
    [
        "Mice deficient in TGF-b2 display fourth aortic arch artery defects.",
        "Cells contribute to the smooth muscle of the aortic arch arteries.",
    ],
)
def test_a_synonym_of_another_structure_is_a_longer_name(linker, sentence):
    """UBERON lists "aortic arch artery" as a name of the pharyngeal arch artery."""
    result = linker.link("aortic arch", sentence)
    assert result.status == al.ABSTAINED
    assert result.reason.startswith("mention_is_inside_a_longer_name")
    assert linker.link("aortic arch", "Calcified plaque in the aortic arch.").cid == "aorta"


# -- developing organism ------------------------------------------------------------------------


def test_the_stage_words_are_a_closed_class():
    frame = DevelopmentalFrame.load()
    assert frame.terms("Embryos at E12.5 and fetal heart") == {"embryos", "e12.5", "fetal"}
    assert frame.terms("A 62 year old man with chest pain") == set()
    assert frame.developmental("The aortic arch is normal.")[0] is False
    assert frame.developmental("E17 embryos were analysed.")[0] is True
    # a document of embryos counts even when the sentence says nothing
    document = "Embryos at E10.5 and E12.5 were fixed; fetal tissue and somites."
    assert frame.developmental("The aortic arch is left-sided.", document)[0] is True


@pytest.mark.skipif(not UBERON.exists(), reason="UBERON file missing")
def test_a_name_shared_with_a_developing_structure_is_a_conflict_not_a_veto(linker):
    sentence = "In E17 embryos the aortic arch is left-sided."
    result = linker.link("aortic arch", sentence)
    assert (result.status, result.cid) == (al.ACCEPTED, "aorta")
    assert "name_shared_with_a_developing_structure_in_a_developmental_text" in result.conflicts
    adult = linker.link("aortic arch", "The aortic arch is left-sided.")
    assert adult.status == al.ACCEPTED
    assert not adult.conflicts or "name_shared_with_a_developing_structure_in_a_developmental_text" not in adult.conflicts
