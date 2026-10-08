"""A noun that names a process or a capacity of the structure ("heart development"), 8 October 2026.

The structure bears the process (SNOMED CT "inheres in", GO "has participant"); the name of the
process is not the name of the structure. The linker does not link and records the structure as
the inherent location, as for a measure. The same for every structure and in both languages.
"""

import json
from pathlib import Path

import pytest

from melampo.memory import anatomy_linker as al
from melampo.memory import anatomy_parts as ap
from melampo.memory.anatomy_graph import AnatomyGraph
from melampo.memory.word_senses import SenseInventory

DATA = Path(__file__).resolve().parents[1] / "data" / "linking"
UBERON = DATA / "uberon-basic.obo"
SENSES = SenseInventory.load()


@pytest.mark.parametrize(
    "mention,sentence,head",
    [
        # the process noun is the whole phrase to the right
        ("heart", "Slit signalling has a role in heart development.", "development"),
        ("liver", "Genes active during postnatal liver development.", "development"),
        ("brain", "Changes in brain state dynamics were measured.", "dynamics"),
        ("brain", "The brain connectivity patterns were altered.", "connectivity"),
        ("colon", "Colon motility was measured with MRI.", "motility"),
        # the process noun before the structure, with a preposition, and the structure ends the phrase
        ("pancreas", "The development of the pancreas was studied.", "development"),
        ("heart", "Autonomic control of the heart was impaired.", "control"),
        ("cuore", "Si studia lo sviluppo del cuore.", "sviluppo"),
        ("cuore", "Si descrive la regolazione del cuore.", "regolazione"),
    ],
)
def test_the_process_noun_is_found(mention, sentence, head):
    assert SENSES.process_head(mention, sentence) == head


@pytest.mark.parametrize(
    "mention,sentence",
    [
        # a word between the structure and the process noun: the process is that thing's
        ("brain", "Brain tumour growth was rapid."),
        ("heart", "Heart development disorders are common."),
        # the phrase goes on after the structure: the process is the disease's
        ("left lower lobe", "New development of left lower lobe airspace disease."),
        # the findings of a report: a measure or a quality of the structure, not a process
        ("heart", "The heart size is normal."),
        ("lung", "The lung volumes are low."),
        ("liver", "The liver is enlarged."),
        # an Italian word that is a follow-up examination or a search, not a process
        ("fegato", "Ecografia di controllo del fegato."),
        ("fegato", "Ricerca di lesioni del fegato."),
        ("liver", ""),
    ],
)
def test_the_construction_that_is_not_a_process_is_left_alone(mention, sentence):
    assert SENSES.process_head(mention, sentence) == ""


def test_the_inventory_without_the_list_decides_nothing():
    assert SenseInventory.empty().process_head("heart", "heart development") == ""


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
def test_the_linker_records_the_structure_instead_of_linking(linker):
    result = linker.link("heart", "Slit signalling has a role in heart development.")
    assert result.status == al.ABSTAINED
    assert result.reason == "process_head_names_a_process:development"
    assert (result.role, result.about) == ("inherent_location", "heart")


@pytest.mark.skipif(not UBERON.exists(), reason="UBERON file missing")
@pytest.mark.parametrize(
    "mention,sentence,cid",
    [
        ("heart", "The heart size is normal.", "heart"),
        ("liver", "The liver is enlarged with a 3 cm lesion.", "liver"),
        ("brain", "Brain tumour growth was rapid.", "brain"),
    ],
)
def test_the_ordinary_link_is_kept(linker, mention, sentence, cid):
    result = linker.link(mention, sentence)
    assert (result.status, result.cid) == (al.ACCEPTED, cid)
