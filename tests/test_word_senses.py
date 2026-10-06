import json
from pathlib import Path

import pytest

from melampo.memory import anatomy_linker as al
from melampo.memory import word_senses as ws

DATA = Path(__file__).resolve().parent.parent / "data" / "linking"


@pytest.fixture(scope="module")
def inventory():
    return ws.SenseInventory.load(DATA / "word_senses.json")


# A form that appears nowhere in the shipped data: the mechanism needs no code for a new word.
TOY = {
    "forms": {
        "xyz": {
            "organ": {
                "anatomical": True,
                "languages": ["en"],
                "strong": ["biopsy"],
                "weak": ["tissue"],
            },
            "unit": {
                "anatomical": False,
                "strong": ["kilo"],
                "patterns": [r"\bxyz\s*\d"],
            },
        }
    }
}


def test_a_new_form_is_data_not_code():
    inv = ws.SenseInventory.from_json(TOY)
    assert inv.judge("xyz", "The xyz biopsy shows tissue.").accepted
    verdict = inv.judge("xyz", "The xyz 12 kilo biopsy.")
    assert not verdict.accepted and verdict.reason == "sense_conflict:unit"


def test_silence_is_not_evidence_for_the_anatomical_sense():
    inv = ws.SenseInventory.from_json(TOY)
    verdict = inv.judge("xyz", "xyz is normal.")
    assert not verdict.accepted and verdict.reason == "sense_unresolved"


def test_language_is_a_discriminant_that_strong_evidence_can_outweigh():
    inv = ws.SenseInventory.from_json(TOY)
    # Italian text: the sense attested only in English loses weight, so one cue is not enough...
    assert not inv.judge("xyz", "Il xyz mostra tissue nella sede.").accepted
    assert not inv.judge("xyz", "Il xyz mostra biopsy nella sede.").accepted
    # ...but strong evidence together with a second cue outweighs it.
    assert inv.judge("xyz", "Il xyz mostra biopsy e tissue nella sede.").accepted
    # In English the same weak cue plus the language is enough.
    assert inv.judge("xyz", "The xyz shows tissue.").accepted


def test_wider_context_counts_for_half():
    inv = ws.SenseInventory.from_json(TOY)
    assert not inv.judge("xyz", "xyz: normal.", "biopsy").accepted  # 1.5 < 2
    assert inv.judge("xyz", "xyz: normal.", "biopsy of tissue").accepted  # 2.0


def test_a_mention_without_a_listed_form_is_not_checked(inventory):
    verdict = inventory.judge("fegato", "Il fegato è regolare.")
    assert verdict.accepted and not verdict.scores


def test_every_form_has_an_anatomical_sense_and_a_rival():
    bad = {"forms": {"abc": {"only": {"anatomical": True}}}}
    with pytest.raises(ValueError, match="needs an anatomical sense"):
        ws.SenseInventory.from_json(bad)


def test_the_shipped_inventory_loads_and_every_cue_is_folded(inventory):
    data = json.loads((DATA / "word_senses.json").read_text("utf-8"))
    assert set(inventory.forms) == {ws._fold(f) for f in data["forms"]}
    for senses in inventory.forms.values():
        for sense in senses:
            for cue in sense.strong | sense.weak:
                assert " " not in cue, f"multi-word cue never matches: {cue!r}"
                assert cue == ws._fold(cue)


@pytest.mark.parametrize(
    ("mention", "sentence", "accepted"),
    [
        ("ponte", "Lesione ischemica del ponte e del cervelletto.", True),
        ("ponte", "Ponti aortocoronarici pervi, bypass regolare.", False),
        ("ponte", "Il ponte dentale è in sede.", False),
        ("ileo", "Anse dilatate: ileo paralitico.", False),
        ("ileo", "Ispessimento dell'ileo terminale, morbo di Crohn.", True),
        ("digiuno", "Glicemia a digiuno nei limiti.", False),
        ("digiuno", "Anse del digiuno e dell'ileo senza ispessimenti.", True),
        ("midollo", "Infiltrazione del midollo osseo da mieloma.", False),
        ("midollo", "Compressione del midollo cervicale con mielopatia.", True),
        ("LM", "Stenosi critica del LM e della IVA.", False),
        ("LM", "Atelettasia del LM al polmone destro.", True),
        ("LM", "Torace: calcificazioni del LM.", False),
    ],
)
def test_shipped_senses(inventory, mention, sentence, accepted):
    verdict = inventory.judge(mention, sentence)
    assert verdict.accepted is accepted, (verdict.reason, verdict.scores)


def test_the_reason_names_the_competing_sense(inventory):
    verdict = inventory.judge("GB", "Hb 12,3 g/dL; GB 5040/mmc (N 48%).")
    assert verdict.reason == "sense_conflict:white_blood_cells"


def test_gigabyte_is_a_sense_too(inventory):
    verdict = inventory.judge("GB", "Il file DICOM pesa 5 GB.")
    assert not verdict.accepted and verdict.reason == "sense_conflict:gigabyte"


def test_language_detection_is_cautious():
    assert ws.detect_language("Il fegato è di dimensioni regolari.") == "it"
    assert ws.detect_language("The liver is of normal size.") == "en"
    assert ws.detect_language("GB: normal.") is None
    assert ws.detect_language("Hb 12,3 g/dL; GB 5040/mmc") is None


def test_the_linker_uses_the_senses_and_the_context(lexicon_and_pool):
    lexicon, pool, equivalent = lexicon_and_pool
    linker = al.AnatomyLinker(lexicon, pool, equivalent)
    blocked = linker.link("GB", "GB 5040/mmc.")
    assert blocked.status == al.ABSTAINED and blocked.stage == "senses"
    assert blocked.reason.startswith("sense_conflict")
    ok = linker.link("GB", "GB wall thickening with gallstones.")
    assert ok.cid == "gallbladder"
    # Without the inventory the old silent error comes back: that is what the inventory is for.
    bare = al.AnatomyLinker(lexicon, pool, equivalent, senses=ws.SenseInventory.empty())
    assert bare.link("GB", "GB 5040/mmc.").cid == "gallbladder"


def test_a_sense_that_limits_classes_blocks_another_class(lexicon_and_pool):
    lexicon, pool, equivalent = lexicon_and_pool
    narrow = ws.SenseInventory.from_json(
        {
            "forms": {
                "colecisti": {
                    "organ": {
                        "anatomical": True,
                        "classes": ["liver"],
                        "strong": ["calcoli"],
                    },
                    "other": {"anatomical": False, "strong": ["kilo"]},
                }
            }
        }
    )
    linker = al.AnatomyLinker(lexicon, pool, equivalent, senses=narrow)
    result = linker.link("colecisti", "Calcoli nella colecisti.")
    assert result.status == al.ABSTAINED
    assert result.reason == "sense_organ_names_another_class"


@pytest.fixture(scope="module")
def lexicon_and_pool():
    lexicon = al.Lexicon.from_json(
        json.loads((DATA / "anatomy_lexicon.json").read_text("utf-8"))
    )
    pool, equivalent = al.build_pool(lexicon)
    return lexicon, pool, equivalent
