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
    written = set(data["forms"]) | set(data.get("aliases", {}))
    assert set(inventory.forms) == {ws._fold(f) for f in written}
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


def test_a_form_inside_a_hyphenated_word_is_not_the_form(inventory):
    """Found on the bench: "ileo-psoas" is the iliopsoas muscle, not the ileum or an ileus."""
    for mention in ("muscolo ileo-psoas di sinistra", "valvola ileo-cecale"):
        verdict = inventory.judge(mention, f"Ispessimento del {mention}.")
        assert verdict.accepted and not verdict.scores


def test_no_bench_mention_with_a_target_is_blocked_by_the_senses(inventory):
    """Regression guard: the inventory must not cost coverage on the written bench sets."""
    rows = []
    for name in ("heldout_it", "heldout_en", "heldout2_it", "heldout2_en"):
        path = DATA / f"{name}.jsonl"
        rows += [
            json.loads(line)
            for line in path.read_text("utf-8").splitlines()
            if line.strip()
        ]
    blocked = [
        r["mention"]
        for r in rows
        if r.get("target") and not inventory.judge(r["mention"], r["sentence"]).accepted
    ]
    assert blocked == []


# "paraspinal" is the muscle or the region beside the spine: the sentence decides, the word does not.
@pytest.mark.parametrize(
    ("mention", "sentence", "accepted"),
    [
        ("left paraspinal", "Fatty atrophy of the left paraspinal.", True),
        ("left paraspinal muscles", "Atrophy of the left paraspinal muscles.", True),
        (
            "muscoli paravertebrali",
            "Ipotrofia dei muscoli paravertebrali destri.",
            True,
        ),
        ("paravertebrale", "Contrattura della muscolatura paravertebrale.", True),
        (
            "left paraspinal",
            "Low left paraspinal/retrocrural adenopathy is present.",
            False,
        ),
        ("paravertebrale destro", "Massa paravertebrale destra con linfonodi.", False),
        # silence is not evidence for the muscle
        ("left paraspinal", "The left paraspinal is normal.", False),
    ],
)
def test_paraspinal_is_the_muscle_or_the_region_by_context(
    inventory, mention, sentence, accepted
):
    assert inventory.judge(mention, sentence).accepted is accepted


def test_the_linker_reads_paraspinal_adenopathy_as_a_region_and_atrophy_as_the_muscle():
    lexicon = al.Lexicon.from_json(
        json.loads((DATA / "anatomy_lexicon.json").read_text("utf-8"))
    )
    linker = al.AnatomyLinker(lexicon, [], {})
    region = linker.link(
        "left paraspinal", "Low left paraspinal/retrocrural adenopathy is present."
    )
    assert region.status == al.ABSTAINED and region.reason == "sense_conflict:region"
    muscle = linker.link("left paraspinal", "Fatty atrophy of the left paraspinal.")
    assert (muscle.status, muscle.cid) == (al.ACCEPTED, "autochthon_left")


def test_aliases_share_the_senses_of_their_form_and_must_name_a_known_one():
    inv = ws.SenseInventory.from_json({**TOY, "aliases": {"xyzs": "xyz"}})
    assert inv.judge("xyzs", "The xyzs biopsy shows tissue.").accepted
    with pytest.raises(ValueError, match="unknown form"):
        ws.SenseInventory.from_json({**TOY, "aliases": {"abc": "nope"}})


# --- the neighbour that turns a structure into a measurement ------------------------------------------


@pytest.mark.parametrize(
    ("mention", "sentence", "head"),
    [
        ("heart", "Her heart rate was 144 beats per minute.", "rate"),
        ("liver", "Liver function tests were normal.", "function"),
        ("thyroid", "Normal thyroid-stimulating hormone.", "stimulating"),
        ("thyroid", "Anti-thyroid peroxidase antibodies were raised.", "anti"),
        ("kidney", "Kidney function was preserved.", "function"),
        ("fegato", "Funzione del fegato nella norma.", ""),
        ("fegato", "Enzimi del fegato nella norma.", ""),
        ("fegato", "Fegato enzimi aumentati.", "enzimi"),
        ("liver", "The liver biopsy showed steatosis.", ""),
        ("heart", "She has heart failure.", ""),
        ("thyroid", "Thyroid, parathyroid and vitamin D were normal.", ""),
        ("liver", "Nothing in the liver. Function was not assessed.", ""),
    ],
)
def test_the_word_next_to_a_structure_can_make_it_a_measurement(
    mention, sentence, head
):
    assert ws.SenseInventory.load().attribute_head(mention, sentence) == head


def test_without_the_data_no_neighbour_counts():
    assert ws.SenseInventory.empty().attribute_head("heart", "heart rate 80") == ""


@pytest.mark.parametrize(
    "sentence",
    [
        "The parasternal short axis view demonstrated an enlarged pulmonary artery.",
        "The ECG showed a normal axis and sinus tachycardia.",
        "Refractive power was -1.50 Dcyl Axis 90 in the right eye.",
        "A mass at the 12:00 axis of the left breast.",
    ],
)
def test_axis_in_a_view_an_ecg_or_an_eye_is_not_the_second_vertebra(sentence):
    verdict = ws.SenseInventory.load().judge("axis", sentence)
    assert not verdict.accepted


@pytest.mark.parametrize(
    "sentence",
    [
        "Fracture of the axis with odontoid involvement.",
        "Dens fracture of the axis (C2) with cervical instability.",
    ],
)
def test_axis_with_the_vertebra_around_it_is_the_vertebra(sentence):
    verdict = ws.SenseInventory.load().judge("axis", sentence)
    assert verdict.accepted and verdict.classes == frozenset({"vertebrae_C2"})
