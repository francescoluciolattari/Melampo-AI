"""The chunk lattice reads a phrase as blocks before it decides what the mention is (offline, tiny memory)."""

import pytest

from melampo.memory.chunk_lattice import BlockMemory, ChunkLattice

MEMORY = BlockMemory.from_json(
    {
        "heads": {
            "filter": ["device", 1.0, 0],
            "placement": ["procedure", 0.79, 66],
            "transfusion": ["procedure", 1.0, 0],
            "weight": ["property", 1.0, 0],
            "removal": ["procedure", 0.9, 40],
            "cell": ["anatomy", 1.0, 0],
            "extract": ["chemical", 0.95, 30],
            "cheese": ["food", 1.0, 0],
            "microbiome": ["organism", 1.0, 13],
            "strain": ["organism", 1.0, 20],
        },
        "names": {"liver transplantation": "procedure"},
    },
    {"liver fatty acid binding protein": "protein"},
)
LATTICE = ChunkLattice(MEMORY)


def read(mention, sentence):
    return LATTICE.read(mention, sentence)


def test_a_structure_alone_is_a_link():
    r = read("liver", "The liver is enlarged.")
    assert r.outcome == "link" and r.role == ""


def test_the_block_closes_at_the_first_head_that_can_fuse():
    r = read(
        "inferior vena cava",
        "Office based inferior vena cava filter placement is safe.",
    )
    assert (r.outcome, r.role, r.kind) == ("role", "device_site", "device")
    assert r.block.endswith(
        "filter"
    )  # not "... filter placement": the node closes at "filter"


def test_a_measure_keeps_the_structure_with_a_role():
    r = read("Brain", "Brain weight and striatal volume were estimated.")
    assert (r.outcome, r.role) == ("role", "inherent_location")


def test_an_anatomical_word_does_not_close_the_block():
    r = read("spleen", "Spleen cells transfusion inhibited rejection.")
    assert (r.outcome, r.role) == ("role", "procedure_site")


def test_a_molecule_made_from_the_structure_records_its_source_and_does_not_veto():
    r = read("liver", "Liver extracts were prepared.")
    assert (r.outcome, r.role) == ("role", "source_of")


def test_a_remembered_name_of_a_molecule_is_not_a_structure():
    r = read("Liver", "Liver fatty acid binding protein was measured.")
    assert (r.outcome, r.role, r.source) == ("not_a_site", "inside_a_name", "memory")


def test_part_of_an_object_with_of():
    r = read("heart", "Samples from rind and heart of Maroilles cheese were used.")
    assert (r.outcome, r.why) == ("not_a_site", "part_of_an_object:cheese")


def test_a_procedure_the_structure_is_the_object_of():
    r = read("gallbladder", "Removal of the gallbladder was uneventful.")
    assert (r.outcome, r.role, r.why) == ("role", "procedure_site", "object_of:removal")


def test_a_coarse_kind_acts_only_through_memory_or_of():
    # NCIt files nutrients under "food" and "Whole ..." under "organism": no veto by composition.
    for sentence in (
        "The colon microbiome changed.",
        "Low brain strain differences were seen.",
    ):
        mention = "colon" if "colon" in sentence else "brain"
        assert read(mention, sentence).outcome in ("unread", "role")
    assert read("colon", "The colon microbiome changed.").outcome == "unread"


def test_punctuation_ends_the_block():
    r = read("heart", "1, heart; weight was not measured.")
    assert r.outcome == "link"


def test_the_mention_is_never_split():
    r = read("inferior vena cava", "inferior vena cava filter placement")
    assert r.block.startswith("inferior vena cava")


def test_unknown_words_leave_the_link_alone():
    assert read("liver", "The liver zorbble was seen.").outcome == "link"


@pytest.mark.parametrize("sentence", ["", "x"])
def test_a_mention_that_is_not_in_the_sentence_is_unread(sentence):
    assert read("liver", sentence).outcome == "unread"


def test_the_linker_writes_the_reading_in_the_trace_and_decides_nothing_by_it():
    import json
    from pathlib import Path

    from melampo.memory import anatomy_linker as al
    from melampo.memory import anatomy_parts as ap

    data = Path(__file__).resolve().parents[1] / "data" / "linking"
    lexicon = al.Lexicon.from_json(
        json.loads((data / "anatomy_lexicon.json").read_text("utf-8"))
    )
    parts = ap.PartTable.from_json(
        json.loads((data / "anatomy_parts.json").read_text("utf-8")), lexicon
    )
    sentence = "Brain weight and striatal volume were estimated."
    plain = al.AnatomyLinker(lexicon, [], {}, parts=parts).link("brain", sentence)
    with_blocks = al.AnatomyLinker(
        lexicon, [], {}, parts=parts, chunk_lattice=LATTICE
    ).link("brain", sentence)
    assert (plain.status, plain.cid, plain.reason) == (
        with_blocks.status,
        with_blocks.cid,
        with_blocks.reason,
    )
    assert plain.block == "" and with_blocks.block.startswith(
        "role:inherent_location:brain weight"
    )
