"""Italian is read with its own grammar and vocabulary (15 October 2026): the head of a phrase is on the left,
a closed-class word never types a block, Italian words come before NCIt's English ones, a structure
coordinated with a structure shares what governs it, and the predicate is never an adverb."""

import numpy as np

from melampo.memory.chunk_lattice import BlockMemory, ChunkLattice
from melampo.memory.grammar import Grammar

# "su" and "rx" are chemical / property abbreviations in the English vocabulary (NCIt), as in the real memory
MEMORY = BlockMemory.from_json(
    {"heads": {"su": ["chemical", 0.8, 5], "rx": ["property", 1.0, 5], "torace": ["anatomy", 1.0, 0],
               "addome": ["anatomy", 1.0, 0], "polmone": ["anatomy", 1.0, 0]}, "names": {}},
    None, ["torace", "addome", "polmone"],
)
MEMORY.italian.update({"radiografia": "procedure", "tac": "procedure", "rx": "procedure"})
LATTICE = ChunkLattice(MEMORY)
ITALIAN = LATTICE.in_language("it")


def test_english_is_read_as_before_and_italian_has_its_own_lattice():
    assert LATTICE.in_language("en") is LATTICE
    assert ITALIAN is not LATTICE and ITALIAN.grammar.language == "it"
    assert ITALIAN.in_language("it") is ITALIAN and LATTICE.in_language("it") is ITALIAN  # cached


def test_a_closed_class_word_never_types_a_block():
    assert LATTICE._head("su") == ("chemical", 0.8)      # the English reading of the abbreviation
    assert ITALIAN._head("su") is None                   # the Italian preposition


def test_italian_words_come_before_the_english_vocabulary():
    assert LATTICE._head("rx") == ("property", 1.0)
    assert ITALIAN._head("rx") == ("procedure", 1.0)


def test_the_word_before_the_mention_governs_in_italian():
    r = ITALIAN.read("torace", "La radiografia del torace mostrava un nodulo.")
    assert r.outcome == "role" and r.role == "procedure_site" and not r.exact


def test_a_structure_coordinated_with_a_structure_shares_its_governor():
    r = ITALIAN.read("addome", "La TAC torace ed addome con mdc è nella norma.")
    assert r.outcome == "role" and r.role == "procedure_site"
    plain = ITALIAN.read("addome", "Il torace ed addome sono nella norma.")
    assert plain.outcome != "role"


def test_a_preposition_before_the_mention_does_not_make_a_chemical():
    r = ITALIAN.read("polmone", "Lesioni su polmone destro.")
    assert r.outcome in ("link", "unread") and r.role == ""


def test_the_predicate_of_the_reader_is_not_an_adverb():
    g = Grammar.for_language("en")
    assert g.ends_phrase("approximately") and not g.ends_phrase("wedging")
    assert Grammar.for_language("it").ends_phrase("oltre")
