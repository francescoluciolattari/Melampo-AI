"""Grammar filter, UMLS-typed heads and the left-headed (Italian) profile of the chunk lattice."""

from melampo.memory.chunk_lattice import BlockMemory, ChunkLattice
from melampo.memory.grammar import Grammar
from melampo.memory.head_typing import UmlsHeadTyper, kind_of_tuis

EN = Grammar.for_language("en")
IT = Grammar.for_language("it")


def test_english_filter_drops_verbs_and_adverbs_but_keeps_nouns_that_look_like_them():
    for word in ("removed", "was", "markedly", "shown", "performed"):
        assert EN.ends_phrase(word), word
    for word in ("rate", "donation", "family", "anomaly", "bed", "hundred", "imaging", "swelling"):
        assert not EN.ends_phrase(word), word


def test_italian_filter_is_only_closed_class_and_adverbs():
    for word in ("del", "della", "è", "marcatamente"):
        assert IT.ends_phrase(word), word
    for word in ("tessuto", "tratto", "stato", "frequenza", "polmonare"):
        assert not IT.ends_phrase(word), word


def test_of_patterns_per_language():
    assert EN.of_after.match(" of the heart") and not EN.of_after.match(" del cuore")
    assert IT.of_after.match(" del cuore") and IT.of_after.match(" di maiale") and IT.of_after.match(" dell'omero")
    assert IT.of_before.search("frequenza del ") and EN.of_before.search("rate of the ")


def test_tui_groups_give_the_lattice_kinds():
    assert kind_of_tuis(["T201"]) == "property"
    assert kind_of_tuis(["T058", "T201"]) == "procedure"  # first matching group wins
    assert kind_of_tuis(["T999"]) == ""


def test_the_typer_gives_kind_and_share_and_never_guesses():
    answers = {
        "rate": [{"types": ["T201"]}, {"types": ["T201"]}, {"types": ["T201"]}, {"types": ["T058"]}],
        "mystery": [{"types": ["T999"]}],
        "gone": None,
    }
    typer = UmlsHeadTyper(answers.get)
    assert typer("rate") == ("property", 0.75)
    assert typer("mystery") is None  # a concept of no useful kind says nothing
    assert typer("gone") is None and typer.lost == 1
    assert typer("rate") == ("property", 0.75) and typer.asked == 3  # the first answer is kept


MEMORY = BlockMemory.from_json({"heads": {"filter": ["device", 1.0, 0]}, "names": {}})


def test_a_head_the_memory_does_not_know_is_typed_from_umls():
    plain = ChunkLattice(MEMORY)
    assert plain.read("liver", "Liver donation is safe.").outcome == "link"  # "donation" is unknown: nothing to act on
    typer = UmlsHeadTyper({"donation": [{"types": ["T058"]}] * 4}.get)
    typed = ChunkLattice(MEMORY, typer=typer)
    r = typed.read("liver", "Liver donation is safe.")
    assert r.outcome == "role" and r.role == "procedure_site"


def test_an_uncertain_umls_type_is_not_acted_on():
    mixed = [{"types": ["T058"]}, {"types": ["T201"]}]  # procedure or property: share 0.5
    r = ChunkLattice(MEMORY, typer=UmlsHeadTyper({"donation": mixed}.get)).read("liver", "Liver donation is safe.")
    assert r.outcome == "link"  # unchanged: nothing is acted on


def test_the_grammar_stops_the_phrase_at_a_verb_form():
    typer = UmlsHeadTyper({"removed": [{"types": ["T058"]}] * 4}.get)
    sentence = "The liver removed from the donor was weighed."
    without = ChunkLattice(MEMORY, typer=typer).read("liver", sentence)
    assert without.outcome == "role"  # "liver removed" was read as a procedure
    with_grammar = ChunkLattice(MEMORY, grammar=EN, typer=typer).read("liver", sentence)
    assert with_grammar.outcome == "link"


IT_MEMORY = BlockMemory.from_json({"heads": {"frequenza": ["property", 1.0, 0], "prelievo": ["procedure", 1.0, 0]},
                                   "names": {"cuore artificiale": "device"}})
ITL = ChunkLattice(IT_MEMORY, grammar=IT)


def test_italian_head_is_on_the_left():
    r = ITL.read("cardiaca", "La frequenza cardiaca è normale.")
    assert r.outcome == "role" and r.role == "inherent_location"
    # the same words read as English (right-headed) do not give the property
    assert ChunkLattice(IT_MEMORY).read("cardiaca", "La frequenza cardiaca è normale.").outcome != "role"


def test_italian_of_constructions_and_remembered_names():
    r = ITL.read("fegato", "Il prelievo del fegato è stato eseguito.")
    assert r.outcome == "role" and r.role == "procedure_site" and r.source == "of"
    assert ITL.read("cuore", "Impianto di cuore artificiale.").outcome in ("not_a_site", "role")
