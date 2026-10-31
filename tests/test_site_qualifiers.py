"""What the words around a structure say about it: location, degree and character, certainty (15 Oct 2026)."""

import pytest

from melampo.memory.site_qualifiers import read_site_qualifiers


def q(sentence, mention, language="en"):
    at = sentence.index(mention)
    return read_site_qualifiers(sentence, at, at + len(mention), language)


def test_an_adverb_of_approximation_makes_the_site_approximate():
    r = q("There is mild anterior wedging of a lower thoracic vertebral body,\napproximately T11 level.", "T11 level")
    assert r.location == "approximate" and r.location_cues[0].word == "approximately"
    assert r.degree == "low" and r.certainty == "factual" and r.polarity == "affirmed"


def test_nothing_around_the_name_leaves_it_as_named():
    r = q("The heart is normal in size.", "heart")
    assert r.location == "as_named" and r.degree == "" and r.character == ()


@pytest.mark.parametrize("sentence,mention,language", [
    ("Compression deformities from T11-T12 are unchanged.", "T11", "en"),
    ("Marcata riduzione dello spazio discale tra L4 e L5.", "L5", "it"),
])
def test_a_range_of_levels(sentence, mention, language):
    assert q(sentence, mention, language).location == "range"


def test_a_region_and_a_possible_finding_with_its_character():
    r = q("Possible small hypodense opacity in the region of the right lower lobe; no effusion.", "right lower lobe")
    assert r.location == "region" and r.degree == "low" and r.certainty == "possible"
    assert [c.value for c in r.character] == ["density:low", "density:high"]


def test_italian_degree_and_density():
    r = q("Lesione lievemente iperdensa del fegato.", "fegato", "it")
    assert r.degree == "low" and [c.value for c in r.character] == ["density:high"]
    r = q("Cisti anecogena del rene destro.", "rene", "it")
    assert [c.value for c in r.character] == ["echogenicity:absent"]
    r = q("Linfonodi aumentati di volume in prossimità dell'ilo epatico.", "ilo", "it")
    assert r.location == "approximate"


def test_negation_is_carried_and_the_clause_bounds_the_scope():
    r = q("No focal hyperechoic lesion in the liver.", "liver")
    assert r.polarity == "negated" and [c.value for c in r.character] == ["echogenicity:high"]
    r = q("Marked cardiomegaly; the liver is normal.", "liver")
    assert r.degree == ""


def test_a_word_inside_another_word_is_not_a_cue():
    # "zio" (uncle) inside "spazio" made the finding a relative's (assertion cues matched substrings)
    r = q("Marcata riduzione dello spazio discale tra L4 e L5.", "L5", "it")
    assert not any(c.startswith("experiencer") for c in r.certainty_cues)


def test_qualifiers_as_flat_values():
    d = q("Mild hyperintense signal approximately at L3.", "L3").as_dict()
    assert d["location"] == "approximate" and d["degree"] == "low"
    assert d["character"] == ["signal:high"] and all(isinstance(x, str) for x in d["location_cues"])


def test_the_phrase_of_the_mention_comes_before_the_rest_of_the_clause():
    r = q("Vie biliari non dilatate, colecisti alitiasica e piccola cisti renale.", "Vie biliari", "it")
    assert r.degree == "low" and r.degree_scope == "clause"
    r = q("Lesione lievemente iperdensa del fegato, piccola cisti renale.", "fegato", "it")
    assert r.degree_scope == "phrase" and [c.word for c in r.degree_cues] == ["lievemente"]
    r = q("There is mild anterior wedging of a lower thoracic vertebral body, approximately T11 level.", "T11 level")
    assert r.degree == "low" and r.degree_scope == "clause"


def test_a_decimal_point_is_not_the_end_of_the_clause():
    """31 October 2026: "Cyst measuring 1.1 cm in the middle lobe" was read as "1 cm in the middle lobe"."""
    from melampo.memory.site_qualifiers import read_site_qualifiers

    text = "Colonic diverticula. Cyst measuring 1.1 cm in the middle lobe of the right lung. Single nodules."
    at = text.find("middle lobe")
    lo, hi = read_site_qualifiers(text, at, at + len("middle lobe"), "en").clause
    assert text[lo:hi].strip() == "Cyst measuring 1.1 cm in the middle lobe of the right lung"
