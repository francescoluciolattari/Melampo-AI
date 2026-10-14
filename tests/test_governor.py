"""The governor of a mention, read by spaCy (skipped when spaCy or its models are not installed)."""

import pytest

spacy = pytest.importorskip("spacy")

from melampo.memory.governor import GovernorParser  # noqa: E402


def _parser(language, model):
    try:
        spacy.load(model)
    except OSError:
        pytest.skip(f"{model} not installed")
    return GovernorParser()


def _gov(parser, sentence, mention, language):
    at = sentence.index(mention)
    return parser.governor(sentence, at, at + len(mention), language)


def test_english_of_compound_and_place():
    p = _parser("en", "en_core_web_md")
    assert _gov(p, "A plain radiograph of the patient's knees revealed densities.", "knees", "en").lemma == "radiograph"
    g = _gov(p, "In this manner, the abdominal wall incisions were located at the rectus sheath.", "abdominal wall", "en")
    assert (g.lemma, g.relation) == ("incision", "compound")
    g = _gov(p, "Small, nodular opacity in the right upper lobe.", "right upper lobe", "en")
    assert g.lemma == "opacity" and g.locative and not g.nominal


def test_italian_coordination_takes_the_governor_of_the_first_member():
    p = _parser("it", "it_core_news_md")
    g = _gov(p, "Lisi delle aderenze e liberazione del cieco e dell’appendice che appaiono normali.", "appendice", "it")
    assert g.lemma == "liberazione" and g.nominal
    g = _gov(p, "Nel 2017 esegue un intervento chirurgico di piede torto congenito.", "piede", "it")
    assert g.lemma == "intervento"
