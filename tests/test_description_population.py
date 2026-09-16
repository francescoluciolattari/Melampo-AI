"""Tests for HPO definition parsing, bulk population, and the canonical shared format."""

import json
import tempfile
from pathlib import Path

from melampo.memory.concept_resolution import parse_obo
from melampo.memory.description_population import (
    CANONICAL_EXAMPLE,
    canonical_format_example,
    parse_model_emitted_structure,
    populate_from_ontology,
)
from melampo.memory.structural_comparison import (
    ConceptDescriptionStore,
    ExtractedRelation,
    ExtractedStructure,
    StructuralResolver,
)

_OBO = (
    "[Term]\n"
    "id: HP:0000256\n"
    "name: Macrocephaly\n"
    'def: "Occipitofrontal circumference greater than 97th centile." [pmid:19125436]\n'
    "\n"
    "[Term]\n"
    "id: HP:0000001\n"
    "name: All\n"
)


def _simple_extractor(text: str) -> ExtractedStructure:
    words = [w.strip(".,;()").lower() for w in text.split() if len(w) > 7][:3]
    if not words:
        return ExtractedStructure(source_text=text)
    return ExtractedStructure(
        source_text=text,
        entities=tuple(words),
        relations=(ExtractedRelation(words[0], "related_to", words[-1]),) if len(words) > 1 else (),
    )


# --------------------------------------------------------------------------
# HPO definitions: 17,441 curated texts that were downloaded and never parsed
# --------------------------------------------------------------------------


def test_a_definition_is_parsed_with_its_reference_stripped():
    """The bracketed PMID is provenance, not clinical text -- leaving it in
    would put "pmid:19125436" in front of every extraction."""
    term = next(parse_obo(_OBO.splitlines()))
    assert term.definition == "Occipitofrontal circumference greater than 97th centile."


def test_a_term_with_no_definition_has_an_empty_one():
    terms = {t.term_id: t for t in parse_obo(_OBO.splitlines())}
    assert terms["HP:0000001"].definition == ""


def test_a_definition_does_not_leak_into_the_next_term():
    terms = {t.term_id: t for t in parse_obo(_OBO.splitlines())}
    assert terms["HP:0000256"].definition
    assert not terms["HP:0000001"].definition


# --------------------------------------------------------------------------
# Bulk population
# --------------------------------------------------------------------------


def test_population_describes_terms_that_have_definitions():
    store = ConceptDescriptionStore()
    report = populate_from_ontology(parse_obo(_OBO.splitlines()), store, _simple_extractor)

    assert report.described == 1
    assert store.get("Macrocephaly") is not None


def test_a_term_with_no_definition_is_counted_separately_from_a_failed_extraction():
    """Two different situations that a single 'skipped' count would blur:
    nothing to work from, versus something that yielded nothing."""
    store = ConceptDescriptionStore()
    report = populate_from_ontology(parse_obo(_OBO.splitlines()), store, _simple_extractor)

    assert report.skipped_no_definition == 1
    assert report.skipped_extraction_empty == 0


def test_an_existing_description_is_never_overwritten():
    """Bulk population fills gaps; a description already built from
    literature or a promotion's justification keeps what it has."""
    store = ConceptDescriptionStore()
    original = ExtractedStructure(source_text="curated", entities=("curated entity",))
    store.add("Macrocephaly", original)

    populate_from_ontology(parse_obo(_OBO.splitlines()), store, _simple_extractor)

    assert store.get("Macrocephaly") is original


def test_only_concepts_restricts_the_run():
    """17,441 terms is still 17,441 model calls -- a run scoped to what is
    actually being reasoned about reaches useful coverage first."""
    store = ConceptDescriptionStore()
    report = populate_from_ontology(
        parse_obo(_OBO.splitlines()), store, _simple_extractor, only_concepts=["something else entirely"]
    )
    assert report.described == 0


def test_one_failing_term_does_not_abort_the_run():
    calls = {"n": 0}

    def flaky_extractor(text):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("model hiccup")
        return _simple_extractor(text)

    obo = _OBO + '\n[Term]\nid: HP:0000002\nname: Second\ndef: "Another definition entirely here." [pmid:1]\n'
    store = ConceptDescriptionStore()
    report = populate_from_ontology(parse_obo(obo.splitlines()), store, flaky_extractor)

    assert report.errors
    assert report.described == 1, "the second term still got described"


def test_population_persists_when_given_a_path():
    store = ConceptDescriptionStore()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "descriptions.jsonl"
        populate_from_ontology(parse_obo(_OBO.splitlines()), store, _simple_extractor, store_path=path)

        assert len(ConceptDescriptionStore.load(path)) == 1


def test_obsolete_terms_are_not_described():
    obo = '[Term]\nid: HP:0000003\nname: Retired\ndef: "An obsolete definition here." [pmid:1]\nis_obsolete: true\n'
    store = ConceptDescriptionStore()
    report = populate_from_ontology(parse_obo(obo.splitlines()), store, _simple_extractor)
    assert report.described == 0


# --------------------------------------------------------------------------
# The canonical format: one definition shared by extractor and vetting model
# --------------------------------------------------------------------------


def test_the_prompt_example_is_generated_from_the_structure_shape_itself():
    """Hand-writing it would let the instruction a model is given drift from
    the format the extractor produces -- and compare_structures matches
    strings literally, so drift reads as disagreement."""
    example = canonical_format_example()

    assert "STRUCTURE:" in example
    for entity in CANONICAL_EXAMPLE.entities:
        assert entity in example


def test_the_example_round_trips_through_its_own_parser():
    """The strongest check that the two sides agree: the format shown to a
    model, parsed back by the code that reads a model's answer, must yield
    the structure it was generated from."""
    example = canonical_format_example()
    parsed = parse_model_emitted_structure(example)

    assert parsed is not None
    assert set(parsed.entities) == set(CANONICAL_EXAMPLE.entities)
    assert len(parsed.relations) == len(CANONICAL_EXAMPLE.relations)


def test_a_model_emitted_structure_is_parsed_from_a_prose_answer():
    answer = (
        "Excess calcitriol raises intestinal calcium absorption.\n"
        'STRUCTURE: {"entities": ["calcitriol"], '
        '"relations": [{"subject": "calcitriol", "relation": "increases", "object": "calcium absorption"}]}'
    )
    parsed = parse_model_emitted_structure(answer)

    assert parsed.entities == ("calcitriol",)
    assert parsed.relations[0].relation == "increases"


def test_prose_with_no_structure_line_yields_none_not_an_error():
    """A normal outcome, not a fault: a model that answers in prose only
    falls through to the extractor exactly as before."""
    assert parse_model_emitted_structure("Just a prose answer with no structure.") is None


def test_a_malformed_structure_line_yields_none():
    assert parse_model_emitted_structure("STRUCTURE: not json at all") is None


def test_an_empty_structure_line_yields_none():
    assert parse_model_emitted_structure('STRUCTURE: {"entities": [], "relations": []}') is None


# --------------------------------------------------------------------------
# The resolver prefers what the model emitted, and falls back cleanly
# --------------------------------------------------------------------------


def _store_with_hypercalcaemia() -> ConceptDescriptionStore:
    store = ConceptDescriptionStore()
    store.add(
        "hypercalcaemia",
        ExtractedStructure(
            source_text="definition",
            entities=("calcitriol", "intestinal calcium absorption"),
            relations=(ExtractedRelation("calcitriol", "increases", "intestinal calcium absorption"),),
        ),
    )
    return store


def test_a_model_emitted_structure_skips_the_extractor_entirely():
    """Not just cheaper -- more faithful: the model that formed the claim
    reports its structure better than a second model re-reading the prose."""
    calls = {"n": 0}

    def counting_extractor(text):
        calls["n"] += 1
        return ExtractedStructure(source_text=text)

    resolver = StructuralResolver(store=_store_with_hypercalcaemia(), extractor=counting_extractor)
    answer = (
        "Excess calcitriol raises absorption.\n"
        'STRUCTURE: {"entities": ["calcitriol", "intestinal calcium absorption"], '
        '"relations": [{"subject": "calcitriol", "relation": "increases", "object": "intestinal calcium absorption"}]}'
    )

    result = resolver(answer, ["hypercalcaemia"])

    assert result == "hypercalcaemia"
    assert calls["n"] == 0
    assert resolver.last_structure_source == "model_emitted"


def test_prose_without_structure_falls_back_to_the_extractor():
    resolver = StructuralResolver(store=_store_with_hypercalcaemia(), extractor=_simple_extractor)
    resolver("prose about calcitriol and absorption", ["hypercalcaemia"])
    assert resolver.last_structure_source == "extractor"


def test_no_extractor_and_no_emitted_structure_resolves_nothing():
    resolver = StructuralResolver(store=_store_with_hypercalcaemia(), extractor=None)
    assert resolver("plain prose", ["hypercalcaemia"]) is None
    assert resolver.last_structure_source is None


def test_a_model_cannot_grade_itself_through_an_emitted_structure():
    """Emitting structure is reporting, not judging: the comparison is still
    arithmetic against a cached description the model never saw, so a
    structure shaped to look agreeable has no target to shape toward."""
    resolver = StructuralResolver(store=_store_with_hypercalcaemia(), extractor=None)
    flattering = (
        "STRUCTURE: "
        + json.dumps({"entities": ["something unrelated"], "relations": [
            {"subject": "something unrelated", "relation": "is", "object": "definitely correct"}]})
    )

    assert resolver(flattering, ["hypercalcaemia"]) is None
