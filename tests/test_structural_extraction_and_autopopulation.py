"""Tests for the structural extractor, description-store persistence, and
automatic description creation on promotion."""

import tempfile
from pathlib import Path

from melampo.evaluation.enumeration_bench import DIFFERENTIAL_GRAPH_EDGES
from melampo.governance.confirmation_registry import SOURCE_HISTOPATHOLOGY, Confirmation
from melampo.memory.structural_comparison import (
    ConceptDescriptionStore,
    ExtractedRelation,
    ExtractedStructure,
)
from melampo.memory.structural_extraction import (
    ExtractionConfig,
    StructuralExtractor,
    _parse_extraction,
)
from melampo.reasoning.diagnostic_assembly import assemble


class _Trajectory:
    case_id = "case-001"
    final_answer = "Bilateral hilar lymphadenopathy."

    def evidence(self):
        return [
            {"record_id": "r1", "text": "bilateral hilar lymphadenopathy"},
            {"record_id": "r2", "text": "hypercalcaemia"},
        ]


def _fake_extractor(text: str) -> ExtractedStructure:
    return ExtractedStructure(
        source_text=text,
        entities=("sarcoidosis", "hypercalcaemia"),
        relations=(ExtractedRelation("sarcoidosis", "causes", "hypercalcaemia"),),
    )


def _confirm_three_times(assembly, source, target):
    for case_id in ("c1", "c2", "c3"):
        assembly.ledger.test(
            source, target, case_id, Confirmation(case_id=case_id, diagnosis=target, source=SOURCE_HISTOPATHOLOGY)
        )


# --------------------------------------------------------------------------
# The extractor: real, HTTP-backed, degrading gracefully throughout
# --------------------------------------------------------------------------


def test_unconfigured_extractor_returns_empty_structure_not_an_error():
    extractor = StructuralExtractor()
    result = extractor("some clinical text")
    assert result.is_empty


def test_empty_text_is_not_sent_to_the_model():
    calls = {"n": 0}
    extractor = StructuralExtractor(ExtractionConfig(endpoint="https://x", api_key="k", model="m"))
    extractor._call_model = lambda text: calls.__setitem__("n", calls["n"] + 1) or "{}"
    extractor("   ")
    assert calls["n"] == 0


def test_a_failing_model_call_degrades_without_raising():
    extractor = StructuralExtractor(ExtractionConfig(endpoint="https://x", api_key="k", model="m"))
    extractor._call_model = lambda text: (_ for _ in ()).throw(RuntimeError("endpoint down"))
    result = extractor("some text")
    assert result.is_empty


def test_a_configured_but_unimplemented_transport_degrades_gracefully():
    """The transport is deliberately left unimplemented -- deployment
    specific -- and must not turn into an unhandled exception for a caller
    that configures endpoint/key/model without also overriding _call_model."""
    extractor = StructuralExtractor(ExtractionConfig(endpoint="https://x", api_key="k", model="m"))
    assert extractor("some text").is_empty


def test_markdown_fences_around_the_json_response_are_stripped():
    """Models asked for 'only JSON' reliably wrap it in fences anyway."""
    raw = '```json\n{"entities": ["a", "b"], "relations": []}\n```'
    result = _parse_extraction("source", raw)
    assert result.entities == ("a", "b")


def test_a_malformed_response_degrades_rather_than_crashing():
    result = _parse_extraction("source", "not json at all")
    assert result.is_empty


def test_a_relation_missing_a_required_field_is_skipped():
    raw = '{"entities": ["a"], "relations": [{"subject": "a", "relation": "causes"}]}'
    result = _parse_extraction("source", raw)
    assert result.relations == ()


def test_extracted_entities_and_relations_are_parsed_correctly():
    raw = '{"entities": ["x", "y"], "relations": [{"subject": "x", "relation": "enables", "object": "y"}]}'
    result = _parse_extraction("source", raw)
    assert result.entities == ("x", "y")
    assert result.relations[0].subject == "x"
    assert result.relations[0].object == "y"


# --------------------------------------------------------------------------
# Description store persistence: append-only, survives a restart
# --------------------------------------------------------------------------


def test_a_description_round_trips_through_persistence():
    store = ConceptDescriptionStore()
    store.add(
        "impaired myelin synthesis",
        ExtractedStructure(
            source_text="literature text",
            entities=("methylcobalamin", "myelin"),
            relations=(ExtractedRelation("methylcobalamin", "required_for", "myelin"),),
        ),
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "descriptions.jsonl"
        assert store.append_to(path, "impaired myelin synthesis") is True

        loaded = ConceptDescriptionStore.load(path)

    assert loaded.get("impaired myelin synthesis").entities == ("methylcobalamin", "myelin")
    assert loaded.get("impaired myelin synthesis").relations[0].relation == "required_for"


def test_a_concept_with_no_description_writes_nothing():
    store = ConceptDescriptionStore()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "descriptions.jsonl"
        assert store.append_to(path, "never added") is False
        assert not path.exists()


def test_loading_a_missing_file_yields_an_empty_store_not_an_error():
    with tempfile.TemporaryDirectory() as directory:
        loaded = ConceptDescriptionStore.load(Path(directory) / "never_written.jsonl")
    assert len(loaded) == 0


def test_appending_does_not_rewrite_earlier_entries():
    store = ConceptDescriptionStore()
    store.add("first", ExtractedStructure(source_text="a", entities=("a",)))
    store.add("second", ExtractedStructure(source_text="b", entities=("b",)))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "descriptions.jsonl"
        store.append_to(path, "first")
        store.append_to(path, "second")

        loaded = ConceptDescriptionStore.load(path)

    assert len(loaded) == 2
    assert loaded.get("first").entities == ("a",)


# --------------------------------------------------------------------------
# Auto-population on promotion: every new promoted edge gets a description
# --------------------------------------------------------------------------


def test_promoting_an_edge_builds_descriptions_for_its_concepts(tmp_path):
    """The concrete behaviour requested: every graph derived from a Nexus
    Engine exploration or a newly confirmed assumption gets a pre-built
    description created and added automatically."""
    assembly = assemble(list(DIFFERENTIAL_GRAPH_EDGES), tmp_path / "learned.jsonl")
    assembly.run_case(_Trajectory())
    _confirm_three_times(assembly, "bilateral hilar lymphadenopathy", "sarcoidosis")

    store = ConceptDescriptionStore()
    desc_path = tmp_path / "descriptions.jsonl"

    promoted = assembly.promote_confirmed(
        description_store=store, description_extractor=_fake_extractor, description_store_path=desc_path
    )

    assert len(promoted) == 1
    assert store.get("sarcoidosis") is not None
    assert store.get("bilateral hilar lymphadenopathy") is not None


def test_descriptions_created_on_promotion_survive_a_restart(tmp_path):
    assembly = assemble(list(DIFFERENTIAL_GRAPH_EDGES), tmp_path / "learned.jsonl")
    assembly.run_case(_Trajectory())
    _confirm_three_times(assembly, "bilateral hilar lymphadenopathy", "sarcoidosis")

    desc_path = tmp_path / "descriptions.jsonl"
    assembly.promote_confirmed(
        description_store=ConceptDescriptionStore(), description_extractor=_fake_extractor,
        description_store_path=desc_path,
    )

    reloaded = ConceptDescriptionStore.load(desc_path)
    assert len(reloaded) == 2


def test_a_concept_that_already_has_a_description_is_not_overwritten(tmp_path):
    """A curated, literature-derived description must not be silently
    replaced by one generated from a promotion's own justification text."""
    assembly = assemble(list(DIFFERENTIAL_GRAPH_EDGES), tmp_path / "learned.jsonl")
    assembly.run_case(_Trajectory())
    _confirm_three_times(assembly, "bilateral hilar lymphadenopathy", "sarcoidosis")

    store = ConceptDescriptionStore()
    original = ExtractedStructure(source_text="curated", entities=("curated entity",))
    store.add("sarcoidosis", original)

    assembly.promote_confirmed(description_store=store, description_extractor=_fake_extractor, description_store_path=None)

    assert store.get("sarcoidosis") is original


def test_without_both_store_and_extractor_promotion_behaves_exactly_as_before(tmp_path):
    """Additive, not a new requirement: a caller not passing the description
    arguments gets the same promotion behaviour that existed before this
    feature."""
    assembly = assemble(list(DIFFERENTIAL_GRAPH_EDGES), tmp_path / "learned.jsonl")
    assembly.run_case(_Trajectory())
    _confirm_three_times(assembly, "bilateral hilar lymphadenopathy", "sarcoidosis")

    promoted = assembly.promote_confirmed()

    assert len(promoted) == 1


def test_promoting_nothing_calls_the_extractor_zero_times(tmp_path):
    calls = {"n": 0}

    def counting_extractor(text):
        calls["n"] += 1
        return ExtractedStructure(source_text=text)

    assembly = assemble(list(DIFFERENTIAL_GRAPH_EDGES), tmp_path / "learned.jsonl")
    assembly.promote_confirmed(description_store=ConceptDescriptionStore(), description_extractor=counting_extractor)

    assert calls["n"] == 0
