"""Tests for the graph source loader and the tiered normalisation cascade."""

import tempfile
from pathlib import Path

import pytest

from melampo.evaluation.vetting_bench import vetting_graph
from melampo.memory.concept_normalisation import (
    TIER_EMBEDDING,
    TIER_LEXICAL,
    TIER_NONE,
    TIER_STRUCTURAL,
    NormalisationCascade,
    cosine_similarity,
)
from melampo.memory.graph_sources import (
    SOURCE_EMPTY,
    SOURCE_FIXTURE,
    SOURCE_HPOA,
    load_verification_graph,
)
from melampo.memory.structural_comparison import (
    ConceptDescriptionStore,
    ExtractedRelation,
    ExtractedStructure,
    StructuralResolver,
    compare_structures,
)

_HPOA_SAMPLE = (
    "#description: HPO annotations\n"
    "database_id\tdisease_name\tqualifier\thpo_id\treference\tevidence\tonset\t"
    "frequency\tsex\tmodifier\taspect\tbiocuration\n"
    "OMIM:154700\tMarfan syndrome\t\tHP:0002616\tOMIM:154700\tTAS\t\t\t\t\tP\tHPO:skoehler"
)


# --------------------------------------------------------------------------
# Graph source: never a silent fixture fallback
# --------------------------------------------------------------------------


def test_a_fixture_graph_reports_itself_as_a_fixture():
    """The audit that produced this module found a 33-edge hand-written
    fixture being reported as grounding against the concept graph."""
    source = load_verification_graph(fixture_factory=vetting_graph)

    assert source.source == SOURCE_FIXTURE
    assert source.is_real_data is False
    assert "not a measurement" in source.detail


def test_requiring_real_data_refuses_to_fall_back(monkeypatch, tmp_path):
    """The setting a bench run making a selection decision should use:
    falling back is allowed, falling back silently is not."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MELAMPO_HPOA_PATH", raising=False)

    with pytest.raises(FileNotFoundError, match="Refusing to fall back"):
        load_verification_graph(require_real_data=True, fixture_factory=vetting_graph)


def test_a_real_hpoa_file_is_reported_as_real_data():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "phenotype.hpoa"
        path.write_text(_HPOA_SAMPLE)

        source = load_verification_graph(explicit_path=path)

    assert source.source == SOURCE_HPOA
    assert source.is_real_data is True
    assert source.edge_count > 0


def test_no_file_and_no_fixture_yields_an_empty_graph_not_a_crash(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MELAMPO_HPOA_PATH", raising=False)

    source = load_verification_graph()

    assert source.source == SOURCE_EMPTY
    assert source.edge_count == 0


def test_the_env_var_is_honoured(monkeypatch, tmp_path):
    path = tmp_path / "somewhere.hpoa"
    path.write_text(_HPOA_SAMPLE)
    monkeypatch.setenv("MELAMPO_HPOA_PATH", str(path))

    assert load_verification_graph().source == SOURCE_HPOA


# --------------------------------------------------------------------------
# Tier 1: unchanged lexical matching, tried first
# --------------------------------------------------------------------------


def test_an_exact_concept_resolves_lexically_without_touching_later_tiers():
    calls = {"n": 0}

    def counting_embedder(text):
        calls["n"] += 1
        return [1.0, 0.0]

    cascade = NormalisationCascade(graph=vetting_graph(), embedder=counting_embedder)
    result = cascade.resolve("secondary hyperparathyroidism")

    assert result.tier == TIER_LEXICAL
    assert calls["n"] == 0, "the embedding tier must not run when lexical matching succeeded"


def test_a_lexical_resolution_is_deterministic():
    cascade = NormalisationCascade(graph=vetting_graph())
    assert cascade.resolve("secondary hyperparathyroidism").is_deterministic is True


# --------------------------------------------------------------------------
# Tier 2: embedding similarity, with both a threshold and a margin
# --------------------------------------------------------------------------


def _paraphrase_embedder(text: str):
    lowered = text.lower()
    if "myelin" in lowered or "methionine synthase" in lowered:
        return [1.0, 0.0, 0.0]
    if "calcitriol" in lowered:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def test_a_paraphrase_lexical_matching_cannot_reach_resolves_by_embedding():
    """The measured gap: candidates wrote 'impaired methylcobalamin-dependent
    methionine synthase activity' where the node reads 'impaired myelin
    synthesis'."""
    cascade = NormalisationCascade(graph=vetting_graph(), embedder=_paraphrase_embedder)

    result = cascade.resolve("impaired methylcobalamin-dependent methionine synthase activity")

    assert result.tier == TIER_EMBEDDING
    assert result.concept == "impaired myelin synthesis"
    assert result.is_deterministic is True, "a fixed embedder is reproducible even though it was learned"


def test_a_degenerate_embedder_resolves_nothing_rather_than_picking_arbitrarily():
    """A threshold judges the winner in isolation; the margin guard judges
    whether there was a winner at all. Found by a test whose mock embedder
    returned the same vector for everything -- every concept scored 1.0, and
    the first one encountered won by accident."""
    cascade = NormalisationCascade(graph=vetting_graph(), embedder=lambda text: [0.0, 0.0, 1.0])

    result = cascade.resolve("something entirely unrelated to this graph")

    assert result.resolved is False


def test_a_best_match_below_threshold_is_not_resolved():
    def weak_embedder(text):
        return [1.0, 0.0] if "hyperparathyroid" in text.lower() else [0.5, 0.86]

    cascade = NormalisationCascade(graph=vetting_graph(), embedder=weak_embedder, embedding_threshold=0.99)
    result = cascade.resolve("a phrase that only weakly resembles anything")

    assert result.resolved is False


def test_a_failing_embedder_degrades_the_cascade_without_breaking_it():
    def broken_embedder(text):
        raise RuntimeError("model unavailable")

    cascade = NormalisationCascade(graph=vetting_graph(), embedder=broken_embedder)
    result = cascade.resolve("a phrase no lexical rule matches")

    assert result.resolved is False
    assert "failed" in result.detail


def test_cosine_similarity_handles_degenerate_input():
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Tier 3: structural comparison -- the model extracts, arithmetic decides
# --------------------------------------------------------------------------


def _myelin_structure(source_text: str = "literature") -> ExtractedStructure:
    return ExtractedStructure(
        source_text=source_text,
        entities=("methylcobalamin", "methionine synthase", "methylation", "myelin"),
        relations=(
            ExtractedRelation("methylcobalamin", "required_for", "methionine synthase"),
            ExtractedRelation("methionine synthase", "enables", "methylation"),
            ExtractedRelation("methylation", "required_for", "myelin"),
        ),
    )


def test_identical_structures_overlap_completely():
    comparison = compare_structures(_myelin_structure(), _myelin_structure())
    assert comparison.score == pytest.approx(1.0)


def test_shared_entities_alone_score_lower_than_shared_relations():
    """Two texts about the same clinical area share entities easily; sharing
    a relation between the same two entities is the stronger signal."""
    same_entities_different_relations = ExtractedStructure(
        source_text="other",
        entities=("methylcobalamin", "methionine synthase", "methylation", "myelin"),
        relations=(ExtractedRelation("myelin", "unrelated_to", "methylation"),),
    )
    comparison = compare_structures(_myelin_structure(), same_entities_different_relations)

    assert comparison.entity_overlap == pytest.approx(1.0)
    assert comparison.relation_overlap < 0.2
    assert comparison.score < 0.5, "topical similarity must not masquerade as mechanistic agreement"


def test_comparison_is_symmetric():
    """One side is cached and the other is not; an asymmetric measure would
    make the cache observable in the results."""
    other = ExtractedStructure(source_text="x", entities=("myelin",), relations=())
    assert compare_structures(_myelin_structure(), other).score == pytest.approx(
        compare_structures(other, _myelin_structure()).score
    )


def test_the_structural_tier_resolves_what_neither_earlier_tier_could():
    store = ConceptDescriptionStore()
    store.add("impaired myelin synthesis", _myelin_structure())

    resolver = StructuralResolver(store=store, extractor=lambda text: _myelin_structure(text))
    cascade = NormalisationCascade(graph=vetting_graph(), embedder=None, structural_resolver=resolver)

    result = cascade.resolve("impaired methylcobalamin-dependent methionine synthase activity")

    assert result.tier == TIER_STRUCTURAL
    assert result.concept == "impaired myelin synthesis"


def test_a_structural_resolution_is_marked_non_deterministic():
    """It involved a model extracting structure, and must be readable as
    such rather than presented alongside an exact lexical match as if the
    two carried the same weight."""
    store = ConceptDescriptionStore()
    store.add("impaired myelin synthesis", _myelin_structure())
    resolver = StructuralResolver(store=store, extractor=lambda text: _myelin_structure(text))
    cascade = NormalisationCascade(graph=vetting_graph(), structural_resolver=resolver)

    result = cascade.resolve("impaired methylcobalamin-dependent methionine synthase activity")

    assert result.is_deterministic is False


def test_a_resolver_naming_a_concept_outside_the_pool_resolves_nothing():
    """The failure mode tier 3 most needs guarding against: inventing a
    concept rather than recognising one."""
    store = ConceptDescriptionStore()
    store.add("impaired myelin synthesis", _myelin_structure())
    cascade = NormalisationCascade(
        graph=vetting_graph(),
        structural_resolver=lambda phrase, candidates: "a concept that is not in the graph",
    )

    assert cascade.resolve("anything").resolved is False


def test_the_structural_tier_does_nothing_without_an_extractor():
    store = ConceptDescriptionStore()
    store.add("impaired myelin synthesis", _myelin_structure())
    resolver = StructuralResolver(store=store, extractor=None)

    assert resolver("some phrase", ["impaired myelin synthesis"]) is None


def test_no_model_call_is_made_when_no_candidate_has_a_description():
    """Extracting the phrase anyway would spend a model call to learn
    nothing."""
    calls = {"n": 0}

    def counting_extractor(text):
        calls["n"] += 1
        return _myelin_structure(text)

    resolver = StructuralResolver(store=ConceptDescriptionStore(), extractor=counting_extractor)
    resolver("some phrase", ["a concept with no stored description"])

    assert calls["n"] == 0


def test_a_failing_extractor_degrades_without_breaking():
    store = ConceptDescriptionStore()
    store.add("impaired myelin synthesis", _myelin_structure())

    def broken_extractor(text):
        raise RuntimeError("extraction model unavailable")

    resolver = StructuralResolver(store=store, extractor=broken_extractor)
    assert resolver("some phrase", ["impaired myelin synthesis"]) is None


# --------------------------------------------------------------------------
# Tier usage reporting: how much of a result rested on which tier
# --------------------------------------------------------------------------


def test_usage_report_shows_how_much_rested_on_the_non_deterministic_tier():
    """A result where tier 3 carried most of the load rests on a far less
    reproducible foundation than one resolved lexically, and the headline
    grounding rate looks identical either way."""
    store = ConceptDescriptionStore()
    store.add("impaired myelin synthesis", _myelin_structure())
    resolver = StructuralResolver(store=store, extractor=lambda text: _myelin_structure(text))
    cascade = NormalisationCascade(graph=vetting_graph(), structural_resolver=resolver)

    cascade.resolve("secondary hyperparathyroidism")
    cascade.resolve("impaired methylcobalamin-dependent methionine synthase activity")

    report = cascade.usage_report()
    assert report["total"] == 2
    assert report["by_tier"][TIER_LEXICAL] == 1
    assert report["by_tier"][TIER_STRUCTURAL] == 1
    assert report["deterministic_fraction"] == pytest.approx(0.5)


def test_an_unresolvable_phrase_is_recorded_as_unresolved():
    cascade = NormalisationCascade(graph=vetting_graph())
    cascade.resolve("nothing in this graph resembles this phrase at all")
    assert cascade.usage_report()["by_tier"].get(TIER_NONE) == 1
