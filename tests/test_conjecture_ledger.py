from melampo.governance.confirmation_registry import (
    SOURCE_HISTOPATHOLOGY,
    SOURCE_SYSTEM_ACCEPTED,
    Confirmation,
)
from melampo.memory.concept_paths import ConceptEdge, InMemoryConceptGraph
from melampo.training.conjecture_ledger import (
    RELATION_CONJECTURED,
    Conjecture,
    ConjectureLedger,
    leaps_in,
)
from melampo.training.mechanism_enumeration import MechanismEnumerator


def _leap(case: str = "c1") -> Conjecture:
    return Conjecture(
        source="bibasilar opacities",
        target="amyloidosis",
        via=("pulmonary oedema", "cardiac failure"),
        hops=3,
        strength_upper=0.3,
        origin_case=case,
    )


def _histology(case: str, diagnosis: str) -> Confirmation:
    return Confirmation(case, diagnosis, source=SOURCE_HISTOPATHOLOGY)


# --------------------------------------------------------------------------
# Recording is free
# --------------------------------------------------------------------------


def test_every_leap_is_recorded_as_a_candidate():
    ledger = ConjectureLedger()
    ledger.record(_leap("c1"))
    ledger.record(_leap("c2"))
    assert len(ledger.records) == 1
    assert ledger.records[_leap().key].raised_in == ["c1", "c2"]


def test_an_untested_conjecture_has_the_full_interval():
    ledger = ConjectureLedger()
    entry = ledger.record(_leap())
    assert entry.interval() == (0.0, 1.0)
    assert entry.is_promotable(ledger.min_confirmations) is False


def test_leaps_are_read_off_enumerated_hypotheses():
    graph = InMemoryConceptGraph.from_edges(
        [
            ConceptEdge("finding", "caused_by", "mechanism", 0.9),
            ConceptEdge("mechanism", "caused_by", "rare condition", 0.4),
        ]
    )
    hypotheses = MechanismEnumerator(graph=graph, max_hops=3).enumerate(
        findings=["finding"], candidate_conditions=["rare condition"]
    )
    ledger = ConjectureLedger()
    assert leaps_in(hypotheses, "c1", ledger) == 1
    entry = next(iter(ledger.records.values()))
    assert entry.conjecture.source == "finding"
    assert entry.conjecture.target == "rare condition"
    assert "mechanism" in [item.lower() for item in entry.conjecture.via]


# --------------------------------------------------------------------------
# Only independent confirmation tests a leap
# --------------------------------------------------------------------------


def test_an_independent_confirmation_tests_the_conjecture():
    ledger = ConjectureLedger()
    ledger.record(_leap("c1"))
    held = ledger.test("bibasilar opacities", "amyloidosis", "c1", _histology("c1", "amyloidosis"))
    assert held is True
    assert ledger.records[_leap().key].confirmed_in == ["c1"]


def test_a_different_confirmed_diagnosis_refutes_the_leap():
    ledger = ConjectureLedger()
    ledger.record(_leap("c1"))
    held = ledger.test("bibasilar opacities", "amyloidosis", "c1", _histology("c1", "pneumonia"))
    assert held is False
    assert ledger.records[_leap().key].refuted_in == ["c1"]


def test_an_accepted_suggestion_does_not_test_anything():
    """It was produced by the system being tested."""
    ledger = ConjectureLedger()
    ledger.record(_leap("c1"))
    accepted = Confirmation("c1", "amyloidosis", source=SOURCE_SYSTEM_ACCEPTED)
    assert ledger.test("bibasilar opacities", "amyloidosis", "c1", accepted) is None
    assert ledger.records[_leap().key].tested == 0


def test_a_case_tests_a_conjecture_once():
    ledger = ConjectureLedger()
    ledger.record(_leap("c1"))
    ledger.test("bibasilar opacities", "amyloidosis", "c1", _histology("c1", "amyloidosis"))
    assert ledger.test("bibasilar opacities", "amyloidosis", "c1", _histology("c1", "amyloidosis")) is None
    assert ledger.records[_leap().key].tested == 1


def test_an_unrecorded_conjecture_cannot_be_tested():
    assert ConjectureLedger().test("a", "b", "c1", _histology("c1", "b")) is None


# --------------------------------------------------------------------------
# Promotion: intuition becomes knowledge, with its provenance
# --------------------------------------------------------------------------


def _ledger_with(confirmed: int, refuted: int = 0) -> ConjectureLedger:
    ledger = ConjectureLedger()
    ledger.record(_leap("c0"))
    for index in range(confirmed):
        ledger.test("bibasilar opacities", "amyloidosis", f"ok{index}", _histology(f"ok{index}", "amyloidosis"))
    for index in range(refuted):
        ledger.test("bibasilar opacities", "amyloidosis", f"no{index}", _histology(f"no{index}", "pneumonia"))
    return ledger


def test_a_leap_confirmed_enough_times_becomes_an_edge():
    edges = _ledger_with(confirmed=3).promotable()
    assert len(edges) == 1
    edge = edges[0]
    assert edge.relation == RELATION_CONJECTURED
    assert edge.source == "bibasilar opacities"
    assert edge.target == "amyloidosis"


def test_the_promoted_edge_carries_the_confirming_cases_as_provenance():
    edge = _ledger_with(confirmed=3).promotable()[0]
    assert "confirmed=ok0,ok1,ok2" in edge.provenance
    assert "via=pulmonary oedema>cardiac failure" in edge.provenance


def test_the_promoted_edge_interval_follows_the_confirmations():
    three = _ledger_with(confirmed=3).promotable()[0]
    thirty = _ledger_with(confirmed=30).promotable()[0]
    assert (thirty.upper - thirty.lower) < (three.upper - three.lower)
    assert thirty.lower > three.lower


def test_too_few_confirmations_do_not_promote():
    assert _ledger_with(confirmed=2).promotable() == []
    assert _ledger_with(confirmed=2).pending()


def test_a_leap_that_keeps_failing_is_never_promoted():
    ledger = _ledger_with(confirmed=3, refuted=40)
    edges = ledger.promotable()
    entry = ledger.records[_leap().key]
    _, upper = entry.interval()
    assert upper < 0.2, "the interval reflects how often the leap actually held"
    assert edges and edges[0].upper < 0.2


def test_a_leap_is_not_traversable_before_promotion():
    """Otherwise the branch could reach the next leap through the previous one."""
    ledger = _ledger_with(confirmed=1)
    assert ledger.promotable() == []
    graph = InMemoryConceptGraph.from_edges(ledger.promotable())
    assert graph.edges_from("bibasilar opacities") == []


def test_the_report_separates_recorded_tested_and_promotable():
    ledger = _ledger_with(confirmed=3)
    ledger.record(Conjecture("x", "y", (), 1, 0.5, "c9"))
    report = ledger.report()
    assert report["conjectures"] == 2
    assert report["tested"] == 1
    assert report["promotable"] == 1
