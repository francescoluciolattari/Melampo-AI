"""Tests for wiring MechanismEnumerator into ClinicalInferencePipeline's dream branch.

Uses a small fixture graph via monkeypatching, never the real 1.27M-edge
HPO graph -- a first version of this change loaded the real graph
unconditionally on every _build_runtime_components() call, regressing the
whole test suite from ~13s to ~99s, since a dozen pre-existing tests
exercise this pipeline with no findings at all and each paid the real
graph's ~6.6s load cost for an enumerator they were never going to use.
Fixed by attaching the enumerator only when a case actually supplies
findings; these tests pin that behaviour so it cannot regress silently.
"""

from dataclasses import fields

from melampo.evaluation.enumeration_bench import differential_graph
from melampo.memory.graph_sources import GraphSource
from melampo.reasoning.clinical_pipeline import ClinicalInferencePipeline
from melampo.training.dream_trainer import DreamTrainer


def _minimal_pipeline(monkeypatch) -> ClinicalInferencePipeline:
    """A ClinicalInferencePipeline with every field the dream branch does not
    touch left as a bare object, and the real graph loader replaced with a
    small, fast fixture -- fast enough to run in every test invocation."""

    class _StubIngestion:
        def from_payload(self, payload):
            from melampo.types import CaseContext

            return CaseContext(
                case_id=payload.get("case_id", "case"),
                report_text=payload.get("report_text", ""),
                ehr_text="",
                demographics={},
                provenance={},
            )

    class _StubNormalizer:
        def to_fhir_bundle(self, case):
            return {"resourceType": "Bundle"}

    def _fake_load_graph(*args, **kwargs):
        graph = differential_graph()
        return GraphSource(graph=graph, source="fixture", edge_count=len(graph.edges), detail="test fixture")

    monkeypatch.setattr("melampo.reasoning.clinical_pipeline.load_verification_graph", _fake_load_graph)

    field_names = {f.name for f in fields(ClinicalInferencePipeline)}
    kwargs = dict.fromkeys(field_names, object())
    kwargs["ingestion"] = _StubIngestion()
    kwargs["normalizer"] = _StubNormalizer()
    kwargs["_dream_graph_source"] = None
    kwargs["_dream_enumerator"] = None
    return ClinicalInferencePipeline(**kwargs)


def _run(pipeline, payload):
    """Exercise only the pieces _run_dream_branch actually needs, bypassing
    the rest of run() (multimodal encoding, retrieval, specialist signals)
    which this test's stub dependencies cannot support."""
    from melampo.types import CaseContext

    case = CaseContext(case_id=payload.get("case_id", "c"), report_text="", ehr_text="", demographics={}, provenance={})
    components = pipeline._build_runtime_components()
    return pipeline._run_dream_branch(
        components=components, payload=payload, case=case, bundle={"a": 1},
        area_dynamics={}, governance_scores={"dream_coherence": 0.5, "risk": 0.2}, visual_imprints=[],
    )


# --------------------------------------------------------------------------
# The regression this whole change was caught by: no findings supplied
# means no real graph touched, fast and unchanged from before
# --------------------------------------------------------------------------


def test_no_findings_never_attaches_an_enumerator(monkeypatch):
    pipeline = _minimal_pipeline(monkeypatch)
    _run(pipeline, {"case_id": "c1"})
    assert pipeline._dream_enumerator is None


def test_no_findings_produces_rehearsal_labels_not_enumerated_hypotheses(monkeypatch):
    pipeline = _minimal_pipeline(monkeypatch)
    result = _run(pipeline, {"case_id": "c1"})
    hypotheses = result["alternative_hypotheses"]
    assert all(item.get("kind") != "enumerated_mechanism" for item in hypotheses)


def test_an_empty_findings_list_behaves_the_same_as_no_findings_key(monkeypatch):
    pipeline = _minimal_pipeline(monkeypatch)
    _run(pipeline, {"case_id": "c1", "findings": []})
    assert pipeline._dream_enumerator is None


# --------------------------------------------------------------------------
# With findings, the real enumerator attaches and real hypotheses come back
# --------------------------------------------------------------------------


def test_findings_attach_a_real_enumerator(monkeypatch):
    pipeline = _minimal_pipeline(monkeypatch)
    _run(pipeline, {"case_id": "c1", "findings": ["bilateral hilar lymphadenopathy", "hypercalcaemia"]})
    assert pipeline._dream_enumerator is not None


def test_findings_produce_real_enumerated_hypotheses(monkeypatch):
    pipeline = _minimal_pipeline(monkeypatch)
    result = _run(
        pipeline,
        {
            "case_id": "c1",
            "findings": ["bilateral hilar lymphadenopathy", "hypercalcaemia", "erythema nodosum"],
        },
    )
    hypotheses = result["alternative_hypotheses"]
    assert any(item.get("kind") == "enumerated_mechanism" for item in hypotheses)


def test_an_enumerated_hypothesis_carries_a_real_graph_path_as_provenance(monkeypatch):
    pipeline = _minimal_pipeline(monkeypatch)
    result = _run(
        pipeline,
        {"case_id": "c1", "findings": ["bilateral hilar lymphadenopathy", "hypercalcaemia", "erythema nodosum"]},
    )
    enumerated = next(h for h in result["alternative_hypotheses"] if h.get("kind") == "enumerated_mechanism")
    assert enumerated["paths"], "an enumerated hypothesis must carry the path that found it"


# --------------------------------------------------------------------------
# The graph loads once per pipeline instance, not once per call
# --------------------------------------------------------------------------


def test_the_graph_loads_only_once_across_multiple_calls(monkeypatch):
    calls = {"n": 0}

    def _counting_load_graph(*args, **kwargs):
        calls["n"] += 1
        graph = differential_graph()
        return GraphSource(graph=graph, source="fixture", edge_count=len(graph.edges), detail="test")

    monkeypatch.setattr("melampo.reasoning.clinical_pipeline.load_verification_graph", _counting_load_graph)
    pipeline = _minimal_pipeline(monkeypatch)
    monkeypatch.setattr("melampo.reasoning.clinical_pipeline.load_verification_graph", _counting_load_graph)

    _run(pipeline, {"case_id": "c1", "findings": ["bilateral hilar lymphadenopathy", "hypercalcaemia"]})
    _run(pipeline, {"case_id": "c2", "findings": ["erythema nodosum"]})

    assert calls["n"] == 1


# --------------------------------------------------------------------------
# The per-candidate cost cap, in place since MechanismEnumerator.run() was
# measured at roughly 3.5-4s per candidate against the real graph
# --------------------------------------------------------------------------


def test_candidate_conditions_are_capped():
    from melampo.reasoning.clinical_pipeline import DREAM_ENUMERATION_CANDIDATE_CAP

    assert 1 <= DREAM_ENUMERATION_CANDIDATE_CAP <= 20, "cap must be small enough to keep the branch usable"


# --------------------------------------------------------------------------
# The DreamTrainer field itself: unset by default, matching its own
# documented graceful-degradation contract
# --------------------------------------------------------------------------


def test_dream_trainer_enumerator_field_defaults_to_none():
    """Confirms the construction site change did not quietly hard-wire an
    enumerator at construction time -- it must still be attachable, or not,
    per case."""
    import inspect

    signature = inspect.signature(DreamTrainer.__init__)
    assert signature.parameters["enumerator"].default is None


# --------------------------------------------------------------------------
# _graph_candidates_for: real IC-weighted candidates for IntuitionEngine,
# reusing the same cached graph as the dream branch's enumerator
# --------------------------------------------------------------------------


def test_no_findings_returns_an_empty_list_without_touching_the_graph(monkeypatch):
    calls = {"n": 0}

    def _counting_load_graph(*args, **kwargs):
        calls["n"] += 1
        graph = differential_graph()
        return GraphSource(graph=graph, source="fixture", edge_count=len(graph.edges), detail="test")

    monkeypatch.setattr("melampo.reasoning.clinical_pipeline.load_verification_graph", _counting_load_graph)
    pipeline = _minimal_pipeline(monkeypatch)
    monkeypatch.setattr("melampo.reasoning.clinical_pipeline.load_verification_graph", _counting_load_graph)

    result = pipeline._graph_candidates_for([])

    assert result == []
    assert calls["n"] == 0


def test_real_findings_produce_real_ic_weighted_candidates(monkeypatch):
    pipeline = _minimal_pipeline(monkeypatch)

    result = pipeline._graph_candidates_for(["bilateral hilar lymphadenopathy", "hypercalcaemia", "erythema nodosum"])

    assert result
    assert all("condition" in item and "specificity_score" in item for item in result)


def test_the_ic_table_loads_only_once_across_multiple_calls(monkeypatch):
    pipeline = _minimal_pipeline(monkeypatch)

    pipeline._graph_candidates_for(["bilateral hilar lymphadenopathy"])
    table_after_first = pipeline._dream_ic_table
    pipeline._graph_candidates_for(["hypercalcaemia"])

    assert pipeline._dream_ic_table is table_after_first
