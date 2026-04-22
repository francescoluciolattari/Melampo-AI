from melampo.areas.case_context_area import CaseContextArea
from melampo.areas.language_listening_area import LanguageListeningArea
from melampo.areas.visual_diagnostic_area import VisualDiagnosticArea
from melampo.models.quantum_belief_layer import QuantumBeliefLayer
from melampo.reasoning.intuition_engine import IntuitionEngine


def test_area_signals_feed_intuition_engine():
    visual = VisualDiagnosticArea().integrate({"study_id": "s1"}, {"slide_id": "p1"})
    language = LanguageListeningArea().integrate(report_text="possible lesion", patient_complaints="cough")
    context = CaseContextArea().integrate({"age": 64, "sex": "F"})

    engine = IntuitionEngine(belief_layer=QuantumBeliefLayer())
    payload = engine.infer(
        case_id="case-2",
        ranked_evidence=[{"rank": 1, "weight": 2, "item": {"source": "semantic_memory"}}],
        dream={"belief": {"mode": "quantum_like_belief_update"}},
        quantum_allowed=True,
        area_signals={
            "visual_diagnostic": visual,
            "language_listening": language,
            "case_context": context,
        },
    )
    assert payload["intuition"] == "candidate_1"
    assert sorted(payload["area_signals"].keys()) == ["case_context", "language_listening", "visual_diagnostic"]
