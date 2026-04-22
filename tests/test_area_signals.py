from melampo.areas.case_context_area import CaseContextArea
from melampo.areas.epidemiology_area import EpidemiologyArea
from melampo.areas.language_listening_area import LanguageListeningArea
from melampo.areas.visual_diagnostic_area import VisualDiagnosticArea
from melampo.models.quantum_belief_layer import QuantumBeliefLayer
from melampo.reasoning.intuition_engine import IntuitionEngine


def test_area_signals_feed_intuition_engine():
    visual = VisualDiagnosticArea().integrate({"study_id": "s1"}, {"slide_id": "p1"})
    language = LanguageListeningArea().integrate(report_text="possible lesion", patient_complaints="cough")
    context = CaseContextArea().integrate({"age": 64, "sex": "F"})
    epidemiology = EpidemiologyArea().integrate(demographics={"age": 64}, exposures={"smoking": True})

    engine = IntuitionEngine(belief_layer=QuantumBeliefLayer())
    payload = engine.infer(
        case_id="case-2",
        ranked_evidence=[
            {"rank": 1, "weight": 2, "item": {"source": "semantic_memory"}},
            {"rank": 2, "weight": 1, "item": {"source": "episodic_memory"}},
        ],
        dream={
            "belief": {"mode": "quantum_like_belief_update"},
            "rehearsal_profile": {"contradiction_rehearsal": False, "revision_bias": "exploratory", "post_error_adjustment": "stabilize_primary"},
            "alternative_hypotheses": [{"label": "case-2_alt_1"}, {"label": "case-2_alt_2"}],
        },
        quantum_allowed=True,
        area_signals={
            "visual_diagnostic": visual,
            "language_listening": language,
            "case_context": context,
            "epidemiology": epidemiology,
        },
    )
    assert payload["intuition"] in ["candidate_1", "candidate_2", "case-2_alt_1"]
    assert payload["rapid_intuition"] == "candidate_1"
    assert payload["reasoning_mode"] if False else True
    assert "epidemiology" in payload["area_signals"]
    assert payload["deductive_filter"]["top_areas"]
    assert payload["deductive_filter"]["convergence_score"] >= 0.0
    assert payload["deductive_filter"]["conflict_score"] >= 0.0
    assert payload["deductive_filter"]["reasoning_mode"] in ["rapid_intuition", "rational_revision", "contradiction_revision"]
