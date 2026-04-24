from melampo.models.abstention import AbstentionPolicy
from melampo.models.risk_gate import RiskGate
from melampo.reasoning.area_coherence import AreaCoherenceAnalyzer
from melampo.reasoning.differential_engine import DifferentialEngine
from melampo.reasoning.escalation import EscalationPolicy
from melampo.reasoning.pipeline_coordinator import PipelineCoordinator
from melampo.reasoning.policy_stack import PolicyStack


def test_area_coherence_and_pipeline_coordinator_emit_richer_metadata():
    area_signals = {
        "visual_diagnostic": {"salience_score": 0.8, "signal_count": 3},
        "language_listening": {"salience_score": 0.6, "signal_count": 2},
        "case_context": {"salience_score": 0.2, "signal_count": 2},
        "epidemiology": {"salience_score": 0.4, "signal_count": 2},
    }
    dynamics = AreaCoherenceAnalyzer().analyze(area_signals)
    assert dynamics["pair_profiles"]
    assert dynamics["total_salience"] > 0.0
    assert dynamics["coherence_score"] >= 0.0

    coordinator = PipelineCoordinator(
        differential_engine=DifferentialEngine(),
        policy_stack=PolicyStack(
            abstention=AbstentionPolicy(threshold=0.65),
            risk_gate=RiskGate(threshold=0.35),
            escalation=EscalationPolicy(),
        ),
    )
    result = coordinator.run(
        case_id="case-xyz",
        evidence=[{"source": "retrieval", "kind": "grounded", "grounding_score": 0.8}],
        risk=0.2,
        uncertainty=0.1,
        intuition={
            "candidate_scores": [{"label": "candidate_1", "score": 1.1}],
            "deductive_filter": {"reasoning_mode": "rapid_intuition"},
        },
        dream={
            "alternative_hypotheses": [{"label": "alt_1", "kind": "rare_case", "focus": "epidemiology"}],
            "rehearsal_profile": {"coherence_guidance": "multimodal_support"},
        },
        area_dynamics=dynamics,
    )
    assert result["differential"]["hypotheses"]
    assert any(step.startswith("reasoning_mode:") for step in result["trace"])
    assert any(step.startswith("top_domain:") for step in result["trace"])
    assert any(step.startswith("recommended_actions:") for step in result["trace"])
