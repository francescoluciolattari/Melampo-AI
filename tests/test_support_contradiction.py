from melampo.reasoning.support_contradiction import SupportContradictionAnalyzer


def test_support_contradiction_analyzer_returns_typed_profiles():
    analyzer = SupportContradictionAnalyzer()
    result = analyzer.analyze(
        evidence=[
            {"source": "retrieval", "kind": "grounded"},
            {"source": "bundle", "kind": "bundle_keys"},
            {"source": "intuition", "kind": "candidate"},
        ],
        intuition={"deductive_filter": {"reasoning_mode": "contradiction_revision"}},
        dream={"rehearsal_profile": {"boundary_case_hint": True, "coherence_guidance": "multimodal_support"}},
        area_dynamics={
            "mismatch_score": 0.65,
            "coherence_score": 0.5,
            "coherence_pairs": [("language_listening", "visual_diagnostic")],
            "mismatch_pairs": [("epidemiology", "language_listening")],
        },
    )
    assert result["support_profiles"]
    assert result["contradiction_profiles"]
    classes = {item["class"] for item in result["contradiction_profiles"]}
    assert "useful_contradiction" in classes
