from melampo.reasoning.support_contradiction import SupportContradictionAnalyzer


def test_support_contradiction_analyzer_returns_typed_profiles():
    analyzer = SupportContradictionAnalyzer()
    result = analyzer.analyze(
        evidence=[
            {"source": "retrieval", "kind": "grounded"},
            {"source": "bundle", "kind": "bundle_keys"},
            {"source": "intuition", "kind": "candidate"},
        ],
        nexus={
            "rehearsal_profile": {
                "boundary_case_hint": True,
                "coherence_guidance": "multimodal_support",
            }
        },
        discordances={"count": 3, "kinds": {"card": 2, "reference": 0, "image": 1, "document": 0}},
    )
    assert result["support_profiles"]
    assert result["contradiction_profiles"]
    classes = {item["class"] for item in result["contradiction_profiles"]}
    assert "useful_contradiction" in classes
    labels = [item["label"] for item in result["contradiction_profiles"]]
    assert "reading:card_discordance" in labels and "reading:image_discordance" in labels
    assert "reading:reference_discordance" not in labels
