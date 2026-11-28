from melampo.reasoning.critique_loop import CritiqueLoop


class _Logger:
    def info(self, *args, **kwargs):
        return None


def test_critique_loop_flags_the_discordances_of_the_reading():
    loop = CritiqueLoop(config=object(), logger=_Logger())
    result = loop.review(
        {
            "discordances": {"count": 2, "kinds": {"card": 2}},
            "nexus": {"rehearsal_profile": {"boundary_case_hint": True}},
            "coordinated": {
                "policy": {"escalate": True},
                "differential": {
                    "contradiction_profiles": [
                        {"class": "useful_contradiction"},
                        {"class": "spurious_conflict"},
                    ],
                    "recommended_actions": [
                        {
                            "category": "multimodal_reconciliation",
                            "label": "recheck multimodal alignment",
                        },
                        {
                            "category": "disambiguation_test",
                            "label": "review alternative hypotheses",
                        },
                    ],
                },
            },
        }
    )
    assert result["status"] == "reviewed"
    assert "discordances_in_the_reading" in result["warnings"]
    assert "recheck multimodal consistency" in result["suggestions"]
    assert "useful_contradiction_detected" in result["warnings"]
    assert "possible_spurious_conflict" in result["warnings"]
    assert result["suggestions"]
    assert result["prioritized_actions"]
    assert result["prioritized_actions"][0]["priority"] == "high"
