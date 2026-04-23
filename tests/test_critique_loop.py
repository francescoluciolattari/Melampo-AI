from melampo.reasoning.critique_loop import CritiqueLoop


class _Logger:
    def info(self, *args, **kwargs):
        return None


def test_critique_loop_flags_mismatch_and_revision():
    loop = CritiqueLoop(config=object(), logger=_Logger())
    result = loop.review(
        {
            "area_dynamics": {"mismatch_score": 0.8},
            "intuition": {"deductive_filter": {"reasoning_mode": "contradiction_revision"}},
            "dream": {"rehearsal_profile": {"boundary_case_hint": True}},
            "coordinated": {
                "policy": {"escalate": True},
                "differential": {
                    "contradiction_profiles": [
                        {"class": "useful_contradiction"},
                        {"class": "spurious_conflict"},
                    ]
                },
            },
        }
    )
    assert result["status"] == "reviewed"
    assert "high_cross_area_mismatch" in result["warnings"]
    assert "contradiction_revision_active" in result["warnings"]
    assert "useful_contradiction_detected" in result["warnings"]
    assert "possible_spurious_conflict" in result["warnings"]
    assert result["suggestions"]
