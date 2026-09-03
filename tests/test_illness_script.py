import pytest

from melampo.governance.confirmation_registry import (
    REJECT_DUPLICATE,
    REJECT_NOT_INDEPENDENT,
    REJECT_REVIEWER_SAW_SUGGESTION,
    REJECT_UNSPECIFIED_SOURCE,
    SOURCE_CLINICAL_OUTCOME,
    SOURCE_HISTOPATHOLOGY,
    SOURCE_INDEPENDENT_REVIEW,
    SOURCE_SYSTEM_ACCEPTED,
    Confirmation,
    ConfirmationRegistry,
    assert_independent,
)
from melampo.memory.concept_paths import ConceptEdge, InMemoryConceptGraph
from melampo.reasoning.illness_script import (
    ORIGIN_HYPOTHESIS_CHANNEL,
    ORIGIN_MODEL,
    VERDICT_GROUNDED_IN_CASE,
    VERDICT_KNOWLEDGE_MEDIATED,
    VERDICT_UNSUPPORTED,
    IllnessScript,
    ScriptVerifier,
    merge_hypotheses,
)


def _graph() -> InMemoryConceptGraph:
    return InMemoryConceptGraph.from_edges(
        [
            ConceptEdge("pulmonary oedema", "causes", "bibasilar opacities", 0.9, "radiology_ontology"),
            ConceptEdge("congestive cardiac failure", "causes", "pulmonary oedema", 0.9, "cardiology_ontology"),
            ConceptEdge("pneumonia", "causes", "bibasilar opacities", 0.7, "radiology_ontology"),
        ]
    )


SCRIPT_PAYLOAD = {
    "enabling_conditions": [{"label": "advanced age"}],
    "fault": {"label": "pulmonary oedema", "term_id": "HP:0100598"},
    "consequences": [{"label": "bibasilar opacities", "term_id": "HP:0002093"}],
    "differential": [
        {"condition": "congestive cardiac failure", "rank": 1, "discriminating_features": ["bnp assay"]},
        {"condition": "pneumonia", "rank": 2},
    ],
}


# --------------------------------------------------------------------------
# The frame
# --------------------------------------------------------------------------


def test_a_model_payload_becomes_an_addressable_script():
    """Free text gives the graph nothing to check; a script gives every element an address."""
    script = IllnessScript.from_payload(SCRIPT_PAYLOAD)

    assert script.fault.label == "pulmonary oedema"
    assert [item.label for item in script.consequences] == ["bibasilar opacities"]
    assert [item.condition for item in script.differential] == [
        "congestive cardiac failure",
        "pneumonia",
    ]
    assert script.differential[0].discriminating_features == ("bnp assay",)


def test_a_partial_payload_does_not_raise():
    script = IllnessScript.from_payload({"differential": [{"condition": "pneumonia"}]})
    assert script.fault is None
    assert script.consequences == []
    assert script.differential[0].rank == 1


def test_malformed_entries_are_skipped():
    script = IllnessScript.from_payload(
        {"differential": [{"condition": ""}, "not a dict", {"condition": "pneumonia"}]}
    )
    assert [item.condition for item in script.differential] == ["pneumonia"]


def test_the_leading_entry_ignores_channel_candidates():
    script = IllnessScript.from_payload(SCRIPT_PAYLOAD)
    merge_hypotheses(script, [{"label": "amyloidosis"}])
    assert script.leading.condition == "congestive cardiac failure"
    assert script.leading.origin == ORIGIN_MODEL


# --------------------------------------------------------------------------
# Verification: what each element rests on
# --------------------------------------------------------------------------


def test_an_observed_finding_is_grounded_in_the_case():
    verification = ScriptVerifier(graph=_graph()).verify(
        IllnessScript.from_payload(SCRIPT_PAYLOAD), case_findings=["bibasilar opacities"]
    )
    assert verification.consequences[0].verdict == VERDICT_GROUNDED_IN_CASE
    assert verification.consequences[0].strength_lower == 1.0


def test_a_condition_reachable_through_the_graph_is_knowledge_mediated():
    """Absent from the case is not automatically wrong."""
    verification = ScriptVerifier(graph=_graph()).verify(
        IllnessScript.from_payload(SCRIPT_PAYLOAD), case_findings=["bibasilar opacities"]
    )
    leading = verification.differential[0]

    assert leading.element == "congestive cardiac failure"
    assert leading.verdict == VERDICT_KNOWLEDGE_MEDIATED
    assert leading.is_admissible is True
    assert leading.path is not None
    assert "pulmonary oedema" in leading.path.describe().lower()


def test_a_condition_with_no_path_is_unsupported():
    payload = {**SCRIPT_PAYLOAD, "differential": [{"condition": "fractured radius", "rank": 1}]}
    verification = ScriptVerifier(graph=_graph()).verify(
        IllnessScript.from_payload(payload), case_findings=["bibasilar opacities"]
    )
    assert verification.differential[0].verdict == VERDICT_UNSUPPORTED
    assert verification.differential[0].is_admissible is False
    assert verification.unsupported


def test_the_verifier_reports_a_grounding_ratio_rather_than_a_pass_or_fail():
    payload = {
        **SCRIPT_PAYLOAD,
        "differential": [
            {"condition": "congestive cardiac failure", "rank": 1},
            {"condition": "fractured radius", "rank": 2},
        ],
    }
    verification = ScriptVerifier(graph=_graph()).verify(
        IllnessScript.from_payload(payload), case_findings=["bibasilar opacities"]
    )
    assert 0.0 < verification.grounding_ratio < 1.0
    assert verification.as_dict()["unsupported_count"] == 1


def test_only_admitted_findings_ground_an_element():
    """A negated mention is not an observation and must ground nothing."""
    verification = ScriptVerifier(graph=_graph()).verify(
        IllnessScript.from_payload(SCRIPT_PAYLOAD), case_findings=[]
    )
    assert verification.consequences[0].verdict == VERDICT_UNSUPPORTED


def test_the_verification_payload_carries_the_supporting_path():
    verification = ScriptVerifier(graph=_graph()).verify(
        IllnessScript.from_payload(SCRIPT_PAYLOAD), case_findings=["bibasilar opacities"]
    )
    payload = verification.as_dict()["differential"][0]
    assert payload["path"]["kind"] == "concept_graph_path"
    assert payload["path"]["edges"][0]["provenance"]


# --------------------------------------------------------------------------
# Dream branch integration
# --------------------------------------------------------------------------


def test_channel_hypotheses_join_the_differential_marked_as_candidates():
    script = IllnessScript.from_payload(SCRIPT_PAYLOAD)
    merge_hypotheses(script, [{"label": "amyloidosis", "term_id": "HP:0011034"}])

    added = script.differential[-1]
    assert added.condition == "amyloidosis"
    assert added.origin == ORIGIN_HYPOTHESIS_CHANNEL
    assert added.is_candidate_only is True
    assert script.differential[0].is_candidate_only is False


def test_hypotheses_are_appended_after_the_model_entries():
    script = IllnessScript.from_payload(SCRIPT_PAYLOAD)
    merge_hypotheses(script, [{"label": "amyloidosis"}])
    assert script.differential[-1].rank > script.differential[0].rank


def test_a_condition_the_model_already_raised_is_not_re_listed():
    """Re-listing spends review attention on something already under consideration."""
    script = IllnessScript.from_payload(SCRIPT_PAYLOAD)
    merge_hypotheses(script, [{"label": "Pneumonia"}, {"label": "amyloidosis"}])
    conditions = [item.condition for item in script.differential]
    assert conditions.count("pneumonia") + conditions.count("Pneumonia") == 1
    assert "amyloidosis" in conditions


def test_merged_hypotheses_are_verified_like_any_other_element():
    script = merge_hypotheses(IllnessScript.from_payload(SCRIPT_PAYLOAD), [{"label": "pneumonia alt"}])
    verification = ScriptVerifier(graph=_graph()).verify(script, case_findings=["bibasilar opacities"])
    assert verification.differential[-1].verdict == VERDICT_UNSUPPORTED


# --------------------------------------------------------------------------
# Confirmation registry
# --------------------------------------------------------------------------


def test_histology_and_outcome_are_independent_confirmations():
    registry = ConfirmationRegistry()
    assert registry.register(Confirmation("c1", "sarcoidosis", source=SOURCE_HISTOPATHOLOGY)) is True
    assert registry.register(Confirmation("c2", "pneumonia", source=SOURCE_CLINICAL_OUTCOME)) is True
    assert len(registry.learning_set()) == 2


def test_an_accepted_suggestion_is_not_a_confirmation():
    """The failure mode: the system learns from its own proposals."""
    registry = ConfirmationRegistry()
    assert registry.register(Confirmation("c1", "pneumonia", source=SOURCE_SYSTEM_ACCEPTED)) is False
    assert registry.rejected[0].reason == REJECT_NOT_INDEPENDENT
    assert registry.learning_set() == []


def test_an_unrecorded_source_is_rejected_rather_than_assumed():
    registry = ConfirmationRegistry()
    assert registry.register(Confirmation("c1", "pneumonia")) is False
    assert registry.rejected[0].reason == REJECT_UNSPECIFIED_SOURCE


def test_a_review_counts_only_when_the_reviewer_was_blinded():
    registry = ConfirmationRegistry()
    unblinded = Confirmation("c1", "pneumonia", source=SOURCE_INDEPENDENT_REVIEW)
    blinded = Confirmation(
        "c2", "pneumonia", source=SOURCE_INDEPENDENT_REVIEW, reviewer_blinded_to_suggestion=True
    )
    assert registry.register(unblinded) is False
    assert registry.rejected[0].reason == REJECT_REVIEWER_SAW_SUGGESTION
    assert registry.register(blinded) is True


def test_an_unrecorded_blinding_status_is_treated_as_unblinded():
    """Reading the suggestion and agreeing is the failure, not a weaker confirmation."""
    registry = ConfirmationRegistry()
    partial = Confirmation(
        "c1", "pneumonia", source=SOURCE_INDEPENDENT_REVIEW, reviewer_blinded_to_suggestion=None
    )
    assert registry.register(partial) is False
    assert partial.is_independent is False


def test_a_duplicate_case_is_registered_once():
    registry = ConfirmationRegistry()
    registry.register(Confirmation("c1", "pneumonia", source=SOURCE_HISTOPATHOLOGY))
    assert registry.register(Confirmation("c1", "pneumonia", source=SOURCE_HISTOPATHOLOGY)) is False
    assert registry.rejected[0].reason == REJECT_DUPLICATE


def test_the_independence_rate_is_watchable_over_time():
    registry = ConfirmationRegistry()
    registry.register_many(
        [
            Confirmation("c1", "a", source=SOURCE_HISTOPATHOLOGY),
            Confirmation("c2", "b", source=SOURCE_SYSTEM_ACCEPTED),
            Confirmation("c3", "c", source=SOURCE_SYSTEM_ACCEPTED),
        ]
    )
    assert registry.independence_rate() == pytest.approx(1 / 3)
    report = registry.report()
    assert report["rejected_by_reason"][REJECT_NOT_INDEPENDENT] == 2
    assert report["admitted_by_source"][SOURCE_HISTOPATHOLOGY] == 1


def test_the_guard_raises_on_a_contaminated_learning_set():
    with pytest.raises(ValueError):
        assert_independent([Confirmation("c1", "pneumonia", source=SOURCE_SYSTEM_ACCEPTED)])


def test_the_guard_passes_a_clean_learning_set():
    assert_independent([Confirmation("c1", "pneumonia", source=SOURCE_HISTOPATHOLOGY)])
