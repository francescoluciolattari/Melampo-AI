import pytest

from melampo.evaluation.falsification_program import (
    CLAIM_CORROBORATED,
    CLAIM_OPEN,
    CLAIM_REFUTED,
    FalsifiableClaim,
    FalsificationProgram,
)


def test_legacy_summary_keys_are_preserved():
    summary = FalsificationProgram().summarize()
    for key in ("quantum_like", "open_systems", "biophysical_frontier"):
        assert key in summary


def test_every_claim_carries_a_refutation_criterion_and_a_source_record():
    program = FalsificationProgram()
    assert program.claims
    for claim in program.claims:
        assert claim.refutation_criterion.strip(), f"{claim.claim_id} has no refutation criterion"
        assert claim.decision_record, f"{claim.claim_id} is not traceable to a decision record"


def test_claims_start_open_unless_explicitly_resolved_with_a_reason():
    program = FalsificationProgram()
    for claim in program.claims:
        if claim.status == CLAIM_OPEN:
            assert claim.evidence == []
        else:
            assert claim.evidence, f"{claim.claim_id} left {claim.status} without recorded evidence"


def test_blocking_claims_gate_the_strategy():
    program = FalsificationProgram()
    blocking = {claim.claim_id for claim in program.blocking_claims()}
    assert "rlm.dual_path_beats_single_path" in blocking
    assert "rlm.disagreement_is_informative" in blocking

    program.resolve("rlm.dual_path_beats_single_path", CLAIM_CORROBORATED, evidence="run_2026_09_01")
    assert "rlm.dual_path_beats_single_path" not in {claim.claim_id for claim in program.blocking_claims()}


def test_route_conditional_claims_are_still_registered_as_blocking():
    """Dormant is not the same as non-blocking: the route may yet be taken."""
    program = FalsificationProgram()
    conditional = program.get("rlm.open_weight_root_is_sufficient")
    assert conditional.blocking is True
    assert conditional.conditional_on is not None


def test_resolving_a_claim_requires_evidence():
    program = FalsificationProgram()
    with pytest.raises(ValueError):
        program.resolve("rlm.coverage_predicts_grounding", CLAIM_CORROBORATED, evidence="")


def test_invalid_resolution_status_is_rejected():
    program = FalsificationProgram()
    with pytest.raises(ValueError):
        program.resolve("rlm.coverage_predicts_grounding", "accepted", evidence="run_1")


def test_refutation_is_recorded_rather_than_removing_the_claim():
    program = FalsificationProgram()
    program.resolve("rlm.disagreement_is_informative", CLAIM_REFUTED, evidence="no association observed")
    refuted = program.refuted_claims()
    assert [claim.claim_id for claim in refuted] == ["rlm.disagreement_is_informative"]
    assert refuted[0].evidence == ["no association observed"]
    assert program.report()["refuted"] == 1


def test_duplicate_claim_ids_are_rejected():
    program = FalsificationProgram()
    with pytest.raises(ValueError):
        program.register(
            FalsifiableClaim(
                claim_id="rlm.coverage_predicts_grounding",
                statement="duplicate",
                refutation_criterion="n/a",
            )
        )


def test_unknown_claim_raises():
    with pytest.raises(KeyError):
        FalsificationProgram().get("rlm.does_not_exist")
