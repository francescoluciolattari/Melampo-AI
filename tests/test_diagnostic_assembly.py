"""Tests for the assembly that wires every component into one working chain."""

from melampo.evaluation.enumeration_bench import DIFFERENTIAL_GRAPH_EDGES
from melampo.governance.confirmation_registry import SOURCE_HISTOPATHOLOGY, Confirmation
from melampo.memory.graph_store import is_learned
from melampo.reasoning.diagnostic_assembly import (
    assemble,
    candidate_conditions_for,
    nexus_context_for,
)


class _Trajectory:
    case_id = "case-001"
    final_answer = "Bilateral hilar lymphadenopathy on imaging."

    def evidence(self):
        return [
            {"record_id": "r1", "text": "CT chest: bilateral hilar lymphadenopathy."},
            {"record_id": "r2", "text": "hypercalcaemia confirmed on repeat."},
            {"record_id": "r3", "text": "erythema nodosum on shins."},
        ]


def _assembly(tmp_path):
    return assemble(list(DIFFERENTIAL_GRAPH_EDGES), tmp_path / "learned.jsonl")


# --------------------------------------------------------------------------
# The chain runs end to end, with nothing supplied by hand
# --------------------------------------------------------------------------


def test_a_case_produces_a_ranked_differential_from_a_trajectory_alone(tmp_path):
    result = _assembly(tmp_path).run_case(_Trajectory())
    assert result.outcome is not None
    assert result.outcome.hypotheses[0].condition == "sarcoidosis"


def test_running_a_case_records_the_conjectures_its_hypotheses_embody(tmp_path):
    """The ledger's value comes from accumulating across many cases, so
    recording happens by default rather than only on interesting ones."""
    result = _assembly(tmp_path).run_case(_Trajectory())
    assert result.conjectures_recorded > 0


def test_conjecture_recording_can_be_turned_off(tmp_path):
    result = _assembly(tmp_path).run_case(_Trajectory(), record_conjectures=False)
    assert result.conjectures_recorded == 0


def test_an_assembly_over_an_empty_store_reports_no_learned_edges(tmp_path):
    assert _assembly(tmp_path).learned_edge_count == 0


# --------------------------------------------------------------------------
# The learning loop: promotion, persistence, and surviving a restart
# --------------------------------------------------------------------------


def _confirm_three_times(assembly, source, target):
    for case_id in ("c1", "c2", "c3"):
        assembly.ledger.test(
            source, target, case_id,
            Confirmation(case_id=case_id, diagnosis=target, source=SOURCE_HISTOPATHOLOGY),
        )


def test_a_conjecture_confirmed_three_times_is_promoted_to_an_edge(tmp_path):
    assembly = _assembly(tmp_path)
    assembly.run_case(_Trajectory())
    _confirm_three_times(assembly, "bilateral hilar lymphadenopathy", "sarcoidosis")

    promoted = assembly.promote_confirmed()

    assert len(promoted) == 1
    assert promoted[0].target == "sarcoidosis"


def test_a_promoted_edge_carries_an_interval_not_a_point_estimate(tmp_path):
    assembly = _assembly(tmp_path)
    assembly.run_case(_Trajectory())
    _confirm_three_times(assembly, "bilateral hilar lymphadenopathy", "sarcoidosis")

    edge = assembly.promote_confirmed()[0]

    assert edge.lower is not None and edge.upper is not None
    assert edge.lower < edge.upper, "three confirmations is not certainty, and the interval must show it"


def test_a_promoted_edge_is_marked_as_learned_not_imported(tmp_path):
    assembly = _assembly(tmp_path)
    assembly.run_case(_Trajectory())
    _confirm_three_times(assembly, "bilateral hilar lymphadenopathy", "sarcoidosis")

    edge = assembly.promote_confirmed()[0]

    assert is_learned(edge) is True
    assert "confirmations=3" in edge.provenance


def test_what_the_system_learned_survives_a_restart(tmp_path):
    """The bottleneck this whole sequence existed to remove: before
    persistence, a promoted edge lived until the process exited and was then
    silently lost."""
    path = tmp_path / "learned.jsonl"

    first = assemble(list(DIFFERENTIAL_GRAPH_EDGES), path)
    first.run_case(_Trajectory())
    _confirm_three_times(first, "bilateral hilar lymphadenopathy", "sarcoidosis")
    first.promote_confirmed()

    second = assemble(list(DIFFERENTIAL_GRAPH_EDGES), path)

    assert second.learned_edge_count == 1, "a fresh process sees what the previous one learned"


def test_promotion_is_a_separate_call_not_a_side_effect_of_answering_a_case(tmp_path):
    """A change to the shared knowledge base should happen when someone runs
    it, not silently while answering one patient's case."""
    assembly = _assembly(tmp_path)
    assembly.run_case(_Trajectory())
    _confirm_three_times(assembly, "bilateral hilar lymphadenopathy", "sarcoidosis")

    assert assembly.store.count() == 0, "running the case alone wrote nothing"

    assembly.promote_confirmed()
    assert assembly.store.count() == 1


def test_promoting_with_nothing_confirmed_writes_nothing(tmp_path):
    assembly = _assembly(tmp_path)
    assembly.run_case(_Trajectory())
    assert assembly.promote_confirmed() == []
    assert assembly.store.count() == 0


# --------------------------------------------------------------------------
# The NexusTrainer hook that was never valorised
# --------------------------------------------------------------------------


def test_nexus_context_supplies_the_candidates_the_trainer_needed(tmp_path):
    """NexusTrainer._enumerated returns None unless case_context carries
    candidate_conditions, and nothing in the pipeline put them there -- which
    is why it fell through to rehearsal labels on every real case."""
    assembly = _assembly(tmp_path)
    context = nexus_context_for(["bilateral hilar lymphadenopathy", "hypercalcaemia"], assembly.graph)

    assert context["findings"]
    assert context["candidate_conditions"], "the field that was always empty before"
    assert "sarcoidosis" in context["candidate_conditions"]


def test_nexus_context_passes_through_extra_keys(tmp_path):
    assembly = _assembly(tmp_path)
    context = nexus_context_for(["hypercalcaemia"], assembly.graph, already_considered=["lymphoma"])
    assert context["already_considered"] == ["lymphoma"]


def test_candidate_conditions_for_is_the_same_retrieval_the_bridge_uses(tmp_path):
    assembly = _assembly(tmp_path)
    findings = ["bilateral hilar lymphadenopathy", "hypercalcaemia"]
    assert "sarcoidosis" in candidate_conditions_for(findings, assembly.graph)


# --------------------------------------------------------------------------
# Assembly choices
# --------------------------------------------------------------------------


def test_information_content_falls_back_to_graph_structure_without_frequencies(tmp_path):
    """A caller with no frequency data still gets specificity weighting --
    the documented intrinsic-IC path, not a degraded one."""
    assembly = _assembly(tmp_path)
    assert len(assembly.table) > 0


def test_supplied_frequencies_are_used_when_given(tmp_path):
    assembly = assemble(
        list(DIFFERENTIAL_GRAPH_EDGES),
        tmp_path / "learned.jsonl",
        frequencies={"sarcoidosis": 130, "hypercalcaemia": 500},
    )
    assert assembly.table.value("sarcoidosis") > assembly.table.value("hypercalcaemia")


def test_case_result_as_dict_carries_what_a_reviewer_needs(tmp_path):
    payload = _assembly(tmp_path).run_case(_Trajectory()).as_dict()
    for key in ("findings_from_documents", "hypotheses", "conjectures_recorded"):
        assert key in payload
