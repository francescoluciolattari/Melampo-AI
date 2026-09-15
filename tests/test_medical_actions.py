"""Tests for MAxO medical action annotations, kept apart from the concept graph."""

from melampo.memory.graph_sources import load_medical_actions
from melampo.memory.medical_actions import (
    CAUTIONARY_RELATIONS,
    MedicalAction,
    MedicalActionIndex,
    parse_maxo_annotations,
)

_HEADER = (
    "disease_id\tdisease_name\tsource_id\tmaxo_id\tmaxo_name\thpo_id\trelation\tevidence\t"
    "extension_id\textension_name\tcomment\tother\tauthor\tlast_updated\tcreated"
)
_TREATS_ROW = (
    "MONDO:0008854\tBardet-biedl Syndrome 1\tPMID:20301537\tMAXO:0000088\tdietary intervention\t"
    "HP:0000819\tTREATS\tTAS\t\t\t\t\tORCID:0000-0001\t2023-01-31\t2022-01-06"
)
_CONTRA_ROW = (
    "MONDO:0100135\tDravet Syndrome\tPMID:9596203\tMAXO:0000058\tsodium channel inhibitor therapy\t"
    "HP:0001250\tCONTRAINDICATED\tTAS\t\t\t\t\tORCID:0000-0001\t2023-01-31\t2022-01-06"
)


def _parse(*rows: str) -> list[MedicalAction]:
    return list(parse_maxo_annotations([_HEADER, *rows]))


# --------------------------------------------------------------------------
# Parsing the real file's shape
# --------------------------------------------------------------------------


def test_a_treats_row_parses():
    action = _parse(_TREATS_ROW)[0]
    assert action.disease_id == "MONDO:0008854"
    assert action.maxo_name == "dietary intervention"
    assert action.relation == "TREATS"


def test_a_row_with_no_disease_or_action_is_skipped():
    assert _parse("\t\tPMID:1\t\t\t\tTREATS\t\t\t\t\t\t\t\t") == []


def test_a_byte_order_mark_does_not_break_the_header():
    """The shipped file is UTF-8 with BOM; a header read as '\\ufeffdisease_id'
    would make every field lookup miss and yield zero rows."""
    actions = list(parse_maxo_annotations(["\ufeff" + _HEADER, _TREATS_ROW]))
    assert len(actions) == 1


def test_carriage_returns_do_not_end_up_inside_field_values():
    """The real file uses CRLF line endings."""
    actions = list(parse_maxo_annotations([_HEADER, _TREATS_ROW + "\r"]))
    assert "\r" not in actions[0].relation


# --------------------------------------------------------------------------
# Cautionary rows: the asymmetric case worth surfacing
# --------------------------------------------------------------------------


def test_a_contraindication_is_flagged_as_cautionary():
    assert _parse(_CONTRA_ROW)[0].is_cautionary is True


def test_a_treatment_is_not_cautionary():
    assert _parse(_TREATS_ROW)[0].is_cautionary is False


def test_cautions_for_returns_only_the_warnings():
    index = MedicalActionIndex.from_annotations(_parse(_TREATS_ROW, _CONTRA_ROW))
    assert [action.relation for action in index.cautions_for("MONDO:0100135")] == ["CONTRAINDICATED"]
    assert index.cautions_for("MONDO:0008854") == []


def test_both_cautionary_relations_are_recognised():
    assert "CONTRAINDICATED" in CAUTIONARY_RELATIONS
    assert "NO_OBSERVED_BENEFIT" in CAUTIONARY_RELATIONS


def test_a_row_with_a_pmid_is_citable():
    assert _parse(_TREATS_ROW)[0].is_citable is True


def test_a_row_with_no_resolvable_source_is_not_citable():
    row = _TREATS_ROW.replace("PMID:20301537", "internal-note")
    assert _parse(row)[0].is_citable is False


# --------------------------------------------------------------------------
# Lookup by id or name; absence means unannotated, never "none exists"
# --------------------------------------------------------------------------


def test_lookup_works_by_id_and_by_name():
    index = MedicalActionIndex.from_annotations(_parse(_TREATS_ROW))
    assert index.for_disease("MONDO:0008854")
    assert index.for_disease("bardet-biedl syndrome 1")


def test_an_unannotated_disease_returns_empty():
    index = MedicalActionIndex.from_annotations(_parse(_TREATS_ROW))
    assert index.for_disease("something not in maxo") == []


def test_the_coverage_note_states_that_absence_means_nothing():
    """At 1.6% coverage a reader who does not know the figure cannot tell
    'nothing is recommended' from 'nobody has annotated this'."""
    index = MedicalActionIndex.from_annotations(_parse(_TREATS_ROW))
    note = index.coverage_note(total_diseases=12_880)
    assert "no clinical meaning" in note
    assert "12,880" in note


# --------------------------------------------------------------------------
# The separation from the concept graph, which is the point
# --------------------------------------------------------------------------


def test_the_index_is_not_a_concept_graph():
    """Treatment relations in the diagnostic graph would let a differential
    traverse disease -> therapy -> other disease and call two conditions
    related because they share a treatment."""
    index = MedicalActionIndex.from_annotations(_parse(_TREATS_ROW))
    assert not hasattr(index, "edges_from")


def test_the_real_shipped_file_loads():
    index = load_medical_actions()
    if index is None:
        return  # no data/ in this checkout; the loader's None path is tested elsewhere
    assert len(index) > 0
    assert index.disease_count > 0
