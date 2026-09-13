"""Tests for the ClinicalTrials.gov connector."""

from melampo.connectors.clinical_trials import (
    DEFAULT_STATUSES,
    ClinicalTrialsConnector,
    _passage_from_study,
)


def _study(clear: str | None = None, **overrides):
    """Build a realistic study record, optionally blanking or overriding one field.

    ``clear`` takes a dotted path ("moduleName.fieldName") and blanks that
    field, for testing what happens when a required piece is missing.
    ``overrides`` takes the same dotted-path keys to set a specific value.
    """
    base = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT01234567", "briefTitle": "Calcitriol trial in sarcoidosis"},
            "descriptionModule": {"briefSummary": "Trial studying hypercalcaemia treatment."},
            "statusModule": {"startDateStruct": {"date": "2023-05"}},
            "conditionsModule": {"conditions": ["Sarcoidosis", "Hypercalcaemia"]},
        }
    }
    if clear:
        module, field_name = clear.split(".")
        base["protocolSection"][module][field_name] = None
    for dotted, value in overrides.items():
        module, field_name = dotted.split(".")
        base["protocolSection"].setdefault(module, {})[field_name] = value
    return base


# --------------------------------------------------------------------------
# Building a passage from a study record
# --------------------------------------------------------------------------


def test_a_complete_study_becomes_a_checkable_passage():
    passage = _passage_from_study(_study())
    assert passage.source_id == "nct:NCT01234567"
    assert passage.is_independently_checkable is True


def test_conditions_are_appended_to_the_searchable_text():
    passage = _passage_from_study(_study())
    assert "Sarcoidosis" in passage.text
    assert "Hypercalcaemia" in passage.text


def test_a_study_with_no_summary_is_not_constructed():
    assert _passage_from_study(_study(clear="descriptionModule.briefSummary")) is None


def test_a_study_with_no_nct_id_is_not_constructed():
    assert _passage_from_study(_study(clear="identificationModule.nctId")) is None


def test_a_study_with_no_title_is_not_constructed():
    assert _passage_from_study(_study(clear="identificationModule.briefTitle")) is None


def test_a_malformed_start_date_does_not_crash_parsing():
    passage = _passage_from_study(_study(**{"statusModule.startDateStruct": {}}))
    assert passage is not None
    assert passage.year is None


def test_a_study_with_no_conditions_still_constructs():
    passage = _passage_from_study(_study(**{"conditionsModule.conditions": []}))
    assert passage is not None
    assert "Conditions studied" not in passage.text


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


def test_search_returns_passages_from_studies():
    connector = ClinicalTrialsConnector()
    connector._fetch_page = lambda query: {"studies": [_study()]}

    results = connector.search("sarcoidosis")

    assert len(results) == 1
    assert results[0].source_id == "nct:NCT01234567"


def test_search_stops_at_max_results():
    connector = ClinicalTrialsConnector()
    connector._fetch_page = lambda query: {
        "studies": [
            _study(**{"identificationModule.nctId": f"NCT0000000{i}"}) for i in range(5)
        ]
    }

    results = connector.search("sarcoidosis", max_results=2)

    assert len(results) == 2


def test_search_for_concepts_with_nothing_makes_no_call():
    connector = ClinicalTrialsConnector()
    connector._fetch_page = lambda *a: (_ for _ in ()).throw(AssertionError("should not be called"))
    assert connector.search_for_concepts([]) == []


def test_populate_adds_results_to_an_index():
    from melampo.memory.literature_index import LiteratureIndex

    connector = ClinicalTrialsConnector()
    connector._fetch_page = lambda query: {"studies": [_study()]}
    index = LiteratureIndex()

    added = connector.populate(index, "sarcoidosis")

    assert added == 1
    assert len(index) == 1


def test_terminated_trials_are_excluded_by_default():
    """A stopped trial is not a place to refer a case or a source of an
    outcome to cite -- excluded by default, not by oversight."""
    assert "TERMINATED" not in DEFAULT_STATUSES
    assert "WITHDRAWN" not in DEFAULT_STATUSES


def test_the_status_filter_is_sent_as_part_of_the_query_config():
    connector = ClinicalTrialsConnector()
    assert connector.config.statuses == DEFAULT_STATUSES
