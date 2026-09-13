"""Tests for the Europe PMC literature connector."""

from melampo.connectors.europe_pmc import (
    EuropePmcConnector,
    RateLimiter,
    _passage_from_result,
)


def _record(**overrides):
    base = {
        "id": "1", "pmid": "12345678", "pmcid": "PMC1234567",
        "title": "Hypercalcaemia in sarcoidosis: mechanisms and management",
        "abstractText": "Granulomatous macrophages express 1-alpha-hydroxylase, producing excess calcitriol.",
        "journalInfo": {"journal": {"title": "Chest"}}, "pubYear": "2024",
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# Building a passage from a search result
# --------------------------------------------------------------------------


def test_a_complete_record_becomes_a_checkable_passage():
    passage = _passage_from_result(_record())
    assert passage.source_id == "pmid:12345678"
    assert passage.is_independently_checkable is True


def test_a_record_with_no_abstract_is_not_constructed():
    """An empty passage would never match a search and would only inflate a
    count past what is actually retrievable."""
    assert _passage_from_result(_record(abstractText="")) is None


def test_a_record_with_no_title_is_not_constructed():
    assert _passage_from_result(_record(title="")) is None


def test_pmid_is_preferred_over_pmcid_and_doi():
    passage = _passage_from_result(_record(pmid="111", pmcid="PMC222", doi="10.1/x"))
    assert passage.source_id == "pmid:111"


def test_pmcid_is_used_when_pmid_is_absent():
    passage = _passage_from_result(_record(pmid=None, pmcid="PMC222"))
    assert passage.source_id == "pmcid:PMC222"


def test_doi_is_used_when_pmid_and_pmcid_are_both_absent():
    passage = _passage_from_result(_record(pmid=None, pmcid=None, doi="10.1000/xyz"))
    assert passage.source_id == "doi:10.1000/xyz"


def test_a_record_with_no_identifier_at_all_still_constructs_but_is_not_checkable():
    """Constructed anyway rather than dropped -- is_independently_checkable
    already exists to flag this case; dropping it here would duplicate that
    logic in a second place."""
    passage = _passage_from_result(_record(pmid=None, pmcid=None, doi=None))
    assert passage is not None
    assert passage.is_independently_checkable is False


def test_a_non_numeric_year_does_not_crash_parsing():
    passage = _passage_from_result(_record(pubYear="n.d."))
    assert passage.year is None


# --------------------------------------------------------------------------
# Search: pagination, concept-query building, index population
# --------------------------------------------------------------------------


def test_search_paginates_until_max_results_or_no_more_pages():
    connector = EuropePmcConnector()
    pages = {"*": {"resultList": {"result": [_record()]}, "nextCursorMark": "next", "request": {"cursorMark": "*"}},
             "next": {"resultList": {"result": []}, "nextCursorMark": None}}
    connector._fetch_page = lambda query, cursor: pages[cursor]

    results = connector.search("sarcoidosis", max_results=10)
    assert len(results) == 1


def test_search_stops_once_max_results_is_reached_without_fetching_more_pages():
    connector = EuropePmcConnector()
    calls = {"n": 0}

    def fetch(query, cursor):
        calls["n"] += 1
        return {"resultList": {"result": [_record(id=str(i)) for i in range(5)]}, "nextCursorMark": "x", "request": {"cursorMark": cursor}}

    connector._fetch_page = fetch
    results = connector.search("sarcoidosis", max_results=2)

    assert len(results) == 2
    assert calls["n"] == 1


def test_multi_word_concepts_are_quoted_as_phrases():
    """Unquoted, Europe PMC would match records containing all the words
    anywhere, not the phrase -- a materially different, looser search."""
    connector = EuropePmcConnector()
    captured = {}

    def fetch(query, cursor):
        captured["query"] = query
        return {"resultList": {"result": []}, "nextCursorMark": None}

    connector._fetch_page = fetch
    connector.search_for_concepts(["sarcoidosis", "connective tissue weakness"])

    assert '"connective tissue weakness"' in captured["query"]
    assert "sarcoidosis" in captured["query"] and '"sarcoidosis"' not in captured["query"]


def test_search_for_concepts_with_nothing_makes_no_call():
    connector = EuropePmcConnector()
    connector._fetch_page = lambda *a: (_ for _ in ()).throw(AssertionError("should not be called"))
    assert connector.search_for_concepts([]) == []


def test_populate_adds_results_directly_to_an_index():
    from melampo.memory.literature_index import LiteratureIndex

    connector = EuropePmcConnector()
    connector._fetch_page = lambda q, c: {"resultList": {"result": [_record()]}, "nextCursorMark": None}
    index = LiteratureIndex()

    added = connector.populate(index, "sarcoidosis")

    assert added == 1
    assert len(index) == 1


# --------------------------------------------------------------------------
# RateLimiter: paced, never blocking when unnecessary
# --------------------------------------------------------------------------


def test_a_non_positive_rate_never_sleeps():
    limiter = RateLimiter(0)
    limiter.wait()
    limiter.wait()  # would hang if this ever tried to sleep on a bad interval


def test_first_call_never_waits():
    import time

    limiter = RateLimiter(1.0)
    start = time.monotonic()
    limiter.wait()
    assert time.monotonic() - start < 0.05
