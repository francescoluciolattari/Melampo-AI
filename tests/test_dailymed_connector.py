"""Tests for the DailyMed connector."""

from melampo.connectors.dailymed import DailyMedConnector, _passage_from_spl
from melampo.memory.literature_index import LiteratureIndex


def _record(**overrides):
    base = {
        "setid": "fdbfe194-b845-42c5-bb87-a48118bc72e7",
        "spl_version": "25",
        "title": "ZOCOR (SIMVASTATIN) TABLET, FILM COATED [MERCK SHARP & DOHME CORP.]",
        "published_date": "Nov 01, 2013",
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# Building a passage from a real SPL search result
# --------------------------------------------------------------------------


def test_a_complete_record_becomes_a_checkable_passage():
    passage = _passage_from_spl(_record(), ["Simvastatin"])
    assert passage.source_id == "dailymed:fdbfe194-b845-42c5-bb87-a48118bc72e7"
    assert passage.is_independently_checkable is True


def test_active_ingredients_are_included_in_the_searchable_text():
    passage = _passage_from_spl(_record(), ["Simvastatin"])
    assert "Simvastatin" in passage.text


def test_a_record_with_no_active_ingredients_still_constructs():
    """Packaging lookup can fail independently of the search itself -- a
    passage without ingredients is still a usable, citable record."""
    passage = _passage_from_spl(_record(), [])
    assert passage is not None
    assert "Active ingredients" not in passage.text


def test_a_record_with_no_title_is_not_constructed():
    assert _passage_from_spl(_record(title=""), ["Simvastatin"]) is None


def test_a_record_with_no_setid_is_not_constructed():
    """Nothing to build a citable identifier or a packaging lookup from."""
    assert _passage_from_spl(_record(setid=""), ["Simvastatin"]) is None


def test_the_publication_year_is_parsed_from_the_date_string():
    passage = _passage_from_spl(_record(published_date="Nov 01, 2013"), [])
    assert passage.year == 2013


def test_a_malformed_date_does_not_crash_parsing():
    passage = _passage_from_spl(_record(published_date="not a date"), [])
    assert passage.year is None


# --------------------------------------------------------------------------
# Search: the two-call pattern (search, then packaging per result)
# --------------------------------------------------------------------------


def test_search_returns_passages_with_ingredients_looked_up_automatically():
    connector = DailyMedConnector()
    connector._fetch_search_page = lambda name: {"data": [_record()]}
    connector._fetch_packaging = lambda setid: {"data": {"active_ingredients": [{"name": "Simvastatin"}]}}

    results = connector.search("simvastatin")

    assert len(results) == 1
    assert "Simvastatin" in results[0].text


def test_a_failing_packaging_lookup_still_yields_a_passage():
    """The search result alone is enough for a citable passage; a broken
    packaging call degrades detail, not existence."""
    connector = DailyMedConnector()
    connector._fetch_search_page = lambda name: {"data": [_record()]}

    def broken_packaging(setid):
        raise RuntimeError("packaging endpoint unavailable")

    connector._fetch_packaging = broken_packaging
    results = connector.search("simvastatin")

    assert len(results) == 1
    assert "Active ingredients" not in results[0].text


def test_search_stops_at_max_results():
    connector = DailyMedConnector()
    connector._fetch_search_page = lambda name: {
        "data": [_record(setid=f"id-{i}") for i in range(5)]
    }
    connector._fetch_packaging = lambda setid: {"data": {"active_ingredients": []}}

    results = connector.search("simvastatin", max_results=2)

    assert len(results) == 2


def test_search_for_concepts_queries_once_per_concept():
    connector = DailyMedConnector()
    calls = []

    def fake_search_page(name):
        calls.append(name)
        return {"data": [_record(setid=f"id-{name}")]}

    connector._fetch_search_page = fake_search_page
    connector._fetch_packaging = lambda setid: {"data": {"active_ingredients": []}}

    connector.search_for_concepts(["simvastatin", "metformin"])

    assert calls == ["simvastatin", "metformin"]


def test_populate_adds_results_directly_to_an_index():
    connector = DailyMedConnector()
    connector._fetch_search_page = lambda name: {"data": [_record()]}
    connector._fetch_packaging = lambda setid: {"data": {"active_ingredients": []}}
    index = LiteratureIndex()

    added = connector.populate(index, "simvastatin")

    assert added == 1
    assert len(index) == 1


# --------------------------------------------------------------------------
# Persistence, matching the other two connectors
# --------------------------------------------------------------------------


def test_populate_persists_when_given_a_store(tmp_path):
    from melampo.memory.vector_memory import PersistentJsonlVectorStore

    connector = DailyMedConnector()
    connector._fetch_search_page = lambda name: {"data": [_record()]}
    connector._fetch_packaging = lambda setid: {"data": {"active_ingredients": []}}
    store = PersistentJsonlVectorStore(path=tmp_path / "lit.jsonl")
    index = LiteratureIndex()

    connector.populate(index, "simvastatin", store=store)

    assert len(store.records) == 1


def test_populate_without_a_store_behaves_exactly_as_before():
    connector = DailyMedConnector()
    connector._fetch_search_page = lambda name: {"data": [_record()]}
    connector._fetch_packaging = lambda setid: {"data": {"active_ingredients": []}}
    index = LiteratureIndex()

    added = connector.populate(index, "simvastatin")

    assert added == 1
