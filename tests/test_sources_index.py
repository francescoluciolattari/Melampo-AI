"""The index of the sources (8 November 2026): well formed, written only when it has no problem, the weekly search
adds candidates only, and the rows of the reference library are verified only by their words on the page."""

import json
import shutil
from pathlib import Path

from melampo.memory import sources_index as si

ROOT = Path(__file__).resolve().parents[1]


def test_the_index_in_the_repository_is_well_formed():
    index = si.SourcesIndex.load()
    assert index.problems() == []
    assert {"europe-pmc", "pubmed", "radiopaedia", "openstax-anatomy-physiology"} <= set(index.by_id())
    closed = [s for s in index.sources if s.access in ("subscription", "institutional", "purchase")]
    assert closed and all(s.how_to_reach for s in closed)


def test_a_source_is_saved_only_without_problems(tmp_path):
    path = tmp_path / "sources.yml"
    shutil.copy(si.SOURCES, path)
    index = si.SourcesIndex.load(path)
    bad = si.Source(id="Bad Id", kind="journal", title="", url="nope", access="subscription", status="available")
    assert len(index.put(bad)) >= 4 and "Bad Id" not in si.SourcesIndex.load(path).by_id()
    doc = si.Source(id="doc-x", kind="document", title="A guideline", url="https://example.org", access="open",
                    status="available", of="nowhere")
    assert any("of:" in p for p in index.put(doc))
    doc.of = "europe-pmc"
    assert index.put(doc) == []
    again = si.SourcesIndex.load(path)
    assert again.by_id()["doc-x"].title == "A guideline" and again.problems() == []
    assert again.header.startswith("# The index of the sources")


def test_the_weekly_search_adds_candidates_and_keeps_a_rejection(tmp_path):
    path = tmp_path / "sources.yml"
    shutil.copy(si.SOURCES, path)
    index = si.SourcesIndex.load(path)
    answer = {"resultList": {"result": [
        {"pmcid": "PMC123", "title": "Normal values of the spleen &amp; liver", "journalTitle": "J"},
        {"title": "no identifier"}]}}
    found = si.candidates_from_europe_pmc("spleen", lambda url: answer, today="2026-11-09")
    assert [(s.id, s.status, s.kind, s.title) for s in found] == [
        ("doc-pmc123", "candidate", "document", "Normal values of the spleen & liver")]
    assert si.add_candidates(index, found) == 1
    index.by_id()["doc-pmc123"].status = "rejected"
    index.save()
    assert si.add_candidates(si.SourcesIndex.load(path), found) == 0
    assert si.SourcesIndex.load(path).by_id()["doc-pmc123"].status == "rejected"


def test_a_row_is_verified_only_by_its_words_on_the_page(tmp_path):
    page = si.page_text(b"<html><p>On sonographic assessment, a length of <b>12 cm</b> is generally considered "
                        b"the upper limit of normal <sup>15</sup></p><script>x</script></html>")
    quoted = '"On sonographic assessment, a length of 12 cm is generally considered the upper limit of normal 15"'
    assert si.check_quote(quoted, page)["found"]
    assert not si.check_quote('"a length of 13 cm is generally considered"', page)["found"]
    values = tmp_path / "values.yml"
    shutil.copy(ROOT / "data/reference/values.yml", values)
    n = si.mark_rows(values, {"spleen-length-us": si.check_quote(quoted, page),
                              "kidney-cortex-us": {"found": False, "missing": ["cortex"]}}, today="2026-11-09")
    assert n == 2
    import yaml

    rows = {r["id"]: r for r in yaml.safe_load(values.read_text("utf-8"))["rows"]}
    assert rows["spleen-length-us"]["status"] == "verified" and "2026-11-09" in rows["spleen-length-us"]["page_check"]
    assert rows["kidney-cortex-us"]["status"] == "to_verify" and "cortex" in rows["kidney-cortex-us"]["page_check"]
    assert len(rows) == 29 and rows["spleen-length-us"]["normal"] == {"high": 120}
    json.dumps(rows)


def test_the_mask_shows_the_index_and_refuses_a_bad_source():
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import sources_mask

    html = sources_mask.page(si.SourcesIndex.load(), {"status": ["to_register"]})
    assert "Indice delle fonti" in html and "ClinicalKey" in html and "PubMed</b>" not in html
    bad = si.Source(id="x", kind="journal", title="", url="nope", access="open", status="available")
    shown = sources_mask.page(si.SourcesIndex.load(), {}, "non salvato: title: required", True, draft=bad)
    assert 'class="msg bad"' in shown and 'value="nope"' in shown


def test_a_missing_quotation_shows_what_the_page_says():
    page = "Normal values. The portal vein measures 7-13 mm, and up to 15 mm in deep inspiration. Other text."
    res = si.check_quote('"normal diameter of 7-13 mm" ... "but may vary up to 15 mm 11"', page)
    assert not res["found"] and res["missing"] == ["normal diameter of 7-13 mm", "but may vary up to 15 mm 11"]
    assert "up to 15 mm in deep inspiration" in res["nearest"]["but may vary up to 15 mm 11"]
    assert si.quoted_pieces('table 4, "16-34" mL/m2') == ["16-34"]       # a short piece with a number counts
