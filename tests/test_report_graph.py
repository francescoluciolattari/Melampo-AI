"""Sites of a report as SITE edges with their qualifiers, in an embedded FalkorDB (15 Oct 2026)."""

from types import SimpleNamespace

import pytest

from melampo.memory.report_graph import load_sites, site_row, sites_of
from melampo.memory.site_qualifiers import read_site_qualifiers


def _result(cid, sentence, mention, status="accepted"):
    at = sentence.index(mention)
    return SimpleNamespace(status=status, cid=cid, relation="equal", part="", role="", roles=(),
                           qualifiers=read_site_qualifiers(sentence, at, at + len(mention)).as_dict()), at


def test_only_an_accepted_link_becomes_a_row():
    r, at = _result("T11", "Wedging, approximately T11 level.", "T11 level", status="abstained")
    assert site_row("rep1", "T11 level", at, at + 9, r) is None
    r, at = _result("T11", "Wedging, approximately T11 level.", "T11 level")
    row = site_row("rep1", "T11 level", at, at + 9, r)
    assert row["location"] == "approximate" and row["cid"] == "T11"


def test_sites_round_trip_through_falkordb(tmp_path):
    pytest.importorskip("redislite")
    from redislite.falkordb_client import FalkorDB

    graph = FalkorDB(str(tmp_path / "g.db")).select_graph("reports_test")
    sentence = "There is mild anterior wedging of a lower thoracic vertebral body, approximately T11 level."
    r, at = _result("vertebrae_T11", sentence, "T11 level")
    assert load_sites(graph, [site_row("rep1", "T11 level", at, at + 9, r)]) == 1
    got = sites_of(graph, "rep1")
    assert len(got) == 1 and got[0]["cid"] == "vertebrae_T11"
    assert got[0]["location"] == "approximate" and got[0]["degree"] == "low"
    assert got[0]["location_cues"] == ["approximate:approximately@67"]


def test_the_qualifiers_of_an_examined_structure_are_about_the_findings_of_the_exam():
    r, at = _result("thorax", "Alla radiografia del torace lievi addensamenti polmonari.", "torace")
    r.roles = ("procedure_site",)
    assert site_row("rep2", "torace", at, at + 6, r)["applies_to"] == "exam_findings"
    r.roles = ()
    assert site_row("rep2", "torace", at, at + 6, r)["applies_to"] == "site"
