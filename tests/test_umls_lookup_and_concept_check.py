"""UMLS lookups for the probes (cache, pacing, lost state) and the E1 recount by concept."""

import sys
from pathlib import Path
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import concept_check as cc  # noqa: E402
from umls_lookup import UmlsLookup, tuis  # noqa: E402

from melampo.memory.anatomy_graph import AnatomyGraph  # noqa: E402


def _transport(table, calls):
    def fetch(url, params):
        calls.append((url, dict(params)))
        key = url.rsplit("/rest/", 1)[1]
        if params.get("string"):
            key += "?" + params["string"]
        if params.get("sabs"):
            key += "#" + params["sabs"]
        value = table.get(key)
        if isinstance(value, Exception):
            raise value
        if value is None:
            raise HTTPError(url, 404, "not found", None, None)
        return value

    return fetch


TABLE = {
    "content/current/CUI/C0003489": {"result": {"name": "Aortic arch", "semanticTypes": [
        {"name": "Body Part, Organ, or Organ Component", "uri": "https://uts-ws.nlm.nih.gov/rest/semantic-network/2024AA/TUI/T023"}]}},
    "search/current?inferior vena cava filter placement": {"result": {"results": [
        {"ui": "C0750159", "name": "Inferior vena cava filter placement",
         "semanticTypes": [{"uri": ".../TUI/T061"}]}]}},
    "search/current?the heart": {"result": {"results": [{"ui": "NONE", "name": "NO RESULTS"}]}},
    "content/current/CUI/C0003489/atoms#FMA": {"result": [{"code": "https://uts-ws.nlm.nih.gov/rest/content/2024AA/source/FMA/3768"}]},
    "content/current/CUI/C0003489/atoms#NCI": {"result": []},
    "content/current/CUI/C0003489/definitions": {"result": [
        {"rootSource": "MSH", "value": "The <b>curved</b> portion of the aorta."},
        {"rootSource": "NCI", "value": "The part of the aorta between the ascending and descending aorta."}]},
}


def _lookup(tmp_path, table=TABLE, calls=None):
    calls = [] if calls is None else calls
    return UmlsLookup(api_key="k", cache_path=tmp_path / "c.json", transport=_transport(table, calls),
                      min_interval=0.0, retries=2, sleep=lambda s: None), calls


def test_concept_exact_definition_codes_and_cache(tmp_path):
    lookup, calls = _lookup(tmp_path)
    assert lookup.concept("C0003489") == {"name": "Aortic arch", "types": ["T023"]}
    found = lookup.exact("Inferior  vena cava filter placement")
    assert found == [{"cui": "C0750159", "name": "Inferior vena cava filter placement", "types": ["T061"]}]
    assert lookup.exact("the heart") == []
    assert lookup.definition("C0003489").startswith("The part of the aorta")  # NCI preferred, tags removed
    assert lookup.codes("C0003489", "FMA") == ["3768"]
    n = len(calls)
    lookup.exact("inferior vena cava filter placement")
    assert len(calls) == n  # cached, normalised
    lookup.save()
    again, calls2 = _lookup(tmp_path)
    assert again.concept("C0003489")["name"] == "Aortic arch" and not calls2


def test_unknown_is_empty_and_failure_is_lost_not_nothing(tmp_path):
    lookup, _ = _lookup(tmp_path)
    assert lookup.concept("C9999999") == {}  # 404: UTS has nothing
    broken = dict(TABLE, **{"search/current?liver donation": URLError("down")})
    lookup2, _ = _lookup(tmp_path, broken)
    assert lookup2.exact("liver donation") is None
    assert lookup2.lost == 1
    assert "exact:liver donation" not in lookup2.cache  # a lost lookup is asked again next time


def test_without_key_nothing_is_asked(tmp_path, monkeypatch):
    monkeypatch.delenv("UMLS_API_KEY", raising=False)
    lookup = UmlsLookup(cache_path=tmp_path / "c.json")
    assert not lookup.available and lookup.concept("C0003489") is None


def test_tuis_any_shape():
    assert tuis([{"uri": "x/TUI/T023"}, "T061", {"name": "no uri"}]) == ["T023", "T061"]


def _graph():
    parents = {
        "UBERON:arch": [("UBERON:aorta", "part_of")],
        "UBERON:urinary_bladder": [("UBERON:bladder_organ", "is_a")],
        "UBERON:paa": [("UBERON:embryo_artery", "is_a")],
    }
    anchors = {"UBERON:aorta": "aorta", "UBERON:urinary_bladder": "urinary_bladder"}
    return AnatomyGraph(names={}, parents=parents, disjoint={}, anchors=anchors, families={})


def _row(corpus, mention, cid, labels, outcome, rule, relation="equal"):
    return {"corpus": corpus, "mention": mention, "cid": cid, "labels": labels, "outcome": outcome,
            "by_project_rule": rule, "status": "accepted", "relation": relation, "sentence": f"... {mention} ..."}


def test_recount_moves_coarser_labels_out_and_crosswalks_by_concept(tmp_path):
    graph = _graph()
    index = {"FMA:3768": {"UBERON:arch"}, "UMLS:C0003489": {"UBERON:paa"}}
    rows = [
        _row("craft", "bladder", "urinary_bladder", ["UBERON:bladder_organ"], "coarser_label", "error"),
        _row("medmentions", "aortic arch", "aorta", ["C0003489|T023"], "other_anatomy", "error", "part_of"),
        _row("medmentions", "heart", "heart", ["C1254362|T082"], "other_concept", "error"),
        _row("medmentions", "aorta", "aorta", ["C0003489|T023"], "same", "agrees"),
    ]
    lookup, _ = _lookup(tmp_path)
    report = cc.recount(rows, graph, index, lookup)
    moved = {(m["mention"], m["rule"]) for m in report["moved"]}
    assert moved == {("bladder", "label_coarser"), ("aortic arch", "crosswalk")}
    craft, mm = report["by_corpus"]["craft"], report["by_corpus"]["medmentions"]
    assert craft["judged_by_concept"] == 0 and craft["errors_by_concept"] == 0
    assert mm["errors_by_identifier"] == 2 and mm["errors_by_concept"] == 1
    assert mm["contradicted_right_links"] == 0
    assert "C1254362" in report["label_names"]  # error labels get their UMLS name (here: lost/unknown)


def test_recount_without_umls_only_moves_coarser_labels():
    rows = [_row("craft", "bladder", "urinary_bladder", ["UBERON:bladder_organ"], "coarser_label", "error"),
            _row("medmentions", "aortic arch", "aorta", ["C0003489|T023"], "other_anatomy", "error")]
    report = cc.recount(rows, _graph(), {}, None)
    assert [m["rule"] for m in report["moved"]] == ["label_coarser"]
    assert report["by_corpus"]["medmentions"]["errors_by_concept"] == 1
    assert "E1" in cc.markdown(report)


def test_the_external_check_keeps_the_offset_of_the_occurrence():
    import external_check as ec

    text = "First sentence here. Canine brain phantoms used agarose brain parenchyma. Last one."
    start = text.rindex("brain")
    sentence, at = ec.sentence_and_offset(text, start, start + 5)
    assert sentence.startswith("Canine") and sentence[at : at + 5] == "brain"
    assert sentence[:at].endswith("agarose ")
