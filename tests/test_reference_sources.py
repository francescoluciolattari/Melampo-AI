"""The reference library from primary, open sources (21 November 2026), without network."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "reference_sources", Path(__file__).resolve().parents[1] / "scripts" / "reference_sources.py")
rs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rs)

TEXT = [("Results", "On sonographic assessment, a length of 12 cm is generally considered the upper limit of normal. "
                    "The mean splenic length was 10.6 ± 1.2 cm."),
        ("Table: Normal values", "Spleen | length | ≤ 13 cm\nPortal vein | diameter | 13 mm")]


def item(**kw):
    base = {"aspect": "length", "detail": "sonographic", "technique": "US", "population": {"age": "adult", "sex": "any"},
            "kind": "upper_limit", "low": None, "high": 12, "unit": "cm",
            "quote": "On sonographic assessment, a length of 12 cm is generally considered the upper limit of normal."}
    base.update(kw)
    return base


def test_every_structure_of_the_lexicon_is_searched_sides_and_numbers_joined():
    groups = {g["key"]: g for g in rs.structures(Path(rs.ROOT / "data" / "linking" / "anatomy_lexicon.json"))}
    assert {"kidney_left", "kidney_right"} <= set(groups["kidney"]["cids"])
    assert len(groups["rib"]["cids"]) == 24 and "vertebrae_L" in groups and "lung" in groups
    assert not any(k.startswith(("kidney_cyst", "liver_segment")) for k in groups)
    assert groups["appendix"]["cids"] == [] and groups["portal_vein_and_splenic_vein"]["names"][0] == "portal vein"


def test_a_value_is_kept_only_by_the_rules_fixed_before_the_run():
    v, why = rs.as_value(item(), TEXT)
    assert why == "" and (v["high"], v["unit"]) == (120.0, "mm")
    assert rs.as_value(item(kind="mean_sd", high=10.6,
                            quote="The mean splenic length was 10.6 ± 1.2 cm."), TEXT) == (None, "mean_sd")
    assert rs.as_value(item(high=14), TEXT)[1] == "number_not_in_quote"
    assert rs.as_value(item(quote="A length of 12 cm is the limit."), TEXT)[1] == "quote_not_in_text"
    assert rs.as_value(item(unit="HU"), TEXT)[1].startswith("unit_not_compared")
    row, why = rs.as_value(item(high=13, quote="Spleen | length | ≤ 13 cm"), TEXT)
    assert why == "" and row["high"] == 130.0                     # a table row counts as a sentence


def test_tables_are_read_one_row_per_line():
    xml = ("<article><front><article-meta><abstract><p>Normal values.</p></abstract></article-meta></front><body>"
           "<sec><title>Results</title><p>The upper limit is 6 mm.</p></sec></body><back><table-wrap><label>Table 1"
           "</label><table><tr><th>Structure</th><th>Limit</th></tr><tr><td>Appendix</td><td>6 mm</td></tr></table>"
           "</table-wrap></back></article>")
    sections = rs.text_from_jats(xml)
    assert sections[0] == ("Abstract", "Normal values.") and sections[1] == ("Results", "The upper limit is 6 mm.")
    assert sections[2][1].splitlines() == ["Structure | Limit", "Appendix | 6 mm"]


def test_the_references_of_a_radiopaedia_page_are_numbered_in_order(monkeypatch):
    page = ('<ol class="references"><li>Smith J. Spleen. <a href="https://doi.org/10.1/abc">doi</a></li>'
            '<li>No link here</li><li>Rossi A. <a href="https://pubmed.ncbi.nlm.nih.gov/123456">PubMed</a></li></ol>')
    monkeypatch.setattr(rs, "get", lambda url, accept="": page)
    refs = rs.radiopaedia_references("https://radiopaedia.org/articles/x")
    assert refs["1"]["doi"] == "10.1/abc" and refs["3"]["pmid"] == "123456" and "2" not in refs


def test_a_row_is_confirmed_when_its_primary_states_its_numbers(tmp_path, monkeypatch):
    values = tmp_path / "values.yml"
    values.write_text("rows:\n  - id: spleen-length-us\n    structures: [spleen]\n    aspect: length\n"
                      "    detail: sonographic\n    techniques: [US]\n    normal: {high: 120}\n    unit: mm\n"
                      "    source: {publisher: Radiopaedia, url: \"https://radiopaedia.org/articles/splenomegaly\", ref: \"15\"}\n"
                      "    status: verified\n")
    monkeypatch.setattr(rs, "radiopaedia_references", lambda url: {"15": {"doi": "10.1/abc", "pmid": "", "words": ""}})
    monkeypatch.setattr(rs, "primary_text", lambda ref: {"id": "PMC1", "read": "full_text", "sections": TEXT,
                                                        "source": {"title": "t", "doi": "10.1/abc", "pmcid": "PMC1",
                                                                   "licence": "cc by", "pmid": "", "year": "2020"}})
    calls = []

    def chat(self, model, prompt, *, reasoning, max_tokens):
        calls.append(model)
        if prompt.startswith("Structure:"):
            return '{"states": "yes"}', "stop"
        return json.dumps([item(), item(kind="mean_sd", high=10.6, quote="The mean splenic length was 10.6 ± 1.2 cm.")]), "stop"
    monkeypatch.setattr(rs.Router, "chat", chat)
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    assert rs.main(["--out", str(tmp_path / "out" / "reference.json"), "--values", str(values), "--only", "primary"]) == 0
    result = json.loads((tmp_path / "out" / "reference.json").read_text())
    assert result["primary"][0]["verdict"] == "confirmed"
    assert result["summary"]["refused_by_rule"] == {"mean_sd": 2}
    # resumed: nothing read or judged again
    calls.clear()
    assert rs.main(["--out", str(tmp_path / "again" / "reference.json"), "--values", str(values), "--only", "primary",
                    "--resume", str(tmp_path / "out")]) == 0
    assert calls == [] and json.loads((tmp_path / "again" / "reference.json").read_text())["primary"][0]["verdict"] == "confirmed"


def test_no_credit_stops_at_once(monkeypatch):
    import urllib.error
    import urllib.request

    def refuse(*a, **k):
        raise urllib.error.HTTPError("u", 402, "Payment Required", {}, None)
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr(rs.time, "sleep", lambda s: pytest.fail("it retried"))
    with pytest.raises(rs.OverBudget):
        rs.Router("x", 5).chat("m", "p", reasoning=True, max_tokens=10)


def test_the_budget_is_never_passed_each_call_holds_the_most_it_can_cost(monkeypatch):
    """(25 November 2026, the owner's rule: not over 5 USD) the router refuses a call when what is spent, what
    calls running hold and the most this call can cost would pass the budget."""
    router = rs.Router("x", 1.0, prices={"m": (1e-6, 10e-6)})        # 1 USD in, 10 USD out per million tokens
    assert abs(router.most("m", "p" * 3000, 50000) - (1064e-6 + 0.5)) < 1e-9

    def answered(model, prompt, *, reasoning, max_tokens):
        router.cost += 0.2                     # what OpenRouter says it cost, well under the most
        return "ok", "stop"
    monkeypatch.setattr(router, "_chat", answered)
    for _ in range(3):
        router.chat("m", "p", reasoning=True, max_tokens=50000)        # each holds 0.5 and spends 0.2
    assert abs(router.cost - 0.6) < 1e-9 and router.held == 0
    with pytest.raises(rs.OverBudget, match="at most for this call"):
        router.chat("m", "p", reasoning=True, max_tokens=50000)        # 0.6 + 0.5 would pass 1.0: refused
    router.held = 0.39                                                 # calls running hold 0.39
    with pytest.raises(rs.OverBudget):
        router.chat("m", "p", reasoning=True, max_tokens=2000)         # 0.6 + 0.39 + 0.02 would pass 1.0
    router.held = 0.30
    router.chat("m", "p", reasoning=True, max_tokens=2000)             # 0.6 + 0.30 + 0.02 does not
