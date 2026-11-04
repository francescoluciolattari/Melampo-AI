"""The reference library (4 November 2026): a measure is compared with a cited reference value, not with what a
model remembers; the reader recomputes the comparison of the cited row."""

import json

from melampo.evaluation.llm_reading import claims_prompt, read_claims
from melampo.memory.finding_owner import HERE, UNSURE, FindingOwner
from melampo.memory.reference_values import (
    ABOVE,
    WITHIN,
    ReferenceValues,
    Row,
    check,
    compare,
    measures_in,
)


def _row(**kw):
    base = {"id": "pv", "structures": ("portal_vein_and_splenic_vein",), "aspect": "diameter",
            "detail": "main portal vein", "techniques": ("any",), "population": {"age": "adult", "sex": "any"},
            "low": 7.0, "high": 13.0, "unit": "mm",
            "source": {"publisher": "Radiopaedia", "title": "Portal vein", "section": "gross anatomy"}}
    base.update(kw)
    return Row(**base)


def test_every_row_of_the_table_has_its_source_and_a_status():
    table = ReferenceValues.load()
    assert table.rows
    for r in table.rows:
        assert r.source.get("url") and r.source.get("licence"), r.id
        assert r.status in ("to_verify", "verified", "reviewed"), r.id
        assert r.low is not None or r.high is not None, r.id
    assert len({r.id for r in table.rows}) == len(table.rows)


def test_rows_apply_by_structure_technique_and_sex():
    table = ReferenceValues.load()
    assert {r.id for r in table.rows_for("spleen", "US")} == {"spleen-length-us"}
    assert {r.id for r in table.rows_for("kidney_left", "CT", "female")} == {"kidney-length-female"}
    assert {r.id for r in table.rows_for("kidney_left", "CT")} == {"kidney-length-male", "kidney-length-female"}
    assert table.rows_for("pancreas", "CT") == []


def test_measures_are_read_in_the_units_of_the_rows():
    assert [m["value"] for m in measures_in("measures 1.6 cm in width")] == [16.0]
    assert [(m["value"], m["unit"]) for m in measures_in("18 x 6 mm")] == [(18.0, "mm")]
    assert [(m["value"], m["unit"]) for m in measures_in("volume: 109.01 cm3")] == [(109.01, "mL")]
    assert [m["unit"] for m in measures_in("LA volume index 41 ml/m2")] == ["mL/m2"]


def test_the_reader_recomputes_the_comparison_of_the_cited_row():
    pv = _row()
    assert compare(16, pv) == ABOVE and compare(10, pv) == WITHIN
    assert check("measures approximately 16mm in width", "pv", "above", [pv]) == ""
    assert check("measures approximately 16mm in width", "pv", "within", [pv]) == "the_reference_says_above"
    assert check("measures approximately 16mm", "other", "above", [pv]) == "the_cited_reference_was_not_given"
    assert check("volume 109 cm3", "pv", "above", [pv]) == "no_measure_in_the_unit_of_the_reference"


def test_the_question_gives_the_rows_and_asks_to_cite_them():
    s = "The portal vein measures approximately 16mm in width"
    p = " ".join(claims_prompt(s, 4, 15, "portal vein", references=[_row().line()]).split())
    assert "[pv]" in p and "cite the id" in p and "never compared with a value you remember" in p
    p = " ".join(claims_prompt(s, 4, 15, "portal vein").split())
    assert "none for this structure and exam" in p


def _owner(answer, rows):
    def chat(prompt):
        return json.dumps(answer)

    chats = {"m0": chat, "m1": chat}
    return FindingOwner(None, chats, question="claims", references=ReferenceValues(rows))


def _answer(against):
    return {"marked": "portal vein", "same": "yes", "observed": [], "normal": [], "past": [], "certainty": [],
            "alternatives": [], "device": [], "not_assessable": [],
            "measure": [{"words": "measures approximately 16mm in width", "reference": "pv", "against": against}]}


def test_a_comparison_the_cited_row_supports_is_attached_with_its_reference():
    s = "The portal vein measures approximately 16mm in width"
    read = read_claims(json.dumps(_answer("above")), s, "portal vein")
    assert read["measure_refs"] == [{"words": "measures approximately 16mm in width", "reference": "pv",
                                     "against": "above"}]
    got = _owner(_answer("above"), [_row()])("portal vein", s, 4, "en", "portal_vein_and_splenic_vein")
    assert got.verdict == HERE and got.claims["measure_refs"][0]["reference"] == "pv"


def test_a_comparison_the_cited_row_does_not_support_goes_to_a_person():
    s = "The portal vein measures approximately 16mm in width"
    got = _owner(_answer("within"), [_row()])("portal vein", s, 4, "en", "portal_vein_and_splenic_vein")
    assert got.verdict == UNSURE and "measure_the_reference_says_above" in got.reader
    # a row the model was not given (one it remembers) is not a reference
    got = _owner(_answer("above"), [])("portal vein", s, 4, "en", "portal_vein_and_splenic_vein")
    assert got.verdict == UNSURE and "measure_the_cited_reference_was_not_given" in got.reader


def test_paragraphs_never_a_chapter_and_their_links(tmp_path):
    import subprocess
    import sys

    text = tmp_path / "p.txt"
    text.write_text("word " * 251, "utf-8")
    out = tmp_path / "passages.jsonl"
    cmd = [sys.executable, "scripts/reference_passages.py", "add", "--url", "u", "--section", "s", "--title", "t",
           "--licence", "CC BY", "--text-file", str(text), "--out", str(out)]
    assert subprocess.run(cmd, capture_output=True, check=False).returncode == 2
    text.write_text("The main portal vein has a normal diameter.", "utf-8")
    assert subprocess.run(cmd, capture_output=True, check=False).returncode == 0
    rows = [json.loads(x) for x in out.read_text("utf-8").splitlines()]
    table = ReferenceValues([_row(source={"url": "u", "section": "s"})], rows)
    assert [p["text"] for p in table.passages_for(table.rows)] == ["The main portal vein has a normal diameter."]
