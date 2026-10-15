"""CIPI and ICD-9-CM tables: links found, headers read by their words, unified dataset, Italian heads."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import classifications_it as ci  # noqa: E402

PAGE = """<a href="https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fwww.salute.gov.it%2Fnew%2Fsites%2Fdefault%2Ffiles%2F2026-02%2F6.TAB_CIPI_V.GAMMA_2.0.xlsx&amp;wdOrigin=BROWSELINK">CIPI</a>
<a href="https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fwww.salute.gov.it%2Fnew%2Fsites%2Fdefault%2Ffiles%2F2026-02%2F7.TRANSCOD_ICD-9-CM-CIPI_V.GAMMA_2.0.xlsx&amp;wdOrigin=BROWSELINK">T</a>
<a href="https://www.salute.gov.it/new/sites/default/files/2026-02/3.ELENCO_SISTEMATICO_ICD-10-IM_V.GAMMA_2.1_VALIDITA_DA_16022026.xlsx">ICD10</a>
<a href="/doc.pdf">pdf</a>"""


def test_links_unwrap_the_viewer_and_leave_icd10_out():
    got = ci.links(PAGE, "https://www.reteclassificazioni.it/x")
    assert [g["name"] for g in got] == ["6.TAB_CIPI_V.GAMMA_2.0.xlsx", "7.TRANSCOD_ICD-9-CM-CIPI_V.GAMMA_2.0.xlsx"]
    assert got[0]["url"].startswith("https://www.salute.gov.it/new/sites/default/files/2026-02/")
    assert got[0]["version"] == "gamma 2.0"


def test_the_header_is_found_by_its_words():
    rows = [["Ministero della Salute"], [None], ["Codice ICD-9-CM", "Descrizione ICD-9-CM", "Codice CIPI", "Descrizione CIPI"]]
    assert ci.header_row(rows) == 2
    assert ci.columns(rows[2]) == {"icd9_code": 0, "icd9_title": 1, "cipi_code": 2, "cipi_title": 3}


def test_build_unified_and_italian_heads(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    raw = tmp_path / "raw"
    raw.mkdir()
    book = openpyxl.Workbook()
    sh = book.active
    sh.append(["Tabella di transcodifica"])
    sh.append(["Codice ICD-9-CM", "Descrizione ICD-9-CM", "Codice CIPI", "Descrizione CIPI"])
    rows = [("47.09", "Altra appendicectomia", "47.0A", "Appendicectomia laparoscopica"),
            ("54.5", "Lisi di aderenze peritoneali", "54.5B", "Lisi di aderenze peritoneali per via laparoscopica"),
            ("43.5", "Gastrectomia parziale", "43.5A", "Resezione gastrica prossimale"),
            ("43.6", "Gastrectomia parziale con anastomosi", "43.6A", "Resezione gastrica distale"),
            ("45.7", "Asportazione parziale dell'intestino crasso", "45.7A", "Resezione del colon")]
    for r in rows:
        sh.append(list(r))
    book.save(raw / "7.TRANSCOD_ICD-9-CM-CIPI_V.GAMMA_2.0.xlsx")
    found = tmp_path / "found.json"
    found.write_text(json.dumps([{"name": "7.TRANSCOD_ICD-9-CM-CIPI_V.GAMMA_2.0.xlsx", "url": "u", "version": "gamma 2.0",
                                  "sha256": "x"}]))
    out = tmp_path / "out"
    assert ci.main(["build", "--raw", str(raw), "--found", str(found), "--out", str(out)]) == 0
    unified = [json.loads(x) for x in (out / "unified.jsonl").read_text().splitlines()]
    colon = next(u for u in unified if u["code"] == "45.7A")
    assert colon["icd9"] == [{"code": "45.7", "title": "Asportazione parziale dell'intestino crasso"}]
    heads = json.loads((out / "italian_heads.json").read_text())
    assert heads["kinds"].get("resezione") == "procedure"       # 3 titles
    assert "lisi" not in heads["kinds"]                         # 2 titles: not enough to be a correspondence


def test_check_says_what_changed(tmp_path, capsys):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"files": [{"url": "a", "sha256": "1"}]}))
    found = tmp_path / "f.json"
    found.write_text(json.dumps([{"url": "a", "name": "a", "sha256": "1"}]))
    ci.main(["check", "--found", str(found), "--manifest", str(manifest)])
    assert "changed=false" in capsys.readouterr().out
    found.write_text(json.dumps([{"url": "a", "name": "a", "sha256": "2"}]))
    ci.main(["check", "--found", str(found), "--manifest", str(manifest)])
    assert "changed=true" in capsys.readouterr().out


def test_a_page_instead_of_the_file_falls_back_to_the_browser_then_to_the_repository_copy(tmp_path, monkeypatch):
    found = tmp_path / "found.json"
    found.write_text(json.dumps([{"url": "https://x/6.TAB_CIPI.xlsx", "name": "6.TAB_CIPI.xlsx"}]))
    monkeypatch.setattr(ci, "_get", lambda url, **k: b"<!doctype html>")
    def no_browser(url, **k):
        raise RuntimeError("no chromium")
    monkeypatch.setattr(ci, "browser_get", no_browser)
    with pytest.raises(SystemExit, match="not downloaded"):
        ci.main(["fetch", "--found", str(found), "--dir", str(tmp_path / "raw"), "--fallback", str(tmp_path / "none")])
    copies = tmp_path / "copies"
    copies.mkdir()
    (copies / "6.TAB_CIPI.xlsx").write_bytes(b"PK\x03\x04copy")
    assert ci.main(["fetch", "--found", str(found), "--dir", str(tmp_path / "raw"), "--fallback", str(copies)]) == 0
    item = json.loads(found.read_text())[0]
    assert item["fetched_by"] == "repository copy"
    monkeypatch.setattr(ci, "browser_get", lambda url, **k: b"PK\x03\x04real")
    assert ci.main(["fetch", "--found", str(found), "--dir", str(tmp_path / "raw"), "--fallback", str(copies)]) == 0
    assert json.loads(found.read_text())[0]["fetched_by"] == "browser"
