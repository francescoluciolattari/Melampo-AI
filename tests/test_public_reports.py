import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "fetch_public_reports",
    Path(__file__).resolve().parent.parent / "scripts" / "fetch_public_reports.py",
)
fpr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fpr)

XML = """<eCitation><MedlineCitation><Article><Abstract>
<AbstractText Label="COMPARISON">None.</AbstractText>
<AbstractText Label="INDICATION">Cough.</AbstractText>
<AbstractText Label="FINDINGS">The heart  is normal in size.
No pleural effusion.</AbstractText>
<AbstractText Label="IMPRESSION">No acute disease.</AbstractText>
</Abstract></Article></MedlineCitation></eCitation>"""


def _tgz(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, body in files.items():
            data = body.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_iuxray_keeps_findings_and_impression_only():
    rows = fpr.iu_reports(_tgz({"ecgen-radiology/7.xml": XML, "readme.txt": "x"}))
    assert rows == [
        {
            "report_id": "iuxray-7",
            "text": "The heart is normal in size. No pleural effusion. No acute disease.",
            "language": "en",
            "site": "iu-xray",
        }
    ]


def test_iuxray_skips_reports_without_text():
    empty = (
        "<eCitation><AbstractText Label='COMPARISON'>None.</AbstractText></eCitation>"
    )
    assert fpr.iu_reports(_tgz({"1.xml": empty})) == []


def test_parrot_csv_keeps_italian_rows_only(tmp_path):
    table = tmp_path / "reports.csv"
    table.write_text(
        "ID,Language,Report\n1,Italian,Fegato nei limiti.\n2,German,Leber normal.\n3,it,Milza ingrandita.\n",
        "utf-8",
    )
    rows = fpr.parrot_reports(fpr.read_table(table), table.name, fpr._ITALIAN)
    assert [r["report_id"] for r in rows] == ["parrot-reports-1", "parrot-reports-3"]
    assert {r["language"] for r in rows} == {"it"}


def test_parrot_jsonl_and_nested_json(tmp_path):
    (tmp_path / "a.jsonl").write_text(
        json.dumps({"id": "x", "lang": "it", "text": "Rene destro normale."}) + "\n",
        "utf-8",
    )
    (tmp_path / "b.json").write_text(
        json.dumps(
            {"reports": [{"language": "it", "report": "Colecisti alitiasica."}]}
        ),
        "utf-8",
    )
    found = [
        r
        for p in fpr._tables(tmp_path)
        for r in fpr.parrot_reports(fpr.read_table(p), p.name, fpr._ITALIAN)
    ]
    assert {r["text"] for r in found} == {
        "Rene destro normale.",
        "Colecisti alitiasica.",
    }


def test_parrot_unknown_columns_fail_loudly(tmp_path):
    table = tmp_path / "odd.csv"
    table.write_text("a,b\n1,2\n", "utf-8")
    with pytest.raises(ValueError, match="cannot find the text and language columns"):
        fpr.parrot_reports(fpr.read_table(table), table.name, fpr._ITALIAN)


def test_discover_lists_columns_and_ignores_git(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "x.json").write_text("{}", "utf-8")
    (tmp_path / "d.csv").write_text("a,b\n1,2\n", "utf-8")
    out = fpr.discover(tmp_path)
    assert "d.csv: 1 rows, columns ['a', 'b']" in out and ".git" not in out
