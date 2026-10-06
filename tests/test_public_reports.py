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


def test_e3c_reads_plain_texts_and_keeps_the_layer(tmp_path):
    base = tmp_path / "data_collection" / "Italian" / "layer1"
    base.mkdir(parents=True)
    (base / "c1.txt").write_text(
        "Donna di 40 anni.\nTC addome:  fegato nei limiti.", "utf-8"
    )
    (tmp_path / "data_collection" / "French" / "layer1").mkdir(parents=True)
    (tmp_path / "data_collection" / "French" / "layer1" / "f.txt").write_text(
        "Femme.", "utf-8"
    )
    rows = fpr.e3c_reports(tmp_path, "Italian")
    assert rows == [
        {
            "report_id": "e3c-it-layer1-c1",
            "text": "Donna di 40 anni. TC addome: fegato nei limiti.",
            "language": "it",
            "site": "e3c-layer1",
        }
    ]


def test_e3c_falls_back_to_annotation_files(tmp_path):
    base = tmp_path / "data_annotation" / "Italian" / "layer2"
    base.mkdir(parents=True)
    (base / "c2.xmi").write_text(
        '<xmi:XMI xmlns:xmi="http://www.omg.org/XMI" xmlns:cas="http:///uima/cas.ecore">'
        '<cas:Sofa xmi:id="1" sofaString="Milza ingrandita.   Rene destro normale."/></xmi:XMI>',
        "utf-8",
    )
    rows = fpr.e3c_reports(tmp_path, "Italian")
    assert [r["text"] for r in rows] == ["Milza ingrandita. Rene destro normale."]
    assert rows[0]["site"] == "e3c-layer2"


def test_e3c_missing_language_gives_nothing(tmp_path):
    assert fpr.e3c_reports(tmp_path, "Italian") == []


def test_multicare_keeps_cases_that_mention_imaging():
    rows = [
        {
            "case_id": "a",
            "case_text": "A 5-year-old boy. Abdominal CT showed a liver mass.",
        },
        {"case_id": "b", "case_text": "The patient was given aspirin."},
        {"case_id": "c", "case_text": "Ultrasound of the right kidney was normal."},
    ]
    assert [r["report_id"] for r in fpr.multicare_reports(rows)] == [
        "multicare-a",
        "multicare-c",
    ]


def test_multicare_imaging_words_need_word_boundaries():
    rows = [{"case_id": "x", "case_text": "Scanty discharge and a doctor's note."}]
    assert fpr.multicare_reports(rows) == []


def test_multicare_unknown_columns_fail_loudly():
    with pytest.raises(ValueError, match="cannot find the case text column"):
        fpr.multicare_reports([{"a": "1"}])


def test_zenodo_files_both_api_shapes_and_cases_pick():
    old = {"files": [{"key": "cases.csv", "size": 10, "links": {"self": "u1"}}]}
    new = {
        "files": {
            "entries": {
                "a": {
                    "key": "big_cases.csv",
                    "size": 9_000_000_000,
                    "links": {"content": "u2"},
                },
                "b": {"key": "images.zip", "size": 5, "links": {"content": "u3"}},
            }
        }
    }
    assert fpr.zenodo_files(old) == [{"key": "cases.csv", "size": 10, "url": "u1"}]
    files = fpr.zenodo_files(new)
    assert len(files) == 2
    assert fpr.pick_cases_file(files, 1_000_000) is None
    assert (
        fpr.pick_cases_file(files + fpr.zenodo_files(old), 1_000_000)["key"]
        == "cases.csv"
    )


def test_tree_counts_extensions_and_skips_git(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "x.txt").write_text("x", "utf-8")
    (tmp_path / "a.txt").write_text("a", "utf-8")
    (tmp_path / "b.json").write_text("{}", "utf-8")
    out = fpr.tree(tmp_path)
    assert out.startswith("2 files") and ".git" not in out


def test_jsonl_is_split_on_newlines_only(tmp_path):
    """U+2028 and U+0085 inside a JSON string must not cut a record (PARROT v1.0 broke on this)."""
    path = tmp_path / "d.jsonl"
    rows = [
        {"language": "it", "report": "Fegato\u2028nei limiti.\x85Milza normale."},
        {"language": "it", "report": "Rene destro normale."},
    ]
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", "utf-8"
    )
    assert fpr.read_table(path) == rows


def test_multicare_picks_a_parquet_cases_table_and_skips_images_and_captions():
    files = [
        {"key": "case_images.parquet", "size": 50, "url": "a"},
        {"key": "captions_and_labels.csv", "size": 40, "url": "b"},
        {"key": "cases.parquet", "size": 168, "url": "c"},
        {"key": "PMC1.zip", "size": 9, "url": "d"},
    ]
    assert fpr.pick_cases_file(files, 1000)["key"] == "cases.parquet"


def test_tree_shows_the_start_of_one_file_per_text_kind(tmp_path):
    (tmp_path / "a.json").write_text('{"text": "Fegato nei limiti."}', "utf-8")
    out = fpr.tree(tmp_path)
    assert "--- start of a.json" in out and "Fegato nei limiti." in out
