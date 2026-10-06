"""Public radiology report corpora -> reports.jsonl for scripts/gold_set.py sample.

  discover  DIR            list the tabular files under DIR with their columns and row counts
  tree      DIR            count files by extension and list the first paths (find the real layout)
  parrot    DIR  --out F   PARROT (fictional reports written by radiologists, 13 languages) -> jsonl
  iuxray    --out F        Indiana University chest X-ray reports (NLM Open-i, anonymised) -> jsonl
  e3c       DIR  --out F   E3C clinical cases (FBK, CC BY-NC), one language, one case per report -> jsonl
  multicare-files          list the files of the MultiCaRe record on Zenodo (CC BY 4.0)
  multicare --out F        MultiCaRe case reports that mention imaging (English) -> jsonl

All corpora are public. PARROT is fictional: no patient is involved, so it is safe to share, but it is
written by clinicians who know the task, so it is a development and stress set, not proof of real-world
accuracy. IU X-ray is real, English, chest radiographs only. Neither replaces the hospital's own
pseudonymised Italian reports for the certification set. E3C and MultiCaRe are clinical case narratives,
not radiology reports: they carry imaging findings in running text. Check each corpus's licence before any
use that goes beyond internal evaluation (PARROT and E3C may be non-commercial).
"""

import argparse
import csv
import gzip
import io
import json
import re
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

IU_URL = "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz"
_TABULAR = (".csv", ".tsv", ".json", ".jsonl")
_LANG_COLUMNS = ("language", "lang", "lingua")
_TEXT_COLUMNS = (
    "report",
    "report_text",
    "text",
    "reporttext",
    "full_report",
    "referto",
)
_ID_COLUMNS = ("report_id", "id", "uid", "reportid", "report_uid")
_ITALIAN = {"it", "ita", "italian", "italiano"}
ZENODO_MULTICARE = "https://zenodo.org/api/records/10079370/versions/latest"
_IMAGING = re.compile(
    r"\b(?:ct|mri|mr|ultrasound|ultrasonograph\w*|radiograph\w*|x-?ray|tomograph\w*"
    r"|magnetic resonance|scan|sonograph\w*)\b",
    re.IGNORECASE,
)


def iu_reports(archive: bytes) -> list[dict]:
    """Findings and impression of each IU X-ray XML report. Anonymisation placeholders (XXXX) are kept."""
    rows = []
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".xml"):
                continue
            handle = tar.extractfile(member)
            if handle is None:
                continue
            root = ET.fromstring(handle.read())
            parts = {}
            for node in root.iter("AbstractText"):
                label = (node.get("Label") or "").upper()
                if label in ("FINDINGS", "IMPRESSION"):
                    parts[label] = " ".join((node.text or "").split())
            text = " ".join(
                parts[k] for k in ("FINDINGS", "IMPRESSION") if parts.get(k)
            )
            if text:
                rows.append(
                    {
                        "report_id": "iuxray-" + Path(member.name).stem,
                        "text": text,
                        "language": "en",
                        "site": "iu-xray",
                    }
                )
    return sorted(rows, key=lambda r: r["report_id"])


def read_table(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    raw = path.read_text("utf-8-sig")
    if suffix == ".jsonl":
        return [json.loads(x) for x in raw.splitlines() if x.strip()]
    if suffix == ".json":
        data = json.loads(raw)
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [])
        return [r for r in data if isinstance(r, dict)]
    delimiter = "\t" if suffix == ".tsv" else ","
    return list(csv.DictReader(io.StringIO(raw), delimiter=delimiter))


def _column(row: dict, names: tuple[str, ...]) -> str | None:
    lowered = {k.strip().lower().replace(" ", "_"): k for k in row}
    for name in names:
        if name in lowered:
            return lowered[name]
    return None


def parrot_reports(rows: list[dict], source: str, language: set[str]) -> list[dict]:
    """Rows of one PARROT file in the wanted languages. Fails loudly if the columns are not recognised."""
    if not rows:
        return []
    text_key = _column(rows[0], _TEXT_COLUMNS)
    lang_key = _column(rows[0], _LANG_COLUMNS)
    id_key = _column(rows[0], _ID_COLUMNS)
    if text_key is None or lang_key is None:
        raise ValueError(
            f"{source}: cannot find the text and language columns among {list(rows[0])}"
        )
    out = []
    for index, row in enumerate(rows):
        code = str(row.get(lang_key, "")).strip().lower()
        text = " ".join(str(row.get(text_key, "")).split())
        if code in language and text:
            rid = str(row[id_key]).strip() if id_key and row.get(id_key) else str(index)
            out.append(
                {
                    "report_id": f"parrot-{Path(source).stem}-{rid}",
                    "text": text,
                    "language": "it" if code in _ITALIAN else code,
                    "site": "parrot",
                }
            )
    return out


def _tables(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in _TABULAR
        and ".git" not in p.parts
        and p.stat().st_size > 0
    )


def discover(root: Path) -> str:
    lines = []
    for path in _tables(root):
        try:
            rows = read_table(path)
            columns = list(rows[0]) if rows else []
            lines.append(
                f"{path.relative_to(root)}: {len(rows)} rows, columns {columns}"
            )
        except (ValueError, OSError, csv.Error) as exc:
            lines.append(f"{path.relative_to(root)}: unreadable ({exc})")
    return "\n".join(lines) or "no tabular files found"


def tree(root: Path, limit: int = 40) -> str:
    """File counts by extension and the first paths: the real layout of a cloned corpus."""
    files = sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts)
    counts: dict[str, int] = {}
    for p in files:
        counts[p.suffix.lower() or "(none)"] = (
            counts.get(p.suffix.lower() or "(none)", 0) + 1
        )
    lines = [f"{len(files)} files; by extension: {dict(sorted(counts.items()))}"]
    lines += [str(p.relative_to(root)) for p in files[:limit]]
    return "\n".join(lines)


def _sofa(path: Path) -> str:
    """The document text of a UIMA XMI annotation file (the Sofa string)."""
    for node in ET.parse(path).getroot().iter():
        if node.tag.endswith("Sofa") and node.get("sofaString"):
            return " ".join(node.get("sofaString", "").split())
    return ""


def e3c_reports(root: Path, language: str = "Italian") -> list[dict]:
    """E3C clinical cases of one language. Plain texts from data_collection, else the annotation files' text."""
    code = "it" if language.lower() in _ITALIAN else language[:2].lower()
    rows = []
    for folder, suffixes in (
        ("data_collection", (".txt",)),
        ("data_annotation", (".xml", ".xmi")),
    ):
        base = root / folder / language
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if suffixes == (".txt",):
                text = " ".join(path.read_text("utf-8", errors="replace").split())
            else:
                text = _sofa(path)
            layer = path.relative_to(base).parts[0]
            if text:
                rows.append(
                    {
                        "report_id": f"e3c-{code}-{layer}-{path.stem}",
                        "text": text,
                        "language": code,
                        "site": f"e3c-{layer}",
                    }
                )
        if rows:
            return rows
    return rows


def zenodo_files(record: dict) -> list[dict]:
    """Name, size and download link of each file of a Zenodo record (old and new API shapes)."""
    files = record.get("files", [])
    if isinstance(files, dict):
        files = list(files.get("entries", {}).values())
    out = []
    for f in files:
        key = f.get("key") or f.get("filename")
        link = (f.get("links") or {}).get("self") or (f.get("links") or {}).get(
            "content"
        )
        if key and link:
            out.append({"key": key, "size": int(f.get("size", 0)), "url": link})
    return out


def pick_cases_file(files: list[dict], max_bytes: int) -> dict | None:
    """The smallest CSV that looks like the cases table and fits the size limit."""
    fits = [
        f
        for f in files
        if "case" in f["key"].lower()
        and f["key"].lower().endswith((".csv", ".csv.gz"))
        and f["size"] <= max_bytes
    ]
    return min(fits, key=lambda f: f["size"]) if fits else None


def multicare_reports(rows: list[dict]) -> list[dict]:
    """English case reports whose text mentions imaging; other cases carry no anatomy to link."""
    if not rows:
        return []
    text_key = _column(rows[0], ("case_text", "text", "case"))
    id_key = _column(rows[0], ("case_id", "id"))
    if text_key is None:
        raise ValueError(f"cannot find the case text column among {list(rows[0])}")
    out = []
    for index, row in enumerate(rows):
        text = " ".join(str(row.get(text_key, "")).split())
        if text and _IMAGING.search(text):
            rid = str(row[id_key]).strip() if id_key and row.get(id_key) else str(index)
            out.append(
                {
                    "report_id": f"multicare-{rid}",
                    "text": text,
                    "language": "en",
                    "site": "multicare",
                }
            )
    return out


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310 (https URL from the Zenodo API)
        return response.read()


def _write(rows: list[dict], out: Path) -> None:
    out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8"
    )
    print(f"{len(rows)} reports -> {out}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("discover")
    d.add_argument("root")
    p = sub.add_parser("parrot")
    p.add_argument("root")
    p.add_argument("--out", default="parrot_it.jsonl")
    p.add_argument(
        "--lang", default="it", help="comma-separated language codes or names"
    )
    i = sub.add_parser("iuxray")
    i.add_argument("--out", default="iuxray.jsonl")
    t = sub.add_parser("tree")
    t.add_argument("root")
    e = sub.add_parser("e3c")
    e.add_argument("root")
    e.add_argument("--out", default="e3c_it.jsonl")
    e.add_argument("--language", default="Italian")
    sub.add_parser("multicare-files")
    m = sub.add_parser("multicare")
    m.add_argument("--out", default="multicare.jsonl")
    m.add_argument("--max-mb", type=int, default=800)
    args = parser.parse_args(argv)

    if args.cmd == "discover":
        print(discover(Path(args.root)))
        return 0
    if args.cmd == "tree":
        print(tree(Path(args.root)))
        return 0
    if args.cmd == "e3c":
        rows = e3c_reports(Path(args.root), args.language)
        if not rows:
            print("no E3C case found: run 'tree' and adapt the layout", file=sys.stderr)
            return 2
        _write(rows, Path(args.out))
        return 0
    if args.cmd in ("multicare-files", "multicare"):
        files = zenodo_files(json.loads(_get(ZENODO_MULTICARE)))
        if args.cmd == "multicare-files":
            for f in sorted(files, key=lambda f: f["size"]):
                print(f"{f['size'] / 1e6:10.1f} MB  {f['key']}")
            return 0
        chosen = pick_cases_file(files, args.max_mb * 1_000_000)
        if chosen is None:
            print(
                "no cases CSV within the size limit: run 'multicare-files'",
                file=sys.stderr,
            )
            return 2
        raw = _get(chosen["url"])
        if chosen["key"].lower().endswith(".gz"):
            raw = gzip.decompress(raw)
        csv.field_size_limit(sys.maxsize)
        table = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        _write(multicare_reports(table), Path(args.out))
        return 0
    if args.cmd == "parrot":
        wanted = {x.strip().lower() for x in args.lang.split(",")}
        if wanted & _ITALIAN:
            wanted |= _ITALIAN
        rows = []
        for path in _tables(Path(args.root)):
            try:
                rows += parrot_reports(read_table(path), path.name, wanted)
            except ValueError as exc:
                print(f"skipped: {exc}", file=sys.stderr)
        if not rows:
            print(
                "no report recognised: run 'discover' and adapt the columns",
                file=sys.stderr,
            )
            return 2
        _write(rows, Path(args.out))
        return 0
    with urllib.request.urlopen(IU_URL, timeout=120) as response:  # noqa: S310 (fixed https URL)
        _write(iu_reports(response.read()), Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
