"""Public radiology report corpora -> reports.jsonl for scripts/gold_set.py sample.

  discover  DIR            list the tabular files under DIR with their columns and row counts
  parrot    DIR  --out F   PARROT (fictional reports written by radiologists, 13 languages) -> jsonl
  iuxray    --out F        Indiana University chest X-ray reports (NLM Open-i, anonymised) -> jsonl

Both corpora are public. PARROT is fictional: no patient is involved, so it is safe to share, but it is
written by clinicians who know the task, so it is a development and stress set, not proof of real-world
accuracy. IU X-ray is real, English, chest radiographs only. Neither replaces the hospital's own
pseudonymised Italian reports for the certification set. Check each corpus's licence before any use that
goes beyond internal evaluation (PARROT may be non-commercial).
"""

import argparse
import csv
import io
import json
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
    args = parser.parse_args(argv)

    if args.cmd == "discover":
        print(discover(Path(args.root)))
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
