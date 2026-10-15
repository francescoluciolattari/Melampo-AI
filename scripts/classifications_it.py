#!/usr/bin/env python
"""Italian classifications of procedures: CIPI and ICD-9-CM, kept up to date (14 October 2026).

Why: the Kintsch reader has no Italian vocabulary; "asportazione", "liberazione", "intervento", "TC" have no
kind for it, and it reads the structure they govern as a plain structure. The Ministry of Health publishes the
new Italian classification of procedures (CIPI, DM 23 October 2025, in force from 1 January 2027) with the
transcoding tables from ICD-9-CM, the classification in use until then. Their titles are the Italian names
of procedures, written by coders: the source of the kinds of Italian words. Licence: confirmed by Frank for
CIPI (14 October 2026); ICD-10-IM (diagnoses) is not downloaded until its licence is checked.

Steps (``.github/workflows/classifications-it.yml`` runs them every week):

* ``discover``  the pages that list the files (reteclassificazioni.it, the Ministry): every link to a CIPI or
                transcoding table (xlsx), unwrapping the Office viewer links to the file itself;
* ``fetch``     the files, retried when the site answers with a page instead of the file; SHA-256 of each;
* ``build``     the tables read with their header found by its words (the column order may change between
                versions): one row per code (system, code, title, version), the transcoding links between
                ICD-9-CM and CIPI, a unified dataset (``unified.jsonl``: each CIPI code with its ICD-9-CM codes
                and titles) and the kinds of Italian words (``italian_heads.json``: the first word of every
                procedure title, with its count: "asportazione", "resezione", "biopsia" -> procedure);
* ``check``     compares with ``data/classifications_it/manifest.json``: new or changed files -> changed=true;
* ``falkordb``  loads the unified dataset into FalkorDB (graph ``classifications_it``: Code nodes, MAPS_TO edges).

    python scripts/classifications_it.py discover --out found.json
    python scripts/classifications_it.py fetch --found found.json --dir raw
    python scripts/classifications_it.py build --raw raw --found found.json --out data/classifications_it
    python scripts/classifications_it.py check --found found.json --manifest data/classifications_it/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PAGES = (
    # NSIS-CLASS documentation (lists the gamma prototypes with their links; 9 March 2026, updated since)
    "https://www.reteclassificazioni.it/portal_main.php?portal_view=news_dettaglio&id=116",
    "https://www.reteclassificazioni.it/portal_main.php?portal_view=news_dettaglio&id=115",
    # The Ministry's page of the CIPI (may answer with a browser check to a script: then it gives no link)
    "https://www.salute.gov.it/new/it/tema/assistenza-ospedaliera-sdo/la-classificazione-delle-procedure-cipi/",
)
# Files kept: CIPI and its transcoding to ICD-9-CM (licence confirmed). ICD-10-IM is left out on purpose.
WANTED = re.compile(r"(CIPI|TRANSCOD[^/]*CIPI)", re.I)
LEFT_OUT = re.compile(r"ICD-10-IM", re.I)
UA = {"User-Agent": "Melampo-AI classification update check (GitHub Actions)"}


def _get(url: str, attempts: int = 4, wait: float = 20.0) -> bytes:
    last = ""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180) as r:  # noqa: S310
                return r.read()
        except OSError as error:
            last = f"{type(error).__name__}: {error}"
        print(f"attempt {attempt}/{attempts} {url}: {last}", file=sys.stderr)
        if attempt < attempts:
            time.sleep(wait * attempt)
    raise OSError(last)


def file_url(href: str, base: str) -> str:
    """The file a link points to: the Office viewer link (view.officeapps.live.com/...?src=...) unwrapped."""
    href = html.unescape(href)
    url = urllib.parse.urljoin(base, href)
    parsed = urllib.parse.urlparse(url)
    if "officeapps.live.com" in parsed.netloc:
        src = urllib.parse.parse_qs(parsed.query).get("src", [""])[0]
        if src:
            return src
    return url


def version_of(url: str) -> str:
    m = re.search(r"V\.?\s*(ALFA|BETA|GAMMA|DELTA)[_\s.]*([0-9]+(?:\.[0-9]+)*)", urllib.parse.unquote(url), re.I)
    return f"{m.group(1).lower()} {m.group(2)}" if m else ""


def links(page: str, base: str) -> list[dict]:
    out = []
    for m in re.finditer(r"""href\s*=\s*["']([^"']+)["']""", page, re.I):
        url = file_url(m.group(1), base)
        name = urllib.parse.unquote(url.rsplit("/", 1)[-1])
        if not re.search(r"\.(xlsx?|csv|zip)$", name, re.I):
            continue
        if LEFT_OUT.search(name) or not WANTED.search(name):
            continue
        out.append({"url": url, "name": name, "version": version_of(url), "listed_on": base})
    return out


def discover(args) -> int:
    found, seen = [], set()
    for page in args.pages or PAGES:
        try:
            text = _get(page, attempts=2).decode("utf-8", "replace")
        except OSError as error:
            print(f"page not read: {page}: {error}", file=sys.stderr)
            continue
        for item in links(text, page):
            if item["url"] not in seen:
                seen.add(item["url"])
                found.append(item)
    Path(args.out).write_text(json.dumps(found, indent=1, ensure_ascii=False), "utf-8")
    print(json.dumps(found, indent=1, ensure_ascii=False))
    if not found:
        print("::warning::no CIPI file found on the pages: the pages may have changed", file=sys.stderr)
    return 0


CIPI_PAGE = "https://www.salute.gov.it/new/it/tema/assistenza-ospedaliera-sdo/la-classificazione-delle-procedure-cipi/"
BROWSER_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36")


def _is_file(data: bytes) -> bool:
    return data[:2] == b"PK"  # xlsx and zip are zip archives: anything else is a page, not the file


def browser_get(url: str, page_first: str = CIPI_PAGE, wait_ms: int = 10000) -> bytes:
    """The file through a real browser (Playwright, headless Chromium). salute.gov.it answers a script with a
    page that checks the browser (JavaScript and cookies) before it lets the file through (15 Oct 2026): the
    browser opens the Ministry's page first, passes the check, then asks for the file with the same cookies."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            context = browser.new_context(user_agent=BROWSER_UA, accept_downloads=True, locale="it-IT")
            page = context.new_page()
            page.goto(page_first, wait_until="domcontentloaded", timeout=120000)
            page.wait_for_timeout(wait_ms)
            data = context.request.get(url, timeout=120000).body()
            if _is_file(data):
                return data
            with page.expect_download(timeout=120000) as download:
                page.evaluate("u => { const a = document.createElement('a'); a.href = u; a.click(); }", url)
            return Path(download.value.path()).read_bytes()
        finally:
            browser.close()


def fetch(args) -> int:
    """Each file: a plain request first; if the site answers with a page, a real browser; if that fails too,
    the copy kept in the repository (``--fallback``), marked as such: the check then says it could not look."""
    found = json.loads(Path(args.found).read_text("utf-8"))
    raw = Path(args.dir)
    raw.mkdir(parents=True, exist_ok=True)
    fallback = Path(args.fallback) if args.fallback else None
    missing = []
    for item in found:
        data, how = b"", ""
        try:
            data, how = _get(item["url"]), "request"
        except OSError as error:
            print(f"request failed: {error}", file=sys.stderr)
        if not _is_file(data):
            print(f"{item['name']}: the site answered with a page ({data[:40]!r}); trying a browser", file=sys.stderr)
            try:
                data, how = browser_get(item["url"]), "browser"
            except Exception as error:  # noqa: BLE001 - any browser failure leads to the fallback
                print(f"browser failed: {type(error).__name__}: {error}", file=sys.stderr)
                data = b""
        if not _is_file(data) and fallback is not None and (fallback / item["name"]).exists():
            data, how = (fallback / item["name"]).read_bytes(), "repository copy"
            print(f"::warning::{item['name']}: not downloadable now; using the copy in the repository "
                  "(the check for a new version could not be done for this file)", file=sys.stderr)
        if not _is_file(data):
            missing.append(item["name"])
            continue
        (raw / item["name"]).write_bytes(data)
        item["sha256"] = hashlib.sha256(data).hexdigest()
        item["bytes"] = len(data)
        item["fetched_by"] = how
    Path(args.found).write_text(json.dumps(found, indent=1, ensure_ascii=False), "utf-8")
    if missing:
        raise SystemExit(f"not downloaded and no copy in the repository: {missing}. Download them by hand from "
                         f"{CIPI_PAGE} and put them in {fallback or 'the fallback folder'}.")
    return 0


# -- reading the tables -------------------------------------------------------------------------------


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text).strip().lower()


def header_row(rows: list[list]) -> int:
    """The first row whose cells name codes and descriptions (the files have title rows above it)."""
    for i, row in enumerate(rows[:30]):
        cells = [_fold(c) for c in row if c is not None]
        if sum(any(k in c for k in ("cod", "descr", "titolo")) for c in cells) >= 2:
            return i
    return 0


def columns(header: list) -> dict[str, int]:
    """Which column holds what, by the words of its header: CIPI or ICD-9-CM, code or description."""
    found: dict[str, int] = {}
    for i, cell in enumerate(header):
        h = _fold(cell)
        system = "icd9" if re.search(r"icd[- ]?9|9[- ]?cm", h) else "cipi" if "cipi" in h else ""
        # "descrizione ICD-9-CM_primo codice" is a description: the description words are looked for first
        what = "title" if any(k in h for k in ("descr", "titolo", "denomin")) else "code" if "cod" in h else ""
        if what:
            found.setdefault(f"{system or 'any'}_{what}", i)
    return found


def read_table(path: Path) -> tuple[list[str], list[list]]:
    import openpyxl

    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = book.worksheets[0]
    rows = [list(r) for r in sheet.iter_rows(values_only=True)]
    h = header_row(rows)
    return [str(c or "") for c in rows[h]], [r for r in rows[h + 1:] if any(c not in (None, "") for c in r)]


def _code(value) -> str:
    return str(value).strip() if value not in (None, "") else ""


def build(args) -> int:
    found = {f["name"]: f for f in json.loads(Path(args.found).read_text("utf-8"))}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    codes: dict[tuple[str, str], dict] = {}
    maps: list[dict] = []
    report = {}
    for path in sorted(Path(args.raw).glob("*.xls*")):
        header, rows = read_table(path)
        cols = columns(header)
        version = found.get(path.name, {}).get("version", "")
        report[path.name] = {"header": header, "columns": cols, "rows": len(rows), "version": version}
        print(f"{path.name}: {len(rows)} rows, columns {cols}, header {header}", file=sys.stderr)
        # The CIPI list has the CIPI code and its description (and the ICD-9-CM codes it comes from, in one
        # cell); the transcoding table has both codes and both descriptions.
        transcoding = "icd9_title" in cols and "cipi_code" in cols
        for row in rows:
            def cell(key):
                i = cols.get(key)
                return row[i] if i is not None and i < len(row) else None
            cc = _code(cell("cipi_code"))
            tc = _code(cell("cipi_title") or cell("any_title"))
            if transcoding:
                c9, t9 = _code(cell("icd9_code")), _code(cell("icd9_title"))
                if c9 and c9.lower() != "nd":
                    codes.setdefault(("ICD-9-CM", c9), {"system": "ICD-9-CM", "code": c9, "title": t9, "version": "2007"})
                if cc and ("CIPI", cc) not in codes:
                    codes[("CIPI", cc)] = {"system": "CIPI", "code": cc, "title": tc, "version": version}
                if c9 and cc and c9.lower() != "nd":
                    maps.append({"icd9": c9, "cipi": cc, "source": path.name})
                continue
            if not cc:
                continue
            extra = {_fold(header[i]).split(" (")[0]: _code(row[i]) for i in range(len(header))
                     if i < len(row) and i not in cols.values() and row[i] not in (None, "")}
            codes[("CIPI", cc)] = {"system": "CIPI", "code": cc, "title": tc, "version": version, **extra}
            for c9 in re.split(r"[;,\s]+", _code(cell("icd9_code"))):
                if c9 and c9.lower() not in ("nd", "n.d."):
                    maps.append({"icd9": c9, "cipi": cc, "source": path.name})
    by_cipi: dict[str, list[str]] = defaultdict(list)
    for m in maps:
        by_cipi[m["cipi"]].append(m["icd9"])
    unified = []
    for (system, code), row in sorted(codes.items()):
        if system != "CIPI":
            continue
        icd9 = sorted(set(by_cipi.get(code, [])))
        unified.append({**row, "icd9": [{"code": c, "title": codes.get(("ICD-9-CM", c), {}).get("title", "")}
                                         for c in icd9]})
    (out / "codes.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                             for r in sorted(codes.values(), key=lambda r: (r["system"], r["code"]))), "utf-8")
    (out / "transcoding.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in maps), "utf-8")
    (out / "unified.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in unified), "utf-8")
    heads = italian_heads(codes.values())
    (out / "italian_heads.json").write_text(json.dumps(heads, ensure_ascii=False, indent=1), "utf-8")
    manifest = {
        "built": time.strftime("%Y-%m-%d"),
        "files": [{k: f.get(k) for k in ("name", "url", "version", "sha256", "bytes")} for f in found.values()],
        "tables": report,
        "counts": {"codes": Counter(r["system"] for r in codes.values()), "transcoding_links": len(maps),
                   "unified": len(unified), "italian_heads": len(heads["kinds"])},
        "licence": "CIPI and its ICD-9-CM transcoding: Ministero della Salute; use confirmed by the project owner "
                   "(14 October 2026). ICD-10-IM not included.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False, default=dict), "utf-8")
    print(json.dumps(manifest["counts"], default=dict))
    return 0


_STOP = frozenset("di del della dei degli delle dello dell d il lo la i gli le l un una uno e o con per su a al alla "
                  "ai agli alle in nel nella nei negli nelle da dal dalla altra altro altri altre fino oltre entro tra fra senza "
                  "mediante".split())


def italian_heads(rows) -> dict:
    """The head word of every Italian procedure title (the first content word: Italian noun phrases are
    left-headed), with how many titles start with it. A word is given the kind "procedure" only when it heads
    at least ``MIN_TITLES`` titles: a correspondence, not a guess from one title."""
    count: Counter[str] = Counter()
    for r in rows:
        words = [w for w in re.findall(r"[a-zà-ù]+", _fold(r.get("title", ""))) if w not in _STOP]
        if words:
            count[words[0]] += 1
            # the abbreviation a title gives in brackets for its head ("Tomografia assiale computerizzata (TAC)")
            # is another word for the same kind of procedure
            for short in re.findall(r"\(([A-Za-z]{2,6})\)", r.get("title", "")):
                count[_fold(short)] += 1
    kinds = {w: "procedure" for w, n in count.items() if n >= MIN_TITLES}
    return {"kinds": dict(sorted(kinds.items())), "titles_per_word": dict(count.most_common()),
            "min_titles": MIN_TITLES, "source": "CIPI and ICD-9-CM (Italian titles of procedures)"}


MIN_TITLES = 3


def check(args) -> int:
    found = json.loads(Path(args.found).read_text("utf-8"))
    manifest = Path(args.manifest)
    old = {f["url"]: f.get("sha256") for f in json.loads(manifest.read_text("utf-8")).get("files", [])} \
        if manifest.exists() else {}
    changed = [f["name"] for f in found if f.get("fetched_by") != "repository copy" and old.get(f["url"]) != f.get("sha256")]
    print(f"changed={'true' if changed else 'false'}")
    print(json.dumps({"changed_files": changed}, ensure_ascii=False), file=sys.stderr)
    return 0


def falkordb(args) -> int:
    from melampo.memory.falkordb_connection import connect

    graph = connect().select_graph(args.graph)
    rows = [json.loads(x) for x in Path(args.unified).read_text("utf-8").splitlines() if x.strip()]
    if args.clear:
        graph.query("MATCH (n:Code) DETACH DELETE n")
    for start in range(0, len(rows), 500):
        batch = rows[start:start + 500]
        graph.query(
            "UNWIND $rows AS row "
            "MERGE (c:Code {system: 'CIPI', code: row.code}) SET c.title = row.title, c.version = row.version "
            "WITH c, row UNWIND row.icd9 AS old "
            "MERGE (o:Code {system: 'ICD-9-CM', code: old.code}) SET o.title = old.title "
            "MERGE (o)-[:MAPS_TO]->(c)",
            params={"rows": batch},
        )
    n = graph.query("MATCH (c:Code) RETURN count(c)").result_set[0][0]
    e = graph.query("MATCH ()-[m:MAPS_TO]->() RETURN count(m)").result_set[0][0]
    print(f"FalkorDB graph {args.graph}: {n} codes, {e} MAPS_TO links")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("discover")
    p.add_argument("--pages", nargs="*")
    p.add_argument("--out", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("--found", required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--fallback", default="data/classifications_it/source",
                   help="copies kept in the repository, used when the site cannot be reached")
    p = sub.add_parser("build")
    p.add_argument("--raw", required=True)
    p.add_argument("--found", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("check")
    p.add_argument("--found", required=True)
    p.add_argument("--manifest", required=True)
    p = sub.add_parser("falkordb")
    p.add_argument("--unified", required=True)
    p.add_argument("--graph", default="classifications_it")
    p.add_argument("--clear", action="store_true")
    args = ap.parse_args(argv)
    return {"discover": discover, "fetch": fetch, "build": build, "check": check, "falkordb": falkordb}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
