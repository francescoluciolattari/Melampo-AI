"""The index of the sources (8 November 2026): which literature Melampo reads, how it may be read, and what it found.

Every reference value, every paragraph a model is given, comes from a source: literature, textbooks, research,
consolidated documentation, never a memory. This index says, for each source, what it is, how it may be reached
(open, free registration, subscription, institutional access, public domain), under which licence, and what a
person must do to reach it when it is closed. It grows in two ways:

* **by hand**, through the mask (``scripts/sources_mask.py``) or the command line: a source, or a document of a
  source (an article, a guideline, a chapter), with its identifier;
* **by the project's own search**, every week (workflow ``sources-index``): Europe PMC is asked for reference
  values and normal ranges of the structures and techniques the reference library covers; what it finds comes in
  as a **candidate** document, never as a row of the library. A person, or the next step of the project, decides.

The same workflow checks the library's rows against their pages (``check_quote``): the page is fetched, its text
compared with the words the row cites (``quoted`` in ``values.yml``); found word for word, the row is
``verified`` with the date and the fingerprint of the page; a page that has changed since is said so.

A source can also be **read for the disease profiles** (18 November 2026): ``profiles: true`` and a ``connector``
(``memory/literature_connectors.py``) say how; ``max_documents`` how many documents per disease; ``research_only``
that its licence allows research only (it is read only by a run that asks for research sources); ``path`` the
folder of its PDFs; ``query`` what narrows its search. A document added by its link (``kind: document``) names the
diseases it is read for (``diseases``), or its title does. The mask switches a source on and off for the profiles,
adds, edits and removes sources, and takes a PDF.

Nothing here decides a diagnosis. The file is ``data/reference/sources.yml``: one place, versioned, readable.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SOURCES = ROOT / "data" / "reference" / "sources.yml"

KINDS = ("database", "journal", "guideline", "textbook", "open_textbook", "reference_site", "terminology",
         "library", "document")
ACCESS = ("open", "free_registration", "subscription", "institutional", "public_domain", "purchase")
STATUS = ("in_use", "available", "to_register", "not_reachable", "candidate", "rejected")
# the connectors of memory/literature_connectors.py (kept here so that the index is checked without importing them)
# (wikipedia_index is read by the profiles script itself: an index of publications, never a source of text)
CONNECTOR_NAMES = ("europe_pmc_reviews", "pubmed_abstracts", "ncbi_bookshelf", "documents", "pdf_folder",
                   "wikipedia_index")
FIELDS = ("anatomy", "physiology", "pathology", "radiology", "reference_values", "clinical", "terminology",
          "guidelines", "oncology", "cardiology", "laboratory")


def _today() -> str:
    return _dt.datetime.now(_dt.UTC).date().isoformat()


@dataclass
class Source:
    id: str
    kind: str
    title: str
    url: str
    access: str
    status: str
    publisher: str = ""
    licence: str = ""
    fields: list[str] = field(default_factory=list)
    how_to_reach: str = ""
    note: str = ""
    of: str = ""                # for a document: the id of its source
    identifier: str = ""        # PMID, PMCID, DOI, ISBN
    added: str = ""
    added_by: str = ""          # "manual", "search", "claude"
    last_checked: str = ""
    # how the disease profiles read it (18 November 2026)
    profiles: bool = False      # read for the disease profiles
    connector: str = ""         # one of CONNECTOR_NAMES
    max_documents: int = 0      # documents per disease (0: the connector's default)
    research_only: bool = False  # its licence allows research only
    path: str = ""              # a folder of PDFs, or a document's file
    query: str = ""             # what narrows the connector's search
    diseases: list[str] = field(default_factory=list)   # for a document: the diseases it is read for

    def problems(self) -> list[str]:
        out = []
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,80}", self.id or ""):
            out.append("id: lower-case letters, digits, '.', '_' or '-'")
        for name, allowed in (("kind", KINDS), ("access", ACCESS), ("status", STATUS)):
            if getattr(self, name) not in allowed:
                out.append(f"{name}: one of {', '.join(allowed)}")
        if not self.title.strip():
            out.append("title: required")
        if not re.match(r"https?://", self.url or ""):
            out.append("url: an http(s) address")
        unknown = [f for f in self.fields if f not in FIELDS]
        if unknown:
            out.append(f"fields: unknown {unknown}")
        if self.kind == "document" and not self.of:
            out.append("of: a document belongs to a source")
        if self.access in ("subscription", "institutional", "purchase") and not self.how_to_reach.strip():
            out.append("how_to_reach: say how a person reaches a closed source")
        if self.connector and self.connector not in CONNECTOR_NAMES:
            out.append(f"connector: one of {', '.join(CONNECTOR_NAMES)}")
        if self.profiles and not self.connector and self.kind != "document":
            out.append("connector: a source read for the profiles says how it is read")
        if self.connector == "pdf_folder" and not self.path.strip():
            out.append("path: the folder of the PDFs")
        if not 0 <= int(self.max_documents or 0) <= 100:
            out.append("max_documents: from 0 to 100")
        if self.path and re.match(r"https?://", self.path):
            out.append("path: a file or folder on this computer (an address goes in url)")
        return out

    def as_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if v not in ("", [], None)}
        return {k: d[k] for k in ("id", "kind", "title", "publisher", "url", "access", "licence", "status",
                                  "fields", "how_to_reach", "of", "identifier", "note", "added", "added_by",
                                  "last_checked", "profiles", "connector", "max_documents", "research_only", "path",
                                  "query", "diseases") if k in d and d[k] is not False and d[k] != 0}


def _flag(value: Any) -> bool:
    return value is True or str(value).strip().lower() in ("true", "yes", "1", "on", "si", "sì")


def source_from(row: dict[str, Any]) -> Source:
    """A row of the file, or of the mask's form, as a Source: lists, flags and numbers in their own type."""
    known = {k: row[k] for k in Source.__dataclass_fields__ if k in row and row[k] is not None}
    out: dict[str, Any] = {k: "" for k in ("id", "kind", "title", "url", "access", "status")}
    for k, v in known.items():
        if k in ("fields", "diseases"):
            out[k] = [str(x).strip() for x in (v if isinstance(v, list) else str(v).split(",")) if str(x).strip()]
        elif k in ("profiles", "research_only"):
            out[k] = _flag(v)
        elif k == "max_documents":
            try:
                out[k] = int(v or 0)
            except (TypeError, ValueError):
                out[k] = -1                       # reported by problems()
        else:
            out[k] = str(v)
    return Source(**out)


class SourcesIndex:
    def __init__(self, sources: list[Source], path: Path = SOURCES, header: str = ""):
        self.sources = sources
        self.path = path
        self.header = header

    @classmethod
    def load(cls, path: Path | None = None) -> SourcesIndex:
        import yaml

        p = Path(path or SOURCES)
        text = p.read_text("utf-8") if p.exists() else ""
        header = "".join(line + "\n" for line in text.splitlines() if line.startswith("#"))
        data = yaml.safe_load(text) or {}
        out = []
        for row in data.get("sources", []):
            row = dict(row)
            out.append(source_from(row))
        return cls(out, p, header)

    def remove(self, source_id: str) -> list[str]:
        """Take a source out of the index; not one that documents still belong to."""
        if source_id not in self.by_id():
            return [f"{source_id}: not in the index"]
        children = [s.id for s in self.sources if s.of == source_id]
        if children:
            return [f"{source_id}: its documents first ({', '.join(children[:5])})"]
        self.sources = [s for s in self.sources if s.id != source_id]
        self.save()
        return []

    def by_id(self) -> dict[str, Source]:
        return {s.id: s for s in self.sources}

    def problems(self) -> list[str]:
        out, seen = [], set()
        ids = self.by_id()
        for s in self.sources:
            out += [f"{s.id}: {p}" for p in s.problems()]
            if s.id in seen:
                out.append(f"{s.id}: twice")
            seen.add(s.id)
            if s.of and s.of not in ids:
                out.append(f"{s.id}: of {s.of}, not in the index")
        return out

    def put(self, source: Source, today: str | None = None) -> list[str]:
        """Add a source, or replace the one with its id; the index is saved only when it has no problem."""
        source.added = source.added or (today or _today())
        problems = source.problems()
        if source.of and source.of not in self.by_id():
            problems.append(f"of: {source.of} is not in the index")
        if problems:
            return problems
        self.sources = [s for s in self.sources if s.id != source.id] + [source]
        self.save()
        return []

    def save(self) -> None:
        import yaml

        body = yaml.safe_dump({"version": _today(),
                               "sources": [s.as_dict() for s in self.sources]},
                              allow_unicode=True, sort_keys=False, width=110)
        self.path.write_text(self.header + body, "utf-8")


# --- the project's weekly search ---------------------------------------------------------------------------------

def search_queries(rows: list[Any]) -> list[str]:
    """What the weekly search asks: the reference values of the structures and techniques the library covers."""
    seen, out = set(), []
    for r in rows:
        for cid in r.structures:
            name = cid.replace("_", " ").replace(" left", "").replace(" right", "")
            q = f'("reference values" OR "normal values" OR "normal range") AND "{name}" AND {r.aspect}'
            if q not in seen:
                seen.add(q)
                out.append(q)
    return out


def candidates_from_europe_pmc(query: str, fetch_json, limit: int = 5, today: str | None = None,
                               of: str = "europe-pmc") -> list[Source]:
    """Candidate documents for a query (guidelines, reviews and consensus first): never rows of the library."""
    from urllib.parse import urlencode

    q = f"({query}) AND (PUB_TYPE:\"guideline\" OR PUB_TYPE:\"review\" OR consensus) AND OPEN_ACCESS:y"
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urlencode(
        {"query": q, "format": "json", "pageSize": limit, "resultType": "lite", "sort": "CITED desc"})
    data = fetch_json(url) or {}
    out = []
    for hit in (data.get("resultList") or {}).get("result", []):
        ident = hit.get("pmcid") or hit.get("pmid") or hit.get("doi") or ""
        if not ident or not hit.get("title"):
            continue
        link = (f"https://europepmc.org/article/PMC/{hit['pmcid']}" if hit.get("pmcid")
                else f"https://europepmc.org/article/MED/{hit['pmid']}" if hit.get("pmid")
                else f"https://doi.org/{hit['doi']}")
        out.append(Source(id=("doc-" + re.sub(r"[^a-z0-9]+", "-", ident.lower())).strip("-"), kind="document",
                          title=html.unescape(hit["title"]).strip(), url=link, access="open", status="candidate",
                          publisher=hit.get("journalTitle", ""), fields=["reference_values"], of=of,
                          identifier=ident, note=f"found by the weekly search: {query}",
                          added=today or _today(), added_by="search"))
    return out


def add_candidates(index: SourcesIndex, found: list[Source]) -> int:
    """Add what the search found and the index does not have (a rejected document stays rejected)."""
    have = index.by_id()
    new = [s for s in found if s.id not in have]
    if new:
        index.sources += new
        index.save()
    return len(new)


# --- the rows of the library checked against their pages ----------------------------------------------------------

def page_text(raw: bytes, content_type: str = "") -> str:
    """The words of a page: an HTML page without its markup; a PDF through ``pdftotext`` when it is there."""
    if raw[:4] == b"%PDF" or "pdf" in content_type:
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf") as fh:
            fh.write(raw)
            fh.flush()
            try:
                return subprocess.run(["pdftotext", "-layout", fh.name, "-"], capture_output=True, check=True,
                                      timeout=120).stdout.decode("utf-8", "replace")
            except (OSError, subprocess.SubprocessError):
                return ""
    return _html_words(raw.decode("utf-8", "replace"))


def _html_words(page: str) -> str:
    """The words of an HTML page, by an HTML parser (20 November 2026): the regular expression that removed
    "<...>" also removed the words after a bare "<" written in the text ("depth: <2 cm"), up to the next ">", and
    rows whose words are on the page were reported missing."""
    from html.parser import HTMLParser

    class Words(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.out: list[str] = []
            self.skip = 0

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style", "noscript", "template"):
                self.skip += 1
            self.out.append(" ")

        def handle_endtag(self, tag):
            if tag in ("script", "style", "noscript", "template") and self.skip:
                self.skip -= 1
            self.out.append(" ")

        def handle_data(self, data):
            if not self.skip:
                self.out.append(data)

    parser = Words()
    parser.feed(page)
    parser.close()
    return "".join(parser.out)


def _squash(text: str) -> str:
    """Letters and digits only, lower case: a quotation and a page compared whatever their spacing and markup
    (a reference number in superscript, a line break, a non-breaking space)."""
    text = text.replace("–", "-").replace("±", "+")
    return re.sub(r"[^0-9a-zà-ÿ≤≥<>+-]", "", text.lower())


def quoted_pieces(quoted: str) -> list[str]:
    """The words a row cites, between double quotes."""
    return [q for q in re.findall(r'"([^"]+)"', quoted or "") if len(q) >= 6 or re.search(r"\d", q)]


def nearest(piece: str, text: str, width: int = 240) -> str:
    """The words of the page closest to a piece that is not there word for word: what the page does say, so that
    the row can be corrected from the page itself (9 November 2026: a web reading had paraphrased some)."""
    import difflib

    sentences = [x.strip() for x in re.split(r"(?<=[.;:])\s+|\n+", " ".join(text.split(" "))) if len(x.strip()) > 8]
    numbers = set(re.findall(r"\d+(?:[.,]\d+)?", piece))
    pool = [x for x in sentences if numbers & set(re.findall(r"\d+(?:[.,]\d+)?", x))] or sentences
    best = max(pool, key=lambda x: difflib.SequenceMatcher(None, piece.lower(), x.lower()).ratio(), default="")
    return " ".join(best.split())[:width]


def check_quote(quoted: str, text: str) -> dict[str, Any]:
    """Is every piece the row cites on the page, word for word? For a piece that is not, the nearest words of
    the page."""
    page = _squash(text)
    pieces = quoted_pieces(quoted)
    missing = [p for p in pieces if _squash(p.replace("...", "")) not in page]
    return {"pieces": len(pieces), "missing": missing, "nearest": {p: nearest(p, text) for p in missing},
            "found": bool(pieces) and not missing, "page_sha256": hashlib.sha256(text.encode()).hexdigest()[:16]}


def mark_rows(values: Path, results: dict[str, dict[str, Any]], today: str | None = None) -> int:
    """Write the result of the check in values.yml, row by row, without rewriting the rest of the file: a row
    whose words are all on its page becomes ``verified``; one that is not stays ``to_verify`` and says why."""
    today = today or _today()
    text = values.read_text("utf-8")
    changed = 0
    for rid, res in results.items():
        m = re.search(rf"(?m)^  - id: {re.escape(rid)}\n(?:    .*\n|      .*\n|             .*\n)*", text)
        if not m:
            continue
        block = m.group(0)
        if re.search(r"(?m)^    status: doubt$", block):
            continue        # a row in doubt says why; the check does not promote it (20 November 2026)
        new = re.sub(r"(?m)^    (status|checked_by|page_check): .*\n", "", block)
        if res.get("found"):
            line = (f"    status: verified\n    checked_by: \"the words found on the page by the sources-index check, "
                    f"{today}\"\n    page_check: \"{today}, page {res['page_sha256']}\"\n")
        else:
            why = ("the page could not be read" if res.get("error") else
                   "not found on the page; the nearest words of the page: "
                   + json.dumps(res.get("nearest") or res.get("missing") or "no quotation", ensure_ascii=False)[:600])
            line = f"    status: to_verify\n    checked_by: \"\"\n    page_check: {json.dumps(today + ', ' + why, ensure_ascii=False)}\n"
        new = new.rstrip("\n") + "\n" + line
        if new != block:
            text = text.replace(block, new, 1)
            changed += 1
    values.write_text(text, "utf-8")
    return changed


__all__ = ["ACCESS", "CONNECTOR_NAMES", "FIELDS", "source_from", "KINDS", "SOURCES", "STATUS", "Source", "SourcesIndex", "add_candidates",
           "candidates_from_europe_pmc", "check_quote", "mark_rows", "page_text", "quoted_pieces",
           "search_queries"]
