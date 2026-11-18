"""The connectors of the literature (18 November 2026): how each source of the index is read for a disease profile.

The disease profiles (``scripts/disease_profiles.py``) read the sources the index (``data/reference/sources.yml``)
marks for them (``profiles: true``), each through the connector the index names (``connector:``). A source is
switched on or off, and given its number of documents, in the index or its mask (``scripts/sources_mask.py``); a
new kind of source is one class here, registered in ``CONNECTORS``, and nothing else changes.

Every connector gives the same thing: ``Document``s, each with its citation (source, identifier, PMID, PMCID, DOI,
title, year, licence) and the sections of its text that describe the disease. What the readers do with a document
does not depend on where it came from.

The licence travels with every document. ``open_for_use`` (CC0, CC BY, CC BY-SA) is what a product may use; a
source marked ``research_only`` (StatPearls' CC BY-NC-ND, a PDF read under a personal subscription, PubMed
abstracts, whose copyright stays with the publishers) is read only in a run that asks for research sources
(``--research``), and every citation it gives says so. A closed PDF is read where it is, on the computer that has
it (``data/private/``, never committed): only the sentence cited and its reference leave it, as for any source.

The connectors:

* ``europe_pmc_reviews``: open reviews and guidelines of Europe PMC (CC BY, CC0), their clinical sections;
* ``pubmed_abstracts``: PubMed abstracts on the disease's presentation and diagnosis (research only);
* ``ncbi_bookshelf``: the chapter of a Bookshelf book on the disease (StatPearls by default: research only);
* ``documents``: the documents the index lists (``kind: document``) by PMCID, PMID, DOI or address, read for the
  diseases they name (``diseases:``) or whose title names them: a source added by its link;
* ``pdf_folder``: the PDFs of a folder (``path:``), a publication or a textbook chapter each, with a ``.yml`` beside
  it for its citation when there is one: a source added by its file.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import disease_profiles as dp
from .sources_index import ROOT, Source, SourcesIndex

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CLINICAL_TITLE = ('(TITLE:"diagnosis" OR TITLE:"clinical" OR TITLE:"presentation" OR TITLE:"imaging" '
                  'OR TITLE:"evaluation" OR TITLE:"approach")')
DEFAULT_DOCUMENTS = 5
MIN_CLINICAL_CHARACTERS = 1500     # a text that says almost nothing of the presentation is skipped

Fetch = Callable[[str, str], str]


def fetch(url: str, accept: str = "application/json") -> str:
    """The body of a public address, '' when it cannot be read (a connector never stops the run)."""
    time.sleep(0.2)
    req = urllib.request.Request(url, headers={"User-Agent": "melampo-disease-profiles/1.1", "Accept": accept})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read().decode("utf-8", errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return ""


@dataclass
class Document:
    citation: dict[str, Any]
    sections: list[tuple[str, str]]

    @property
    def size(self) -> int:
        return sum(len(x) for _, x in self.sections)


@dataclass
class Skipped:
    source: str
    reason: str


@dataclass
class Reading:
    """What the sources gave for one disease: the documents, and the sources that gave none, with why."""
    documents: list[Document] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)
    per_source: dict[str, int] = field(default_factory=dict)


class Connector:
    """One way of reading a source. ``source`` is its row of the index (``max_documents``, ``path``, ``query``)."""

    name = ""
    needs_network = True

    def __init__(self, source: Source, get: Fetch = fetch):
        self.source, self.get = source, get

    @property
    def limit(self) -> int:
        return int(self.source.max_documents or DEFAULT_DOCUMENTS)

    def documents(self, disease: str, words: Iterable[str]) -> list[Document]:
        raise NotImplementedError

    def citation(self, **values: Any) -> dict[str, Any]:
        out = {"source": self.source.title, "source_id": self.source.id, **{k: v for k, v in values.items() if v}}
        if self.source.research_only:
            out["use"] = "research only"
        return out


def _json(text: str) -> dict:
    try:
        value = json.loads(text or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


class EuropePmcReviews(Connector):
    """Open reviews and guidelines (CC BY, CC0) on the diagnosis, whose title speaks of its diagnosis, clinical
    picture or imaging first, systematic reviews and meta-analyses left out; their clinical sections. (Until 18
    November 2026 the pilot read at most two, and stopped at 6,000 characters: one source per manifestation.)"""

    name = "europe_pmc_reviews"

    def documents(self, disease: str, words: Iterable[str]) -> list[Document]:
        base = (f'TITLE:"{disease}" AND (PUB_TYPE:"review" OR PUB_TYPE:"guideline") AND OPEN_ACCESS:y '
                f'AND IN_EPMC:y AND (LICENSE:"cc by" OR LICENSE:"cc0") AND NOT TITLE:"meta-analysis" '
                f'AND NOT TITLE:"systematic"')
        if self.source.query:
            base += f" AND ({self.source.query})"
        results, seen, out = [], set(), []
        for query in (f"{base} AND {CLINICAL_TITLE}", base):
            q = {"query": query, "format": "json", "resultType": "core", "pageSize": 25, "sort": "CITED desc"}
            results += _json(self.get(f"{EPMC}/search?{urllib.parse.urlencode(q)}", "application/json")).get(
                "resultList", {}).get("result", [])
        for r in results:
            pmcid = r.get("pmcid", "")
            if not pmcid or pmcid in seen or not dp.open_for_use(str(r.get("license", ""))):
                continue
            seen.add(pmcid)
            sections = dp.sections_from_jats(self.get(f"{EPMC}/{pmcid}/fullTextXML", "application/xml"))
            doc = Document(self.citation(id=pmcid, pmid=r.get("pmid", ""), doi=r.get("doi", ""),
                                         title=r.get("title", ""), licence=r.get("license", ""),
                                         year=r.get("pubYear", ""),
                                         source="Europe PMC review"), sections)
            if doc.size >= MIN_CLINICAL_CHARACTERS:
                out.append(doc)
            if len(out) >= self.limit:
                break
        return out


class PubmedAbstracts(Connector):
    """PubMed abstracts on the disease's clinical presentation and diagnosis, reviews first: short, many, and
    independent of one another. Their copyright stays with the publishers: research only."""

    name = "pubmed_abstracts"

    def documents(self, disease: str, words: Iterable[str]) -> list[Document]:
        term = (f'"{disease}"[Title] AND (diagnosis[Title/Abstract] OR "clinical features"[Title/Abstract] OR '
                f'presentation[Title/Abstract] OR imaging[Title/Abstract]) AND hasabstract')
        if self.source.query:
            term += f" AND ({self.source.query})"
        q = {"db": "pubmed", "term": term, "retmode": "json", "retmax": self.limit, "sort": "relevance"}
        ids = _json(self.get(f"{EUTILS}/esearch.fcgi?{urllib.parse.urlencode(q)}", "application/json")).get(
            "esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        xml = self.get(f"{EUTILS}/efetch.fcgi?db=pubmed&retmode=xml&id={','.join(ids)}", "application/xml")
        return [Document(self.citation(id=pmid, pmid=pmid, doi=doi, title=title, year=year,
                                       licence="PubMed abstract (publisher's copyright)", source="PubMed abstract"),
                         [("Abstract", text)]) for pmid, doi, title, year, text in pubmed_articles(xml) if text]


def pubmed_articles(xml: str) -> list[tuple[str, str, str, str, str]]:
    """(PMID, DOI, title, year, abstract) of the articles of a PubMed efetch answer."""
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(xml or "<x/>")
    except ET.ParseError:
        return []
    out = []
    for art in root.iter("PubmedArticle"):
        pmid = (art.findtext(".//MedlineCitation/PMID") or "").strip()
        title = " ".join("".join(art.find(".//ArticleTitle").itertext()).split()) if art.find(
            ".//ArticleTitle") is not None else ""
        doi = next((" ".join(e.itertext()).strip() for e in art.iter("ArticleId") if e.get("IdType") == "doi"), "")
        year = (art.findtext(".//PubDate/Year") or art.findtext(".//PubDate/MedlineDate") or "")[:4]
        text = " ".join(" ".join("".join(a.itertext()).split()) for a in art.iter("AbstractText"))
        out.append((pmid, doi, title, year, text))
    return out


class NcbiBookshelf(Connector):
    """The chapter of a Bookshelf book (``query``: the book, StatPearls by default) whose title is the disease, its
    sections on history, examination and evaluation."""

    name = "ncbi_bookshelf"
    SECTIONS = ("History and Physical", "Evaluation", "Clinical Features", "Presentation", "Diagnosis")

    def documents(self, disease: str, words: Iterable[str]) -> list[Document]:
        book = self.source.query or "statpearls"
        q = {"db": "books", "term": f'{book}[book] AND "{disease}"[title]', "retmode": "json",
             "retmax": self.limit}
        ids = _json(self.get(f"{EUTILS}/esearch.fcgi?{urllib.parse.urlencode(q)}", "application/json")).get(
            "esearchresult", {}).get("idlist", [])
        out = []
        for uid in ids[: self.limit]:
            doc = _json(self.get(f"{EUTILS}/esummary.fcgi?db=books&retmode=json&id={uid}", "application/json")).get(
                "result", {}).get(uid, {})
            nbk = doc.get("chapteraccessionid") or doc.get("accessionid") or ""
            page = self.get(f"https://www.ncbi.nlm.nih.gov/books/{nbk}/?report=printable", "text/html") if nbk else ""
            sections = dp.html_sections(page, self.SECTIONS)
            if sections:
                out.append(Document(self.citation(id=nbk, title=doc.get("title", ""), licence=self.source.licence,
                                                  source=self.source.title), sections))
        return out


def about(disease_words: Iterable[str], title: str, named: Iterable[str]) -> bool:
    """A document is read for a disease when the index names the disease for it, or its title names it."""
    words = [w for w in disease_words if w]
    named = [n for n in named if n]
    if named:
        return any(dp.norm(n) == dp.norm(w) for n in named for w in words)
    return dp.names_the_disease(title, words)


def clinical_sections(text: str, title: str = "Text") -> list[tuple[str, str]]:
    """The clinical part of a text without structure (a PDF, a web page): the paragraphs under headings that speak
    of the presentation, the diagnosis or the imaging when it has such headings; otherwise the whole text, up to
    ``dp.SECTION_CHARACTERS``. The readers are told to list only what the text says of the disease."""
    lines = [x.strip() for x in (text or "").splitlines()]
    out, current, buffer, used = [], "", [], 0

    def flush() -> None:
        nonlocal used
        body = " ".join(" ".join(buffer).split())
        if current and body and dp.SECTION_WORDS.search(current) and not dp.SECTION_EXCLUDED.search(current):
            body = body[: max(0, dp.SECTION_CHARACTERS - used)]
            if body:
                out.append((current, body))
                used += len(body)

    for line in lines:
        heading = (0 < len(line) <= 80 and not line.endswith((".", ",", ";")) and len(line.split()) <= 8
                   and (line.isupper() or line.istitle() or re.match(r"^\d+(\.\d+)*\s+\S", line)))
        if heading:
            flush()
            current, buffer = re.sub(r"^\d+(\.\d+)*\s+", "", line), []
        else:
            buffer.append(line)
    flush()
    if out:
        return out
    whole = " ".join(" ".join(lines).split())[: dp.SECTION_CHARACTERS]
    return [(title, whole)] if whole else []


def pdf_text(path: Path) -> str:
    """The words of a PDF through ``pdftotext`` (poppler), '' when it cannot be read."""
    try:
        return subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, check=True,
                              timeout=180).stdout.decode("utf-8", "replace")
    except (OSError, subprocess.SubprocessError):
        return ""


class Documents(Connector):
    """The documents of this source in the index (``kind: document``, ``of:`` this source) added by their link:
    a PMCID or PMID read through Europe PMC (the open ones in full, the others by their abstract), a DOI the same
    way, any other address as a page or a PDF."""

    name = "documents"

    def __init__(self, source: Source, get: Fetch = fetch, index: SourcesIndex | None = None):
        super().__init__(source, get)
        self.index = index

    def documents(self, disease: str, words: Iterable[str]) -> list[Document]:
        words = list(words)
        rows = [s for s in (self.index.sources if self.index else []) if s.kind == "document" and
                s.of == self.source.id and s.status not in ("rejected", "not_reachable") and
                about(words, s.title, s.diseases)]
        out = []
        for row in rows[: self.limit]:
            doc = self.read(row)
            if doc and doc.size >= 200:
                out.append(doc)
        return out

    def read(self, row: Source) -> Document | None:
        ident = (row.identifier or "").strip()
        if row.path:
            path = Path(row.path) if Path(row.path).is_absolute() else ROOT / row.path
            text = pdf_text(path) if path.suffix.lower() == ".pdf" else path.read_text("utf-8", "replace") \
                if path.exists() else ""
            return Document(self._cite(row, ident), clinical_sections(text, row.title)) if text else None
        query = (f"PMCID:{ident.upper()}" if re.fullmatch(r"(?i)pmc\d+", ident) else
                 f"EXT_ID:{ident} AND SRC:MED" if re.fullmatch(r"\d{5,9}", ident) else
                 f'DOI:"{ident.removeprefix("doi:")}"' if ident.lower().removeprefix("doi:").startswith("10.") else "")
        if query:
            q = {"query": query, "format": "json", "resultType": "core", "pageSize": 1}
            hits = _json(self.get(f"{EPMC}/search?{urllib.parse.urlencode(q)}", "application/json")).get(
                "resultList", {}).get("result", [])
            if hits:
                r = hits[0]
                cite = self._cite(row, r.get("pmcid") or r.get("pmid") or ident, pmid=r.get("pmid", ""),
                                  doi=r.get("doi", ""), year=r.get("pubYear", ""),
                                  licence=r.get("license", "") or row.licence)
                if r.get("pmcid") and r.get("inEPMC") == "Y":
                    sections = dp.sections_from_jats(self.get(f"{EPMC}/{r['pmcid']}/fullTextXML", "application/xml"))
                    if sections:
                        return Document(cite, sections)
                abstract = " ".join(re.sub(r"<[^>]+>", " ", r.get("abstractText", "")).split())
                return Document({**cite, "read": "abstract"}, [("Abstract", abstract)]) if abstract else None
        page = self.get(row.url, "text/html,application/pdf") if row.url else ""
        from .sources_index import page_text

        text = page_text(page.encode("utf-8", "replace")) if page else ""
        return Document(self._cite(row, ident or row.url), clinical_sections(text, row.title)) if text else None

    def _cite(self, row: Source, ident: str, **more: Any) -> dict[str, Any]:
        return self.citation(id=ident or row.id, title=row.title, licence=more.pop("licence", row.licence),
                             document=row.id, url=row.url, **more)


class PdfFolder(Connector):
    """The PDFs of a folder (``path:``; ``data/private/pdfs`` by default, never committed). Each PDF is a
    publication or a chapter; a ``.yml`` beside it (same name) gives its citation: title, identifier (PMID, DOI,
    ISBN and page), licence, diseases. Without it the file name is the title and the disease must be in it."""

    name = "pdf_folder"
    needs_network = False

    def documents(self, disease: str, words: Iterable[str]) -> list[Document]:
        import yaml

        words = list(words)
        folder = Path(self.source.path or "data/private/pdfs")
        folder = folder if folder.is_absolute() else ROOT / folder
        out = []
        for pdf in sorted(folder.glob("*.pdf")) if folder.is_dir() else []:
            meta_file = pdf.with_suffix(".yml")
            meta = yaml.safe_load(meta_file.read_text("utf-8")) if meta_file.exists() else {}
            meta = meta if isinstance(meta, dict) else {}
            title = str(meta.get("title") or pdf.stem.replace("_", " "))
            if not about(words, title, meta.get("diseases") or []):
                continue
            text = pdf_text(pdf)
            sections = clinical_sections(text, title)
            if not sections:
                continue
            out.append(Document(self.citation(id=str(meta.get("identifier") or pdf.name), title=title,
                                              licence=str(meta.get("licence") or self.source.licence),
                                              year=str(meta.get("year") or ""), file=pdf.name), sections))
            if len(out) >= self.limit:
                break
        return out


CONNECTORS: dict[str, type[Connector]] = {c.name: c for c in (EuropePmcReviews, PubmedAbstracts, NcbiBookshelf,
                                                               Documents, PdfFolder)}


def profile_sources(index: SourcesIndex, research: bool = False) -> tuple[list[Source], list[Skipped]]:
    """The sources a disease profile reads: marked for the profiles, reachable now, with a known connector; a
    research-only source only when the run asks for research sources. The others are listed with why."""
    use, skipped = [], []
    for s in index.sources:
        if not s.profiles or s.kind == "document" or s.connector == "wikipedia_index":
            continue
        if s.connector not in CONNECTORS:
            skipped.append(Skipped(s.id, f"connector {s.connector or '(none)'} unknown"))
        elif s.status not in ("in_use", "available"):
            skipped.append(Skipped(s.id, f"status {s.status}: {s.how_to_reach or 'not reachable now'}"))
        elif s.research_only and not research:
            skipped.append(Skipped(s.id, "research only: read with --research"))
        else:
            use.append(s)
    return use, skipped


def read_sources(disease: str, words: Iterable[str], index: SourcesIndex, *, research: bool = False,
                 get: Fetch = fetch) -> Reading:
    """Every document the profile sources give for a disease, the same publication once (by PMID, PMCID, DOI)."""
    words = list(words)
    sources, skipped = profile_sources(index, research)
    reading = Reading(skipped=list(skipped))
    seen: set[str] = set()
    for s in sources:
        cls = CONNECTORS[s.connector]
        connector = cls(s, get, index) if cls is Documents else cls(s, get)
        try:
            docs = connector.documents(disease, words)
        except Exception as error:          # a broken source is reported, the others are still read
            reading.skipped.append(Skipped(s.id, f"error: {type(error).__name__}: {error}"[:300]))
            continue
        kept = 0
        for d in docs:
            keys = {str(d.citation.get(k) or "").lower() for k in ("pmid", "pmcid", "doi")} - {""}
            ident = str(d.citation.get("id") or "").lower()
            if ident.startswith("pmc"):
                keys.add(ident)
            if keys & seen:
                continue
            seen |= keys
            reading.documents.append(d)
            kept += 1
        reading.per_source[s.id] = kept
    return reading


__all__ = ["CONNECTORS", "Connector", "Document", "Documents", "EuropePmcReviews", "NcbiBookshelf", "PdfFolder",
           "PubmedAbstracts", "Reading", "Skipped", "about", "clinical_sections", "fetch", "pdf_text",
           "profile_sources", "pubmed_articles", "read_sources"]
