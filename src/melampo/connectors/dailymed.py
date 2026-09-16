"""Populate the literature index from DailyMed -- FDA drug labels, the complement to MAxO's thin coverage.

Built after measuring a real gap this project already found and could not
fill: MAxO's own contraindication data covers 202 of 12,880 diseases (1.6%),
one row is a genuine CONTRAINDICATED annotation. DailyMed carries the
Structured Product Label (SPL) for every FDA-approved drug -- active
ingredients, packaging, and (in the full SPL document, see the limitation
below) the actual contraindications and warnings text -- for the entire US
formulary, not for the handful of diseases MAxO happened to annotate.

**The real, official API, verified directly against NIH's own documentation
rather than a third-party scraper.** Base URL
`https://dailymed.nlm.nih.gov/dailymed/services/v2/`, JSON or XML by file
extension, GET only, no API key required or accepted -- confirmed the same
way Europe PMC and ClinicalTrials.gov were: read the primary source, not a
wrapper's description of it.

**What this connector deliberately does not attempt.** The `/spls.json` and
`/spls/{SETID}/packaging.json` endpoints return structured metadata --
title, active ingredients, packaging -- not the prose contraindications and
warnings sections themselves. Those live inside the full SPL document (an
HL7-standard XML structure, downloadable as ZIP via
`getFile.cfm?type=zip&setid=...`), which needs its own LOINC-coded-section
parser, a distinct and larger piece of work not built here. This connector
gives citable, checkable drug identification and composition -- enough to
recognise a drug a vetting model names and link it to its official
label page -- not yet the contraindication text itself. Documented as a
limitation rather than silently claiming more than it delivers.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..memory.literature_index import LiteratureIndex, LiteraturePassage
from .europe_pmc import RateLimiter

DAILYMED_BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

# No official rate limit is published for DailyMed, the same situation as
# ClinicalTrials.gov -- kept conservative for the same reason: no key exists
# to raise it or fall back on if this guess is wrong.
DEFAULT_REQUESTS_PER_SECOND = 3.0
DEFAULT_PAGE_SIZE = 25


@dataclass(frozen=True)
class DailyMedConfig:
    """Politeness identification for DailyMed. No key required or accepted."""

    tool: str = "melampo-literature-connector"
    requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND
    page_size: int = DEFAULT_PAGE_SIZE


def _passage_from_spl(record: dict[str, Any], active_ingredients: Sequence[str]) -> LiteraturePassage | None:
    """Build a passage from one SPL search result plus its packaging lookup.

    A record with no title is skipped -- the same standard every other
    connector in this project applies: nothing to match a search against,
    nothing worth a citation with no content behind it.
    """
    title = str(record.get("title") or "").strip()
    setid = str(record.get("setid") or "").strip()
    if not title or not setid:
        return None

    ingredients_text = f" Active ingredients: {', '.join(active_ingredients)}." if active_ingredients else ""
    published = str(record.get("published_date") or "")
    year = None
    for token in published.replace(",", " ").split():
        if token.isdigit() and len(token) == 4:
            year = int(token)
            break

    return LiteraturePassage(
        passage_id=f"dailymed:{setid}",
        text=f"{title}.{ingredients_text}",
        title=title,
        # The PDF download URL is DailyMed's own confirmed, stable,
        # directly resolvable reference for a given SET ID -- a reviewer
        # opening it gets the official label, the strongest form of
        # "independently checkable" this connector can offer without the
        # full SPL parser.
        source_id=f"dailymed:{setid}",
        year=year,
        publication="DailyMed (FDA)",
    )


@dataclass
class DailyMedConnector:
    """Search DailyMed and produce checkable drug-label passages."""

    config: DailyMedConfig = field(default_factory=DailyMedConfig)

    def __post_init__(self) -> None:
        self._limiter = RateLimiter(self.config.requests_per_second)

    def search(self, drug_name: str, *, max_results: int = 25) -> list[LiteraturePassage]:
        """Search DailyMed by drug name, returning usable passages.

        One page per call, matching `ClinicalTrialsConnector.search`'s own
        posture: this connector surfaces a handful of relevant labels for a
        case's concepts, not the full history of every label DailyMed has
        ever published for a drug.
        """
        page = self._fetch_search_page(drug_name)
        results = page.get("data", [])
        passages: list[LiteraturePassage] = []
        for record in results:
            setid = str(record.get("setid") or "")
            ingredients = self._active_ingredients(setid) if setid else []
            passage = _passage_from_spl(record, ingredients)
            if passage is not None:
                passages.append(passage)
                if len(passages) >= max_results:
                    break
        return passages

    def search_for_concepts(self, concepts: Sequence[str], *, max_results: int = 25) -> list[LiteraturePassage]:
        """Search once per concept, since DailyMed's drug_name filter takes one name, not a query language."""
        passages: list[LiteraturePassage] = []
        for concept in concepts:
            if not concept:
                continue
            passages.extend(self.search(concept, max_results=max_results))
            if len(passages) >= max_results:
                break
        return passages[:max_results]

    def populate(self, index: LiteratureIndex, drug_name: str, *, max_results: int = 25, store: Any = None) -> int:
        """Search and add results directly to an index, optionally persisting to the shared vector store."""
        passages = self.search(drug_name, max_results=max_results)
        added = index.add_many(passages)
        if store is not None:
            from ..memory.literature_persistence import persist_passage

            for passage in passages:
                persist_passage(store, passage)
        return added

    def _active_ingredients(self, setid: str) -> list[str]:
        try:
            payload = self._fetch_packaging(setid)
        except Exception:  # noqa: BLE001 - missing packaging data degrades a passage, never breaks the search
            return []
        ingredients = payload.get("data", {}).get("active_ingredients", [])
        return [str(item.get("name", "")).strip() for item in ingredients if item.get("name")]

    def _fetch_search_page(self, drug_name: str) -> dict[str, Any]:  # pragma: no cover - network call
        self._limiter.wait()
        params = {"drug_name": drug_name, "pagesize": str(self.config.page_size)}
        return self._get(f"{DAILYMED_BASE}/spls.json?{urlencode(params)}")

    def _fetch_packaging(self, setid: str) -> dict[str, Any]:  # pragma: no cover - network call
        self._limiter.wait()
        return self._get(f"{DAILYMED_BASE}/spls/{setid}/packaging.json")

    def _get(self, url: str) -> dict[str, Any]:  # pragma: no cover - network call
        request = Request(url, headers={"User-Agent": self.config.tool})
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8", errors="ignore"))
