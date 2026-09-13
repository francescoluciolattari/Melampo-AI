"""Populate the literature index from Europe PMC, the primary literature source.

Chosen over raw PubMed E-utilities for this specific job. PubMed does not
index full text -- Europe PMC does, aggregating PubMed's citations with PMC
full text and life-science preprints (bioRxiv, medRxiv) in one search, no API
key required. `connectors/pmc_case_reports.py` already wraps E-utilities for
a different job -- fetching case-report full text for evaluation cases, via
the OAI full-text endpoint -- and is left as-is; this module is a separate
connector for a separate purpose, not a replacement.

**What this produces, and what it does not decide.** Every result becomes a
`LiteraturePassage` with a checkable identifier (PMID, PMCID or DOI, in that
preference order -- a PMID is the most universally resolvable of the three).
Passages without a real title and abstract are not constructed at all: a
passage search cannot match against nothing, and a citation with no content
behind it would be a reference with nothing to check it against, defeating
the reason retrieval was chosen over training in the first place. This
module does not decide when to query, how often to refresh, or which
concepts matter enough to search for -- those are deployment and clinical
curation decisions, not a connector's job.
"""

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..memory.literature_index import LiteratureIndex, LiteraturePassage

EUROPE_PMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# Europe PMC publishes no strict per-second contract the way NCBI does; this
# is a conservative default for an unregistered client, matched to the same
# discipline `connectors/pmc_case_reports.py` applies to NCBI -- polite by
# choice, not because the API enforces it.
DEFAULT_REQUESTS_PER_SECOND = 2.0

DEFAULT_PAGE_SIZE = 25


class RateLimiter:
    """Sleep exactly as long as the chosen politeness contract requires, no more.

    A second implementation of `pmc_case_reports.RateLimiter` rather than an
    import from there: that class sits alongside NCBI-specific config it does
    not need here. Shared instead between this module and
    `clinical_trials.py`, since both are plain N-requests-per-second pacing
    with no source-specific fields -- the right amount of sharing is between
    connectors with the same actual contract, not all connectors regardless
    of what each one's politeness rules require.
    """

    def __init__(self, requests_per_second: float) -> None:
        self._interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self._last_call: float | None = None

    def wait(self) -> None:
        if self._interval <= 0:
            return
        now = time.monotonic()
        if self._last_call is not None:
            remaining = self._interval - (now - self._last_call)
            if remaining > 0:
                time.sleep(remaining)
        self._last_call = time.monotonic()


@dataclass(frozen=True)
class EuropePmcConfig:
    """Politeness identification for Europe PMC. No key required or accepted."""

    tool: str = "melampo-literature-connector"
    requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND
    page_size: int = DEFAULT_PAGE_SIZE
    # Europe PMC's own `src` filter: MED (PubMed-indexed) plus PMC and PPR
    # (preprints) is the broadest reasonable default. Restricting to MED only
    # would silently drop exactly the preprint coverage that is Europe PMC's
    # advantage over PubMed alone.
    result_type: str = "core"


def _passage_from_result(record: dict[str, Any]) -> LiteraturePassage | None:
    """Build a passage from one Europe PMC search result, or None if unusable.

    A record with a title but no abstract is common (some record types carry
    only metadata) and is skipped rather than stored with empty text -- an
    empty passage would never match a search and would only inflate a count
    that should reflect what is actually retrievable.
    """
    title = str(record.get("title") or "").strip()
    abstract = str(record.get("abstractText") or "").strip()
    if not title or not abstract:
        return None

    pmid = record.get("pmid")
    pmcid = record.get("pmcid")
    doi = record.get("doi")
    if pmid:
        source_id = f"pmid:{pmid}"
    elif pmcid:
        source_id = f"pmcid:{pmcid}"
    elif doi:
        source_id = f"doi:{doi}"
    else:
        # No independently checkable identifier at all -- constructed anyway
        # rather than dropped, since is_independently_checkable already
        # exists precisely to flag this case, and dropping it here would
        # duplicate that logic in a second place.
        source_id = f"europepmc:{record.get('id', 'unknown')}"

    journal_info = record.get("journalInfo") or {}
    journal_title = ((journal_info.get("journal") or {}).get("title")) if isinstance(journal_info, dict) else None
    year_raw = record.get("pubYear")
    year = int(year_raw) if isinstance(year_raw, str) and year_raw.isdigit() else None

    return LiteraturePassage(
        passage_id=f"europepmc:{record.get('id', source_id)}",
        text=f"{title}. {abstract}",
        title=title,
        source_id=source_id,
        year=year,
        publication=journal_title,
    )


@dataclass
class EuropePmcConnector:
    """Search Europe PMC and produce checkable literature passages."""

    config: EuropePmcConfig = field(default_factory=EuropePmcConfig)

    def __post_init__(self) -> None:
        self._limiter = RateLimiter(self.config.requests_per_second)

    def search(self, query: str, *, max_results: int = 25) -> list[LiteraturePassage]:
        """Search Europe PMC for a query string, returning usable passages.

        Fewer passages than ``max_results`` is expected and not an error:
        some results lack an abstract and are skipped by
        `_passage_from_result`. A caller wanting exactly N should ask for
        more than N and take what is usable, the same posture
        `retrieve_candidates` takes toward its own `max_candidates`.
        """
        passages: list[LiteraturePassage] = []
        cursor = "*"
        while len(passages) < max_results:
            page = self._fetch_page(query, cursor)
            results = page.get("resultList", {}).get("result", [])
            if not results:
                break
            for record in results:
                passage = _passage_from_result(record)
                if passage is not None:
                    passages.append(passage)
                    if len(passages) >= max_results:
                        break
            cursor = page.get("nextCursorMark")
            if not cursor or cursor == page.get("request", {}).get("cursorMark"):
                break
        return passages

    def search_for_concepts(self, concepts: Sequence[str], *, max_results: int = 25) -> list[LiteraturePassage]:
        """Build a reasonable query from case concepts and search for them.

        Multi-word concepts are quoted so Europe PMC treats them as a
        phrase, not an implicit AND of separate words -- "connective tissue
        weakness" unquoted would match any record containing all three words
        anywhere, not the phrase.
        """
        terms = [f'"{concept}"' if " " in concept else concept for concept in concepts if concept]
        if not terms:
            return []
        return self.search(" OR ".join(terms), max_results=max_results)

    def populate(self, index: LiteratureIndex, query: str, *, max_results: int = 25) -> int:
        """Search and add results directly to an index, returning how many were added."""
        return index.add_many(self.search(query, max_results=max_results))

    def _fetch_page(self, query: str, cursor: str) -> dict[str, Any]:  # pragma: no cover - network call
        self._limiter.wait()
        params = {
            "query": query,
            "format": "json",
            "resultType": self.config.result_type,
            "pageSize": str(self.config.page_size),
            "cursorMark": cursor,
        }
        request = Request(
            f"{EUROPE_PMC_BASE}?{urlencode(params)}",
            headers={"User-Agent": self.config.tool},
        )
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8", errors="ignore"))


def populate_index_for_concepts(
    index: LiteratureIndex, concepts: Iterable[str], *, connector: EuropePmcConnector | None = None,
    max_results: int = 25,
) -> int:
    """Convenience entry point: search Europe PMC for these concepts, add what is usable.

    A thin wrapper rather than the only way to use this module -- a caller
    curating a specific query has `EuropePmcConnector.search` directly, and
    this exists for the common case of refreshing the index around a
    known set of clinical concepts.
    """
    connector = connector or EuropePmcConnector()
    return connector.populate(index, " OR ".join(f'"{c}"' if " " in c else c for c in concepts), max_results=max_results)
