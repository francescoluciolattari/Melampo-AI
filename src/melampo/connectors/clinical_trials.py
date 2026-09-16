"""Populate the literature index from ClinicalTrials.gov, a different kind of source.

Not a second copy of the literature connector. A trial record is not a
finding about established mechanism the way a published paper is -- it is a
registration of an ongoing or completed investigation, with eligibility
criteria, interventions, and a status. Its relevance to this project is
different in kind: where `europe_pmc` answers "does the literature support
this mechanism", a trial record can answer "is there a study a case like
this could be referred to, or that already reports an outcome relevant
here" -- closer to the additional-testing and further-investigation role
this project's `OpenQuestion` machinery already serves for the concept graph,
applied to what the wider research landscape is currently doing rather than
to what the graph itself already contains.

**Kept in `LiteraturePassage`'s own shape rather than a new type.** A trial
record still has a title, a body of text worth matching against case
concepts, and a checkable identifier -- an NCT number is exactly as openly
resolvable as a PMID, at https://clinicaltrials.gov/study/<NCT>. Introducing
a second passage type would fragment `LiteratureIndex.search` into two
code paths for what is, from the retrieval side, the same operation.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..memory.literature_index import LiteratureIndex, LiteraturePassage
from .europe_pmc import RateLimiter

CLINICAL_TRIALS_BASE = "https://clinicaltrials.gov/api/v2/studies"

DEFAULT_REQUESTS_PER_SECOND = 2.0
DEFAULT_PAGE_SIZE = 25

# Recruitment statuses worth surfacing for "further investigation" purposes.
# Terminated and withdrawn trials are excluded by default -- a stopped trial
# is not a place to refer a case or a source of an outcome to cite, and
# including it without flagging why would look like an oversight rather than
# a choice.
DEFAULT_STATUSES = ("RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED")


@dataclass(frozen=True)
class ClinicalTrialsConfig:
    """Politeness identification for ClinicalTrials.gov. No key required or accepted."""

    tool: str = "melampo-literature-connector"
    requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND
    page_size: int = DEFAULT_PAGE_SIZE
    statuses: tuple[str, ...] = DEFAULT_STATUSES


def _passage_from_study(study: dict[str, Any]) -> LiteraturePassage | None:
    """Build a passage from one ClinicalTrials.gov study record, or None if unusable.

    A study with no brief summary is skipped for the same reason a
    literature record with no abstract is: an empty passage cannot be
    matched against, and storing one would inflate a count past what is
    actually retrievable.
    """
    protocol = study.get("protocolSection") or {}
    identification = protocol.get("identificationModule") or {}
    description = protocol.get("descriptionModule") or {}
    status_module = protocol.get("statusModule") or {}
    conditions_module = protocol.get("conditionsModule") or {}

    nct_id = identification.get("nctId")
    title = str(identification.get("briefTitle") or "").strip()
    summary = str(description.get("briefSummary") or "").strip()
    if not nct_id or not title or not summary:
        return None

    conditions = conditions_module.get("conditions") or []
    conditions_text = f" Conditions studied: {', '.join(conditions)}." if conditions else ""

    year_raw = str(status_module.get("startDateStruct", {}).get("date") or "")
    year = int(year_raw[:4]) if year_raw[:4].isdigit() else None

    return LiteraturePassage(
        passage_id=f"nct:{nct_id}",
        text=f"{title}. {summary}{conditions_text}",
        title=title,
        source_id=f"nct:{nct_id}",
        year=year,
        publication="ClinicalTrials.gov",
    )


@dataclass
class ClinicalTrialsConnector:
    """Search ClinicalTrials.gov and produce checkable trial-registry passages."""

    config: ClinicalTrialsConfig = field(default_factory=ClinicalTrialsConfig)

    def __post_init__(self) -> None:
        self._limiter = RateLimiter(self.config.requests_per_second)

    def search(self, condition_query: str, *, max_results: int = 25) -> list[LiteraturePassage]:
        """Search trials for a condition, returning usable passages.

        One page only. Bulk pagination across the full registry is a
        different job -- "how many trials exist for X" -- from this
        connector's job of surfacing a handful of relevant, currently
        meaningful trials for a specific case's concepts.
        """
        page = self._fetch_page(condition_query)
        studies = page.get("studies", [])
        passages: list[LiteraturePassage] = []
        for study in studies:
            passage = _passage_from_study(study)
            if passage is not None:
                passages.append(passage)
                if len(passages) >= max_results:
                    break
        return passages

    def search_for_concepts(self, concepts: Sequence[str], *, max_results: int = 25) -> list[LiteraturePassage]:
        terms = [concept for concept in concepts if concept]
        if not terms:
            return []
        return self.search(" OR ".join(terms), max_results=max_results)

    def populate(self, index: LiteratureIndex, condition_query: str, *, max_results: int = 25, store: Any = None) -> int:
        """Search and add results directly to an index, returning how many were added.

        ``store`` persists each added passage into the same
        `PersistentJsonlVectorStore` `europe_pmc.py`'s connector writes to --
        one durable backend for both sources, not two.
        """
        passages = self.search(condition_query, max_results=max_results)
        added = index.add_many(passages)
        if store is not None:
            from ..memory.literature_persistence import persist_passage

            for passage in passages:
                persist_passage(store, passage)
        return added

    def _fetch_page(self, condition_query: str) -> dict[str, Any]:  # pragma: no cover - network call
        self._limiter.wait()
        params = {
            "query.cond": condition_query,
            "filter.overallStatus": "|".join(self.config.statuses),
            "pageSize": str(self.config.page_size),
            "format": "json",
        }
        request = Request(
            f"{CLINICAL_TRIALS_BASE}?{urlencode(params)}",
            headers={"User-Agent": self.config.tool},
        )
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8", errors="ignore"))
