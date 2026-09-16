"""Query the EMA's Product Management Service (PMS) Public API -- EU-authorised medicinal products, in FHIR R5.

Verified directly against EMA's own July 2026 FAQ document, not a
third-party summary: the API is live in public beta, with a working Swagger
UI at `https://api.pms.ema.europa.eu/public/v1/swagger`, exposing a limited
public dataset of `MedicinalProductDefinition` resources in FHIR R5 (5.0.0)
format. Access requires a registered API key -- unlike Europe PMC,
ClinicalTrials.gov and DailyMed, this one is not open by default.

**What this complements, not replaces.** DailyMed carries the FDA's US
formulary; PMS carries the EU's centrally-authorised one. The two markets
overlap substantially for major international drugs but are not identical
-- a drug centrally authorised in the EU and prescribed in Italy may have no
DailyMed entry at all, and PMS is the more directly authoritative source
for what is actually available in an Italian clinical context. Both
connectors are kept, feeding the same literature index, because neither
alone covers what the other does.

**Beta status, stated plainly.** EMA's own documentation calls this a beta
release of a public dataset -- endpoints and response shapes may change
before a stable release. This connector isolates the request shape into one
method (`_fetch_search_page`) for exactly that reason: when EMA's beta
stabilises into a different contract, one method needs updating, not every
caller of this connector.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..memory.literature_index import LiteraturePassage
from .europe_pmc import RateLimiter

PMS_EMA_BASE = "https://api.pms.ema.europa.eu/public/v1"

# No published rate limit was found in EMA's beta documentation. Kept
# conservative for the same reason ClinicalTrials.gov's default is: no
# established relationship with the service to fall back on if this guesses
# wrong, and a beta service is the last one worth stressing.
DEFAULT_REQUESTS_PER_SECOND = 2.0


@dataclass(frozen=True)
class PmsEmaConfig:
    """Authentication for the EMA PMS Public API. Requires a registered API key -- see EMA's PMS registration process."""

    api_key: str | None = None
    tool: str = "melampo-literature-connector"
    requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND


@dataclass(frozen=True)
class PmsEmaAvailability:
    available: bool
    reason: str = ""


def _passage_from_product(resource: dict[str, Any]) -> LiteraturePassage | None:
    """Build a passage from one FHIR MedicinalProductDefinition resource.

    FHIR resources are nested and verbose by design; only the fields this
    project's downstream comparisons actually use (name, identifier,
    ingredient names when present) are extracted, not the full resource --
    the same restraint `document_processing.py` shows toward Nemotron-Parse
    output, taking what is needed rather than everything offered.
    """
    resource_id = str(resource.get("id") or "").strip()
    name_entries = resource.get("name") or []
    title = ""
    if name_entries and isinstance(name_entries, list):
        title = str(name_entries[0].get("productName") or "").strip()
    if not resource_id or not title:
        return None

    ingredients: list[str] = []
    for ingredient in resource.get("ingredient") or []:
        substance = (ingredient.get("substance") or {}).get("code", {}).get("concept", {})
        text = substance.get("text") or ""
        if text:
            ingredients.append(str(text))
    ingredients_text = f" Active ingredients: {', '.join(ingredients)}." if ingredients else ""

    return LiteraturePassage(
        passage_id=f"pms-ema:{resource_id}",
        text=f"{title}.{ingredients_text}",
        title=title,
        source_id=f"pms-ema:{resource_id}",
        year=None,
        publication="EMA Product Management Service",
    )


@dataclass
class PmsEmaConnector:
    """Search the EMA PMS Public API for EU-authorised medicinal products."""

    config: PmsEmaConfig = field(default_factory=PmsEmaConfig)

    def __post_init__(self) -> None:
        self._limiter = RateLimiter(self.config.requests_per_second)

    def availability(self) -> PmsEmaAvailability:
        if not self.config.api_key:
            return PmsEmaAvailability(available=False, reason="PMS_EMA_API_KEY not configured")
        return PmsEmaAvailability(available=True)

    def search(self, product_name: str, *, max_results: int = 25) -> list[LiteraturePassage]:
        """Search MedicinalProductDefinition resources by name."""
        if not self.availability().available or not product_name:
            return []
        try:
            bundle = self._fetch_search_page(product_name)
        except Exception:  # noqa: BLE001 - a failing call degrades this connector, never raises to a caller
            return []
        entries = bundle.get("entry", []) if isinstance(bundle, dict) else []
        passages: list[LiteraturePassage] = []
        for entry in entries:
            resource = entry.get("resource", {})
            passage = _passage_from_product(resource)
            if passage is not None:
                passages.append(passage)
                if len(passages) >= max_results:
                    break
        return passages

    def search_for_concepts(self, concepts: Sequence[str], *, max_results: int = 25) -> list[LiteraturePassage]:
        passages: list[LiteraturePassage] = []
        for concept in concepts:
            if not concept:
                continue
            passages.extend(self.search(concept, max_results=max_results))
            if len(passages) >= max_results:
                break
        return passages[:max_results]

    def populate(self, index: Any, product_name: str, *, max_results: int = 25, store: Any = None) -> int:
        passages = self.search(product_name, max_results=max_results)
        added = index.add_many(passages)
        if store is not None:
            from ..memory.literature_persistence import persist_passage

            for passage in passages:
                persist_passage(store, passage)
        return added

    def _fetch_search_page(self, product_name: str) -> dict[str, Any]:  # pragma: no cover - network call
        self._limiter.wait()
        params = {"name": product_name}
        url = f"{PMS_EMA_BASE}/MedicinalProductDefinition?{urlencode(params)}"
        request = Request(
            url,
            headers={"User-Agent": self.config.tool, "Authorization": f"Bearer {self.config.api_key}"},
        )
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8", errors="ignore"))
