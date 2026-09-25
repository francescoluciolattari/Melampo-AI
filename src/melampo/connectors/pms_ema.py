"""Query the EMA's Product Management Service (PMS) Public API -- EU-authorised medicinal products, in FHIR R5.

Verified directly against EMA's own PMS Public API FAQ and the UPD
Registration Guide for UI and API users (both ema.europa.eu documents), not
a third-party summary: the API is live in public beta, with a working
Swagger UI at `https://api.pms.ema.europa.eu/public/v1/swagger`, exposing a
limited public dataset of `MedicinalProductDefinition` resources in FHIR R5
(5.0.0) format.

**Authentication changed from a static key to OAuth2, and this connector
follows that change.** EMA originally issued a single registered API key
(`PMS_EMA_API_KEY`); the FAQ and Registration Guide both now describe an
OAuth2 **client-credentials** flow through a Microsoft Entra ID tenant
instead -- a client id and secret are exchanged for a short-lived bearer
token (documented as valid one hour), which is what actually authorises
each call. `PMS_EMA_API_KEY` was retired accordingly and replaced by
`PMS_EMA_CLIENT_ID` / `PMS_EMA_CLIENT_SECRET`. The token endpoint and scope
are this project's own registered tenant and API, not discovered at
runtime, so they are module constants rather than environment variables --
the credentials that vary per deployment are the client id and secret.

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

**Rate limit, now documented rather than guessed.** The FAQ states the
limit plainly: 500 requests per minute per IP address, with a 429 response
and a `Retry-After` header past that. The conservative default below stays
well under it -- this project's own usage is a handful of calls per tracked
concept in a nightly batch, nowhere near the ceiling -- rather than
matching it exactly the way `pmc_case_reports.py` matches NCBI's published
contract; NCBI publishes a hard ceiling this project deliberately runs at,
while nothing here needs the extra throughput 500/minute would allow.
"""

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..memory.literature_index import LiteraturePassage
from .europe_pmc import RateLimiter

PMS_EMA_BASE = "https://api.pms.ema.europa.eu/public/v1"

# EMA's registered Microsoft Entra ID tenant and API scope for this
# project's PMS Public API registration -- not a per-deployment secret,
# unlike the client id/secret pair, so it lives here rather than in the
# environment.
PMS_EMA_TOKEN_URL = "https://login.microsoftonline.com/bc9dc15c-61bc-4f03-b60b-e5b6d8922839/oauth2/v2.0/token"
PMS_EMA_SCOPE = "api://euema.onmicrosoft.com/upd-apim-secured/.default"

# A token is refreshed this many seconds before its stated expiry, so a
# request that starts just before the deadline is never sent with a token
# that turns invalid mid-flight.
TOKEN_REFRESH_MARGIN_SECONDS = 60.0

# See the module docstring: verified against EMA's own PMS Public API FAQ
# (500 requests/minute/IP), kept well under it rather than matched exactly.
DEFAULT_REQUESTS_PER_SECOND = 2.0


@dataclass(frozen=True)
class PmsEmaConfig:
    """Authentication for the EMA PMS Public API: an OAuth2 client-credentials pair.

    ``client_id`` and ``client_secret`` are issued by EMA's own registration
    process (see the module docstring) and are exchanged for a bearer token
    -- neither is sent directly on a search or fetch call.
    """

    client_id: str | None = None
    client_secret: str | None = None
    token_url: str = PMS_EMA_TOKEN_URL
    scope: str = PMS_EMA_SCOPE
    tool: str = "melampo-literature-connector"
    requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND

    @classmethod
    def from_env(cls) -> "PmsEmaConfig":
        """Read PMS_EMA_CLIENT_ID / PMS_EMA_CLIENT_SECRET from the environment.

        Replaces the single ``PMS_EMA_API_KEY`` this project read until
        EMA's registration moved to Entra ID OAuth2: that secret was
        removed from this project's GitHub configuration, and
        ``PMS_EMA_CLIENT_ID`` / ``PMS_EMA_CLIENT_SECRET`` took its place --
        the same bridge pattern ``UmlsConfig.from_env()`` uses for its own
        single key.
        """
        import os

        return cls(
            client_id=os.environ.get("PMS_EMA_CLIENT_ID"),
            client_secret=os.environ.get("PMS_EMA_CLIENT_SECRET"),
        )


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
        self._access_token: str | None = None
        self._token_expiry: float = 0.0

    def availability(self) -> PmsEmaAvailability:
        if not self.config.client_id or not self.config.client_secret:
            return PmsEmaAvailability(
                available=False, reason="PMS_EMA_CLIENT_ID/PMS_EMA_CLIENT_SECRET not configured"
            )
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

    def populate(
        self, index: Any, product_name: str, *, max_results: int = 25, store: Any = None, graph: Any = None
    ) -> int:
        """Superseded by ``graph`` -- see europe_pmc.py's populate() for why passing it skips the separate JSONL write."""
        passages = self.search(product_name, max_results=max_results)
        added = index.add_many(passages, source_graph=graph) if graph is not None else index.add_many(passages)
        if store is not None and graph is None:
            from ..memory.literature_persistence import persist_passage

            for passage in passages:
                persist_passage(store, passage)
        return added

    def _get_access_token(self) -> str:
        """Return a cached bearer token, refreshing it once it is close to expiry.

        Verified against EMA's own PMS Public API FAQ: the token endpoint is
        a standard Entra ID OAuth2 client-credentials grant, and the token
        it returns is valid for one hour (3600 seconds). Caching here means
        a batch of searches (`search_for_concepts`, or the nightly
        literature refresh looping over many concepts) exchanges a token
        once, not on every call.
        """
        now = time.monotonic()
        if self._access_token is not None and now < self._token_expiry:
            return self._access_token
        payload = self._request_token()
        token = str(payload["access_token"])
        expires_in = float(payload.get("expires_in", 3600))
        self._access_token = token
        self._token_expiry = now + max(expires_in - TOKEN_REFRESH_MARGIN_SECONDS, 0.0)
        return token

    def _request_token(self) -> dict[str, Any]:  # pragma: no cover - network call
        body = urlencode(
            {
                "grant_type": "client_credentials",
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
                "scope": self.config.scope,
            }
        ).encode("ascii")
        request = Request(
            self.config.token_url,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": self.config.tool},
            method="POST",
        )
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8", errors="ignore"))

    def _fetch_search_page(self, product_name: str) -> dict[str, Any]:  # pragma: no cover - network call
        self._limiter.wait()
        token = self._get_access_token()
        params = {"name": product_name}
        url = f"{PMS_EMA_BASE}/MedicinalProductDefinition?{urlencode(params)}"
        request = Request(
            url,
            headers={"User-Agent": self.config.tool, "Authorization": f"Bearer {token}"},
        )
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8", errors="ignore"))
