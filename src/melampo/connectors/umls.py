"""Query the UMLS Metathesaurus, and bridge HPO codes to every vocabulary UMLS asserts a synonym in.

Verified directly against NLM's own UTS API documentation. Base URI
`https://uts-ws.nlm.nih.gov/rest`, `apiKey` as a required query parameter on
every call, authenticated with the same key a UTS account issues after the
license agreement is accepted.

**The crosswalk endpoint is why this connector exists.** NLM's own
documentation uses this project's exact scenario as its worked example:
"crosswalk the Human Phenotype Ontology (HPO) code HP:0001947... to see if
there are any SNOMEDCT codes for which the UMLS has asserted synonymy."
Every concept this project's graph already has is an HPO code -- crosswalk
takes one directly and returns whatever other vocabularies (RxNorm, MeSH,
LOINC, and yes, SNOMED CT when available) share its CUI, with no need to
search by free text first. This is the cross-vocabulary bridge raised in
discussion, built on the mechanism NLM itself designed it for.

**Search remains available for the case crosswalk cannot cover**: a phrase
that has no HPO code at all (a clinician's own wording, not yet mapped to
anything). `/search` takes free text and returns CUIs directly.

**UMLS's own disclaimer, carried forward rather than dropped.** NLM states
plainly that "the synonymy asserted by the UMLS... has not been rigorously
tested and maintained in actual clinical care" and that results "should be
carefully reviewed for relevancy". A crosswalk hit is not a clinical
equivalence claim; `CrosswalkResult` keeps the source CUI as its own field
so a downstream caller can see why two codes were linked, not just that
they were.
"""

import json
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

UMLS_BASE = "https://uts-ws.nlm.nih.gov/rest"


@dataclass(frozen=True)
class UmlsConfig:
    """UMLS UTS authentication. Requires a licensed UTS account's API key."""

    api_key: str | None = None
    tool: str = "melampo-literature-connector"
    version: str = "current"


@dataclass(frozen=True)
class UmlsConcept:
    """One CUI, with its preferred name."""

    cui: str
    name: str
    root_source: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"cui": self.cui, "name": self.name, "root_source": self.root_source}


@dataclass(frozen=True)
class CrosswalkResult:
    """A code in another vocabulary that shares a CUI with the source code queried.

    ``source_cui`` is kept explicitly rather than left implicit: two codes
    sharing a CUI is *why* they crosswalked, and dropping it would turn a
    traceable claim ("these share CUI C0009044") into an opaque one
    ("UMLS says these are the same").
    """

    ui: str
    name: str
    root_source: str
    source_cui: str

    def as_dict(self) -> dict[str, Any]:
        return {"ui": self.ui, "name": self.name, "root_source": self.root_source, "source_cui": self.source_cui}


@dataclass(frozen=True)
class UmlsAvailability:
    available: bool
    reason: str = ""


@dataclass
class UmlsConnector:
    """Search UMLS and crosswalk HPO codes to other vocabularies via shared CUIs."""

    config: UmlsConfig = field(default_factory=UmlsConfig)

    def availability(self) -> UmlsAvailability:
        if not self.config.api_key:
            return UmlsAvailability(available=False, reason="UMLS_API_KEY not configured")
        return UmlsAvailability(available=True)

    def search(self, term: str, *, max_results: int = 10) -> list[UmlsConcept]:
        """Search by free text, returning CUIs and preferred names.

        Used for a phrase that has no HPO code to crosswalk from -- see the
        module docstring for why `crosswalk_from_hpo` is preferred whenever
        one is available.
        """
        if not self.availability().available or not term:
            return []
        try:
            payload = self._get(f"{UMLS_BASE}/search/{self.config.version}", {"string": term})
        except Exception:  # noqa: BLE001 - a failing call degrades this connector, never breaks a caller
            return []
        results = payload.get("result", {}).get("results", [])
        concepts = [
            UmlsConcept(cui=str(item.get("ui", "")), name=str(item.get("name", "")), root_source=str(item.get("rootSource", "")))
            for item in results
            if item.get("ui") and item.get("ui") != "NONE"
        ]
        return concepts[:max_results]

    def crosswalk_from_hpo(self, hpo_id: str, *, target_source: str | None = None) -> list[CrosswalkResult]:
        """Find codes in other vocabularies sharing a CUI with this HPO code.

        ``target_source`` restricts to one vocabulary (e.g. "RXNORM", "MSH",
        "SNOMEDCT_US"); omitted, every source UMLS knows a synonym in is
        returned. ``hpo_id`` is expected in HPO's own form ("HP:0001947"),
        matching what `parse_obo` already gives every term in this project's
        graph.
        """
        if not self.availability().available or not hpo_id:
            return []
        params: dict[str, str] = {}
        if target_source:
            params["targetSource"] = target_source
        try:
            payload = self._get(f"{UMLS_BASE}/crosswalk/{self.config.version}/source/HPO/{hpo_id}", params)
        except Exception:  # noqa: BLE001
            return []
        results = payload.get("result", [])
        if not isinstance(results, list):
            return []
        source_cui = ""
        crosswalked = []
        for item in results:
            cui = str(item.get("concepts", [{}])[0].get("ui", "")) if item.get("concepts") else ""
            source_cui = source_cui or cui
            crosswalked.append(
                CrosswalkResult(
                    ui=str(item.get("ui", "")),
                    name=str(item.get("name", "")),
                    root_source=str(item.get("rootSource", "")),
                    source_cui=source_cui,
                )
            )
        return crosswalked

    def atoms_for_cui(self, cui: str, *, max_results: int = 50) -> list[str]:
        """Every surface form (synonym) UMLS records for a CUI.

        The direct feed for `NormalisationCascade`'s lexical tier -- a
        concept's atoms are exactly the kind of curated synonym set
        `_synonyms_for` already checks a claim against for HPO's own
        synonyms; UMLS atoms extend that same check across every vocabulary
        UMLS bundles.
        """
        if not self.availability().available or not cui:
            return []
        try:
            payload = self._get(f"{UMLS_BASE}/content/{self.config.version}/CUI/{cui}/atoms", {"pageSize": str(max_results)})
        except Exception:  # noqa: BLE001
            return []
        results = payload.get("result", [])
        if not isinstance(results, list):
            return []
        return [str(item.get("name", "")).strip() for item in results if item.get("name")]

    def _get(self, url: str, params: dict[str, str]) -> dict[str, Any]:  # pragma: no cover - network call
        full_params = {**params, "apiKey": self.config.api_key}
        request = Request(f"{url}?{urlencode(full_params)}", headers={"User-Agent": self.config.tool})
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8", errors="ignore"))
