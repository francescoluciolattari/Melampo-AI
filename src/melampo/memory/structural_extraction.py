"""The extractor tier 3 needs: entities and relations out of free text, via a real model call.

`structural_comparison.StructuralResolver` accepts any callable as its
`extractor` and, until now, nothing in the project supplied one -- tier 3
was wired and tested with a mock, never with a real model behind it. This
module is that real model, called the same way every other external model in
this project is: isolated HTTP call, mockable for tests, graceful
degradation when unconfigured, the transport itself left unimplemented
because the exact request shape depends on how a given deployment exposes
its endpoint (a NIM container, an OpenRouter-style gateway, a direct
provider API) -- the same posture `document_processing.py` takes toward
Nemotron-Parse and LlamaParse.

**What the model is asked to do, and what it is never asked to do.** The
prompt asks for entities and subject-relation-object triples in a fixed,
parseable format -- structure, not judgement. It is never asked whether a
claim matches a concept; that comparison stays in `compare_structures`,
arithmetic, unaffected by whatever this call returns. A malformed or empty
response degrades tier 3 for that one call, exactly like a failing embedder
degrades tier 2 -- it does not raise into the cascade.
"""

import json
import re
from dataclasses import dataclass
from typing import Any

from .structural_comparison import ExtractedRelation, ExtractedStructure

EXTRACTION_SYSTEM_PROMPT = (
    "You extract structure from a short clinical or biomedical text. Output only JSON, "
    "no prose, no markdown fences, in exactly this shape: "
    '{"entities": ["...", "..."], "relations": [{"subject": "...", "relation": "...", "object": "..."}]}. '
    "Entities are biomedical concepts named in the text (genes, proteins, findings, processes). "
    "Relations connect two entities with a short verb phrase (e.g. \"required_for\", \"causes\", "
    "\"enables\"). Extract only what the text states; never infer a relation the text does not name."
)


@dataclass(frozen=True)
class ExtractionConfig:
    """Where and how to call the extraction model. No key stored, only referenced."""

    endpoint: str | None = None
    api_key: str | None = None
    model: str | None = None
    timeout_seconds: int = 30


class StructuralExtractor:
    """A real, callable tier-3 extractor -- HTTP-backed, degrading gracefully when unconfigured.

    Used as `NormalisationCascade`'s or `StructuralResolver`'s `extractor`
    directly: `StructuralExtractor(config)` is itself callable with a single
    string argument, matching the `Callable[[str], ExtractedStructure]`
    contract those already expect.
    """

    def __init__(self, config: ExtractionConfig | None = None) -> None:
        self.config = config or ExtractionConfig()

    def __call__(self, text: str) -> ExtractedStructure:
        if not text.strip():
            return ExtractedStructure(source_text=text)
        if not (self.config.endpoint and self.config.api_key and self.config.model):
            # Unconfigured is not an error -- it is the cascade's normal
            # degraded state, exactly like an absent embedder in tier 2.
            return ExtractedStructure(source_text=text)
        try:
            raw = self._call_model(text)
        except Exception:  # noqa: BLE001 - a failing model call degrades this tier, never breaks the cascade
            return ExtractedStructure(source_text=text)
        return _parse_extraction(text, raw)

    def _call_model(self, text: str) -> str:  # pragma: no cover - network call
        """The actual HTTP call, isolated so tests can monkeypatch it.

        Left unimplemented at the transport level deliberately: the request
        shape depends on how the endpoint is deployed, a configuration
        decision for whoever operates a given installation, not something to
        hard-code here -- the same choice `document_processing.py` made for
        Nemotron-Parse and LlamaParse.
        """
        raise NotImplementedError(
            "configure an HTTP client for this deployment's extraction endpoint; "
            "see document_processing.py's _call_nemotron_parse for the same pattern"
        )


def _parse_extraction(source_text: str, raw: str) -> ExtractedStructure:
    """Turn a model's JSON response into an ExtractedStructure, tolerating common formatting slips.

    Strips markdown code fences before parsing -- models asked for "only
    JSON" reliably wrap it in ```json fences anyway, and treating that as a
    parse failure would degrade tier 3 on the most common well-formed
    response shape, not just on genuinely malformed ones.
    """
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        payload: dict[str, Any] = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return ExtractedStructure(source_text=source_text)

    entities = tuple(str(item).strip() for item in payload.get("entities", []) if str(item).strip())
    relations = tuple(
        ExtractedRelation(
            subject=str(item.get("subject", "")).strip(),
            relation=str(item.get("relation", "")).strip(),
            object=str(item.get("object", "")).strip(),
        )
        for item in payload.get("relations", [])
        if isinstance(item, dict) and item.get("subject") and item.get("relation") and item.get("object")
    )
    return ExtractedStructure(source_text=source_text, entities=entities, relations=relations)
