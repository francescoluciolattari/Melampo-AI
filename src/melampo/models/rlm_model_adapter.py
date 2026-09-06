"""Bind the recursive engine to a real model through the audited client.

`RlmEngine` takes a plain callable so it can be driven by scripted stand-ins in
tests. This adapter turns `SafeModelClient` into that callable, which is the
last piece between an engine that has only ever spoken to fixtures and one that
has navigated a case with a real model.

Nothing here relaxes the client's posture. The client keeps its own gates —
`enabled`, `mode`, `allow_remote`, the endpoint host allowlist, secret
redaction, and the execution trace — and this adapter only translates between
its payload shape and the engine's `str -> str` contract.

Two translations deserve naming because they decide what the engine sees.

**A refused or failed call becomes empty text, not an exception.** The engine
already treats an empty response as `model_emitted_no_action` and ends the run
with that reason recorded. Raising instead would lose the trajectory built so
far, and a run that stopped because the provider was unreachable and a run that
stopped because the model had nothing to say are both legitimately "no action" —
the client's own trace holds the distinction for whoever needs it.

**The candidate registry is data, not behaviour.** It records which models are
worth putting on the bench and what constrains each, so that a licence
restriction is visible next to the model rather than living in someone's memory.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

LICENCE_APACHE_2 = "Apache-2.0"
LICENCE_GEMMA_TERMS = "Gemma Terms of Use"
LICENCE_LLAMA_COMMUNITY = "Llama Community License"
LICENCE_ANTHROPIC_COMMERCIAL = "Anthropic Commercial Terms of Service"

# Whether a licence permits commercial use in the EU without further review.
# Not a legal opinion: a flag that makes an open question visible at the point
# of choosing, so a model is never benched, liked and adopted before anyone
# checks whether it can ship.
LICENCE_CLEARED_FOR_EU_COMMERCIAL = {
    LICENCE_APACHE_2: True,
    LICENCE_GEMMA_TERMS: None,
    LICENCE_LLAMA_COMMUNITY: None,
    # A commercial API terms-of-service agreement is a contract entered
    # deliberately, unlike an open-weight licence whose EU applicability may be
    # buried in an acceptable-use policy. Still recorded rather than assumed,
    # since "cleared" here means "the agreement was read", not "no terms apply".
    LICENCE_ANTHROPIC_COMMERCIAL: None,
}


@dataclass(frozen=True)
class RootModelCandidate:
    """A model worth benching as the recursive root, with what constrains it."""

    name: str
    provider: str
    licence: str
    note: str = ""

    @property
    def eu_commercial_cleared(self) -> bool | None:
        """True, False, or None when the licence needs review before shipping."""
        return LICENCE_CLEARED_FOR_EU_COMMERCIAL.get(self.licence)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "licence": self.licence,
            "eu_commercial_cleared": self.eu_commercial_cleared,
            "note": self.note,
        }


DEFAULT_CANDIDATES = (
    RootModelCandidate(
        name="mistral-small-3.1",
        provider="mistral",
        licence=LICENCE_APACHE_2,
        note="Reported as the most instruction-obedient of its class on exact output formats.",
    ),
    RootModelCandidate(
        name="mistral-small-4",
        provider="mistral",
        licence=LICENCE_APACHE_2,
        note="Newer sparse MoE; instruction following claimed but the adherence figure is on 3.1.",
    ),
    RootModelCandidate(
        name="qwen-3.5",
        provider="qwen",
        licence=LICENCE_APACHE_2,
        note="Leads open-weight comparisons overall; same permissive licence, so cheap to include.",
    ),
    RootModelCandidate(
        name="gemma-3-27b",
        provider="google",
        licence=LICENCE_GEMMA_TERMS,
        note="Terms are more restrictive than Apache 2.0 and need review before any commercial use.",
    ),
    RootModelCandidate(
        name="llama-3.3-70b",
        provider="meta",
        licence=LICENCE_LLAMA_COMMUNITY,
        note=(
            "Dense and text-only, so unaffected by the Llama 4 restriction on EU-based "
            "companies. Benched for comparison; the community licence still needs review "
            "before shipping. Llama 4 is deliberately absent."
        ),
    ),
    RootModelCandidate(
        name="claude",
        provider="anthropic",
        licence=LICENCE_ANTHROPIC_COMMERCIAL,
        note=(
            "Commercial terms of service need review before shipping, same as any other "
            "candidate here. Reached through OpenRouter, a named and established aggregator "
            "that proxies to the real provider -- not the unverified gateway (oneprovider.dev) "
            "considered and rejected: a public review of that service states the model actually "
            "served behind it is not Claude at all, which is the exact failure this bench exists "
            "to avoid propagating."
        ),
    ),
)

# Benching a model is not adopting it. A candidate whose licence is unresolved
# belongs on the bench — comparison is how you learn what a permissive model
# costs you in capability — but must not pass silently into deployment on the
# strength of a good score.
BENCH_ONLY_UNTIL_LICENCE_REVIEW = frozenset(
    {LICENCE_GEMMA_TERMS, LICENCE_LLAMA_COMMUNITY, LICENCE_ANTHROPIC_COMMERCIAL}
)


@dataclass
class RootModelAdapter:
    """Expose a `SafeModelClient` as the `str -> str` callable the engine expects."""

    client: Any
    max_tokens: int = 512
    temperature: float = 0.0
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, prompt: str) -> str:
        response = self.client.execute(
            {
                "prompt": prompt,
                "max_tokens": self.max_tokens,
                # Zero temperature: the loop is navigation, not composition, and
                # a deterministic decode makes a trajectory reproducible for the
                # audit record.
                "temperature": self.temperature,
            }
        )
        text = _extract_text(response)
        self.calls.append({"status": response.get("status"), "characters": len(text)})
        return text

    def report(self) -> dict[str, Any]:
        return {
            "calls": len(self.calls),
            "not_called": sum(1 for item in self.calls if item["status"] != "completed"),
        }


def _extract_text(response: dict[str, Any]) -> str:
    """Pull generated text from the client's response, whatever shape it took.

    Returns empty string on a refused or failed call so the engine records
    `model_emitted_no_action` and keeps the trajectory, rather than losing it to
    an exception.
    """
    if not isinstance(response, dict) or response.get("status") != "completed":
        return ""
    for key in ("text", "output", "completion", "content"):
        value = response.get(key)
        if isinstance(value, str):
            return value
    payload = response.get("response")
    if isinstance(payload, dict):
        for key in ("text", "output", "completion", "content"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
    return ""


def root_model_from_client(client: Any, **kwargs: Any) -> Callable[[str], str]:
    """Convenience: the adapter as a bare callable."""
    return RootModelAdapter(client=client, **kwargs)
