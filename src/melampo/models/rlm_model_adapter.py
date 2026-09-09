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
LICENCE_MIT = "MIT"
LICENCE_ANTHROPIC_COMMERCIAL = "Anthropic Commercial Terms of Service"
LICENCE_OPENAI_COMMERCIAL = "OpenAI Commercial Terms of Service"
LICENCE_XAI_COMMERCIAL = "xAI Commercial Terms of Service"
LICENCE_GOOGLE_COMMERCIAL = "Google Commercial Terms of Service"
# Distinct from LICENCE_GEMMA_TERMS: Gemini is a proprietary commercial API
# (like Claude, GPT, Grok), not an open-weight release with its own terms.
# Conflating the two would misrepresent what governs each.
LICENCE_NVIDIA_OPEN = "NVIDIA Open Model License"
# Open-weight but with NVIDIA's own custom terms, not Apache/MIT -- treated
# with the same review requirement as Llama's and Gemma 3's licences rather
# than assumed permissive because weights are published.
# Terms not directly confirmed from a primary source at the time these
# candidates were added -- a placeholder that forces review rather than a
# guess. Distinct from LICENCE_GEMMA_TERMS/LICENCE_LLAMA_COMMUNITY, whose
# specific restrictions (EU acceptable-use scope, etc.) are documented; this
# one means only "unverified", not "known and restrictive".
LICENCE_UNVERIFIED = "Unverified -- confirm before use"

# Whether a licence permits commercial use in the EU without further review.
# Not a legal opinion: a flag that makes an open question visible at the point
# of choosing, so a model is never benched, liked and adopted before anyone
# checks whether it can ship.
LICENCE_CLEARED_FOR_EU_COMMERCIAL = {
    LICENCE_APACHE_2: True,
    LICENCE_MIT: True,
    LICENCE_GEMMA_TERMS: None,
    LICENCE_LLAMA_COMMUNITY: None,
    # A commercial API terms-of-service agreement is a contract entered
    # deliberately, unlike an open-weight licence whose EU applicability may be
    # buried in an acceptable-use policy. Still recorded rather than assumed,
    # since "cleared" here means "the agreement was read", not "no terms apply".
    LICENCE_ANTHROPIC_COMMERCIAL: None,
    LICENCE_OPENAI_COMMERCIAL: None,
    LICENCE_XAI_COMMERCIAL: None,
    LICENCE_GOOGLE_COMMERCIAL: None,
    LICENCE_NVIDIA_OPEN: None,
    LICENCE_UNVERIFIED: None,
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


# Every Anthropic candidate below is reached through OpenRouter, a named and
# established aggregator that proxies to the real provider. This is not the
# default choice: a third-party gateway advertising Claude access
# (oneprovider.dev) was evaluated and rejected first, because a public review
# of that service states the model actually served behind it is not Claude at
# all -- the exact failure this bench exists to avoid propagating. See
# docs/recursive_engine_decision_record.md for the full reasoning.
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
        name="mistral-large-openrouter",
        provider="mistral",
        licence=LICENCE_APACHE_2,
        note=(
            "Added after mistral-small-3.1's direct-API call hit HTTP 429 on the first live "
            "run -- a genuine rate limit on Mistral's own free evaluation tier, documented as "
            "conservative and intended for prototyping, not a wrong slug or a bug. OpenRouter's "
            "pass-through has its own, separate limits, so this is a real second path rather "
            "than hitting the same wall twice. Mistral Large 3 (Dec 2025), Apache 2.0."
        ),
    ),
    RootModelCandidate(
        name="mistral-small-openrouter",
        provider="mistral",
        licence=LICENCE_APACHE_2,
        note="Same rationale as mistral-large-openrouter, at the small tier: a second path to the model the direct API rate-limited.",
    ),
    RootModelCandidate(
        name="qwen-3.5",
        provider="qwen",
        licence=LICENCE_APACHE_2,
        note=(
            "Slug corrected after the first live run: the original "
            "\"qwen-3.5-72b-instruct\" was invented and never existed as a real model -- "
            "there is no 72B-parameter Qwen 3.5 variant. Verified against OpenRouter's own "
            "listing: qwen/qwen3.5-plus-02-15."
        ),
    ),
    RootModelCandidate(
        name="qwen-3.7",
        provider="qwen",
        licence=LICENCE_APACHE_2,
        note="Flagship of the 3.7 generation, benched alongside 3.5 and 3.8 rather than assuming newer is better.",
    ),
    RootModelCandidate(
        name="qwen-3.8",
        provider="qwen",
        licence=LICENCE_APACHE_2,
        note=(
            "Current Qwen flagship as of September 2026 (2.4T-parameter MoE). The generic "
            "\"qwen/qwen3.8-max\" slug is used rather than a dated snapshot, so it tracks "
            "Alibaba's own updates instead of going stale the way the original 3.5 slug did."
        ),
    ),
    RootModelCandidate(
        name="glm-5",
        provider="z-ai",
        licence=LICENCE_MIT,
        note=(
            "MIT licensed, full weights on Hugging Face -- no acceptable-use policy to review, "
            "unlike Llama or Gemma. Open-weight #1 on Artificial Analysis at release. Trained on "
            "non-NVIDIA hardware, which is unrelated to its suitability here but worth knowing."
        ),
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
            "before shipping."
        ),
    ),
    RootModelCandidate(
        name="llama-4-maverick",
        provider="meta",
        licence=LICENCE_LLAMA_COMMUNITY,
        note=(
            "Commercial terms need review before shipping, same as any other candidate here -- "
            "and separately, benched for comparison only, not for adoption: Llama 4's Acceptable "
            "Use Policy withholds rights from EU-based individuals and companies (see "
            "recursive_engine_decision_record.md), which stands regardless of this bench's "
            "result. Included so the comparison table states a measured gap rather than an "
            "assumed one."
        ),
    ),
    RootModelCandidate(
        name="llama-4-scout",
        provider="meta",
        licence=LICENCE_LLAMA_COMMUNITY,
        note=(
            "Commercial terms need review before shipping, same as any other candidate here. "
            "Same EU restriction and same bench-only status as Maverick; the smaller sibling, "
            "included because a licence-restricted flagship and a licence-restricted small "
            "model are not equally informative about the licence's practical cost."
        ),
    ),
    RootModelCandidate(
        name="claude-sonnet-5",
        provider="anthropic",
        licence=LICENCE_ANTHROPIC_COMMERCIAL,
        note=(
            "Commercial terms of service need review before shipping, same as any other "
            "candidate here. Default Claude tier: mid-tier cost ($2/M in, $10/M out on "
            "OpenRouter) for a task that is format adherence, not depth of reasoning."
        ),
    ),
    RootModelCandidate(
        name="claude-opus-5",
        provider="anthropic",
        licence=LICENCE_ANTHROPIC_COMMERCIAL,
        note=(
            "Commercial terms of service need review before shipping, same as any other "
            "candidate here. Benched alongside Sonnet rather than assumed unnecessary: a 25% "
            "cost premium ($2.50/M in, $12.50/M out) is worth paying only if it also raises "
            "adherence, and that is a question for the bench, not for argument."
        ),
    ),
    RootModelCandidate(
        name="claude-fable-5.1",
        provider="anthropic",
        licence=LICENCE_ANTHROPIC_COMMERCIAL,
        note=(
            "Commercial terms of service need review before shipping, same as any other "
            "candidate here. Mythos-tier, five times Sonnet's cost ($10/M in, $50/M out). "
            "Benched for the same reason as Opus: the premium is justified by a measured "
            "adherence gain or it is not, and guessing either way defeats the purpose of a bench."
        ),
    ),
    RootModelCandidate(
        name="gpt-6-astra",
        provider="openai",
        licence=LICENCE_OPENAI_COMMERCIAL,
        note=(
            "OpenAI was omitted from the first version of this registry alongside Mistral, Qwen "
            "and Llama with no reasoning given -- an oversight, not a decision, corrected here. "
            "Commercial API terms need review before shipping, same as Claude's."
        ),
    ),
    RootModelCandidate(
        name="glm-5.3",
        provider="z-ai",
        licence=LICENCE_UNVERIFIED,
        note=(
            "Licence needs review before shipping: not directly confirmed for this specific "
            "release, marked unverified rather than assumed to match glm-5's MIT. Newer "
            "flagship than glm-5, added after a live run raised the completion-rate question "
            "this bench exists to answer -- its own listing states reasoning \"is always on "
            "and cannot be disabled\", directly relevant to why some candidates used their "
            "full iteration budget without ever finalising."
        ),
    ),
    RootModelCandidate(
        name="gemma-4-31b",
        provider="google",
        licence=LICENCE_APACHE_2,
        note=(
            "Supersedes gemma-3-27b in two ways at once: released April 2026, and shipped "
            "under Apache 2.0 rather than Gemma 3's more restrictive terms -- newer and "
            "licence-cleared in the same release. Dense, #3 on the Arena text leaderboard at "
            "launch."
        ),
    ),
    RootModelCandidate(
        name="gemma-4-26b-a4b",
        provider="google",
        licence=LICENCE_APACHE_2,
        note="Same generation and licence as gemma-4-31b; MoE with only 4B active parameters, cheaper per call. Both sizes benched rather than assuming the larger one wins.",
    ),
    RootModelCandidate(
        name="kimi-k2.6",
        provider="moonshotai",
        licence=LICENCE_UNVERIFIED,
        note=(
            "Licence terms not directly confirmed; review before shipping, same as any "
            "other unresolved candidate here. Reported to sustain the longest correct "
            "open-weight tool-calling sequences available, which is closer to this bench's "
            "actual task -- a multi-step, format-constrained loop -- than a general "
            "capability score. Chinese-developed; reached here through OpenRouter rather than "
            "a China-hosted endpoint directly, and every document this bench sends is "
            "synthetic, so there is no live data-residency exposure in this context. The "
            "consideration becomes live the moment any candidate here is considered for "
            "production use on real case content."
        ),
    ),
    RootModelCandidate(
        name="deepseek-v4-flash",
        provider="deepseek",
        licence=LICENCE_UNVERIFIED,
        note=(
            "Licence terms not directly confirmed; review before shipping. The cheapest "
            "capable candidate here by a wide margin, included with a caveat rather than "
            "assumed reliable: independent integration reports describe the predecessor "
            "generation's structured tool-calling as unreliable and note V4 was too new for a "
            "settled verdict at time of writing. This bench measures exactly that question on "
            "our specific six-verb grammar rather than inheriting the reputation either way. "
            "Flash rather than Pro: a separate report describes Pro hitting a thinking-mode "
            "protocol incompatibility in some harnesses. Same data-residency consideration as "
            "kimi-k2.6."
        ),
    ),
    RootModelCandidate(
        name="grok-4-fast",
        provider="xai",
        licence=LICENCE_XAI_COMMERCIAL,
        note=(
            "Commercial terms need review before shipping, same as any other candidate here. "
            "Verified OpenRouter slug for xAI's cost-efficient tier. A costlier flagship tier "
            "was referenced but unconfirmed when this candidate was added; grok-4.6 below is "
            "that confirmed flagship, added separately once its slug was verified rather than "
            "guessed at."
        ),
    ),
    RootModelCandidate(
        name="grok-4.6",
        provider="xai",
        licence=LICENCE_XAI_COMMERCIAL,
        note=(
            "Commercial terms need review before shipping, same as any other candidate here. "
            "xAI's current flagship: an August 2026 post-training refresh of the Grok 4.5 base "
            "(same $2/M in, $6/M out pricing) rather than a new foundation model. Benched "
            "alongside grok-4-fast as the reasoning-capable tier next to the cost-efficient one, "
            "the same cheap-plus-flagship pattern used for every other family here."
        ),
    ),
    RootModelCandidate(
        name="gemini-3-pro-preview",
        provider="google",
        licence=LICENCE_GOOGLE_COMMERCIAL,
        note=(
            "Commercial terms need review before shipping. First Gemini candidate in this "
            "registry -- every prior Google entry was Gemma, the open-weight sibling; Gemini "
            "itself was never benched until now. Google's flagship: reasoning cannot be fully "
            "disabled (only a 'High'/'Low' effort choice), similar to GLM-5.3's situation, so "
            "the reasoning-disable hint is sent as a best-effort attempt rather than an "
            "expected guarantee, same as for every other reasoning-mandatory candidate here. "
            "Slug corrected after a live HTTP 404: google/gemini-3-pro-preview (without .1) "
            "was deprecated and shut down by Google on March 9, 2026, after this candidate was "
            "first added; google/gemini-3.1-pro-preview is the confirmed current successor."
        ),
    ),
    RootModelCandidate(
        name="nemotron-3-super",
        provider="nvidia",
        licence=LICENCE_NVIDIA_OPEN,
        note=(
            "Licence needs review: open weights under NVIDIA's own terms, not Apache/MIT. "
            "Its own description names \"cross-document reasoning\" and \"multi-step task "
            "planning\" specifically -- closer to this bench's actual demands than most "
            "candidates' general capability marketing. 120B total / 12B active MoE, verified "
            "native tool-calling support."
        ),
    ),
    RootModelCandidate(
        name="muse-glimmer-30b",
        provider="meta",
        licence=LICENCE_APACHE_2,
        note=(
            "Meta Superintelligence Labs' first open-weight release, and the first Meta entry "
            "in this registry under Apache 2.0 rather than the Llama Community Licence -- so "
            "unlike every Llama candidate here, no EU acceptable-use restriction applies. 30B "
            "dense, distilled from the proprietary Muse Spark, described for long-horizon "
            "agentic workflows with multi-step reasoning, reliable tool use and failure "
            "recovery. Benched with a caveat rather than on reputation: an independent reader "
            "of its published scores flagged a high hallucination rate and advised against "
            "critical tasks. This bench measures navigation and format adherence, not answer "
            "correctness, so it cannot confirm or refute that -- the caveat is recorded here "
            "because a good result on this bench would not address it. Muse Spark itself is "
            "deliberately absent: closed-weight, and its cheap 'contributor' tier states that "
            "prompts and outputs may be used to improve Meta's products."
        ),
    ),
)

# Benching a model is not adopting it. A candidate whose licence is unresolved
# belongs on the bench — comparison is how you learn what a permissive model
# costs you in capability — but must not pass silently into deployment on the
# strength of a good score.
BENCH_ONLY_UNTIL_LICENCE_REVIEW = frozenset(
    {
        LICENCE_GEMMA_TERMS,
        LICENCE_LLAMA_COMMUNITY,
        LICENCE_ANTHROPIC_COMMERCIAL,
        LICENCE_OPENAI_COMMERCIAL,
        LICENCE_XAI_COMMERCIAL,
        LICENCE_GOOGLE_COMMERCIAL,
        LICENCE_NVIDIA_OPEN,
        LICENCE_UNVERIFIED,
    }
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
