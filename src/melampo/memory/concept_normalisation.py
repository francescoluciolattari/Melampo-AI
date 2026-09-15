"""Bridge free-text clinical phrasing to graph concepts, in tiers of increasing cost.

The problem, measured rather than assumed. Across live vetting-bench runs,
candidates wrote things like "impaired methylcobalamin-dependent methionine
synthase activity reduces methylation of myelin phospholipids" where the
graph node reads "impaired myelin synthesis". Clinically the same claim;
lexically nothing in common. The existing `concept_names_match` -- exact,
then containment, then word-set equality -- cannot bridge that, by design:
it was built to be safe, and loosening it would readmit the
character-similarity failure this project measured going wrong at the very
start ("pulmonary embolism" scoring closer to "pulmonary oedema" than to a
correct paraphrase).

The field's own name for this is **biomedical entity linking** or **medical
concept normalization**, and the literature's shape is a cascade, not a
single matcher: a cheap lexical tier, then a learned-embedding tier
(SapBERT-style bi-encoder retrieval, optionally cross-encoder re-ranked),
and only then anything more expensive.

**Why the tiers are ordered by determinism, not just by cost.** Tier 1 is
exact and reproducible. Tier 2 is a fixed function once trained -- same
input, same vector, same distance, every time -- so it is deterministic at
runtime even though it was learned. Tier 3 involves a model generating
something, and is the only tier whose output can vary between identical
calls. Ordering them this way means the least reproducible tier only ever
sees what the reproducible ones could not resolve, and every result records
which tier produced it, so a reader can always tell how much of a
verification rested on a learned or generated step.

**No tier is installed here.** Tier 2 needs a sentence-embedding model and
tier 3 needs a language model; both are injected, with the cascade
degrading to whatever is available rather than failing. A deployment with
neither gets exactly today's behaviour, which is the point: this module adds
reach without removing the guarantee that the strict tier still runs first.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .concept_paths import ConceptGraphView, concept_names_match, normalise_concept

TIER_LEXICAL = "lexical"
TIER_EMBEDDING = "embedding"
TIER_STRUCTURAL = "structural"
TIER_NONE = "unresolved"

# Cosine similarity below which a SapBERT-style match is not trusted. Set
# deliberately high: the tier exists to catch genuine paraphrase, and a
# permissive threshold on biomedical embeddings reliably surfaces
# co-occurring-but-distinct concepts -- the same latent-proximity failure
# this project already documented for retrieval, where a query about heart
# failure returns "acute coronary syndrome" because the two sit close in
# embedding space rather than because one answers the other.
DEFAULT_EMBEDDING_THRESHOLD = 0.85

# How much clearer the best match must be than the second-best. A threshold
# alone judges the winner in isolation; this judges whether there was a
# winner at all. Small, because genuine synonyms in a curated concept pool
# usually stand well clear of their neighbours -- and when they do not, the
# honest answer is that the phrase did not resolve, not that the alphabet
# picked one.
DEFAULT_EMBEDDING_MARGIN = 0.05


@dataclass(frozen=True)
class NormalisationResult:
    """Which graph concept a phrase resolved to, and which tier resolved it."""

    phrase: str
    concept: str | None
    tier: str
    score: float = 0.0
    detail: str = ""

    @property
    def resolved(self) -> bool:
        return self.concept is not None

    @property
    def is_deterministic(self) -> bool:
        """Whether this result would be identical on an identical re-run.

        Tiers 1 and 2 are reproducible functions; tier 3 involves generation
        and is not. A verification that rested on tier 3 should be readable
        as such rather than presented alongside an exact lexical match as if
        the two carried the same weight.
        """
        return self.tier in (TIER_LEXICAL, TIER_EMBEDDING, TIER_NONE)

    def as_dict(self) -> dict[str, Any]:
        return {
            "phrase": self.phrase,
            "concept": self.concept,
            "tier": self.tier,
            "score": round(self.score, 4),
            "resolved": self.resolved,
            "is_deterministic": self.is_deterministic,
            "detail": self.detail,
        }


@dataclass
class NormalisationCascade:
    """Lexical, then embedding, then structural -- each only on the previous tier's misses.

    ``embedder`` is any callable turning a string into a vector (a SapBERT
    bi-encoder is the intended one, but nothing here depends on that
    specific model). ``structural_resolver`` is tier 3, kept as an opaque
    callable so this module does not depend on how it works -- see
    `structural_comparison.py` for the intended implementation.
    """

    graph: ConceptGraphView
    embedder: Callable[[str], Sequence[float]] | None = None
    structural_resolver: Callable[[str, Sequence[str]], str | None] | None = None
    embedding_threshold: float = DEFAULT_EMBEDDING_THRESHOLD
    embedding_margin: float = DEFAULT_EMBEDDING_MARGIN
    tier_usage: dict[str, int] = field(default_factory=dict)

    def _record(self, tier: str) -> None:
        self.tier_usage[tier] = self.tier_usage.get(tier, 0) + 1

    def resolve(self, phrase: str, candidates: Sequence[str] | None = None) -> NormalisationResult:
        """Map a phrase to one graph concept, using the cheapest tier that works.

        ``candidates`` narrows the search to concepts already known to be
        relevant -- the mediating concepts a spread surfaced, for instance.
        Without it the whole graph is searched, which is correct but slower
        and, more importantly, more likely to surface a distant concept that
        happens to embed closely.
        """
        pool = list(candidates) if candidates is not None else sorted(self.graph.concepts())
        if not phrase or not pool:
            self._record(TIER_NONE)
            return NormalisationResult(phrase=phrase, concept=None, tier=TIER_NONE)

        lexical = self._resolve_lexical(phrase, pool)
        if lexical.resolved:
            self._record(TIER_LEXICAL)
            return lexical

        embedding = self._resolve_embedding(phrase, pool)
        if embedding.resolved:
            self._record(TIER_EMBEDDING)
            return embedding

        structural = self._resolve_structural(phrase, pool)
        if structural.resolved:
            self._record(TIER_STRUCTURAL)
            return structural

        self._record(TIER_NONE)
        # Carry forward why the later tiers declined, rather than replacing
        # it with a generic message. "No tier resolved this" and "the
        # embedder raised" look identical from the outside otherwise, and
        # only the second is a configuration fault someone should fix.
        reasons = [item.detail for item in (embedding, structural) if item.detail]
        detail = "no tier resolved this phrase to a graph concept"
        if reasons:
            detail = f"{detail} ({'; '.join(reasons)})"
        return NormalisationResult(phrase=phrase, concept=None, tier=TIER_NONE, detail=detail)

    def _resolve_lexical(self, phrase: str, pool: Sequence[str]) -> NormalisationResult:
        """Tier 1: the existing exact/containment/word-set rule, unchanged.

        Reused rather than reimplemented -- it is the same comparison
        `verify_mechanism` already applies, and a second copy here would be
        exactly the drift this project has had to fix once already.
        """
        for concept in sorted(pool, key=len, reverse=True):
            if concept_names_match(phrase, concept):
                return NormalisationResult(
                    phrase=phrase, concept=concept, tier=TIER_LEXICAL, score=1.0,
                    detail="exact, containment, or word-set match",
                )
        return NormalisationResult(phrase=phrase, concept=None, tier=TIER_LEXICAL)

    def _resolve_embedding(self, phrase: str, pool: Sequence[str]) -> NormalisationResult:
        """Tier 2: nearest concept by embedding similarity, above a strict threshold.

        Two guards, not one. The absolute threshold rejects a best match that
        is simply not close to anything. The *margin* guard rejects a best
        match that is barely closer than the runner-up: with dozens of
        clinical concepts in a pool, "nearest" is always something, and an
        embedder under pressure returns a field of near-ties where the winner
        is arbitrary. A threshold alone cannot see that -- it looks at the
        top score in isolation -- and a test with a deliberately degenerate
        embedder confirmed it: every concept scoring 1.0 passes any
        threshold, and the first one encountered wins by accident.
        """
        if self.embedder is None:
            return NormalisationResult(phrase=phrase, concept=None, tier=TIER_EMBEDDING)

        try:
            phrase_vector = self.embedder(phrase)
        except Exception as error:  # noqa: BLE001 - a failing embedder degrades the cascade, never breaks it
            return NormalisationResult(
                phrase=phrase, concept=None, tier=TIER_EMBEDDING, detail=f"embedder failed: {error}"
            )

        scored: list[tuple[float, str]] = []
        for concept in pool:
            try:
                scored.append((cosine_similarity(phrase_vector, self.embedder(concept)), concept))
            except Exception:  # noqa: BLE001, S112 - one bad concept must not abort the whole search
                # Deliberately not logged: a concept whose embedding fails is
                # simply not a candidate, and one line of noise per concept
                # per resolution would bury the signal it is supposed to add.
                continue

        if not scored:
            return NormalisationResult(phrase=phrase, concept=None, tier=TIER_EMBEDDING)

        scored.sort(key=lambda item: -item[0])
        best_score, best_concept = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0

        if best_score < self.embedding_threshold:
            return NormalisationResult(
                phrase=phrase, concept=None, tier=TIER_EMBEDDING, score=best_score,
                detail=f"best embedding similarity {best_score:.3f} below threshold",
            )
        if (best_score - runner_up) < self.embedding_margin:
            return NormalisationResult(
                phrase=phrase, concept=None, tier=TIER_EMBEDDING, score=best_score,
                detail=(
                    f"best match {best_concept!r} ({best_score:.3f}) too close to runner-up "
                    f"({runner_up:.3f}); an arbitrary winner among near-ties is not a resolution"
                ),
            )
        return NormalisationResult(
            phrase=phrase, concept=best_concept, tier=TIER_EMBEDDING, score=best_score,
            detail=f"embedding similarity {best_score:.3f}, clear of runner-up by {best_score - runner_up:.3f}",
        )

    def _resolve_structural(self, phrase: str, pool: Sequence[str]) -> NormalisationResult:
        """Tier 3: hand off to a structural comparison, if one is configured."""
        if self.structural_resolver is None:
            return NormalisationResult(phrase=phrase, concept=None, tier=TIER_STRUCTURAL)
        try:
            concept = self.structural_resolver(phrase, pool)
        except Exception as error:  # noqa: BLE001 - same degradation contract as tier 2
            return NormalisationResult(
                phrase=phrase, concept=None, tier=TIER_STRUCTURAL, detail=f"structural resolver failed: {error}"
            )
        if concept and concept in pool:
            return NormalisationResult(
                phrase=phrase, concept=concept, tier=TIER_STRUCTURAL, score=1.0,
                detail="resolved by structural comparison",
            )
        # A resolver naming something outside the pool has not resolved
        # anything -- it has invented a concept, which is the failure mode
        # tier 3 most needs guarding against.
        return NormalisationResult(
            phrase=phrase, concept=None, tier=TIER_STRUCTURAL,
            detail="structural resolver returned nothing, or a concept not in the graph",
        )

    def usage_report(self) -> dict[str, Any]:
        """How many resolutions each tier handled.

        Worth surfacing in any bench run: a result where tier 3 carried most
        of the load rests on a far less reproducible foundation than one
        resolved almost entirely lexically, and the headline grounding rate
        looks identical either way.
        """
        total = sum(self.tier_usage.values())
        return {
            "total": total,
            "by_tier": dict(sorted(self.tier_usage.items())),
            "deterministic_fraction": (
                sum(count for tier, count in self.tier_usage.items() if tier != TIER_STRUCTURAL) / total
                if total
                else 0.0
            ),
        }


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity between two vectors, stdlib only.

    Written out rather than pulled from numpy: this module is imported by
    code paths that must work without a numeric stack installed, and the
    operation is four lines.
    """
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def sapbert_embedder(model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext") -> Callable[[str], Sequence[float]]:
    """Build a SapBERT embedder, if sentence-transformers is installed.

    SapBERT is a bi-encoder self-aligned on UMLS synonym pairs: it places
    different surface forms of the same biomedical concept close together in
    vector space, which is exactly the gap tier 1 cannot bridge. Returned as
    a plain callable so the cascade never imports the library itself, and
    raises here rather than at first use so a misconfigured deployment fails
    at startup rather than silently mid-run.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "sapbert_embedder needs sentence-transformers installed; "
            "the cascade runs without it, using tier 1 only"
        ) from error

    model = SentenceTransformer(model_name)

    def _embed(text: str) -> Sequence[float]:
        return model.encode(normalise_concept(text)).tolist()

    return _embed
