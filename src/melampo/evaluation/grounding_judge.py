"""Structural detection of overreach in grounded claims.

The existing ``RAGEvaluator`` computes faithfulness as the fraction of answer
terms appearing somewhere in the evidence. That catches fabricated entities, but
it cannot see the failure mode recursive retrieval actually exhibits.

Consider a claim asserting that finding A is consistent with condition B, where
A appears in one fragment and B in another and no fragment relates them. Every
term is supported, so term-overlap faithfulness scores 1.0 and the provenance
check passes at 100%, because both citations are real. The unsupported part is
the *relation* between them — connective tissue the model supplied to make the
fragments cohere. Nothing that inspects evidence individually can detect it.

This module inspects relations instead. Four structural checks:

1. **Unsupported terms** — content words absent from every cited fragment.
2. **Relational overreach** — a relational connective linking two entities that
   never co-occur inside a single fragment.
3. **Modality escalation** — the source hedges, the claim asserts.
4. **Span inflation** — the claim carries far more content than the cited spans.

**A relation absent from the case is not automatically an error.** Most clinical
inference connects a finding to a condition through knowledge outside the case:
bibasilar opacities relate to cardiac failure through pulmonary oedema, and no
report needs to say so for the relation to hold. Checking only against case
fragments would conflate a legitimate inference with a fabrication and reject
both.

The judge therefore consults two corpora. A relation asserted by a fragment is
``grounded_in_case``. A relation absent from every fragment but supported by a
path through the concept graph is ``knowledge_mediated``: admissible as a
hypothesis, carrying the path as its provenance. A relation supported by neither
is ``unsupported_relation`` — fabrication, and the only one of the three that is
a defect.

Speculation is then a gradient rather than a category. A short path over
well-attested edges is a strong inference; a long path over weak edges is
speculative but traceable, and surfaces only where diagnostic indeterminacy
justifies it. No path at all is not speculation.

**Limitation, stated plainly.** These checks are lexical and structural, not
semantic. They cannot recognise a paraphrased relation, and they will flag some
legitimate synthesis. They are a floor: an auditable, model-free, deterministic
signal that catches the common shape of the error and can gate a release without
introducing another model into the evaluation path. A semantic judge remains a
separate open item, and this module does not substitute for it. Where the two
disagree, this one is the conservative side.

Graph coverage is a second limitation with a different character: a relation the
graph does not yet contain is reported as unsupported. That failure is visible
and fixable by extending the graph, unlike a silent acceptance.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..memory.concept_paths import (
    ConceptGraphView,
    ConceptPath,
    find_paths,
    mentioned_concepts,
    shared_mechanisms,
)

RELATIONAL_MARKERS = (
    "because",
    "causes",
    "caused by",
    "due to",
    "therefore",
    "thus",
    "hence",
    "consistent with",
    "suggests",
    "suggestive of",
    "indicates",
    "indicative of",
    "implies",
    "secondary to",
    "attributable to",
    "explains",
    "leads to",
    "results in",
    "compatible with",
    "confirms",
)

HEDGE_MARKERS = (
    "may",
    "might",
    "could",
    "possible",
    "possibly",
    "probable",
    "probably",
    "suspected",
    "cannot be excluded",
    "raises the possibility",
    "appears",
    "apparent",
    "likely",
    "unlikely",
    "equivocal",
)

ASSERTION_MARKERS = (
    "is",
    "are",
    "was",
    "were",
    "confirms",
    "confirmed",
    "demonstrates",
    "establishes",
    "shows",
    "proves",
    "definite",
    "definitive",
    "diagnostic of",
)

STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "for", "from", "had", "has", "have", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their", "there", "these", "this", "to", "was", "were", "will", "with", "without", "which", "who", "whom", "whose", "than", "then"]
)

VERDICT_GROUNDED = "grounded"
VERDICT_KNOWLEDGE_MEDIATED = "knowledge_mediated"
VERDICT_SUSPECTED_OVERREACH = "suspected_overreach"
VERDICT_OVERREACH = "overreach"


@dataclass
class GroundingAssessment:
    """Result of judging one claim against its cited fragments."""

    verdict: str
    overreach_score: float
    unsupported_terms: list[str] = field(default_factory=list)
    unsupported_relations: list[str] = field(default_factory=list)
    mediated_relations: list[dict[str, Any]] = field(default_factory=list)
    modality_escalations: list[str] = field(default_factory=list)
    span_inflation_ratio: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        return self.verdict == VERDICT_GROUNDED

    @property
    def is_admissible(self) -> bool:
        """Whether the claim may enter reasoning at all.

        A knowledge-mediated claim is admissible but not as a fact: it carries a
        graph path rather than a case citation, and belongs in the differential
        as a hypothesis to be examined.
        """
        return self.verdict in {VERDICT_GROUNDED, VERDICT_KNOWLEDGE_MEDIATED}

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "overreach_score": round(self.overreach_score, 3),
            "unsupported_terms": list(self.unsupported_terms),
            "unsupported_relations": list(self.unsupported_relations),
            "mediated_relations": list(self.mediated_relations),
            "modality_escalations": list(self.modality_escalations),
            "span_inflation_ratio": round(self.span_inflation_ratio, 3),
            "notes": list(self.notes),
        }


@dataclass
class GroundingJudge:
    """Deterministic structural judge for claim-to-evidence grounding.

    With no ``concept_graph`` supplied the judge checks against case fragments
    only and every unsupported relation is reported as such. Supplying a graph
    lets it separate inference from fabrication.
    """

    suspected_threshold: float = 0.25
    overreach_threshold: float = 0.5
    max_span_inflation: float = 2.0
    concept_graph: ConceptGraphView | None = None
    max_hops: int = 3
    min_edge_weight: float = 0.0

    def assess(self, claim: str, fragments: Sequence[dict[str, Any] | str]) -> GroundingAssessment:
        texts = [_fragment_text(fragment) for fragment in fragments]
        texts = [text for text in texts if text]

        if not claim.strip():
            return GroundingAssessment(verdict=VERDICT_GROUNDED, overreach_score=0.0, notes=["empty claim"])
        if not texts:
            return GroundingAssessment(
                verdict=VERDICT_OVERREACH,
                overreach_score=1.0,
                notes=["claim has no cited fragments"],
            )

        claim_terms = _content_terms(claim)
        fragment_term_sets = [_content_terms(text) for text in texts]
        all_fragment_terms: set[str] = set().union(*fragment_term_sets) if fragment_term_sets else set()

        unsupported = sorted(claim_terms - all_fragment_terms)
        term_penalty = len(unsupported) / max(len(claim_terms), 1)

        relation_findings = self._unsupported_relations(claim, fragment_term_sets)
        unsupported_relations, mediated_relations = self._split_by_knowledge(relation_findings)
        relation_penalty = min(1.0, len(unsupported_relations) * 0.5)

        escalations = self._modality_escalations(claim, texts)
        modality_penalty = min(1.0, len(escalations) * 0.5)

        inflation = len(claim) / max(sum(len(text) for text in texts), 1)
        inflation_penalty = 1.0 if inflation > self.max_span_inflation else 0.0

        overreach_score = min(
            1.0,
            term_penalty * 0.3 + relation_penalty * 0.4 + modality_penalty * 0.2 + inflation_penalty * 0.1,
        )

        if overreach_score >= self.overreach_threshold:
            verdict = VERDICT_OVERREACH
        elif overreach_score >= self.suspected_threshold:
            verdict = VERDICT_SUSPECTED_OVERREACH
        elif mediated_relations:
            verdict = VERDICT_KNOWLEDGE_MEDIATED
        else:
            verdict = VERDICT_GROUNDED

        notes: list[str] = []
        if unsupported_relations:
            notes.append("relation asserted between entities that never co-occur in a cited fragment")
        if mediated_relations:
            notes.append("relation absent from the case but supported by a concept graph path")
        if escalations:
            notes.append("claim asserts what the source hedges")
        if inflation_penalty:
            notes.append("claim length exceeds the cited spans")

        return GroundingAssessment(
            verdict=verdict,
            overreach_score=overreach_score,
            unsupported_terms=unsupported,
            unsupported_relations=unsupported_relations,
            mediated_relations=mediated_relations,
            modality_escalations=escalations,
            span_inflation_ratio=inflation,
            notes=notes,
        )

    def faithfulness(self, claims: Iterable[tuple[str, Sequence[dict[str, Any] | str]]]) -> float:
        """Fraction of claims judged grounded. Complements term-overlap faithfulness."""
        assessments = [self.assess(claim, fragments) for claim, fragments in claims]
        if not assessments:
            return 0.0
        return sum(1 for item in assessments if item.is_grounded) / len(assessments)

    def _unsupported_relations(
        self, claim: str, fragment_term_sets: list[set[str]]
    ) -> list[tuple[str, set[str], set[str], str, str]]:
        """Relations whose two sides never co-occur inside a single fragment.

        Checking one fragment at a time is the whole point. Pooling the evidence
        into a single bag of terms, which is what term-overlap faithfulness does,
        destroys exactly the information needed here: whether any one source
        actually stated the relation.
        """
        lowered = claim.lower()
        findings: list[tuple[str, set[str], set[str], str, str]] = []
        for marker in RELATIONAL_MARKERS:
            position = lowered.find(f" {marker} ")
            if position < 0:
                continue
            left_text = lowered[:position]
            right_text = lowered[position + len(marker) + 2 :]
            left = _content_terms(left_text)
            right = _content_terms(right_text)
            if not left or not right:
                continue
            if not any(left & terms and right & terms for terms in fragment_term_sets):
                findings.append((marker, left, right, left_text, right_text))
        return findings

    def _split_by_knowledge(
        self, findings: list[tuple[str, set[str], set[str], str, str]]
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Separate fabrication from inference by consulting the concept graph."""
        unsupported: list[str] = []
        mediated: list[dict[str, Any]] = []
        for marker, left_terms, right_terms, left_text, right_text in findings:
            left = self._concepts(left_text, left_terms)
            right = self._concepts(right_text, right_terms)
            path = self._best_path(left, right)
            if path is None:
                unsupported.append(marker)
                continue
            entry = {
                "relation": marker,
                "path": path.as_provenance(),
                "description": path.describe(),
            }
            mechanisms = self._mechanisms(left, right)
            if mechanisms:
                entry["shared_mechanisms"] = mechanisms
            mediated.append(entry)
        return unsupported, mediated

    def _concepts(self, text: str, fallback_terms: set[str]) -> set[str]:
        """Graph concepts named in one side of a relation.

        Falls back to single content terms when the graph names none, so a graph
        with single-word nodes still works.
        """
        if self.concept_graph is None:
            return set(fallback_terms)
        matched = set(mentioned_concepts(text, self.concept_graph))
        return matched or set(fallback_terms)

    def _best_path(self, left: set[str], right: set[str]) -> ConceptPath | None:
        if self.concept_graph is None:
            return None
        best: ConceptPath | None = None
        for start in sorted(left):
            for end in sorted(right):
                for path in find_paths(
                    self.concept_graph,
                    start,
                    end,
                    max_hops=self.max_hops,
                    min_edge_weight=self.min_edge_weight,
                ):
                    if best is None or (path.hops, -path.strength) < (best.hops, -best.strength):
                        best = path
        return best

    def _mechanisms(self, left: set[str], right: set[str]) -> list[str]:
        if self.concept_graph is None:
            return []
        found: list[str] = []
        for start in sorted(left):
            for end in sorted(right):
                for concept in shared_mechanisms(
                    self.concept_graph, start, end, min_edge_weight=self.min_edge_weight
                ):
                    if concept not in found:
                        found.append(concept)
        return found

    def _modality_escalations(self, claim: str, texts: list[str]) -> list[str]:
        lowered = claim.lower()
        source = " ".join(texts).lower()
        claim_hedges = {marker for marker in HEDGE_MARKERS if f" {marker} " in f" {lowered} "}
        source_hedges = {marker for marker in HEDGE_MARKERS if f" {marker} " in f" {source} "}
        if not source_hedges or claim_hedges:
            return []
        asserted = [marker for marker in ASSERTION_MARKERS if f" {marker} " in f" {lowered} "]
        return sorted(set(asserted))


def _fragment_text(fragment: dict[str, Any] | str) -> str:
    if isinstance(fragment, str):
        return fragment
    if isinstance(fragment, dict):
        return str(fragment.get("text", ""))
    return ""


def _content_terms(text: str) -> set[str]:
    tokens = "".join(character if character.isalnum() else " " for character in text.lower()).split()
    return {token for token in tokens if len(token) > 2 and token not in STOPWORDS}
