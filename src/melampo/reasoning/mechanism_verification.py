"""Verify a claimed mechanism against the concept graph instead of comparing strings.

This closes a loop opened much earlier in this project. `FRAME_RELEVANCE` was
built with a `mechanism` slot on the stated reasoning that a relevance
question — "does the family history bear on today's aortic measurement, and
why?" — opens a third mental space, one of clinical consequence, and that the
mapping it asks about is not in the document at any level of careful reading:
it is in the concept graph or nowhere. But `mechanism` was then compared like
every other slot, string against string, which left two failures the frame
was supposed to prevent.

**Agreement that is not corroboration.** Two models writing the same
plausible, invented mechanism agree perfectly by string comparison. Agreement
measures whether two answers match each other; it says nothing about whether
either is grounded. A shared hallucination is indistinguishable from
independent confirmation when the only evidence considered is the two answers
themselves.

**Disagreement that is not conflict.** "connective tissue weakness" and
"inherited aortopathy" may name the same real mechanism in different words.
String comparison reports a conflict and sends a correct case to review.

This module asks a different question of the claim: not "did the two models
write the same thing" but "does the graph independently support a connection
between the factor and the target, and does the claimed mechanism appear on
it". `mediating_concepts` supplies the answer — it spreads from both ends
under the relation, decay and threshold constraints, weights what it reaches
by Information Content, and reports which concepts were reached from both
sides. Those multiply-sourced concepts are the candidate mechanisms.

**What this module refuses to do.** It does not decide which of two models is
right, and it does not overrule agreement or disagreement — it adds an axis.
A claim can be agreed-and-grounded, agreed-but-ungrounded (the dangerous
case, previously invisible), disagreed-but-both-grounded (two valid
descriptions of one real link), or disagreed-and-neither-grounded. Those four
need different responses, and collapsing them into one verdict would discard
exactly what the graph was consulted for.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..memory.concept_paths import ConceptGraphView, normalise_concept
from ..memory.information_content import InformationContentTable
from ..memory.spreading_activation import ActivatedConcept, mediating_concepts
from .frame_answer import FRAME_RELEVANCE, parse_frame_answer

# A claimed mechanism counts as appearing on the graph's connection when it
# matches a multiply-sourced concept at or above this weighted activation.
# Not zero: "pulmonary" is reached from both origins in the Marfan fixture
# and would technically match, at weighted activation 0.004 -- which is
# precisely the near-zero-Information-Content case the whole preceding
# investigation was about, and admitting it would undo that work at the last
# step.
MECHANISM_SUPPORT_THRESHOLD = 0.05

GROUNDING_SUPPORTED = "supported"
GROUNDING_CONNECTION_WITHOUT_THIS_MECHANISM = "connection_without_this_mechanism"
GROUNDING_NO_CONNECTION = "no_connection"
GROUNDING_NOT_CHECKABLE = "not_checkable"


@dataclass
class MechanismVerification:
    """Whether the graph supports a claimed factor-to-target mechanism."""

    factor: str
    target: str
    claimed_mechanism: str
    grounding: str = GROUNDING_NOT_CHECKABLE
    matched_concept: str | None = None
    matched_activation: float = 0.0
    candidate_mechanisms: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        return self.grounding == GROUNDING_SUPPORTED

    @property
    def graph_supports_any_connection(self) -> bool:
        """Whether the graph links factor and target at all, by any mechanism.

        Distinct from `is_grounded`: the graph may connect the two concepts
        strongly by a route the model never mentioned, which is a different
        finding from the graph knowing of no connection at all.
        """
        return self.grounding in (GROUNDING_SUPPORTED, GROUNDING_CONNECTION_WITHOUT_THIS_MECHANISM)

    def as_dict(self) -> dict[str, Any]:
        return {
            "factor": self.factor,
            "target": self.target,
            "claimed_mechanism": self.claimed_mechanism,
            "grounding": self.grounding,
            "is_grounded": self.is_grounded,
            "graph_supports_any_connection": self.graph_supports_any_connection,
            "matched_concept": self.matched_concept,
            "matched_activation": round(self.matched_activation, 4),
            "candidate_mechanisms": list(self.candidate_mechanisms),
            "notes": list(self.notes),
        }


def _mechanism_matches(claimed: str, candidate: ActivatedConcept) -> bool:
    """Whether a claimed mechanism names the same concept as a graph node.

    Three tests, in increasing looseness, all against a *graph node* rather
    than against the other model's answer -- which is what makes the looseness
    safe. A rule this permissive applied between two free-text answers would
    reintroduce the character-comparison failure this line of work removed;
    applied against a concept the graph independently surfaced, the worst case
    is matching a claim to a real node slightly too eagerly, and the node is
    named in the output for a reader to check.

    Exact equality, then containment in either direction, then equality as a
    set of words: "weakness of connective tissue" and "connective tissue
    weakness" are the same concept reordered, which containment misses
    because the words are permuted rather than nested. Word-set equality
    catches that without admitting genuinely different claims -- "aortic
    dilation" and "pulmonary dilation" share a word but not their word sets.

    What this deliberately does not do is resolve synonyms: "inherited
    aortopathy" will not match "connective tissue weakness" here, and the
    result is a recorded miss rather than a silent guess. Synonym expansion is
    real work with its own literature; approximating it with a similarity
    threshold is exactly the shortcut this project already measured going
    wrong.
    """
    left = normalise_concept(claimed)
    right = normalise_concept(candidate.concept)
    if not left or not right:
        return False
    if left == right or left in right or right in left:
        return True
    # Stop words carry no conceptual content and their presence should not
    # decide whether two phrasings name the same thing.
    stop = {"of", "the", "a", "an", "in", "to", "and", "with"}
    left_words = {word for word in left.split() if word not in stop}
    right_words = {word for word in right.split() if word not in stop}
    return bool(left_words) and left_words == right_words


def verify_mechanism(
    graph: ConceptGraphView,
    factor: str,
    target: str,
    claimed_mechanism: str,
    *,
    table: InformationContentTable | None = None,
    support_threshold: float = MECHANISM_SUPPORT_THRESHOLD,
    **spread_kwargs: Any,
) -> MechanismVerification:
    """Check a claimed mechanism against what the graph independently supports."""
    verification = MechanismVerification(
        factor=normalise_concept(factor),
        target=normalise_concept(target),
        claimed_mechanism=normalise_concept(claimed_mechanism),
    )

    if not verification.factor or not verification.target:
        verification.notes.append("factor or target unstated; nothing to check a mechanism between")
        return verification

    result = mediating_concepts(graph, factor, target, table=table, **spread_kwargs)
    supported = [item for item in result.multiply_sourced() if item.weighted_activation >= support_threshold]
    verification.candidate_mechanisms = tuple(item.concept for item in supported)

    if not verification.claimed_mechanism:
        verification.notes.append("no mechanism claimed; reporting what the graph offers instead")
        verification.grounding = (
            GROUNDING_CONNECTION_WITHOUT_THIS_MECHANISM if supported else GROUNDING_NO_CONNECTION
        )
        return verification

    for candidate in supported:
        if _mechanism_matches(verification.claimed_mechanism, candidate):
            verification.grounding = GROUNDING_SUPPORTED
            verification.matched_concept = candidate.concept
            verification.matched_activation = candidate.weighted_activation
            return verification

    if supported:
        verification.grounding = GROUNDING_CONNECTION_WITHOUT_THIS_MECHANISM
        verification.notes.append(
            "the graph connects these concepts, but not by the mechanism claimed -- "
            f"it offers {', '.join(verification.candidate_mechanisms[:3])}"
        )
    else:
        verification.grounding = GROUNDING_NO_CONNECTION
        verification.notes.append(
            "the graph supports no connection between these concepts above the support threshold; "
            "the claimed mechanism rests on nothing the graph knows"
        )
    return verification


# --------------------------------------------------------------------------
# The four combinations of agreement and grounding
# --------------------------------------------------------------------------

DISPOSITION_AGREED_AND_GROUNDED = "agreed_and_grounded"
DISPOSITION_AGREED_BUT_UNGROUNDED = "agreed_but_ungrounded"
DISPOSITION_DISAGREED_BUT_GROUNDED = "disagreed_but_grounded"
DISPOSITION_DISAGREED_AND_UNGROUNDED = "disagreed_and_ungrounded"


@dataclass
class MechanismCrossCheck:
    """Two models' claimed mechanisms, each checked against the graph.

    Agreement and grounding are two axes, not one. Reporting only their
    combination as a single verdict would hide the case this module exists to
    surface: two models agreeing on a mechanism neither the graph nor
    anything else supports.
    """

    primary: MechanismVerification
    secondary: MechanismVerification
    models_agree: bool = False

    @property
    def both_grounded(self) -> bool:
        return self.primary.is_grounded and self.secondary.is_grounded

    @property
    def either_grounded(self) -> bool:
        return self.primary.is_grounded or self.secondary.is_grounded

    @property
    def disposition(self) -> str:
        if self.models_agree:
            return DISPOSITION_AGREED_AND_GROUNDED if self.either_grounded else DISPOSITION_AGREED_BUT_UNGROUNDED
        return (
            DISPOSITION_DISAGREED_BUT_GROUNDED if self.both_grounded else DISPOSITION_DISAGREED_AND_UNGROUNDED
        )

    @property
    def needs_review(self) -> bool:
        """Everything except agreed-and-grounded warrants a human.

        `agreed_but_ungrounded` is the case worth naming: without the graph
        it would have looked identical to clean agreement, since both models
        said the same thing and nothing else was consulted.
        """
        return self.disposition != DISPOSITION_AGREED_AND_GROUNDED

    def as_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition,
            "models_agree": self.models_agree,
            "both_grounded": self.both_grounded,
            "needs_review": self.needs_review,
            "primary": self.primary.as_dict(),
            "secondary": self.secondary.as_dict(),
        }


def cross_check_mechanisms(
    graph: ConceptGraphView,
    primary_answer: str | None,
    secondary_answer: str | None,
    *,
    table: InformationContentTable | None = None,
    **kwargs: Any,
) -> MechanismCrossCheck:
    """Parse two relevance-frame answers and check each mechanism against the graph.

    Each model's own stated factor and target are used for its own check
    rather than one model's being imposed on both: if they disagree about
    what the question is even relating, that is itself a finding, and
    normalising it away would hide it.
    """
    left = parse_frame_answer(FRAME_RELEVANCE, primary_answer)
    right = parse_frame_answer(FRAME_RELEVANCE, secondary_answer)

    primary = verify_mechanism(
        graph, left.value("factor"), left.value("target"), left.value("mechanism"), table=table, **kwargs
    )
    secondary = verify_mechanism(
        graph, right.value("factor"), right.value("target"), right.value("mechanism"), table=table, **kwargs
    )

    claimed_left, claimed_right = left.value("mechanism"), right.value("mechanism")
    agree = bool(claimed_left) and bool(claimed_right) and (
        claimed_left == claimed_right or claimed_left in claimed_right or claimed_right in claimed_left
    )
    # Two models landing on the same graph-supported concept by different
    # wording agree in substance even when their strings differ -- which is
    # the false-alarm case string comparison could never resolve, and the
    # graph resolves without needing a synonym table.
    if not agree and primary.is_grounded and secondary.is_grounded:
        agree = primary.matched_concept == secondary.matched_concept

    return MechanismCrossCheck(primary=primary, secondary=secondary, models_agree=agree)


def summarise(checks: Sequence[MechanismCrossCheck]) -> dict[str, Any]:
    """Counts by disposition across several cases."""
    counts: dict[str, int] = {}
    for check in checks:
        counts[check.disposition] = counts.get(check.disposition, 0) + 1
    return {
        "cases": len(checks),
        "by_disposition": dict(sorted(counts.items())),
        "needing_review": sum(1 for check in checks if check.needs_review),
    }
