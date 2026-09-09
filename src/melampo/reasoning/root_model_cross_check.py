"""Run two root models over the same case and treat their disagreement as a signal.

This is the same principle `retrieval_reconciliation` already applies to the
one-shot and recursive retrieval paths, moved one level up: instead of two
retrieval *strategies* disagreeing, two root *models* navigate the identical
environment independently, and divergence between them is recorded as an
empirical uncertainty estimate rather than resolved silently in favour of
either.

Why two models rather than one, given a bench already picked a winner. The
format-adherence bench measures whether a candidate follows the action grammar
and finishes within budget -- it does not, anywhere, grade whether `final()`
is clinically right. Two models that both navigate competently can still reach
different conclusions from the same documents, and that difference is exactly
what a single model cannot report about itself: a confident wrong answer and a
confident right one look identical from the inside. Independent agreement is
weak evidence of correctness; independent disagreement is strong evidence that
something is worth a human's attention.

The comparison is deliberately deterministic and non-adjudicating. No third
model decides which of the two is right, and neither is designated
authoritative -- `primary` and `secondary` name which ran first for
reproducibility, not which one wins. A disagreement raises
`answers_agree = False` and leaves both answers visible; it does not pick one.
Silently preferring either would discard precisely the signal this module
exists to produce.

Model choice, and why it is configurable rather than hardcoded. Two live runs
of the focused comparison bench put nemotron-3-super and gemma-3-27b as the
two most efficient candidates at 100% adherence and 100% completion, with
near-identical iteration counts -- the pairing with the least redundant cost.
mistral-large-openrouter matched them on adherence and completion but at
roughly twice the iterations; it is kept available as an alternative rather
than dropped, because it is the only one of the three from an EU-based lab,
which the bench cannot measure and which may matter for a system with MDR
ambitions. `DEFAULT_PAIR` and `ALTERNATIVE_PAIR` name these; either can be
passed to `cross_check` as two callables.
"""

import difflib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..memory.context_environment import EnvironmentDocument
from .rlm_engine import STOP_FINAL, Budget, RlmEngine, Trajectory

# Named pairings, from two live runs of the focused comparison bench. Names
# only -- the callables themselves are supplied by the caller, since this
# module has no opinion about how a model is reached (OpenRouter, a local
# endpoint, a scripted stand-in in tests).
DEFAULT_PAIR = ("nemotron-3-super", "gemma-3-27b")
ALTERNATIVE_PAIR = ("nemotron-3-super", "mistral-large-openrouter")

# Two answers are treated as agreeing when their similarity reaches this.
# Not an exact string match: two correct answers to "what dose was started"
# ("40 mg daily" vs "prednisone 40 mg daily") differ in wording while agreeing
# entirely, and flagging that as disagreement would bury real divergence in
# noise. Not semantic comparison either -- that would need a third model to
# adjudicate, which is exactly what this module refuses to do.
ANSWER_AGREEMENT_THRESHOLD = 0.75


def _similarity_ratio(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, left, right).ratio()


@dataclass
class CrossCheckResult:
    """What two root models each did with the same case, and whether they agree."""

    case_id: str
    primary_model: str
    secondary_model: str
    primary_answer: str | None = None
    secondary_answer: str | None = None
    primary_completed: bool = False
    secondary_completed: bool = False
    answer_similarity: float = 0.0
    shared_evidence_ids: list[str] = field(default_factory=list)
    primary_only_evidence_ids: list[str] = field(default_factory=list)
    secondary_only_evidence_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def both_completed(self) -> bool:
        return self.primary_completed and self.secondary_completed

    @property
    def answers_agree(self) -> bool:
        """True only when both models finished AND their answers are similar.

        One model completing and the other not is not agreement, and must not
        read as such: there is only one answer, so there is nothing for a
        second opinion to confirm. That case is a distinct outcome
        (`disposition` reports it separately), not a quiet pass.
        """
        return self.both_completed and self.answer_similarity >= ANSWER_AGREEMENT_THRESHOLD

    @property
    def evidence_agreement_ratio(self) -> float:
        """Shared evidence as a fraction of everything either model looked at.

        Two models reaching the same answer from entirely different fragments
        is a different situation from reaching it from the same ones -- the
        first is genuine independent corroboration, the second may just be
        two models finding the one obvious passage. This does not change the
        verdict; it is recorded so the distinction is visible.
        """
        union = len(self.shared_evidence_ids) + len(self.primary_only_evidence_ids) + len(
            self.secondary_only_evidence_ids
        )
        return len(self.shared_evidence_ids) / union if union else 0.0

    @property
    def disposition(self) -> str:
        """One of four outcomes, kept distinct because they need different responses."""
        if self.answers_agree:
            return "agreed"
        if self.both_completed:
            return "disagreed"
        if self.primary_completed or self.secondary_completed:
            return "single_answer_only"
        return "neither_completed"

    @property
    def needs_review(self) -> bool:
        """Anything other than clean agreement warrants a human looking.

        Deliberately inclusive: `single_answer_only` and `neither_completed`
        are flagged alongside outright disagreement, because "one model could
        not finish" is itself a reason not to trust the other's answer
        unexamined -- it means the case defeated a competent navigator.
        """
        return self.disposition != "agreed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "primary_model": self.primary_model,
            "secondary_model": self.secondary_model,
            "primary_answer": self.primary_answer,
            "secondary_answer": self.secondary_answer,
            "primary_completed": self.primary_completed,
            "secondary_completed": self.secondary_completed,
            "both_completed": self.both_completed,
            "answers_agree": self.answers_agree,
            "answer_similarity": round(self.answer_similarity, 3),
            "evidence_agreement_ratio": round(self.evidence_agreement_ratio, 3),
            "shared_evidence_count": len(self.shared_evidence_ids),
            "primary_only_evidence_count": len(self.primary_only_evidence_ids),
            "secondary_only_evidence_count": len(self.secondary_only_evidence_ids),
            "disposition": self.disposition,
            "needs_review": self.needs_review,
            "notes": list(self.notes),
        }


def normalise_answer(answer: str | None) -> str:
    """Lowercase, collapse whitespace, drop trailing punctuation.

    Minimal on purpose: enough that "40 mg daily" and "40 mg daily." are not
    treated as different answers, not so much that genuinely different answers
    are normalised into agreement.
    """
    if not answer:
        return ""
    return " ".join(answer.lower().split()).rstrip(".,;:!?")


def answer_similarity(first: str | None, second: str | None) -> float:
    """Normalised similarity between two answers, 0.0 to 1.0.

    Character-sequence ratio alone is unreliable here, and measurably so: on
    the bench's own vocabulary, "pulmonary embolism" vs "pulmonary oedema" --
    clinically opposite findings, and a distinction one advanced case exists
    specifically to test -- scores 0.71, while "40 mg daily" vs "prednisone
    40 mg daily" -- the same answer, one more verbose -- scores 0.67. Ranking
    a contradiction above a paraphrase is exactly backwards for this purpose.

    Containment is the correction: when one normalised answer contains the
    other whole, the shorter is a subset of the longer rather than a rival
    claim, which is the common shape of "same answer, different verbosity"
    and never the shape of two different findings. Those score 1.0; everything
    else falls back to the sequence ratio.

    Both empty scores 0.0, not 1.0: two models that said nothing have not
    agreed on anything, and treating that as perfect agreement would turn the
    worst outcome into the best-looking number.

    Known limitation, stated rather than hidden: two answers that mean the
    same thing in entirely different words ("not documented" vs "the report
    does not mention prednisone", 0.38) are reported as disagreement. This is
    a false alarm, and it is the direction the error is deliberately allowed
    to fall -- over-reporting sends a correct case to a human unnecessarily,
    under-reporting lets a genuine divergence through unexamined. Closing this
    gap properly needs semantic comparison, which needs a third model, which
    reintroduces exactly the adjudicating judgement this module is built to
    avoid. `answer_similarity` is recorded alongside the verdict so a reviewer
    can see when a flagged disagreement scored near the threshold rather than
    far from it.
    """
    left, right = normalise_answer(first), normalise_answer(second)
    if not left or not right:
        return 0.0
    if left in right or right in left:
        return 1.0
    return _similarity_ratio(left, right)


def cross_check(
    case_id: str,
    documents: Sequence[EnvironmentDocument],
    question: str,
    primary_model: Callable[[str], str],
    secondary_model: Callable[[str], str],
    *,
    primary_name: str = DEFAULT_PAIR[0],
    secondary_name: str = DEFAULT_PAIR[1],
    budget_factory: Callable[[], Budget] = Budget,
    search_fn: Any = None,
    graph_expand_fn: Any = None,
) -> CrossCheckResult:
    """Run both models over the same case and compare what they produced.

    Each gets its own ``RlmEngine`` and its own fresh ``Budget`` -- sharing
    either would let the first model's navigation influence the second's,
    which would defeat the independence the comparison depends on.
    """
    result = CrossCheckResult(case_id=case_id, primary_model=primary_name, secondary_model=secondary_name)

    primary_trajectory = _run_one(primary_model, case_id, documents, question, budget_factory, search_fn, graph_expand_fn)
    secondary_trajectory = _run_one(
        secondary_model, case_id, documents, question, budget_factory, search_fn, graph_expand_fn
    )

    result.primary_completed = primary_trajectory.stop_reason == STOP_FINAL
    result.secondary_completed = secondary_trajectory.stop_reason == STOP_FINAL
    result.primary_answer = primary_trajectory.final_answer
    result.secondary_answer = secondary_trajectory.final_answer
    result.answer_similarity = answer_similarity(result.primary_answer, result.secondary_answer)

    primary_ids = {item.get("record_id", "") for item in primary_trajectory.evidence() if item.get("record_id")}
    secondary_ids = {item.get("record_id", "") for item in secondary_trajectory.evidence() if item.get("record_id")}
    result.shared_evidence_ids = sorted(primary_ids & secondary_ids)
    result.primary_only_evidence_ids = sorted(primary_ids - secondary_ids)
    result.secondary_only_evidence_ids = sorted(secondary_ids - primary_ids)

    if not result.primary_completed:
        result.notes.append(f"{primary_name} did not complete: {primary_trajectory.stop_reason}")
    if not result.secondary_completed:
        result.notes.append(f"{secondary_name} did not complete: {secondary_trajectory.stop_reason}")
    if result.disposition == "disagreed":
        result.notes.append(
            "both models completed but reached different answers; neither is preferred here -- "
            "both are recorded for review"
        )
    if result.answers_agree and result.evidence_agreement_ratio < 0.5:
        result.notes.append(
            "answers agree but the models read largely different fragments -- stronger independent "
            "corroboration than agreement from identical evidence would be"
        )

    return result


def _run_one(
    model: Callable[[str], str],
    case_id: str,
    documents: Sequence[EnvironmentDocument],
    question: str,
    budget_factory: Callable[[], Budget],
    search_fn: Any,
    graph_expand_fn: Any,
) -> Trajectory:
    engine = RlmEngine(root_model=model, depth=0)
    return engine.run(
        case_id, documents, question,
        budget=budget_factory(), search_fn=search_fn, graph_expand_fn=graph_expand_fn,
    )


@dataclass
class CrossCheckReport:
    """Aggregate of cross-checks across several cases."""

    results: list[CrossCheckResult] = field(default_factory=list)

    @property
    def agreement_rate(self) -> float:
        return sum(1 for item in self.results if item.answers_agree) / len(self.results) if self.results else 0.0

    @property
    def review_rate(self) -> float:
        return sum(1 for item in self.results if item.needs_review) / len(self.results) if self.results else 0.0

    def by_disposition(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.results:
            counts[item.disposition] = counts.get(item.disposition, 0) + 1
        return dict(sorted(counts.items()))

    def flagged(self) -> list[CrossCheckResult]:
        """Every case a human should look at, in the order they were run."""
        return [item for item in self.results if item.needs_review]

    def as_dict(self) -> dict[str, Any]:
        return {
            "cases": len(self.results),
            "agreement_rate": round(self.agreement_rate, 3),
            "review_rate": round(self.review_rate, 3),
            "by_disposition": self.by_disposition(),
            "results": [item.as_dict() for item in self.results],
        }


def cross_check_cases(
    cases: Sequence[tuple[str, Sequence[EnvironmentDocument], str]],
    primary_model: Callable[[str], str],
    secondary_model: Callable[[str], str],
    **kwargs: Any,
) -> CrossCheckReport:
    """Cross-check a sequence of (case_id, documents, question) triples."""
    report = CrossCheckReport()
    for case_id, documents, question in cases:
        report.results.append(cross_check(case_id, documents, question, primary_model, secondary_model, **kwargs))
    return report
