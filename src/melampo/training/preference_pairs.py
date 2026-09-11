"""Extract preference pairs for DPO from what the system already records.

A claim was made in discussion and is verified here rather than assumed: that
this project's existing data already has the shape DPO training needs, so no
separate preference-collection pipeline would have to be built.

The shape DPO requires is (prompt, preferred response, rejected response).
The claim is that it falls out of two things already in place:
`mechanism_enumeration` produces several candidate hypotheses per case, and
`ConfirmationRegistry` records which diagnosis was independently confirmed.
The confirmed one is the preferred response; the other candidates raised for
the same case are the rejected ones.

**This module builds the pairs. It does not train anything.** No PEFT
configuration exists in this project yet -- no LoRA, no DoRA, no fine-tuning
code of any kind -- and writing one before a base model is chosen, and before
the vetting bench says which candidate to choose, would be building a seventh
disconnected module. What this does is answer, now and without a GPU, whether
the data would be there when that decision is made. If the pairs come out
well-formed, the DPO route is open whenever it is wanted. If they do not,
better to know while it is still cheap to find out.

**Two properties the extraction must have, and both are enforced here rather
than left to the caller.** Only independently confirmed cases produce pairs,
via the registry's own `learning_set()` -- a case "confirmed" because the
system suggested something and nobody objected would teach the model that its
own guesses were right, which is the automation-bias failure
`confirmation_registry` exists to prevent. And a case with no rejected
alternative produces no pair at all: DPO learns from the contrast between two
responses, and a pair whose rejected half is empty carries no signal while
still looking like training data.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..governance.confirmation_registry import ConfirmationRegistry


@dataclass(frozen=True)
class PreferencePair:
    """One DPO training pair, with the provenance that produced it."""

    case_id: str
    prompt: str
    preferred: str
    rejected: str
    confirmation_source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "prompt": self.prompt,
            "preferred": self.preferred,
            "rejected": self.rejected,
            "confirmation_source": self.confirmation_source,
        }


@dataclass
class ExtractionReport:
    """What the extraction found, including what it deliberately skipped."""

    pairs: list[PreferencePair] = field(default_factory=list)
    cases_with_no_confirmation: int = 0
    cases_with_no_alternative: int = 0
    cases_where_confirmed_was_not_raised: int = 0

    @property
    def usable_cases(self) -> int:
        return len({pair.case_id for pair in self.pairs})

    def as_dict(self) -> dict[str, Any]:
        return {
            "pairs": len(self.pairs),
            "usable_cases": self.usable_cases,
            "cases_with_no_confirmation": self.cases_with_no_confirmation,
            "cases_with_no_alternative": self.cases_with_no_alternative,
            "cases_where_confirmed_was_not_raised": self.cases_where_confirmed_was_not_raised,
        }


def _normalise(value: str) -> str:
    return " ".join(str(value).lower().split())


def extract_preference_pairs(
    raised_by_case: dict[str, Sequence[str]],
    registry: ConfirmationRegistry,
    *,
    prompt_for_case: dict[str, str] | None = None,
) -> ExtractionReport:
    """Pair each case's confirmed hypothesis against the alternatives raised alongside it.

    ``raised_by_case`` maps a case id to the condition names the system
    surfaced for it -- the shape `mechanism_enumeration`'s output reduces to.
    ``prompt_for_case`` supplies the question each case was asked; a case
    without one gets a neutral placeholder rather than being dropped, since
    the pair's value is in the contrast between the two responses and the
    prompt can be filled in by whoever assembles the training file.

    One pair per rejected alternative, not one per case: a case where the
    confirmed diagnosis was raised alongside four wrong ones carries four
    contrasts, and collapsing them to one would discard three quarters of
    the signal that case produced.
    """
    report = ExtractionReport()
    confirmed_by_case = {item.case_id: item for item in registry.learning_set()}

    for case_id, raised in raised_by_case.items():
        confirmation = confirmed_by_case.get(case_id)
        if confirmation is None:
            report.cases_with_no_confirmation += 1
            continue

        confirmed_name = _normalise(confirmation.diagnosis)
        matching = [item for item in raised if _normalise(item) == confirmed_name]
        alternatives = [item for item in raised if _normalise(item) != confirmed_name]

        if not matching:
            # The system never raised what turned out to be true. This is a
            # real and interesting outcome -- a miss, not a preference -- and
            # it is counted rather than silently turned into a pair whose
            # "preferred" answer the system never actually produced.
            report.cases_where_confirmed_was_not_raised += 1
            continue
        if not alternatives:
            report.cases_with_no_alternative += 1
            continue

        prompt = (prompt_for_case or {}).get(case_id, f"What is the most likely diagnosis for case {case_id}?")
        for alternative in alternatives:
            report.pairs.append(
                PreferencePair(
                    case_id=case_id,
                    prompt=prompt,
                    preferred=matching[0],
                    rejected=alternative,
                    confirmation_source=confirmation.source,
                )
            )

    return report


def as_training_records(pairs: Sequence[PreferencePair]) -> list[dict[str, str]]:
    """The pairs in the field names TRL, Axolotl and LlamaFactory's DPO trainers expect.

    A thin rename, deliberately: the extraction above keeps names that read
    clearly in this codebase, and this converts them at the boundary rather
    than letting a training library's vocabulary leak into the domain code.
    """
    return [
        {"prompt": pair.prompt, "chosen": pair.preferred, "rejected": pair.rejected}
        for pair in pairs
    ]
