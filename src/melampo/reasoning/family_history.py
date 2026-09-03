"""Route family history to screening hypotheses and priors, never to findings.

A condition reported in a relative is not a finding of the patient. Recording it
as one with a reduced score does not soften the error, it only disguises it:
traversal is binary. A finding is an entry point, so the graph would be walked
*from* the relative's condition, producing paths toward its complications in
someone who does not have it. The paths would be correct and the premise false,
which is the hardest kind of error to notice.

Family history has two legitimate destinations instead.

**Screening hypothesis.** The condition may be present in the patient and simply
undiagnosed. This answers a different question from the differential — not
"what explains these findings" but "what else might this patient have,
independent of why they came in" — and therefore does not pass the indeterminacy
gate. That gate exists because an alternative explanation is informative only
when the differential is flat; a screening consideration is valid precisely when
the differential has already settled, and gating it there would lose it in the
clear cases, which are the ones where a patient leaves without anyone looking.

**Prior modifier.** Family history raises the prior on heritable conditions. The
size of the shift is not a constant: it follows the mode of inheritance and the
degree of relatedness, both of which the ontology encodes.

Either way the condition is a destination, not an origin: a candidate to confirm
or exclude, which produces no descendants until it is confirmed.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

DEGREE_FIRST = 1
DEGREE_SECOND = 2
DEGREE_THIRD = 3

INHERITANCE_AUTOSOMAL_DOMINANT = "HP:0000006"
INHERITANCE_AUTOSOMAL_RECESSIVE = "HP:0000007"
INHERITANCE_X_LINKED = "HP:0001417"
INHERITANCE_X_LINKED_DOMINANT = "HP:0001423"
INHERITANCE_X_LINKED_RECESSIVE = "HP:0001419"
INHERITANCE_MITOCHONDRIAL = "HP:0001427"
INHERITANCE_POLYGENIC = "HP:0010982"
INHERITANCE_SPORADIC = "HP:0003745"

# Policy table, not derived truth. These multipliers express how much a family
# history shifts the prior on a heritable condition, by mode of inheritance.
# Intervals rather than points, for the same reason edges carry intervals: the
# strength of the shift is itself uncertain. Requires clinical review before any
# use beyond research, and is versioned in the decision record.
INHERITANCE_PRIOR_SHIFT: dict[str, tuple[float, float]] = {
    INHERITANCE_AUTOSOMAL_DOMINANT: (2.5, 6.0),
    INHERITANCE_X_LINKED_DOMINANT: (2.0, 5.0),
    INHERITANCE_MITOCHONDRIAL: (2.0, 5.0),
    INHERITANCE_AUTOSOMAL_RECESSIVE: (1.3, 2.5),
    INHERITANCE_X_LINKED_RECESSIVE: (1.3, 2.5),
    INHERITANCE_X_LINKED: (1.3, 2.5),
    INHERITANCE_POLYGENIC: (1.1, 1.6),
    INHERITANCE_SPORADIC: (1.0, 1.0),
}

# Attenuation by how far the relative sits from the patient.
DEGREE_ATTENUATION: dict[int, float] = {DEGREE_FIRST: 1.0, DEGREE_SECOND: 0.5, DEGREE_THIRD: 0.25}

ACTIONABLE_INHERITANCE = frozenset(
    {
        INHERITANCE_AUTOSOMAL_DOMINANT,
        INHERITANCE_X_LINKED_DOMINANT,
        INHERITANCE_MITOCHONDRIAL,
        INHERITANCE_AUTOSOMAL_RECESSIVE,
        INHERITANCE_X_LINKED_RECESSIVE,
        INHERITANCE_X_LINKED,
        INHERITANCE_POLYGENIC,
    }
)

ROLE_SCREENING_HYPOTHESIS = "screening_hypothesis"

BLOCK_NOT_HERITABLE = "inheritance mode does not support a screening hypothesis"
BLOCK_ONSET_NOT_REACHED = "documented onset is not yet reached for this patient"
BLOCK_ALREADY_ASSESSED = "already assessed in the record"


@dataclass(frozen=True)
class FamilyHistoryEntry:
    """A condition reported in a relative, with what is known about it."""

    condition: str
    degree: int = DEGREE_FIRST
    inheritance: str | None = None
    onset_age_years: float | None = None
    source: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "degree": self.degree,
            "inheritance": self.inheritance,
            "onset_age_years": self.onset_age_years,
            "source": self.source,
        }


@dataclass(frozen=True)
class ScreeningHypothesis:
    """A condition to consider in the patient, separate from the differential."""

    condition: str
    rationale: str
    prior_shift: tuple[float, float]
    entry: FamilyHistoryEntry

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "role": ROLE_SCREENING_HYPOTHESIS,
            "rationale": self.rationale,
            "prior_shift_lower": round(self.prior_shift[0], 3),
            "prior_shift_upper": round(self.prior_shift[1], 3),
            "family_history": self.entry.as_dict(),
            "usable_as_evidence": False,
            "belongs_in_differential": False,
            "requires_clinician_judgement": True,
        }


@dataclass(frozen=True)
class BlockedEntry:
    entry: FamilyHistoryEntry
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"condition": self.entry.condition, "reason": self.reason}


@dataclass
class FamilyHistoryResult:
    screening: list[ScreeningHypothesis] = field(default_factory=list)
    blocked: list[BlockedEntry] = field(default_factory=list)

    def prior_shifts(self) -> dict[str, tuple[float, float]]:
        """Multipliers to apply to candidate condition priors, by condition."""
        return {item.condition: item.prior_shift for item in self.screening}

    def as_dict(self) -> dict[str, Any]:
        return {
            "screening_considerations": [item.as_dict() for item in self.screening],
            "blocked": [item.as_dict() for item in self.blocked],
            "note": "family history never enters patient findings; these are destinations, not origins",
        }


@dataclass
class FamilyHistoryChannel:
    """Gate family history into screening hypotheses and prior shifts.

    The gate is not the indeterminacy gate. Three conditions instead, each
    readable from the ontology or the record: the mode of inheritance must
    support transmission at all, the documented onset must be reachable at the
    patient's age, and the condition must not already have been assessed. The
    third matters most in practice — without it the same consideration is
    re-proposed at every visit, spending the review attention the channel exists
    to protect.
    """

    prior_shift_by_inheritance: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(INHERITANCE_PRIOR_SHIFT)
    )
    degree_attenuation: dict[int, float] = field(default_factory=lambda: dict(DEGREE_ATTENUATION))
    actionable_inheritance: frozenset[str] = ACTIONABLE_INHERITANCE
    unknown_inheritance_is_actionable: bool = True

    def evaluate(
        self,
        entries: Iterable[FamilyHistoryEntry],
        *,
        patient_age_years: float | None = None,
        already_assessed: Sequence[str] = (),
    ) -> FamilyHistoryResult:
        assessed = {_normalise(item) for item in already_assessed}
        result = FamilyHistoryResult()

        for entry in entries:
            if _normalise(entry.condition) in assessed:
                result.blocked.append(BlockedEntry(entry, BLOCK_ALREADY_ASSESSED))
                continue
            if not self._inheritance_is_actionable(entry.inheritance):
                result.blocked.append(BlockedEntry(entry, BLOCK_NOT_HERITABLE))
                continue
            if not self._onset_is_reachable(entry, patient_age_years):
                result.blocked.append(BlockedEntry(entry, BLOCK_ONSET_NOT_REACHED))
                continue

            result.screening.append(
                ScreeningHypothesis(
                    condition=entry.condition,
                    rationale=self._rationale(entry),
                    prior_shift=self.prior_shift(entry),
                    entry=entry,
                )
            )
        return result

    def prior_shift(self, entry: FamilyHistoryEntry) -> tuple[float, float]:
        """Prior multiplier interval, attenuated by degree of relatedness.

        Attenuation moves the interval toward 1.0 — no effect — rather than
        scaling it, because a distant relative weakens the inference toward
        neutrality and never reverses it.
        """
        base = self.prior_shift_by_inheritance.get(entry.inheritance or "", (1.0, 1.5))
        weight = self.degree_attenuation.get(entry.degree, 0.1)
        return (1.0 + (base[0] - 1.0) * weight, 1.0 + (base[1] - 1.0) * weight)

    def _inheritance_is_actionable(self, inheritance: str | None) -> bool:
        if inheritance is None:
            return self.unknown_inheritance_is_actionable
        return inheritance in self.actionable_inheritance

    def _onset_is_reachable(self, entry: FamilyHistoryEntry, patient_age_years: float | None) -> bool:
        if entry.onset_age_years is None or patient_age_years is None:
            return True
        return patient_age_years >= entry.onset_age_years

    def _rationale(self, entry: FamilyHistoryEntry) -> str:
        lower, upper = self.prior_shift(entry)
        relation = {DEGREE_FIRST: "first-degree", DEGREE_SECOND: "second-degree"}.get(
            entry.degree, "distant"
        )
        return (
            f"Reported in a {relation} relative; prior shift x{lower:.1f}-{upper:.1f}. "
            "Consider whether present and undiagnosed in this patient. Not an explanation of the "
            "presenting findings."
        )


def assert_not_a_finding(items: Sequence[dict[str, Any]]) -> None:
    """Guard at the findings boundary.

    Raises when a screening hypothesis reaches the path reserved for patient
    findings. The distinction is direction, not magnitude: as a finding the
    condition would be a starting point and the traversal would generate its
    complications in a patient who does not have it.
    """
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        if item.get("role") == ROLE_SCREENING_HYPOTHESIS or item.get("belongs_in_differential") is False:
            raise ValueError(
                f"item {index} is a family-history screening hypothesis and is not a patient finding"
            )


def _normalise(value: str) -> str:
    return " ".join(str(value).lower().split())
