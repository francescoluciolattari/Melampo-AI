"""Illness scripts: the structure a diagnostic model must emit, and its verifier.

Expertise research describes clinical knowledge as organised into *illness
scripts* rather than facts, with three components: enabling conditions, the
underlying fault, and the clinical consequences it produces. Experts recognise
the script from its consequences and descend to the fault only when the script
fails to fit — the phenomenon called knowledge encapsulation.

That structure is the reason this module exists, and it is not decoration. A
model emitting free text gives the graph nothing precise to check: "probably
heart failure, consider pneumonia" contains no element that can be matched
against a node or an edge. A model emitting a script gives every element an
address — these findings, this mechanism, these candidates with these
discriminating features — and each one can be classified as grounded, mediated
by knowledge, or unsupported.

**The verifier does not decide the diagnosis.** It says what each part of a
proposed script rests on. The model decides the content and its ordering, the
graph decides what that content is founded on, and the calibrator decides
whether the result may leave. Separating those three is what allows learned
pattern recognition and auditability in the same system: a weight cannot explain
itself, but a path can.

The three-way verdict matters more than a score. A claim absent from the graph
is not automatically wrong — most clinical inference connects findings to
conditions through knowledge outside the case — so the verifier distinguishes
"the graph has no path" from "the case does not state it", and only the first is
a defect.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..memory.concept_paths import ConceptGraphView, ConceptPath, find_paths

RELATION_ENABLES = "enables"
RELATION_CAUSES = "causes"
RELATION_MANIFESTS_AS = "manifests_as"

SCRIPT_RELATIONS = frozenset({RELATION_ENABLES, RELATION_CAUSES, RELATION_MANIFESTS_AS})

VERDICT_GROUNDED_IN_CASE = "grounded_in_case"
VERDICT_KNOWLEDGE_MEDIATED = "knowledge_mediated"
VERDICT_UNSUPPORTED = "unsupported"

ORIGIN_MODEL = "model"
ORIGIN_HYPOTHESIS_CHANNEL = "hypothesis_channel"
ORIGIN_SCREENING = "screening_channel"

# Origins whose entries are candidates rather than proposed diagnoses. Kept
# separate in the differential so that a synthetic alternative is never read as
# the model's own reading of the case.
CANDIDATE_ORIGINS = frozenset({ORIGIN_HYPOTHESIS_CHANNEL, ORIGIN_SCREENING})


@dataclass(frozen=True)
class ScriptElement:
    """One addressable element of a script, with its ontology identifier."""

    label: str
    term_id: str | None = None
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "term_id": self.term_id, "note": self.note}


@dataclass(frozen=True)
class DifferentialEntry:
    """A candidate condition with the features that would discriminate it."""

    condition: str
    term_id: str | None = None
    rank: int = 0
    discriminating_features: tuple[str, ...] = ()
    origin: str = ORIGIN_MODEL

    @property
    def is_candidate_only(self) -> bool:
        """Whether this entry is a candidate rather than a proposed diagnosis."""
        return self.origin in CANDIDATE_ORIGINS

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "term_id": self.term_id,
            "rank": self.rank,
            "discriminating_features": list(self.discriminating_features),
            "origin": self.origin,
            "is_candidate_only": self.is_candidate_only,
        }


@dataclass
class IllnessScript:
    """A structured diagnostic proposal, in the form expertise research describes.

    ``enabling_conditions`` are the predisposing factors that move the prior:
    epidemiology, exposure, family history. ``fault`` is the proposed
    pathophysiological mechanism. ``consequences`` are the observed findings the
    fault would produce. ``differential`` ranks the candidate conditions.
    """

    enabling_conditions: list[ScriptElement] = field(default_factory=list)
    fault: ScriptElement | None = None
    consequences: list[ScriptElement] = field(default_factory=list)
    differential: list[DifferentialEntry] = field(default_factory=list)

    @property
    def leading(self) -> DifferentialEntry | None:
        """The highest-ranked entry the model itself proposed."""
        proposed = [item for item in self.differential if not item.is_candidate_only]
        return min(proposed, key=lambda item: item.rank) if proposed else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabling_conditions": [item.as_dict() for item in self.enabling_conditions],
            "fault": self.fault.as_dict() if self.fault else None,
            "consequences": [item.as_dict() for item in self.consequences],
            "differential": [item.as_dict() for item in self.differential],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "IllnessScript":
        """Build a script from a model's structured output, tolerating omissions."""
        return cls(
            enabling_conditions=[_element(item) for item in _listing(payload, "enabling_conditions")],
            fault=_element(payload["fault"]) if isinstance(payload.get("fault"), dict) else None,
            consequences=[_element(item) for item in _listing(payload, "consequences")],
            differential=[
                DifferentialEntry(
                    condition=str(item.get("condition", "")).strip(),
                    term_id=item.get("term_id"),
                    rank=int(item.get("rank", index + 1) or index + 1),
                    discriminating_features=tuple(item.get("discriminating_features", ()) or ()),
                    origin=str(item.get("origin", ORIGIN_MODEL)),
                )
                for index, item in enumerate(_listing(payload, "differential"))
                if str(item.get("condition", "")).strip()
            ],
        )


@dataclass(frozen=True)
class ElementVerdict:
    """What one element of a script rests on."""

    element: str
    verdict: str
    path: ConceptPath | None = None
    strength_lower: float = 0.0
    strength_upper: float = 0.0

    @property
    def is_admissible(self) -> bool:
        """Whether the element may enter reasoning at all.

        A knowledge-mediated element is admissible but not as an observation: it
        carries a graph path rather than a case citation.
        """
        return self.verdict in {VERDICT_GROUNDED_IN_CASE, VERDICT_KNOWLEDGE_MEDIATED}

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "element": self.element,
            "verdict": self.verdict,
            "admissible": self.is_admissible,
            "strength_lower": round(self.strength_lower, 3),
            "strength_upper": round(self.strength_upper, 3),
        }
        if self.path is not None:
            payload["path"] = self.path.as_provenance()
        return payload


@dataclass
class ScriptVerification:
    """Per-element verdicts over a proposed script."""

    consequences: list[ElementVerdict] = field(default_factory=list)
    differential: list[ElementVerdict] = field(default_factory=list)
    fault: ElementVerdict | None = None

    @property
    def unsupported(self) -> list[ElementVerdict]:
        everything = [*self.consequences, *self.differential]
        if self.fault is not None:
            everything.append(self.fault)
        return [item for item in everything if item.verdict == VERDICT_UNSUPPORTED]

    @property
    def grounding_ratio(self) -> float:
        """Fraction of verified elements that the graph or the case supports."""
        everything = [*self.consequences, *self.differential]
        if self.fault is not None:
            everything.append(self.fault)
        if not everything:
            return 0.0
        return sum(1 for item in everything if item.is_admissible) / len(everything)

    def as_dict(self) -> dict[str, Any]:
        return {
            "fault": self.fault.as_dict() if self.fault else None,
            "consequences": [item.as_dict() for item in self.consequences],
            "differential": [item.as_dict() for item in self.differential],
            "unsupported_count": len(self.unsupported),
            "grounding_ratio": round(self.grounding_ratio, 3),
        }


@dataclass
class ScriptVerifier:
    """Classify each element of a proposed script against case and graph.

    Case findings are the observations admitted by the findings boundary — never
    every mention, because a negated or hypothetical mention is not an
    observation and must not ground anything.
    """

    graph: ConceptGraphView
    max_hops: int = 3
    min_edge_weight: float = 0.0

    def verify(self, script: IllnessScript, case_findings: Sequence[str]) -> ScriptVerification:
        observed = {_normalise(item) for item in case_findings if str(item).strip()}
        verification = ScriptVerification()

        for element in script.consequences:
            verification.consequences.append(self._verify_element(element.label, observed, observed))

        if script.fault is not None:
            verification.fault = self._verify_element(script.fault.label, observed, observed)

        for entry in script.differential:
            verification.differential.append(self._verify_element(entry.condition, observed, observed))

        return verification

    def _verify_element(
        self, label: str, observed: set[str], anchors: set[str]
    ) -> ElementVerdict:
        if _normalise(label) in observed:
            return ElementVerdict(element=label, verdict=VERDICT_GROUNDED_IN_CASE, strength_lower=1.0, strength_upper=1.0)

        best: ConceptPath | None = None
        for anchor in sorted(anchors):
            for path in find_paths(
                self.graph, anchor, label, max_hops=self.max_hops, min_edge_weight=self.min_edge_weight
            ):
                if best is None or (path.hops, -path.strength_upper) < (best.hops, -best.strength_upper):
                    best = path
        if best is None:
            return ElementVerdict(element=label, verdict=VERDICT_UNSUPPORTED)
        return ElementVerdict(
            element=label,
            verdict=VERDICT_KNOWLEDGE_MEDIATED,
            path=best,
            strength_lower=best.strength_lower,
            strength_upper=best.strength_upper,
        )


def merge_hypotheses(
    script: IllnessScript, hypotheses: Sequence[dict[str, Any]], *, origin: str = ORIGIN_HYPOTHESIS_CHANNEL
) -> IllnessScript:
    """Add channel hypotheses to a script's differential, marked as candidates.

    They are appended after the model's own entries and carry their origin, so a
    synthetic alternative is never read as the model's reading of the case. This
    is the integration point for the dream branch: it contributes *entries*, and
    it does so where a differential already exists rather than replacing one.

    Already-present conditions are skipped: a candidate the model has raised is
    no longer an alternative, and re-listing it would spend review attention on
    something already under consideration.
    """
    present = {_normalise(entry.condition) for entry in script.differential}
    next_rank = max((entry.rank for entry in script.differential), default=0)

    for item in hypotheses:
        label = str(item.get("label") or item.get("condition") or "").strip()
        if not label or _normalise(label) in present:
            continue
        next_rank += 1
        present.add(_normalise(label))
        script.differential.append(
            DifferentialEntry(
                condition=label,
                term_id=item.get("term_id"),
                rank=next_rank,
                discriminating_features=tuple(item.get("discriminating_features", ()) or ()),
                origin=origin,
            )
        )
    return script


def _listing(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload.get(key)
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _element(item: dict[str, Any]) -> ScriptElement:
    return ScriptElement(
        label=str(item.get("label", "")).strip(),
        term_id=item.get("term_id"),
        note=item.get("note"),
    )


def _normalise(value: str) -> str:
    return " ".join(str(value).lower().split())
