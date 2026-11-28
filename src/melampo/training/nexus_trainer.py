from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..memory.visual_imprint import VisualImprintMorpher
from ..utils.values import safe_float as _safe_float
from ..utils.values import safe_int as _safe_int
from .counterfactual_sampler import CounterfactualSampler
from .mechanism_enumeration import MODE_HYPOTHESES
from .replay_filter import ReplayFilter


@dataclass(frozen=True, slots=True)
class NexusRuntimeContext:
    case_context: dict[str, Any]
    filter_assessment: dict[str, Any]
    sampled: dict[str, Any]
    accepted: bool
    coherence: float
    risk: float
    bundle_keys: list[Any]
    base_label: str
    exposures: dict[str, Any]
    report_text: str
    patient_complaints: str
    discordances: int
    variant_focus: str
    visual_morphing: dict[str, Any]
    visual_morph_gain: float
    visual_prediction_link_score: float


@dataclass
class NexusTrainer:
    """Offline nexus replay: the alternatives a case's findings reach in the graph (``enumerator``).

    Closed (28 November 2026): it proposes alternatives and rehearsals, and promotes nothing
    (``auto_evolution_plan`` holds every case) until the lateral cycle of the intuition engine has measured
    criteria. The metrics it read until then (pi_score, convergence, mismatch, plasticity) came from the
    coherence between areas, removed; its signal now is the case's discordances.
    """

    replay_filter: ReplayFilter
    sampler: CounterfactualSampler
    visual_morpher: VisualImprintMorpher = field(default_factory=VisualImprintMorpher)
    enumerator: Any = None

    def _runtime_context(
        self, case_context: dict, coherence: float, risk: float
    ) -> NexusRuntimeContext:
        case_context = case_context or {}
        filter_assessment = self.replay_filter.assess(coherence=coherence, risk=risk)
        sampled = self.sampler.sample(case_context)
        discordances = case_context.get("discordances", {})
        discordance_count = _safe_int(
            discordances.get("count", 0) if isinstance(discordances, dict) else 0
        )
        visual_imprints = (
            list(case_context.get("visual_imprints", []))
            if isinstance(case_context.get("visual_imprints", []), list)
            else []
        )
        concept_memory_imprints = (
            list(case_context.get("concept_memory_imprints", []))
            if isinstance(case_context.get("concept_memory_imprints", []), list)
            else []
        )
        diagnostic_visual_imprints = (
            list(case_context.get("diagnostic_visual_imprints", visual_imprints))
            if isinstance(
                case_context.get("diagnostic_visual_imprints", visual_imprints), list
            )
            else visual_imprints
        )
        visual_morphing = self.visual_morpher.nexus_morph(
            concept_imprints=concept_memory_imprints + visual_imprints,
            diagnostic_imprints=diagnostic_visual_imprints,
        )
        return NexusRuntimeContext(
            case_context=case_context,
            filter_assessment=filter_assessment,
            sampled=sampled,
            accepted=bool(filter_assessment["accepted"]),
            coherence=coherence,
            risk=risk,
            bundle_keys=case_context.get("bundle_keys", []),
            base_label=case_context.get("case_id", "case"),
            exposures=case_context.get("exposures", {}),
            report_text=case_context.get("report_text", ""),
            patient_complaints=case_context.get("patient_complaints", ""),
            discordances=discordance_count,
            variant_focus=sampled.get("variant_focus", "context"),
            visual_morphing=visual_morphing,
            visual_morph_gain=_safe_float(
                visual_morphing.get("visual_morph_intuition_gain", 0.0)
            ),
            visual_prediction_link_score=_safe_float(
                visual_morphing.get("visual_prediction_link_score", 0.0)
            ),
        )

    def _rehearsal_profile(self, context: NexusRuntimeContext) -> dict[str, Any]:
        contradiction_rehearsal = bool(
            (not context.accepted)
            or context.risk > 0.2
            or context.discordances > 0
        )
        return {
            "rare_case_hint": bool(context.accepted and len(context.bundle_keys) <= 2),
            "boundary_case_hint": bool(
                context.coherence < 0.95 and context.risk <= 0.15
            ),
            "contradiction_rehearsal": contradiction_rehearsal,
            "revision_bias": "conservative"
            if (
                context.risk > 0.15
                or context.discordances > 0
            )
            else "exploratory",
            "post_error_adjustment": "re-rank_alternatives"
            if contradiction_rehearsal
            else "stabilize_primary",
            "replay_mode": context.filter_assessment["replay_mode"],
            "acceptance_score": context.filter_assessment["acceptance_score"],
            "variant_focus": context.variant_focus,
            "discordances": context.discordances,
            "visual_morphing_active": bool(
                context.visual_morphing.get("morph_count", 0)
            ),
            "visual_morph_intuition_gain": round(context.visual_morph_gain, 3),
            "visual_prediction_link_score": round(
                context.visual_prediction_link_score, 3
            ),
        }

    def _alternative_hypotheses(
        self, context: NexusRuntimeContext, rehearsal_profile: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Alternative hypotheses for this case.

        With an ``enumerator`` configured these are found by path enumeration
        over the concept graph: each candidate is a condition the case has not
        raised, reached from the observed findings, carrying the path as its
        provenance. A hypothesis is therefore found rather than written, and it
        cannot be fluent and baseless at the same time — a path exists in the
        graph or it does not.

        Without one the previous rehearsal labels are produced unchanged. They
        name what kind of alternative *ought* to exist rather than proposing one,
        which is useful as a rehearsal plan and is not a clinical hypothesis.
        """
        enumerated = self._enumerated_hypotheses(context)
        if enumerated is not None:
            return enumerated
        return self._rehearsal_labels(context, rehearsal_profile)

    def _enumerated_hypotheses(
        self, context: NexusRuntimeContext
    ) -> list[dict[str, Any]] | None:
        """Enumerate over the graph, or return None when no enumerator is wired.

        Returns the branch's own register: under sparse local coverage the
        enumerator emits questions for the knowledge base instead of hypotheses
        for the patient, and those are passed through as they are rather than
        being reshaped into hypotheses they are not.
        """
        if self.enumerator is None:
            return None
        findings = [
            str(item)
            for item in (context.case_context.get("findings") or [])
            if str(item).strip()
        ]
        candidates = [
            str(item)
            for item in (context.case_context.get("candidate_conditions") or [])
            if str(item).strip()
        ]
        if not findings or not candidates:
            return None

        outcome = self.enumerator.run(
            findings=findings,
            candidate_conditions=candidates,
            already_considered=[
                str(item)
                for item in (context.case_context.get("already_considered") or [])
            ],
        )
        payload = outcome.as_dict()
        if outcome.mode != MODE_HYPOTHESES:
            return [
                {
                    "label": item["condition"],
                    "kind": "knowledge_gap_question",
                    "focus": "graph_completion",
                    "question": item["question"],
                    "clinical_use": False,
                    "density": payload["density"]["density"],
                }
                for item in payload["open_questions"]
            ]

        return [
            {
                "label": item["condition"],
                "kind": "enumerated_mechanism",
                "focus": context.variant_focus,
                "novelty": item["novelty"],
                "plausibility": item["plausibility"],
                "guaranteed": item["guaranteed"],
                "corroboration": item["corroboration"],
                "paths": item["paths"],
                "density": payload["density"]["density"],
            }
            for item in payload["hypotheses"]
        ]

    def _rehearsal_labels(
        self, context: NexusRuntimeContext, rehearsal_profile: dict[str, Any]
    ) -> list[dict[str, Any]]:
        hypotheses: list[dict[str, Any]] = [
            {
                "label": f"{context.base_label}_alt_1",
                "kind": "rare_case"
                if rehearsal_profile["rare_case_hint"]
                else "adjacent_case",
                "focus": "epidemiology" if context.exposures else context.variant_focus,
            },
            {
                "label": f"{context.base_label}_alt_2",
                "kind": "boundary_case"
                if rehearsal_profile["boundary_case_hint"]
                else "counterfactual_case",
                "focus": "language_listening"
                if (context.report_text or context.patient_complaints)
                else context.variant_focus,
            },
        ]
        if rehearsal_profile["contradiction_rehearsal"]:
            hypotheses.append(
                {
                    "label": f"{context.base_label}_alt_3",
                    "kind": "contradiction_revision",
                    "focus": "multi_area_recheck",
                }
            )
        if context.discordances > 0:
            hypotheses.append(
                {
                    "label": f"{context.base_label}_alt_4",
                    "kind": "discordance_resolution",
                    "focus": "cross_area_alignment",
                }
            )
        if context.visual_prediction_link_score >= 0.45:
            hypotheses.append(
                {
                    "label": f"{context.base_label}_visual_morph_link",
                    "kind": "visual_semantic_morph_correlation",
                    "focus": "visual_diagnostic",
                    "score": round(context.visual_prediction_link_score, 3),
                    "requires_review": True,
                }
            )
        return hypotheses

    def _auto_evolution_plan(self, context: NexusRuntimeContext) -> dict[str, Any]:
        """Whether this case is a candidate for offline auto-evolution: none is, for now.

        ``safety.rails`` reads ``status`` on every case. Until 28 November 2026 a case became a candidate on
        pi_score, convergence and plasticity, measures of the coherence between areas that were never measured
        against anything; with them removed the chain is closed: every case is held, with its reason, until the
        lateral cycle of the intuition engine has criteria measured on published cases.
        """
        return {"status": "hold_for_more_evidence", "reason": "no_measured_promotion_criteria"}

    def run(self, case_context: dict, coherence: float, risk: float) -> dict:
        context = self._runtime_context(
            case_context=case_context, coherence=coherence, risk=risk
        )
        rehearsal_profile = self._rehearsal_profile(context)
        alternative_hypotheses = self._alternative_hypotheses(
            context, rehearsal_profile
        )
        auto_evolution_plan = self._auto_evolution_plan(context)
        return {
            "accepted": context.accepted,
            "filter_assessment": context.filter_assessment,
            "sampled": context.sampled,
            "rehearsal_profile": rehearsal_profile,
            "alternative_hypotheses": alternative_hypotheses,
            "auto_evolution_plan": auto_evolution_plan,
            "visual_morphing": context.visual_morphing,
        }
