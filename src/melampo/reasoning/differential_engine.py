from dataclasses import dataclass, field
from typing import Any

from .support_contradiction import SupportContradictionAnalyzer

_PLACEHOLDER_LABEL_PREFIX = "candidate_"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _best_graph_hypothesis(alternatives: list) -> dict | None:
    """The highest-plausibility real, graph-grounded hypothesis among the nexus branch's alternatives, if any.

    `kind == "enumerated_mechanism"` is MechanismEnumerator's own marker for
    a hypothesis found by real path enumeration over the concept graph
    (`nexus_trainer.py`, `_enumerated_hypotheses`) -- distinct from
    `rare_case`/`adjacent_case`/`mismatch_resolution` rehearsal labels, which
    name what kind of alternative ought to exist rather than proposing one.
    """
    graph_hypotheses = [item for item in alternatives if isinstance(item, dict) and item.get("kind") == "enumerated_mechanism"]
    if not graph_hypotheses:
        return None
    return max(graph_hypotheses, key=lambda item: _safe_float(item.get("plausibility"), 0.0))


@dataclass
class DifferentialEngine:
    """Baseline differential engine using evidence, intuition, nexus alternatives, and area dynamics."""

    support_analyzer: SupportContradictionAnalyzer = field(default_factory=SupportContradictionAnalyzer)

    def _infer_domain(self, support_signals: list, source: str, hypothesis_type: str) -> str:
        joined = " ".join(support_signals)
        if "fusion:" in joined or "visual_diagnostic" in joined or source in ["mismatch_resolution", "contradiction_revision"]:
            return "imaging_led"
        if "nexus:language_listening" in joined:
            return "language_led"
        if "nexus:epidemiology" in joined:
            return "epidemiology_led"
        if hypothesis_type == "mismatch_resolution":
            return "mismatch_resolution_led"
        return "multimodal_led"

    def _recommended_actions_for_domain(self, domain: str, contradiction_classes: set[str], coherence_score: float, reasoning_mode: str) -> list[dict]:
        actions = []
        if domain == "imaging_led":
            actions.extend([
                {"category": "confirmation_test", "label": "repeat targeted imaging review"},
                {"category": "multimodal_reconciliation", "label": "correlate imaging with structured report findings"},
            ])
        elif domain == "language_led":
            actions.extend([
                {"category": "disambiguation_test", "label": "reassess symptom chronology"},
                {"category": "disambiguation_test", "label": "clarify narrative inconsistencies"},
            ])
        elif domain == "epidemiology_led":
            actions.extend([
                {"category": "confirmation_test", "label": "review exposure history"},
                {"category": "disambiguation_test", "label": "check prevalence-guided differential filters"},
            ])
        elif domain == "mismatch_resolution_led":
            actions.extend([
                {"category": "multimodal_reconciliation", "label": "cross-check modality agreement"},
                {"category": "multimodal_reconciliation", "label": "request multimodal reconciliation review"},
            ])
        else:
            actions.append({"category": "confirmation_test", "label": "expand corroborating evidence"})

        if "useful_contradiction" in contradiction_classes:
            actions.append({"category": "multimodal_reconciliation", "label": "recheck multimodal alignment"})
        if coherence_score < 0.5:
            actions.append({"category": "confirmation_test", "label": "expand corroborating evidence"})
        if reasoning_mode == "contradiction_revision":
            actions.append({"category": "disambiguation_test", "label": "review alternative hypotheses"})
        if "weak_contradiction" in contradiction_classes:
            actions.append({"category": "disambiguation_test", "label": "monitor boundary conditions"})

        deduped = []
        seen = set()
        for item in actions:
            key = (item["category"], item["label"])
            if key not in seen:
                deduped.append(item)
                seen.add(key)
        return deduped or [{"category": "confirmation_test", "label": "continue standard differential refinement"}]

    def rank(self, evidence: list, intuition: dict | None = None, nexus: dict | None = None, area_dynamics: dict | None = None) -> dict:
        intuition = intuition or {}
        nexus = nexus or {}
        area_dynamics = area_dynamics or {}

        candidate_scores = intuition.get("candidate_scores", [])
        top_candidate = candidate_scores[0] if candidate_scores else {"label": "working_hypothesis", "score": 0.7}
        alternatives = nexus.get("alternative_hypotheses", [])
        mismatch_score = float(area_dynamics.get("mismatch_score", 0.0))
        coherence_score = float(area_dynamics.get("coherence_score", 0.0))
        reasoning_mode = intuition.get("deductive_filter", {}).get("reasoning_mode", "rapid_intuition")

        signals = self.support_analyzer.analyze(evidence=evidence, intuition=intuition, nexus=nexus, area_dynamics=area_dynamics)
        support_signals = signals["support_signals"]
        contradiction_signals = signals["contradiction_signals"]
        support_profiles = signals["support_profiles"]
        contradiction_profiles = signals["contradiction_profiles"]
        contradiction_classes = {item["class"] for item in contradiction_profiles}

        primary_type = "primary_hypothesis"
        if reasoning_mode == "rational_revision":
            primary_type = "revision_hypothesis"
        elif reasoning_mode == "contradiction_revision":
            primary_type = "contradiction_revision_hypothesis"

        primary_domain = self._infer_domain(support_signals[:4], "intuition_engine", primary_type)
        primary_hypothesis = {
            "label": top_candidate.get("label", "working_hypothesis"),
            "hypothesis_type": primary_type,
            "hypothesis_domain": primary_domain,
            "score": round(_safe_float(top_candidate.get("score", 0.7), 0.7) + coherence_score * 0.1 + signals["support_strength"] * 0.02, 3),
            "support": len(evidence),
            "source": "intuition_engine",
            "support_signals": support_signals[:4],
            "contradiction_signals": contradiction_signals[:3],
            "support_profile_classes": [item["class"] for item in support_profiles[:3]],
            "contradiction_profile_classes": [item["class"] for item in contradiction_profiles[:3]],
        }

        # A real, graph-grounded hypothesis outranks a placeholder one.
        # IntuitionEngine's own candidate labels are literally "candidate_1",
        # "candidate_2" -- indices into ranked_evidence, not diagnosis names
        # -- until the functional areas that feed it produce genuine
        # clinical signal (see ROADMAP.md). Promoting the best real
        # enumerated hypothesis here, when one exists, means the primary
        # slot is never worse than a placeholder while that gap remains,
        # without silently discarding intuition's own contribution: demoted
        # rather than dropped, and tagged with its own real source
        # ("graph_enumeration") so which engine actually produced the
        # primary hypothesis stays distinguishable, the same discipline
        # `rlm_graph_bridge.py` already applies to origin labelling.
        primary_is_placeholder = str(primary_hypothesis["label"]).startswith(_PLACEHOLDER_LABEL_PREFIX)
        best_graph_hypothesis = _best_graph_hypothesis(alternatives) if primary_is_placeholder else None
        if best_graph_hypothesis is not None:
            demoted_intuition_hypothesis = {**primary_hypothesis, "hypothesis_type": "revision_alternative"}
            primary_hypothesis = {
                "label": best_graph_hypothesis["label"],
                "hypothesis_type": primary_type,
                "hypothesis_domain": "graph_enumeration_led",
                "score": round(_safe_float(best_graph_hypothesis.get("plausibility"), 0.7), 3),
                "support": len(best_graph_hypothesis.get("paths", [])) or len(evidence),
                "source": "graph_enumeration",
                "paths": best_graph_hypothesis.get("paths", []),
                "novelty": best_graph_hypothesis.get("novelty"),
                "guaranteed": best_graph_hypothesis.get("guaranteed"),
                "corroboration": best_graph_hypothesis.get("corroboration"),
                "support_signals": support_signals[:4],
                "contradiction_signals": contradiction_signals[:3],
                "support_profile_classes": [item["class"] for item in support_profiles[:3]],
                "contradiction_profile_classes": [item["class"] for item in contradiction_profiles[:3]],
            }
        hypotheses = [primary_hypothesis]
        if best_graph_hypothesis is not None:
            hypotheses.append(demoted_intuition_hypothesis)
            # Excluded from the loop below, which would otherwise list it a
            # second time as one of its own "alternatives" -- identity, not
            # equality, since two distinct hypotheses could coincidentally
            # carry the same condition label.
            alternatives = [item for item in alternatives if item is not best_graph_hypothesis]

        for index, alt in enumerate(alternatives[:3]):
            hypothesis_type = "revision_alternative"
            if alt.get("kind") == "mismatch_resolution":
                hypothesis_type = "mismatch_resolution"
            elif alt.get("kind") == "contradiction_revision":
                hypothesis_type = "contradiction_revision_alternative"
            alt_support = [f"nexus:{alt.get('focus', 'unknown')}"] + support_signals[:1]
            hypotheses.append(
                {
                    "label": alt.get("label", f"alternative_{index + 1}"),
                    "hypothesis_type": hypothesis_type,
                    "hypothesis_domain": self._infer_domain(alt_support, alt.get("kind", "nexus_alternative"), hypothesis_type),
                    "score": round(0.4 + mismatch_score * 0.1 - index * 0.05 + signals["contradiction_strength"] * 0.01, 3),
                    "support": max(len(evidence) - index - 1, 0),
                    "source": alt.get("kind", "nexus_alternative"),
                    "support_signals": alt_support,
                    "contradiction_signals": contradiction_signals[:3],
                    "support_profile_classes": [item["class"] for item in support_profiles[:2]],
                    "contradiction_profile_classes": [item["class"] for item in contradiction_profiles[:3]],
                }
            )

        if len(hypotheses) == 1:
            hypotheses.append(
                {
                    "label": "alternative_hypothesis",
                    "hypothesis_type": "fallback_alternative",
                    "hypothesis_domain": "multimodal_led",
                    "score": round(0.3 + mismatch_score * 0.05, 3),
                    "support": max(len(evidence) - 1, 0),
                    "source": "fallback_alternative",
                    "support_signals": support_signals[:2],
                    "contradiction_signals": contradiction_signals[:2],
                    "support_profile_classes": [item["class"] for item in support_profiles[:2]],
                    "contradiction_profile_classes": [item["class"] for item in contradiction_profiles[:2]],
                }
            )

        recommended_actions = self._recommended_actions_for_domain(primary_domain, contradiction_classes, coherence_score, reasoning_mode)
        recommended_tests = [item["label"] for item in recommended_actions]

        return {
            "status": "grounded_differential_ready",
            "evidence_count": len(evidence),
            "mismatch_score": mismatch_score,
            "coherence_score": coherence_score,
            "reasoning_mode": reasoning_mode,
            "support_strength": signals["support_strength"],
            "contradiction_strength": signals["contradiction_strength"],
            "support_profiles": support_profiles,
            "contradiction_profiles": contradiction_profiles,
            "hypotheses": hypotheses,
            "recommended_actions": recommended_actions,
            "recommended_tests": recommended_tests,
        }
