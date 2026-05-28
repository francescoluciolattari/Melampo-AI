from dataclasses import dataclass

from ..models.quantum_belief_layer import QuantumBeliefLayer


def _float_metric(source: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(source.get(key, default))
    except (TypeError, ValueError):
        return default


@dataclass
class IntuitionEngine:
    """Research scaffold for biologically inspired inductive-deductive intuition."""

    belief_layer: QuantumBeliefLayer

    def _extract_metrics(self, area_dynamics: dict, neuro_metrics: dict) -> tuple[float, ...]:
        return (
            _float_metric(area_dynamics, "pi_score", _float_metric(neuro_metrics, "pi_score")),
            _float_metric(area_dynamics, "prediction_error", _float_metric(neuro_metrics, "prediction_error")),
            _float_metric(area_dynamics, "precision_weighted_coherence", _float_metric(neuro_metrics, "precision_weighted_coherence")),
            _float_metric(neuro_metrics, "convergence_index"),
            _float_metric(neuro_metrics, "mismatch_index"),
            _float_metric(neuro_metrics, "inhibitory_control"),
            _float_metric(neuro_metrics, "deductive_gate"),
            _float_metric(neuro_metrics, "revision_pressure"),
            _float_metric(neuro_metrics, "dream_plasticity"),
            _float_metric(neuro_metrics, "intuition_gain", 1.0),
            _float_metric(neuro_metrics, "bias_suppression_score"),
            _float_metric(neuro_metrics, "candidate_temperature", 1.0),
            _float_metric(neuro_metrics, "belief_update_rate"),
            _float_metric(neuro_metrics, "interdependence_index"),
            _float_metric(neuro_metrics, "evidence_integration_score"),
            _float_metric(neuro_metrics, "noise_suppression_score"),
            _float_metric(neuro_metrics, "action_potential_gate"),
            _float_metric(neuro_metrics, "synaptic_plasticity_index"),
            _float_metric(neuro_metrics, "deep_inference_score"),
            _float_metric(neuro_metrics, "deductive_stability"),
        )

    def _rank_areas(self, area_signals: dict) -> tuple[list[dict], list[str], float, float]:
        area_ranking = []
        for name, payload in area_signals.items():
            signal_size = len(payload) if isinstance(payload, dict) else 1
            signal_count = int(payload.get("signal_count", signal_size)) if isinstance(payload, dict) else signal_size
            salience = float(payload.get("salience_score", 0.0)) if isinstance(payload, dict) else 0.0
            uncertainty = float(payload.get("uncertainty_score", max(0.0, 1.0 - min(salience, 1.0)))) if isinstance(payload, dict) else 1.0
            area_ranking.append({"area": name, "weight": signal_size + signal_count + salience - uncertainty})
        area_ranking.sort(key=lambda item: item["weight"], reverse=True)
        top_areas = [item["area"] for item in area_ranking[:2]]
        if area_ranking:
            top_weight = area_ranking[0]["weight"]
            second_weight = area_ranking[1]["weight"] if len(area_ranking) > 1 else 0
            convergence_score = round(second_weight / max(top_weight, 1), 3)
            conflict_score = round((top_weight - second_weight) / max(top_weight, 1), 3)
        else:
            convergence_score = 0.0
            conflict_score = 0.0
        return area_ranking, top_areas, convergence_score, conflict_score

    def _area_pair_bonus(self, top_areas: list[str], coherence_score: float, pi_score: float, convergence_index: float, interdependence_index: float, evidence_integration_score: float) -> float:
        base_bonus = 0.0
        top_area_pair = tuple(sorted(top_areas))
        if top_area_pair == ("language_listening", "visual_diagnostic"):
            base_bonus = 0.2
        elif top_area_pair == ("epidemiology", "visual_diagnostic"):
            base_bonus = 0.15
        elif top_area_pair == ("case_context", "language_listening"):
            base_bonus = 0.1
        return round(
            base_bonus
            + (0.08 * coherence_score)
            + (0.12 * pi_score)
            + convergence_index * 0.18
            + interdependence_index * 0.12
            + evidence_integration_score * 0.10,
            3,
        )

    def _disagreement_penalty(self, conflict_score: float, mismatch_score: float, prediction_error: float, mismatch_index: float, inhibitory_control: float, noise_suppression_score: float, deductive_stability: float) -> float:
        return round(
            max(
                0.0,
                max(conflict_score - 0.4, 0.0)
                + mismatch_score * 0.08
                + prediction_error * 0.18
                + mismatch_index * 0.22
                - inhibitory_control * 0.1
                - noise_suppression_score * 0.12
                - deductive_stability * 0.08,
            ),
            3,
        )

    def infer(self, case_id: str, ranked_evidence: list, dream: dict, quantum_allowed: bool, area_signals: dict | None = None, area_dynamics: dict | None = None) -> dict:
        area_signals = area_signals or {}
        area_dynamics = area_dynamics or {}
        neuro_metrics = area_dynamics.get("neuro_dynamic_metrics", {}) if isinstance(area_dynamics, dict) else {}
        (
            pi_score,
            prediction_error,
            precision_weighted_coherence,
            convergence_index,
            mismatch_index,
            inhibitory_control,
            deductive_gate,
            revision_pressure,
            dream_plasticity,
            intuition_gain,
            bias_suppression_score,
            candidate_temperature,
            belief_update_rate,
            interdependence_index,
            evidence_integration_score,
            noise_suppression_score,
            action_potential_gate,
            synaptic_plasticity_index,
            deep_inference_score,
            deductive_stability,
        ) = self._extract_metrics(area_dynamics, neuro_metrics)

        rehearsal_profile = dream.get("rehearsal_profile", {}) if isinstance(dream, dict) else {}
        alternative_hypotheses = dream.get("alternative_hypotheses", []) if isinstance(dream, dict) else []
        contradiction_rehearsal = bool(rehearsal_profile.get("contradiction_rehearsal", False))
        revision_bias = rehearsal_profile.get("revision_bias", "exploratory")
        post_error_adjustment = rehearsal_profile.get("post_error_adjustment", "stabilize_primary")
        coherence_score_ext = float(area_dynamics.get("coherence_score", 0.0))
        mismatch_score_ext = float(area_dynamics.get("mismatch_score", 0.0))

        inductive_candidates = [
            {
                "label": f"candidate_{index + 1}",
                "support_weight": item.get("weight", 1),
                "source": item.get("item", {}).get("source", "unknown"),
            }
            for index, item in enumerate(ranked_evidence[:3])
        ]
        area_ranking, top_areas, convergence_score, conflict_score = self._rank_areas(area_signals)
        area_pair_bonus = self._area_pair_bonus(
            top_areas,
            coherence_score_ext,
            pi_score,
            convergence_index,
            interdependence_index,
            evidence_integration_score,
        )
        disagreement_penalty = self._disagreement_penalty(
            conflict_score,
            mismatch_score_ext,
            prediction_error,
            mismatch_index,
            inhibitory_control,
            noise_suppression_score,
            deductive_stability,
        )

        rapid_intuition = inductive_candidates[0]["label"] if inductive_candidates else "no_candidate"
        rational_revision = inductive_candidates[1]["label"] if len(inductive_candidates) > 1 else rapid_intuition
        contradiction_revision = alternative_hypotheses[0]["label"] if alternative_hypotheses else rational_revision

        first_support = float(inductive_candidates[0]["support_weight"] if inductive_candidates else 0.0)
        second_support = float(inductive_candidates[1]["support_weight"] if len(inductive_candidates) > 1 else 0.0)
        temperature_damping = max(candidate_temperature, 0.35)
        rapid_score = round(
            (
                (
                    first_support
                    + area_pair_bonus
                    + convergence_score
                    + deductive_gate
                    + action_potential_gate * 0.16
                    + deep_inference_score * 0.14
                    + bias_suppression_score * 0.18
                    + pi_score * 0.2
                    - disagreement_penalty
                )
                * intuition_gain
            )
            / temperature_damping,
            3,
        )
        rational_score = round(
            second_support
            + conflict_score
            + (0.2 if revision_bias == "conservative" else 0.0)
            + mismatch_score_ext * 0.08
            + revision_pressure * 0.20
            + precision_weighted_coherence * 0.12
            + deductive_gate * 0.15
            + evidence_integration_score * 0.18
            + interdependence_index * 0.12
            + deductive_stability * 0.10,
            3,
        )
        contradiction_score = round(
            (1.0 if contradiction_rehearsal else 0.0)
            + (0.3 if post_error_adjustment == "re-rank_alternatives" else 0.0)
            + (0.1 * len(alternative_hypotheses))
            + mismatch_score_ext * 0.18
            + prediction_error * 0.3
            + mismatch_index * 0.25
            + dream_plasticity * 0.16
            + synaptic_plasticity_index * 0.10
            - noise_suppression_score * 0.12,
            3,
        )

        candidate_scores = [
            {"mode": "rapid_intuition", "label": rapid_intuition, "score": rapid_score},
            {"mode": "rational_revision", "label": rational_revision, "score": rational_score},
            {"mode": "contradiction_revision", "label": contradiction_revision, "score": contradiction_score},
        ]
        candidate_scores.sort(key=lambda item: item["score"], reverse=True)
        selected = candidate_scores[0] if candidate_scores else {"mode": "rapid_intuition", "label": "no_candidate", "score": 0.0}
        intuition = selected["label"]
        reasoning_mode = selected["mode"]

        deductive_filter = {
            "kept": len(inductive_candidates),
            "rejected": max(len(ranked_evidence) - len(inductive_candidates), 0),
            "criterion": "top_ranked_grounded_evidence_with_neuro_dynamic_modulation",
            "active_areas": sorted(area_signals.keys()),
            "top_areas": top_areas,
            "convergence_score": convergence_score,
            "conflict_score": conflict_score,
            "coherence_score": coherence_score_ext,
            "mismatch_score": mismatch_score_ext,
            "pi_score": pi_score,
            "prediction_error": prediction_error,
            "precision_weighted_coherence": precision_weighted_coherence,
            "convergence_index": convergence_index,
            "mismatch_index": mismatch_index,
            "inhibitory_control": inhibitory_control,
            "deductive_gate": deductive_gate,
            "revision_pressure": revision_pressure,
            "dream_plasticity": dream_plasticity,
            "intuition_gain": intuition_gain,
            "candidate_temperature": candidate_temperature,
            "belief_update_rate": belief_update_rate,
            "bias_suppression_score": bias_suppression_score,
            "interdependence_index": interdependence_index,
            "evidence_integration_score": evidence_integration_score,
            "noise_suppression_score": noise_suppression_score,
            "action_potential_gate": action_potential_gate,
            "synaptic_plasticity_index": synaptic_plasticity_index,
            "deep_inference_score": deep_inference_score,
            "deductive_stability": deductive_stability,
            "area_pair_bonus": area_pair_bonus,
            "disagreement_penalty": disagreement_penalty,
            "contradiction_rehearsal": contradiction_rehearsal,
            "revision_bias": revision_bias,
            "post_error_adjustment": post_error_adjustment,
            "reasoning_mode": reasoning_mode,
        }
        belief_context = {
            "dream_mode": "none",
            "quantum_allowed": quantum_allowed,
            "area_count": len(area_signals),
            "top_areas": top_areas,
            "convergence_score": convergence_score,
            "conflict_score": conflict_score,
            "coherence_score": coherence_score_ext,
            "mismatch_score": mismatch_score_ext,
            "pi_score": pi_score,
            "prediction_error": prediction_error,
            "precision_weighted_coherence": precision_weighted_coherence,
            "convergence_index": convergence_index,
            "mismatch_index": mismatch_index,
            "inhibitory_control": inhibitory_control,
            "belief_update_rate": belief_update_rate,
            "candidate_temperature": candidate_temperature,
            "conflict_load": neuro_metrics.get("conflict_load", 0.0),
            "interdependence_index": interdependence_index,
            "evidence_integration_score": evidence_integration_score,
            "noise_suppression_score": noise_suppression_score,
            "action_potential_gate": action_potential_gate,
            "synaptic_plasticity_index": synaptic_plasticity_index,
            "deep_inference_score": deep_inference_score,
            "deductive_stability": deductive_stability,
            "neuro_dynamic_metrics": neuro_metrics,
            "area_pair_bonus": area_pair_bonus,
            "disagreement_penalty": disagreement_penalty,
            "contradiction_rehearsal": contradiction_rehearsal,
            "revision_bias": revision_bias,
            "reasoning_mode": reasoning_mode,
        }
        if isinstance(dream, dict):
            belief = dream.get("belief", {})
            if isinstance(belief, dict):
                belief_context["dream_mode"] = belief.get("mode", "none")
        if quantum_allowed:
            belief_update = self.belief_layer.update(
                prior={"case_id": case_id, "candidate_count": len(inductive_candidates)},
                context=belief_context,
            )
        else:
            belief_update = {"mode": "classical_only", **belief_context}
        return {
            "intuition": intuition,
            "rapid_intuition": rapid_intuition,
            "rational_revision": rational_revision,
            "contradiction_revision": contradiction_revision,
            "candidate_scores": candidate_scores,
            "inductive_candidates": inductive_candidates,
            "dream_alternatives": alternative_hypotheses,
            "deductive_filter": deductive_filter,
            "belief_update": belief_update,
            "area_signals": area_signals,
            "area_ranking": area_ranking,
            "area_dynamics": area_dynamics,
        }

    def summarize_for_trace(self, intuition_payload: dict) -> dict:
        area_signals = intuition_payload.get("area_signals", {})
        deductive = intuition_payload.get("deductive_filter", {})
        return {
            "intuition": intuition_payload.get("intuition", "none"),
            "candidate_count": len(intuition_payload.get("inductive_candidates", [])),
            "belief_mode": intuition_payload.get("belief_update", {}).get("mode", "none"),
            "active_areas": sorted(area_signals.keys()),
            "top_areas": deductive.get("top_areas", []),
            "convergence_score": deductive.get("convergence_score", 0.0),
            "conflict_score": deductive.get("conflict_score", 0.0),
            "coherence_score": deductive.get("coherence_score", 0.0),
            "mismatch_score": deductive.get("mismatch_score", 0.0),
            "pi_score": deductive.get("pi_score", 0.0),
            "prediction_error": deductive.get("prediction_error", 0.0),
            "precision_weighted_coherence": deductive.get("precision_weighted_coherence", 0.0),
            "convergence_index": deductive.get("convergence_index", 0.0),
            "mismatch_index": deductive.get("mismatch_index", 0.0),
            "inhibitory_control": deductive.get("inhibitory_control", 0.0),
            "deductive_gate": deductive.get("deductive_gate", 0.0),
            "dream_plasticity": deductive.get("dream_plasticity", 0.0),
            "candidate_temperature": deductive.get("candidate_temperature", 1.0),
            "belief_update_rate": deductive.get("belief_update_rate", 0.0),
            "interdependence_index": deductive.get("interdependence_index", 0.0),
            "evidence_integration_score": deductive.get("evidence_integration_score", 0.0),
            "noise_suppression_score": deductive.get("noise_suppression_score", 0.0),
            "action_potential_gate": deductive.get("action_potential_gate", 0.0),
            "deep_inference_score": deductive.get("deep_inference_score", 0.0),
            "deductive_stability": deductive.get("deductive_stability", 0.0),
            "area_pair_bonus": deductive.get("area_pair_bonus", 0.0),
            "disagreement_penalty": deductive.get("disagreement_penalty", 0.0),
            "contradiction_rehearsal": deductive.get("contradiction_rehearsal", False),
            "revision_bias": deductive.get("revision_bias", "exploratory"),
            "post_error_adjustment": deductive.get("post_error_adjustment", "stabilize_primary"),
            "reasoning_mode": deductive.get("reasoning_mode", "rapid_intuition"),
        }
