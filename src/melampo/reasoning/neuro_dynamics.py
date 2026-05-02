from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class AreaInteractionPrior:
    """Expected interaction profile between two simulated functional areas.

    These priors are not literal neurobiology. They encode an enterprise-grade,
    inspectable abstraction inspired by predictive processing, active inference,
    recurrent cortical communication, inhibitory control, and cross-modal
    binding: coherent multimodal pairs should raise precision and convergence;
    unresolved high-salience mismatches should raise prediction error and
    revision pressure.
    """

    excitatory_weight: float
    inhibitory_weight: float
    expected_precision: float
    mismatch_sensitivity: float
    rationale: str


@dataclass(slots=True)
class NeuroDynamicMetrics:
    """Neuro-inspired interaction metrics for Melampo functional areas.

    The model converts area salience and pairwise coherence/mismatch into
    explicit, auditable signals used by intuition, dream replay, differential
    ranking, and belief update:

    - pi_score: predictive-inference quality score.
    - precision_weighted_coherence: coherence weighted by signal precision.
    - prediction_error: mismatch load after area-prior modulation.
    - conflict_load: residual unresolved cross-area conflict.
    - deductive_gate: strength of slow, controlled, evidence-preserving thought.
    - revision_pressure: pressure to re-rank alternatives and trigger dream replay.
    - bias_suppression_score: proxy for inhibition of noisy or biased candidates.

    This is a computational abstraction for research software. It must not be
    interpreted as a validated biological or clinical model without empirical
    calibration, prospective testing, and medical governance.
    """

    precision_gain: float = 0.6
    mismatch_gain: float = 0.4
    synchrony_gain: float = 0.25
    conflict_gain: float = 0.35
    area_priors: dict[tuple[str, str], AreaInteractionPrior] = field(default_factory=lambda: {
        ("language_listening", "visual_diagnostic"): AreaInteractionPrior(
            excitatory_weight=0.22,
            inhibitory_weight=0.08,
            expected_precision=0.85,
            mismatch_sensitivity=0.75,
            rationale="cross-modal report-image binding and contradiction checking",
        ),
        ("epidemiology", "visual_diagnostic"): AreaInteractionPrior(
            excitatory_weight=0.16,
            inhibitory_weight=0.06,
            expected_precision=0.72,
            mismatch_sensitivity=0.65,
            rationale="pre-test probability constrains imaging interpretation",
        ),
        ("case_context", "language_listening"): AreaInteractionPrior(
            excitatory_weight=0.14,
            inhibitory_weight=0.05,
            expected_precision=0.78,
            mismatch_sensitivity=0.55,
            rationale="history and narrative context stabilize symptom semantics",
        ),
        ("case_context", "epidemiology"): AreaInteractionPrior(
            excitatory_weight=0.12,
            inhibitory_weight=0.05,
            expected_precision=0.7,
            mismatch_sensitivity=0.5,
            rationale="demographics and exposure priors constrain prevalence reasoning",
        ),
        ("language_listening", "epidemiology"): AreaInteractionPrior(
            excitatory_weight=0.1,
            inhibitory_weight=0.04,
            expected_precision=0.64,
            mismatch_sensitivity=0.48,
            rationale="symptom chronology interacts with exposure and prevalence priors",
        ),
        ("case_context", "visual_diagnostic"): AreaInteractionPrior(
            excitatory_weight=0.09,
            inhibitory_weight=0.04,
            expected_precision=0.62,
            mismatch_sensitivity=0.52,
            rationale="patient context can modulate imaging interpretation but should not dominate pixels",
        ),
    })

    def _prior_for(self, pair: tuple[str, str]) -> AreaInteractionPrior:
        return self.area_priors.get(
            tuple(sorted(pair)),
            AreaInteractionPrior(
                excitatory_weight=0.05,
                inhibitory_weight=0.05,
                expected_precision=0.5,
                mismatch_sensitivity=0.5,
                rationale="generic weak cross-area coupling",
            ),
        )

    def _area_state(self, area_signals: dict[str, Any] | None) -> dict[str, dict[str, float]]:
        area_signals = area_signals or {}
        state: dict[str, dict[str, float]] = {}
        for area, payload in area_signals.items():
            payload = payload if isinstance(payload, dict) else {}
            salience = _safe_float(payload.get("salience_score", 0.0))
            signal_count = _safe_int(payload.get("signal_count", len(payload)))
            uncertainty = _safe_float(payload.get("uncertainty_score", 1.0 - min(salience, 1.0)))
            state[area] = {
                "salience": _clamp(salience),
                "signal_count": max(signal_count, 0),
                "uncertainty": _clamp(uncertainty),
                "precision": _clamp(salience / max(salience + uncertainty + 0.001, 0.001)),
            }
        return state

    def compute(
        self,
        pair_profiles: list[dict],
        coherence_score: float,
        mismatch_score: float,
        total_salience: float,
        dream_pressure: float = 0.0,
        area_signals: dict[str, Any] | None = None,
    ) -> dict:
        coherent_salience = 0.0
        mismatch_salience = 0.0
        coherent_signal_count = 0
        mismatch_signal_count = 0
        excitatory_mass = 0.0
        inhibitory_mass = 0.0
        expected_precision_mass = 0.0
        mismatch_sensitivity_mass = 0.0
        interaction_profiles = []

        for profile in pair_profiles:
            pair = tuple(sorted(profile.get("pair", ("unknown", "unknown"))))
            prior = self._prior_for(pair)
            salience = _safe_float(profile.get("pair_salience", 0.0))
            signals = _safe_int(profile.get("pair_signal_count", 0))
            status = profile.get("status", "mismatch")
            normalized_salience = _clamp(salience / max(total_salience + 1.0, 1.0))
            expected_precision_mass += prior.expected_precision * normalized_salience
            mismatch_sensitivity_mass += prior.mismatch_sensitivity * normalized_salience

            if status == "coherent":
                coherent_salience += salience * (1.0 + prior.excitatory_weight)
                coherent_signal_count += signals
                excitatory_mass += prior.excitatory_weight * normalized_salience
                coupling_score = _clamp(normalized_salience + prior.excitatory_weight + prior.expected_precision * 0.1)
            else:
                mismatch_salience += salience * (1.0 + prior.mismatch_sensitivity)
                mismatch_signal_count += signals
                inhibitory_mass += prior.inhibitory_weight * normalized_salience
                coupling_score = _clamp(normalized_salience - prior.mismatch_sensitivity * 0.2)

            interaction_profiles.append({
                "pair": list(pair),
                "status": status,
                "coupling_score": round(coupling_score, 3),
                "expected_precision": prior.expected_precision,
                "mismatch_sensitivity": prior.mismatch_sensitivity,
                "rationale": prior.rationale,
            })

        area_state = self._area_state(area_signals)
        area_precision_values = [entry["precision"] for entry in area_state.values()]
        mean_area_precision = sum(area_precision_values) / max(len(area_precision_values), 1)
        pair_salience_total = max(coherent_salience + mismatch_salience, 1e-6)
        cross_area_synchrony = coherent_salience / pair_salience_total
        mismatch_ratio = mismatch_salience / pair_salience_total
        signal_precision = total_salience / max(total_salience + mismatch_signal_count + 1.0, 1.0)
        prior_precision = expected_precision_mass / max(expected_precision_mass + mismatch_sensitivity_mass + 1e-6, 1e-6)

        precision_weighted_coherence = coherence_score * (self.precision_gain + signal_precision + mean_area_precision * 0.25)
        prediction_error = mismatch_score * (self.mismatch_gain + mismatch_ratio + mismatch_sensitivity_mass * 0.15)
        conflict_load = prediction_error + (mismatch_signal_count / max(coherent_signal_count + mismatch_signal_count, 1)) * self.conflict_gain
        convergence_index = _clamp(
            coherence_score * 0.35
            + cross_area_synchrony * 0.25
            + signal_precision * 0.2
            + prior_precision * 0.15
            + excitatory_mass * 0.25
        )
        mismatch_index = _clamp(
            prediction_error * 0.45
            + mismatch_ratio * 0.25
            + conflict_load * 0.2
            + inhibitory_mass * 0.2
        )
        inhibitory_control = _clamp(1.0 - conflict_load * 0.55 + signal_precision * 0.25 + mean_area_precision * 0.1)
        deductive_gate = _clamp(
            precision_weighted_coherence * 0.35
            + convergence_index * 0.3
            + inhibitory_control * 0.2
            + cross_area_synchrony * self.synchrony_gain
            - prediction_error * self.conflict_gain
        )
        revision_pressure = _clamp(prediction_error * 0.45 + conflict_load * 0.35 + dream_pressure * 0.2 + mismatch_index * 0.2)
        dream_plasticity = _clamp(0.35 + revision_pressure * 0.4 + (1.0 - inhibitory_control) * 0.15)
        intuition_gain = _clamp(1.0 + deductive_gate * 0.25 - revision_pressure * 0.15 + convergence_index * 0.1, 0.5, 1.35)
        pi_score = _clamp(
            precision_weighted_coherence * 0.35
            + convergence_index * 0.25
            + inhibitory_control * 0.18
            + signal_precision * 0.12
            - prediction_error * 0.22
            - mismatch_index * 0.18
            + dream_plasticity * 0.04
        )
        bias_suppression_score = _clamp(inhibitory_control * 0.65 + signal_precision * 0.2 + prior_precision * 0.15 - mismatch_index * 0.15)
        candidate_temperature = _clamp(1.0 - deductive_gate * 0.25 + revision_pressure * 0.35, 0.35, 1.5)
        belief_update_rate = _clamp(pi_score * 0.45 + dream_plasticity * 0.25 + convergence_index * 0.2 - conflict_load * 0.2)

        return {
            "pi_score": round(pi_score, 3),
            "precision_weighted_coherence": round(_clamp(precision_weighted_coherence), 3),
            "prediction_error": round(_clamp(prediction_error), 3),
            "cross_area_synchrony": round(_clamp(cross_area_synchrony), 3),
            "mismatch_ratio": round(_clamp(mismatch_ratio), 3),
            "signal_precision": round(_clamp(signal_precision), 3),
            "prior_precision": round(_clamp(prior_precision), 3),
            "conflict_load": round(_clamp(conflict_load), 3),
            "convergence_index": round(convergence_index, 3),
            "mismatch_index": round(mismatch_index, 3),
            "inhibitory_control": round(inhibitory_control, 3),
            "deductive_gate": round(deductive_gate, 3),
            "revision_pressure": round(revision_pressure, 3),
            "dream_plasticity": round(dream_plasticity, 3),
            "intuition_gain": round(intuition_gain, 3),
            "bias_suppression_score": round(bias_suppression_score, 3),
            "candidate_temperature": round(candidate_temperature, 3),
            "belief_update_rate": round(belief_update_rate, 3),
            "coherent_signal_count": coherent_signal_count,
            "mismatch_signal_count": mismatch_signal_count,
            "interaction_profiles": interaction_profiles,
            "area_state": area_state,
            "interpretation": "computational_abstraction_not_literal_neurobiology_or_clinical_validation",
        }
