from __future__ import annotations

from dataclasses import dataclass


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass(slots=True)
class NeuroDynamicMetrics:
    """Neuro-inspired interaction metrics for Melampo functional areas.

    This module is a computational abstraction, not a literal biological claim.
    It operationalizes predictive-processing style concepts that are useful for
    a clinical reasoning prototype: precision-weighted coherence, prediction
    error, cross-area synchrony, conflict load, and a predictive-inference score.
    """

    precision_gain: float = 0.6
    mismatch_gain: float = 0.4
    synchrony_gain: float = 0.25
    conflict_gain: float = 0.35

    def compute(
        self,
        pair_profiles: list[dict],
        coherence_score: float,
        mismatch_score: float,
        total_salience: float,
        dream_pressure: float = 0.0,
    ) -> dict:
        coherent_salience = 0.0
        mismatch_salience = 0.0
        coherent_signal_count = 0
        mismatch_signal_count = 0
        for profile in pair_profiles:
            salience = float(profile.get("pair_salience", 0.0))
            signals = int(profile.get("pair_signal_count", 0))
            if profile.get("status") == "coherent":
                coherent_salience += salience
                coherent_signal_count += signals
            else:
                mismatch_salience += salience
                mismatch_signal_count += signals

        pair_salience_total = max(coherent_salience + mismatch_salience, 1e-6)
        cross_area_synchrony = coherent_salience / pair_salience_total
        mismatch_ratio = mismatch_salience / pair_salience_total
        signal_precision = total_salience / max(total_salience + mismatch_signal_count + 1.0, 1.0)
        precision_weighted_coherence = coherence_score * (self.precision_gain + signal_precision)
        prediction_error = mismatch_score * (self.mismatch_gain + mismatch_ratio)
        conflict_load = prediction_error + (mismatch_signal_count / max(coherent_signal_count + mismatch_signal_count, 1)) * self.conflict_gain
        deductive_gate = _clamp(
            precision_weighted_coherence
            + cross_area_synchrony * self.synchrony_gain
            - prediction_error * self.conflict_gain
        )
        revision_pressure = _clamp(prediction_error + conflict_load * 0.35 + dream_pressure * 0.2)
        intuition_gain = _clamp(1.0 + deductive_gate * 0.25 - revision_pressure * 0.15, 0.5, 1.35)
        pi_score = _clamp(
            precision_weighted_coherence * 0.45
            + cross_area_synchrony * 0.25
            + signal_precision * 0.2
            - prediction_error * 0.25
            - dream_pressure * 0.05
        )
        bias_suppression_score = _clamp(1.0 - conflict_load * 0.5 + signal_precision * 0.2)
        return {
            "pi_score": round(pi_score, 3),
            "precision_weighted_coherence": round(_clamp(precision_weighted_coherence), 3),
            "prediction_error": round(_clamp(prediction_error), 3),
            "cross_area_synchrony": round(_clamp(cross_area_synchrony), 3),
            "mismatch_ratio": round(_clamp(mismatch_ratio), 3),
            "signal_precision": round(_clamp(signal_precision), 3),
            "conflict_load": round(_clamp(conflict_load), 3),
            "deductive_gate": round(deductive_gate, 3),
            "revision_pressure": round(revision_pressure, 3),
            "intuition_gain": round(intuition_gain, 3),
            "bias_suppression_score": round(bias_suppression_score, 3),
            "coherent_signal_count": coherent_signal_count,
            "mismatch_signal_count": mismatch_signal_count,
            "interpretation": "computational_abstraction_not_literal_neurobiology",
        }
