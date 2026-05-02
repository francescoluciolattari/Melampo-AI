from dataclasses import dataclass, field

from .neuro_dynamics import NeuroDynamicMetrics


@dataclass
class AreaCoherenceAnalyzer:
    """Compute coherence, mismatch, and neuro-inspired interaction metrics across functional areas."""

    metrics: NeuroDynamicMetrics = field(default_factory=NeuroDynamicMetrics)

    COHERENT_PAIRS = {
        ("language_listening", "visual_diagnostic"),
        ("epidemiology", "visual_diagnostic"),
        ("case_context", "language_listening"),
        ("case_context", "epidemiology"),
        ("language_listening", "epidemiology"),
        ("case_context", "visual_diagnostic"),
    }

    def analyze(self, area_signals: dict, dream_pressure: float = 0.0) -> dict:
        names = sorted(area_signals.keys())
        coherence_pairs = []
        mismatch_pairs = []
        pair_profiles = []
        total_salience = 0.0
        for payload in area_signals.values():
            if isinstance(payload, dict):
                total_salience += float(payload.get("salience_score", 0.0))

        for index, name in enumerate(names):
            for other in names[index + 1 :]:
                pair = tuple(sorted([name, other]))
                first = area_signals.get(name, {}) if isinstance(area_signals.get(name, {}), dict) else {}
                second = area_signals.get(other, {}) if isinstance(area_signals.get(other, {}), dict) else {}
                pair_salience = round(float(first.get("salience_score", 0.0)) + float(second.get("salience_score", 0.0)), 3)
                pair_signal_count = int(first.get("signal_count", 0)) + int(second.get("signal_count", 0))
                first_focus = first.get("focus", name)
                second_focus = second.get("focus", other)
                status = "coherent" if pair in self.COHERENT_PAIRS else "mismatch"
                if status == "coherent":
                    coherence_pairs.append(pair)
                else:
                    mismatch_pairs.append(pair)
                pair_profiles.append({
                    "pair": pair,
                    "status": status,
                    "pair_salience": pair_salience,
                    "pair_signal_count": pair_signal_count,
                    "focus_alignment": first_focus == second_focus,
                    "first_focus": first_focus,
                    "second_focus": second_focus,
                })
        coherence_score = round((len(coherence_pairs) / max(len(names), 1)) + total_salience * 0.05, 3)
        mismatch_score = round((len(mismatch_pairs) / max(len(names), 1)), 3)
        neuro_dynamic_metrics = self.metrics.compute(
            pair_profiles=pair_profiles,
            coherence_score=coherence_score,
            mismatch_score=mismatch_score,
            total_salience=total_salience,
            dream_pressure=dream_pressure,
            area_signals=area_signals,
        )
        return {
            "coherence_pairs": coherence_pairs,
            "mismatch_pairs": mismatch_pairs,
            "pair_profiles": pair_profiles,
            "coherence_score": coherence_score,
            "mismatch_score": mismatch_score,
            "total_salience": round(total_salience, 3),
            "neuro_dynamic_metrics": neuro_dynamic_metrics,
            "pi_score": neuro_dynamic_metrics["pi_score"],
            "prediction_error": neuro_dynamic_metrics["prediction_error"],
            "precision_weighted_coherence": neuro_dynamic_metrics["precision_weighted_coherence"],
            "convergence_index": neuro_dynamic_metrics["convergence_index"],
            "mismatch_index": neuro_dynamic_metrics["mismatch_index"],
            "deductive_gate": neuro_dynamic_metrics["deductive_gate"],
            "revision_pressure": neuro_dynamic_metrics["revision_pressure"],
            "dream_plasticity": neuro_dynamic_metrics["dream_plasticity"],
        }
