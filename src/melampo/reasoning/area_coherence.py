from dataclasses import dataclass


@dataclass
class AreaCoherenceAnalyzer:
    """Compute lightweight coherence and mismatch signals across functional areas."""

    def analyze(self, area_signals: dict) -> dict:
        names = sorted(area_signals.keys())
        coherence_pairs = []
        mismatch_pairs = []
        for index, name in enumerate(names):
            for other in names[index + 1 :]:
                pair = tuple(sorted([name, other]))
                if pair in [
                    ("language_listening", "visual_diagnostic"),
                    ("epidemiology", "visual_diagnostic"),
                    ("case_context", "language_listening"),
                ]:
                    coherence_pairs.append(pair)
                else:
                    mismatch_pairs.append(pair)
        coherence_score = round(len(coherence_pairs) / max(len(names), 1), 3)
        mismatch_score = round(len(mismatch_pairs) / max(len(names), 1), 3)
        return {
            "coherence_pairs": coherence_pairs,
            "mismatch_pairs": mismatch_pairs,
            "coherence_score": coherence_score,
            "mismatch_score": mismatch_score,
        }
