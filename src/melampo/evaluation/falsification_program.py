from dataclasses import dataclass


@dataclass
class FalsificationProgram:
    """Track experimental claims and minimum falsification criteria."""

    def summarize(self) -> dict:
        return {
            "quantum_like": "requires comparative benchmark",
            "open_systems": "requires replicated gains",
            "biophysical_frontier": "requires independent laboratory evidence",
        }
