from melampo.reasoning.differential_engine import DifferentialEngine


def test_differential_engine_counts_evidence():
    engine = DifferentialEngine()
    result = engine.rank(["finding_a", "finding_b"])
    assert result["evidence_count"] == 2
