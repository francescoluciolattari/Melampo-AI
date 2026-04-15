from melampo.reasoning.decision_trace import DecisionTrace


def test_decision_trace_collects_steps():
    trace = DecisionTrace()
    trace.add("step_a")
    trace.add("step_b")
    assert trace.dump() == ["step_a", "step_b"]
