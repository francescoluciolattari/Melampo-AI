from melampo.training.replay_filter import ReplayFilter


def test_replay_filter_accepts_safe_sample():
    filt = ReplayFilter(min_coherence=0.7, max_risk=0.3)
    assert filt.accept(coherence=0.9, risk=0.1) is True
    assert filt.accept(coherence=0.5, risk=0.1) is False
