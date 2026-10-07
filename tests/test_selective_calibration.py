import math

import pytest

from melampo.evaluation import selective_calibration as sc


def test_binomial_cdf_matches_the_direct_sum():
    n, p = 40, 0.03
    for k in (0, 1, 3, 10):
        direct = sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))
        assert sc.binomial_cdf(k, n, p) == pytest.approx(direct, rel=1e-9)


def test_with_no_error_one_percent_at_five_percent_needs_299_links():
    assert sc.links_needed(0.01, 0.05) == 299
    assert sc.p_value(0, 299, 0.01) <= 0.05 < sc.p_value(0, 298, 0.01)


def test_more_errors_need_more_links():
    assert sc.links_needed(0.01, 0.05, 1) > sc.links_needed(0.01, 0.05, 0)


def test_the_loosest_threshold_that_passes_is_certified_and_testing_stops_at_the_first_failure():
    # 400 links with score 3 and no error, 100 with score 2 and 5 errors, 100 with score 1 and 20.
    scores = [3] * 400 + [2] * 100 + [1] * 100
    correct = [True] * 400 + [True] * 95 + [False] * 5 + [True] * 80 + [False] * 20
    cert = sc.certify(scores, correct, alpha=0.01, delta=0.05)
    assert cert.threshold == 3 and cert.kept == 400 and cert.errors == 0
    assert [s.certified for s in cert.steps] == [True, False]  # score 1 never tested
    assert cert.coverage == pytest.approx(400 / 600)


def test_too_few_links_certify_nothing():
    cert = sc.certify([2] * 100, [True] * 100)
    assert cert.threshold is None and cert.kept == 0


def test_mondrian_certificates_hide_no_bad_stratum():
    rows = (
        [("en|lexicon", 2, True)] * 1500
        + [("it|lexicon", 2, True)] * 50
        + [("it|lexicon", 2, False)] * 2
    )
    pooled = sc.certify([s for _, s, _ in rows], [ok for _, _, ok in rows])
    by = sc.certify_by_stratum(rows)
    assert pooled.threshold == 2  # the average passes
    assert by["en|lexicon"].threshold == 2
    assert by["it|lexicon"].threshold is None  # the Italian stratum does not


def test_simultaneous_strata_split_delta():
    rows = [("a", 1, True)] * 299 + [("b", 1, True)] * 299
    assert sc.certify_by_stratum(rows)["a"].threshold == 1
    assert sc.certify_by_stratum(rows, simultaneous=True)["a"].threshold is None
