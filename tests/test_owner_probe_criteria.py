"""Criterion 7 of the claims reading (10 November 2026): the uncertainty of the report, lost or kept."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from owner_probe import criteria


def _row(item, claims, verdict="here", doubt=None):
    return {"item_id": item, "gold_claims": claims, "gold_doubt": False, "language": "en",
            "claims": {"verdict": verdict, "statement": "finding", "claims": {"findings": [{"of": "it"}], **claims_of(
                verdict)}, "doubt": doubt or {}, "models": {}, "reader": ""}}


def claims_of(_):
    return {}


def test_the_uncertainty_of_the_report_lost_is_counted():
    kept = _row("a", ["finding", "uncertain"], doubt={"kind": "clinical"})
    lost = _row("b", ["finding", "uncertain"])
    plain = _row("c", ["finding"])
    in_doubt = _row("d", ["finding", "uncertain"], verdict="doubt", doubt={"kind": "reading", "readings": {}})
    in_doubt["claims"]["claims"]["uncertainty"] = ["?"]
    got = criteria([kept, lost, plain, in_doubt])["without_doubt"]
    assert got["7_uncertainty_lost"] == 1 and got["7_lost_items"] == ["b"]
    assert got["7_uncertain_items"] == 3 and got["7_uncertainty_lost_share"] == 0.25


def test_the_uncertainty_given_where_the_report_states_is_counted():
    given = _row("e", ["finding"])
    given["claims"]["claims"]["uncertainty"] = ["nonspecific"]
    high = _row("f", ["finding", "uncertain_high"])
    high["claims"]["claims"].update(uncertainty=["consistent with"],
                                    uncertain_about=[{"words": "consistent with", "about": "x", "level": "high"}])
    lower = _row("g", ["finding", "uncertain_high"])
    lower["claims"]["claims"].update(uncertainty=["consistent with"],
                                     uncertain_about=[{"words": "consistent with", "about": "x", "level": "possible"}])
    only_high = _row("h", ["finding", "uncertain"])
    only_high["claims"]["claims"].update(uncertainty=["?"],
                                         uncertain_about=[{"words": "?", "about": "x", "level": "high"}])
    got = criteria([given, high, lower, only_high])["without_doubt"]
    assert got["8_given_items"] == ["e", "g"]
    assert got["7_lost_items"] == ["h"]


def test_the_readings_of_the_uncertainty_are_measured_apart():
    from owner_probe import uncertainty_by_reading

    def row(item, gold, a, b):
        return {"item_id": item, "gold_claims": gold, "claims": {"verdict": "here", "models": {
            "m1": {"uncertain_about": a}, "m2": {"uncertain_about": b}}}}

    q = [{"words": "?", "level": "possible"}]
    rows = [row("a", ["finding", "uncertain"], q, q), row("b", ["finding"], [], [{"words": "stable"}])]
    rows += [row(f"c{i}", ["finding"], [], []) for i in range(98)]
    got = uncertainty_by_reading(rows)
    assert got["m1"]["8_given"] == 0 and got["m2"]["8_items"] == ["b"] and got["union"]["8_given"] == 1
    assert got["chosen"] == "m1"
