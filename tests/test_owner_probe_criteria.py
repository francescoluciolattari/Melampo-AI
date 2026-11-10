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
