"""Error levels, granularity states, calibration metrics, provenance (12 Oct 2026)."""


from melampo.evaluation import gold_set as gs
from melampo.evaluation import selective_calibration as sc
from melampo.evaluation.provenance import Provenance
from melampo.memory import anatomy_linker as al


def _result(status=al.ACCEPTED, cid="kidney_left", relation="equal", reason="lexicon", fallback=""):
    return al.LinkResult(status, cid, "lexicon", reason, relation=relation, fallback=fallback)


def test_match_states():
    assert _result().match == "exact"
    assert _result(relation="approx").match == "approximate"
    assert _result(relation="part_of").match == "approximate"
    assert _result(reason="lexicon+graph_equal").match == "parent_candidate"
    assert _result(status=al.ABSTAINED, cid=None, fallback="lung_left").match == "parent_candidate"
    assert _result(status=al.ABSTAINED, cid=None).match == "abstain"


def test_error_levels():
    row = {"structure": "kidney_right", "mention": "rene", "role": "structure", "span_text": ""}
    assert gs.error_level(row, _result(), "wrong_critical", "structure") == "concept:side"
    assert gs.error_level({**row, "structure": "liver"}, _result(), "wrong_critical", "structure") == "concept"
    assert gs.error_level({**row, "span_text": "rene policistico"}, _result(), "wrong_critical", "structure") == "span"
    na = {"structure": gs.NOT_ANATOMY, "mention": "heart", "role": "inherent_location"}
    assert gs.error_level(na, _result(), "wrong_critical", "structure") == "type"
    assert gs.error_level({**na, "role": "not_a_body_site"}, _result(), "wrong_critical", "structure") == "sense"
    assert gs.error_level(row, _result(), "wrong_relation", "structure") == "relation"
    right = {**row, "structure": "kidney_left", "role": "procedure_site"}
    assert gs.error_level(right, _result(), "correct", "structure") == "role"
    assert gs.error_level(right, _result(), "correct", "procedure_site") == ""
    assert gs.error_level({**row, "error_level": "export_convention"}, _result(), "wrong_critical", "x") == "export_convention"


def test_brier_ece_reliability():
    probs, ok = [0.9, 0.9, 0.1, 0.1], [True, True, False, False]
    assert abs(sc.brier(probs, ok) - 0.01) < 1e-9
    assert abs(sc.expected_calibration_error(probs, ok) - 0.1) < 1e-9
    over = sc.expected_calibration_error([0.9] * 10, [True] * 5 + [False] * 5)
    assert abs(over - 0.4) < 1e-9
    assert sc.reliability(probs, ok)[0]["n"] == 2


def test_coverage_risk_and_upper_bound():
    curve = sc.coverage_risk([3, 3, 2, 1], [True, True, False, True], total=8)
    assert [p["kept"] for p in curve] == [2, 3, 4]
    assert curve[0]["risk"] == 0.0 and curve[1]["errors"] == 1 and curve[2]["coverage"] == 0.5
    # zero errors in 299 links: the bound is just under 1% (the rule of three)
    assert 0.0095 < sc.risk_upper_bound(0, 299) < 0.0101


def test_grouped_bootstrap_resamples_reports():
    groups = ["r1"] * 10 + ["r2"] * 10
    ok = [True] * 10 + [False] * 10
    got = sc.grouped_bootstrap(groups, ok, n=500)
    assert got["groups"] == 2 and got["items"] == 20 and got["error"] == 0.5
    assert got["low"] == 0.0 and got["high"] == 1.0  # two reports: the interval is honest about it


def test_provenance_marks_items_touched_by_a_rule_written_after_an_error(tmp_path):
    path = tmp_path / "p.json"
    path.write_text('{"entries": [{"id": "gb", "post_error": true, "surfaces": ["GB"]},'
                    ' {"id": "t", "post_error": false, "surfaces": ["liver"]}]}', "utf-8")
    p = Provenance.load(path)
    assert p.touches("GB", "GB 5040/mmc") == ["gb"]
    assert p.touches("liver", "the liver is normal") == []  # not post-error
    assert p.touches("gbx", "") == []
    assert Provenance.load().post_error()  # the repository file is readable


def test_evaluate_reports_levels_states_and_risk():
    gold = [
        {"item_id": "1", "report_id": "r1", "sentence": "Il rene destro è normale.", "mention": "rene destro",
         "structure": "kidney_right", "relation": "equal", "role": "structure"},
        {"item_id": "2", "report_id": "r1", "sentence": "Frequenza cardiaca normale.", "mention": "cardiaca",
         "structure": gs.NOT_ANATOMY, "relation": "equal", "role": "inherent_location"},
    ]

    class Stub:
        def link(self, mention, sentence, **_):
            return _result(cid="kidney_right") if "rene" in mention else _result(cid="heart")

    out = gs.evaluate(Stub(), gold)
    assert out["error_levels"] == {"type": 1}
    assert out["match_states"] == {"exact": 2}
    assert out["coverage_risk"][0]["kept"] == 2
    assert out["error_by_report_bootstrap"]["groups"] == 1
    assert out["errors"][0]["level"] == "type"


def test_plural_keys_and_the_lexicon_reads_plurals():
    import json
    from pathlib import Path

    from melampo.memory import anatomy_parts as ap
    from melampo.memory.word_senses import plural_keys

    assert ("livers",) in plural_keys(("liver",), "en")
    assert ("femora",) in plural_keys(("femur",), "en") and ("bronchi",) in plural_keys(("bronchus",), "en")
    assert ("vertebrae",) in plural_keys(("vertebra",), "en")
    assert plural_keys(("rene",), "it") == (("reni",),)
    assert plural_keys(("vena", "porta"), "it") == ()  # Italian names of more words are not inflected
    data = Path(__file__).resolve().parents[1] / "data" / "linking"
    lexicon = al.Lexicon.from_json(json.loads((data / "anatomy_lexicon.json").read_text("utf-8")))
    parts = ap.PartTable.from_json(json.loads((data / "anatomy_parts.json").read_text("utf-8")), lexicon)
    linker = al.AnatomyLinker(lexicon, [], {}, parts=parts)
    got = linker.link("livers", "The livers of the mutant mice were enlarged.")
    assert (got.status, got.cid) == (al.ACCEPTED, "liver")
    # a plural of an ambiguous form is judged by the form's senses, never linked straight
    axes = linker.link("axes", "Multiple pituitary hormone axes dysfunction occurs in SAH.")
    assert axes.status != al.ACCEPTED
