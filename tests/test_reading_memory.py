"""The manual, the memory of verified examples and the readers' adapters (11 November 2026)."""

import json
import sys
from pathlib import Path

from melampo.evaluation.llm_reading import claims_prompt
from melampo.memory import reading_memory as rm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _ex(i, sample, sentence, structure="kidney_left", claims=("finding",)):
    return rm.Example(item_id=i, sample=sample, structure=structure, claims=list(claims), sentence=sentence,
                      mention="left kidney")


def test_the_nearest_examples_never_come_from_the_sample_measured_nor_the_sentence_itself():
    memory = rm.ReadingMemory([_ex("a", "s5", "A hypodense lesion in the left kidney (cyst?)", claims=("finding",
                                                                                                        "uncertain")),
                               _ex("b", "s6", "A hypodense lesion in the left kidney (cyst?) seen"),
                               _ex("c", "s5", "Calibration of the esophagus was normal", "esophagus", ("normal",))],
                              exclude_samples=("s6",))
    got = memory.nearest("A hypodense lesion in the left kidney (cyst?)", "kidney_left", k=2)
    assert [e.item_id for e in got] == ["c"]          # "a" is the sentence itself, "b" is of the sample measured
    got = memory.nearest("A small hypodense lesion in the left kidney", "kidney_left", k=1)
    assert got[0].item_id == "a" and "uncertainty below high" in got[0].line()


def test_the_manual_and_the_examples_go_with_the_question():
    q = claims_prompt("A cyst in the left kidney", 10, 21, "left kidney", manual="MANUAL TEXT",
                      examples=["- Sentence: x"])
    assert "Reading manual" in q and "MANUAL TEXT" in q and "already read and verified" in q
    assert q.index("MANUAL TEXT") < q.index("Answer with JSON only")
    assert rm.manual_version() and len(rm.manual_version()) == 12


def test_an_example_enters_only_when_the_label_and_every_reading_agree():
    import owner_probe as op

    def row(gold, verdict="here", a_unc=(), b_unc=(), doubt=False):
        about = lambda ws: [{"words": w, "level": "possible"} for w in ws]  # noqa: E731
        return {"gold_claims": gold, "gold_doubt": doubt,
                "claims": {"verdict": verdict, "statement": "finding", "doubt": {},
                           "claims": {"findings": [{"of": "it"}], "uncertainty": list(a_unc)},
                           "models": {"m1": {"uncertainty": list(a_unc), "uncertain_about": about(a_unc)},
                                      "m2": {"uncertainty": list(b_unc), "uncertain_about": about(b_unc)}}}}

    assert rm.verified(row(["finding", "uncertain"], a_unc=["?"], b_unc=["?"]), op.outcome_of_claims,
                       op.attached_kinds)
    assert not rm.verified(row(["finding", "uncertain"], a_unc=["?"]), op.outcome_of_claims, op.attached_kinds)
    assert not rm.verified(row(["finding"], doubt=True), op.outcome_of_claims, op.attached_kinds)
    assert not rm.verified(row(["finding"], verdict="doubt"), op.outcome_of_claims, op.attached_kinds)
    assert not rm.verified(row(["aside"]), op.outcome_of_claims, op.attached_kinds)


def test_an_adapter_enters_only_if_every_criterion_passes_and_it_errs_less(tmp_path, monkeypatch):
    import reader_training as rt

    def crit(given, person=0):
        return {"items": 293, "1_aside_attached_as_certain": 0, "2_findings_set_aside": 0,
                "3_to_a_person_share": person / 293, "4_certain_attached_otherwise_share": 0.0,
                "6_doubt_without_the_right_reading_share": 0.0, "7_uncertainty_lost": 1, "7_uncertainty_lost_share":
                1 / 293, "8_uncertainty_given": given, "8_uncertainty_given_share": given / 293,
                "counts": {"3_to_a_person": person}}

    def report(path, c):
        path.write_text(json.dumps({"by_set": {"s9": {"criteria": {"without_doubt": c}}}}))
        return str(path)

    monkeypatch.setattr(rt, "LOG", tmp_path / "log.yml")
    base, better, worse = (report(tmp_path / n, c) for n, c in (("b.json", crit(4)), ("a.json", crit(2)),
                                                                 ("w.json", crit(9))))
    import argparse
    rt.cmd_decide(argparse.Namespace(base=base, adapter=better, set="s9", name="r1"))
    rt.cmd_decide(argparse.Namespace(base=base, adapter=worse, set="s9", name="r2"))
    log = rt._log()
    assert [t["enters"] for t in log] == [True, False]
    assert "does not pass" in log[1]["why"]


def test_the_target_is_the_whole_verified_answer():
    import reader_training as rt

    got = rt.target({"marked": "left kidney", "observed": [{"words": "lesion", "of": "it"}],
                     "uncertain_about": [{"words": "?", "about": "cyst", "level": "possible"}],
                     "undetermined": ["nonspecific"], "measure": ["12 mm"], "measure_refs": []})
    assert got["uncertainty"] == [{"words": "?", "about": "cyst", "level": "possible"}]
    assert got["undetermined"] == ["nonspecific"] and got["measure"][0]["words"] == "12 mm"


def test_the_second_labeller_answer_and_the_agreement():
    import second_labeller as sl

    assert sl.read_answer('```json\n{"claims": ["finding", "uncertain", "bogus"], "why": "a|b"}\n```') == \
        (["finding", "uncertain"], "a/b")
    # the independent question (12 November 2026): plain words, mapped to the labels
    assert sl.read_answer('{"says": ["abnormal_in_it", "its_size"], "certainty": "not_certain", "why": "x"}') == \
        (["finding", "measure", "uncertain"], "x")
    assert sl.read_answer('{"says": ["only_where_something_else_is"], "certainty": "very_likely"}')[0] == \
        ["aside", "uncertain_high"]
    assert sl.read_answer('{"says": ["abnormal_in_it"], "certainty": "certain"}')[0] == ["finding"]
    assert sl.read_answer('{"says": ["abnormal_in_it"]}')[0] == []       # no certainty, no label
    prompt = sl.prompt({"clause": "A cyst.", "mention": "kidney", "structure": "kidney_left"})
    assert "consistent with" not in prompt and "Convention" not in prompt and "manual" not in prompt.lower()
    assert sl._kappa([("a", "a"), ("b", "b")]) == 1.0
    assert sl._kappa([("a", "b"), ("b", "a")]) < 0
