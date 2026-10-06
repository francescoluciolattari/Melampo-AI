"""Tests of the gold-set instrument. The reports below are SYNTHETIC fixtures for the code, not a gold set."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from melampo.evaluation import gold_set as gs
from melampo.memory import anatomy_linker as al
from melampo.memory import anatomy_parts as ap

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "linking"


@pytest.fixture(scope="module")
def lexicon():
    return al.Lexicon.from_json(
        json.loads((DATA / "anatomy_lexicon.json").read_text("utf-8"))
    )


@pytest.fixture(scope="module")
def parts(lexicon):
    return ap.PartTable.from_json(
        json.loads((DATA / "anatomy_parts.json").read_text("utf-8")), lexicon
    )


REPORTS = [
    {
        "report_id": "R1",
        "text": "Fegato di dimensioni regolari. Cisti del rene destro. Milza nei limiti.",
    },
    {
        "report_id": "R2",
        "text": "The liver is normal. There is a cyst in the left kidney. No ascites.",
    },
    {
        "report_id": "R3",
        "text": "Nessuna lesione focale. Referto privo di strutture note.",
    },
    {"report_id": "R4", "text": "Ispessimento del sigma con diverticoli."},
]


def test_language_is_guessed_from_function_words():
    assert gs.language_of("Il fegato è di dimensioni regolari con la milza") == "it"
    assert gs.language_of("The liver is normal and there is a cyst") == "en"


def test_known_names_are_proposed_with_their_side(lexicon, parts):
    found = gs.propose_mentions(REPORTS[0]["text"], lexicon, parts)
    assert [m["mention"] for m in found] == [
        "Fegato",
        "Cisti del rene destro",
        "Milza",
    ]
    text = REPORTS[0]["text"]
    assert all(text[m["start"] : m["end"]] == m["mention"] for m in found)


def test_part_names_of_the_table_are_proposed(lexicon, parts):
    assert [
        m["mention"] for m in gs.propose_mentions(REPORTS[3]["text"], lexicon, parts)
    ] == ["sigma"]


def test_a_report_without_known_names_gives_nothing(lexicon, parts):
    assert gs.propose_mentions(REPORTS[2]["text"], lexicon, parts) == []


def test_one_mention_per_report_and_the_seed_makes_it_repeatable(lexicon, parts):
    a = gs.sample_items(REPORTS, lexicon, parts, seed=1)
    b = gs.sample_items(REPORTS, lexicon, parts, seed=1)
    assert a == b
    assert sorted(i["report_id"] for i in a) == ["R1", "R2", "R4"]
    assert len({i["item_id"] for i in a}) == len(a)
    assert {i["language"] for i in a} == {"it", "en"}


def test_n_limits_the_sample_and_keeps_both_languages(lexicon, parts):
    reports = [
        {"report_id": f"I{i}", "text": f"Il fegato è regolare numero {i} con la milza."}
        for i in range(6)
    ]
    reports += [
        {
            "report_id": f"E{i}",
            "text": f"The liver is normal number {i} with the spleen.",
        }
        for i in range(6)
    ]
    items = gs.sample_items(reports, lexicon, parts, n=4, seed=3)
    assert len(items) == 4
    assert {i["language"] for i in items} == {"it", "en"}


def test_the_sentence_is_the_one_holding_the_mention():
    text = "Fegato regolare. Cisti del rene destro. Milza nei limiti."
    start = text.index("rene")
    assert gs.sentence_of(text, start, start + 11) == "Cisti del rene destro."


def test_sheets_are_blind_and_differently_ordered(lexicon, parts, tmp_path):
    items = gs.sample_items(REPORTS, lexicon, parts, seed=1)
    paths = gs.write_sheets(items, tmp_path, class_ids=sorted(lexicon.classes))
    a, b = (gs.read_sheet(p) for p in paths[:2])
    assert set(a) == set(b) == {i["item_id"] for i in items}
    header = paths[0].read_text("utf-8-sig").splitlines()[0].split(",")
    assert not {"system", "cid", "suggestion", "linker", "reason"} & set(header)
    assert all(row["structure"] == "" for row in a.values())
    valid = paths[2].read_text("utf-8").splitlines()
    assert {"spleen", gs.NONE_IN_CLASSES, gs.NOT_ANATOMY, gs.AMBIGUOUS} <= set(valid)

    def order(p):
        return [r["item_id"] for r in csv.DictReader(p.open(encoding="utf-8-sig"))]

    assert sorted(order(paths[0])) == sorted(order(paths[1]))


def _row(item_id, structure, relation="equal", mention="m", sentence="s"):
    return {"item_id": item_id, "report_id": "R", "language": "it", "sentence": sentence, "mention": mention,
            "start": "0", "end": "1", "structure": structure, "relation": relation, "side_in_text": "", "note": ""}  # fmt: skip


def test_validation_flags_empty_unknown_and_missing_relation(lexicon):
    rows = {
        "1": _row("1", ""),
        "2": _row("2", "liverr"),
        "3": _row("3", "liver", relation=""),
        "4": _row("4", "liver", relation="part_of"),
        "5": _row("5", gs.AMBIGUOUS, relation=""),
    }
    problems = gs.validate_sheet(rows, lexicon.classes)
    assert len(problems) == 3
    assert any("1:" in p for p in problems) and any("liverr" in p for p in problems)


def test_kappa_agreement_and_the_adjudication_queue():
    a = {
        str(i): _row(str(i), s)
        for i, s in enumerate(["liver", "spleen", "liver", "colon"])
    }
    b = {
        str(i): _row(str(i), s)
        for i, s in enumerate(["liver", "spleen", "colon", "colon"])
    }
    report = gs.agreement(a, b)
    assert report["items"] == 4 and report["structure_agreement"] == 0.75
    assert 0 < report["structure_kappa"] < 1
    queue = gs.adjudication_queue(a, b)
    assert [q["item_id"] for q in queue] == ["2"]
    assert (queue[0]["structure_A"], queue[0]["structure_B"]) == ("liver", "colon")


def test_a_relation_disagreement_also_goes_to_the_third_reviewer():
    a = {"1": _row("1", "colon", "part_of")}
    b = {"1": _row("1", "colon", "equal")}
    assert len(gs.adjudication_queue(a, b)) == 1


def test_merge_takes_agreed_labels_then_the_adjudicators_and_leaves_open_items_out():
    a = {"1": _row("1", "liver"), "2": _row("2", "liver"), "3": _row("3", "colon")}
    b = {"1": _row("1", "liver"), "2": _row("2", "spleen"), "3": _row("3", "spleen")}
    adjudicated = {"2": _row("2", "spleen")}
    gold = gs.merge_gold(a, b, adjudicated)
    assert {g["item_id"]: (g["structure"], g["agreed"]) for g in gold} == {
        "1": ("liver", True),
        "2": ("spleen", False),
    }


class _Stub:
    """A linker that answers from a table, to test the scoring and not the linker."""

    def __init__(self, answers):
        self.answers = answers

    def link(self, mention, sentence):
        cid, relation = self.answers.get(mention, (None, "equal"))
        if cid is None:
            return al.LinkResult(al.ABSTAINED, None, "lexicon", "unknown_name")
        return al.LinkResult(
            al.ACCEPTED, cid, "lexicon", "recognised", relation=relation
        )


def _gold(mention, structure, relation="equal"):
    return {
        "item_id": mention,
        "mention": mention,
        "sentence": "s",
        "structure": structure,
        "relation": relation,
        "language": "it",
    }


def test_evaluate_counts_every_outcome_separately():
    stub = _Stub(
        {
            "a": ("liver", "equal"),
            "b": ("colon", "equal"),
            "c": ("colon", "equal"),
            "d": ("stomach", "equal"),
            "e": ("heart", "equal"),
        }
    )
    gold = [
        _gold("a", "liver"),  # correct
        _gold("b", "spleen"),  # wrong critical
        _gold("c", "colon", "part_of"),  # wrong relation
        _gold("d", gs.NOT_ANATOMY),  # linked what should not be linked
        _gold(
            "e", gs.NONE_IN_CLASSES
        ),  # an outside term: listed for a human look, not counted as a class error
        _gold("f", "liver"),  # abstained
    ]
    report = gs.evaluate(stub, gold)
    assert report["outcomes"] == {
        "correct": 1,
        "wrong_critical": 2,
        "wrong_relation": 1,
        "accepted_outside_classes": 1,
        "abstained": 1,
    }
    assert report["accepted"] == 4 and report["wrong"] == 3
    assert {e["mention"] for e in report["errors"]} == {"b", "c", "d", "e"}
    assert report["coverage"] == round(4 / 6, 4)


def test_zero_errors_in_299_accepted_certify_one_percent_and_298_do_not():
    assert gs.cases_needed(0.01, 0.95) == 299
    assert "<= 1.00%" in gs.verdict(299, 0) or "<= 0.99%" in gs.verdict(299, 0)
    assert gs.verdict(298, 0).startswith("Not certified")
    assert gs.cases_needed(0.01, 0.95, 5) == 1049


def test_the_verdict_always_states_its_conditions():
    for accepted, wrong in ((0, 0), (500, 0), (50, 3)):
        text = gs.verdict(accepted, wrong)
        assert "random sample" in text and "never used while developing" in text


def test_power_of_the_test_matches_the_published_figures():
    assert gs.power(0.005, 299, 0) == pytest.approx(0.22, abs=0.01)
    assert gs.power(0.005, 1049, 5) == pytest.approx(0.57, abs=0.02)
    assert gs.power(0.005, 2000, 12) == pytest.approx(0.79, abs=0.02)


def test_command_line_runs_the_whole_study(tmp_path, monkeypatch, lexicon):
    spec = importlib.util.spec_from_file_location(
        "gold_cli", ROOT / "scripts" / "gold_set.py"
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    reports = tmp_path / "reports.jsonl"
    reports.write_text("\n".join(json.dumps(r) for r in REPORTS), encoding="utf-8")
    out = tmp_path / "study"
    assert cli.main(["sample", str(reports), "--out", str(out)]) == 0
    sheet_a, sheet_b = out / "annotator_A.csv", out / "annotator_B.csv"
    answers = {"Fegato": ("liver", "equal"), "Cisti del rene destro": ("kidney_cyst_right", "equal"), "Milza": ("spleen", "equal"),
               "The liver": ("liver", "equal"), "liver": ("liver", "equal"), "left kidney": ("kidney_left", "equal"),
               "sigma": ("colon", "part_of")}  # fmt: skip

    def fill(path, disagree=None):
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        for row in rows:
            row["structure"], row["relation"] = answers[row["mention"]]
            if disagree and row["mention"] == disagree:
                row["structure"] = "spleen"
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    fill(sheet_a)
    fill(sheet_b, disagree="sigma")
    assert cli.main(["check", str(sheet_a)]) == 0
    assert (
        cli.main(
            ["agree", str(sheet_a), str(sheet_b), "--out", str(tmp_path / "adj.csv")]
        )
        == 0
    )
    adjudication = list(
        csv.DictReader((tmp_path / "adj.csv").open(encoding="utf-8-sig"))
    )
    assert [r["mention"] for r in adjudication] == ["sigma"]
    adjudication[0]["structure"], adjudication[0]["relation"] = "colon", "part_of"
    with (tmp_path / "adj.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(adjudication[0]))
        writer.writeheader()
        writer.writerows(adjudication)
    gold_path = tmp_path / "gold.jsonl"
    assert (
        cli.main(
            [
                "merge",
                str(sheet_a),
                str(sheet_b),
                str(tmp_path / "adj.csv"),
                "--out",
                str(gold_path),
            ]
        )
        == 0
    )
    assert (
        cli.main(["evaluate", str(gold_path), "--out", str(tmp_path / "report.json")])
        == 0
    )
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["wrong"] == 0 and report["accepted"] == report["n"] == 3
    assert report["verdict"].startswith("Not certified")  # three items certify nothing


def test_a_part_word_is_not_trimmed_as_if_it_were_a_function_word():
    """ "colon discendente" must reach the annotators whole, not as "colon" (found in the E3C dry run)."""
    import json
    from pathlib import Path

    from melampo.memory import anatomy_linker as al
    from melampo.memory import anatomy_parts as ap

    data = Path(__file__).resolve().parent.parent / "data" / "linking"
    real = al.Lexicon.from_json(
        json.loads((data / "anatomy_lexicon.json").read_text("utf-8"))
    )
    table = ap.PartTable.from_json(
        json.loads((data / "anatomy_parts.json").read_text("utf-8")), real
    )
    text = "Dilatazione del colon discendente con diverticoli."
    assert [m["mention"] for m in gs.propose_mentions(text, real, table)] == [
        "colon discendente"
    ]
