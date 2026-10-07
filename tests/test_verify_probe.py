import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "verify_probe", ROOT / "scripts" / "verify_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TEXT = "The thyroid appeared normal. She was tachycardic with a heart beat of 120. The liver was enlarged."


def _items():
    items = []
    for n, mention in enumerate(("thyroid", "heart", "liver")):
        start = TEXT.index(mention)
        items.append(
            {
                "item_id": f"G{n}",
                "report_id": "r1",
                "mention": mention,
                "start": start,
                "end": start + len(mention),
                "language": "en",
            }
        )
    return items


def test_probe_asks_only_links_made_from_the_name_and_reports_the_stops(tmp_path):
    probe = _load()
    answers = {"thyroid": "NO"}  # the models reject only the thyroid sentence

    def chat(prompt):
        for word, answer in answers.items():
            if f"<tgt>{word}</tgt>" in prompt:
                return answer
        return "YES"

    plain, asking, flags = probe.build_linkers(
        tmp_path / "none.obo", {"a": chat, "b": chat}, "all"
    )
    rows = probe.probe(_items(), {"r1": TEXT}, plain, asking, flags, workers=1)
    by_id = {r["item_id"]: r for r in rows}
    assert by_id["G0"]["asked"] and by_id["G0"]["after"]["status"] == "abstained"
    assert by_id["G0"]["after"]["reason"] == "context_check_failed:no"
    assert by_id["G2"]["asked"] and by_id["G2"]["after"]["status"] == "accepted"
    summary = probe.summarise(rows)
    assert summary["asked"] == sum(1 for r in rows if r.get("asked"))
    assert summary["outcome"]["context_check_failed:no"] == 1
    text = probe.render(rows, summary)
    assert "Stopped by the check" in text and "G0" in text
    assert "WARNING" not in text


def test_a_model_that_does_not_answer_is_counted_and_warned(tmp_path):
    probe = _load()

    def down(prompt):
        raise RuntimeError("HTTP 429")

    plain, asking, flags = probe.build_linkers(
        tmp_path / "none.obo", {"a": down, "b": down}, "all"
    )
    rows = probe.probe(_items(), {"r1": TEXT}, plain, asking, flags, workers=1)
    summary = probe.summarise(rows)
    assert summary["outcome"]["model_unavailable"] == summary["asked"] > 0
    assert "WARNING" in probe.render(rows, summary)


def test_without_a_key_the_script_refuses_to_run(tmp_path, monkeypatch, capsys):
    probe = _load()
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    items = tmp_path / "items.jsonl"
    items.write_text(json.dumps(_items()[0]) + "\n", encoding="utf-8")
    code = probe.main(["--items", str(items), "--reports", str(items)])
    assert code == 2
