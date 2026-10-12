"""Point 1 probe: arms, the endpoint guard, scoring (no network)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import role_probe as rp  # noqa: E402


def item(role, structure="liver", stratum="e3c_en", report="r1", sentence="The liver biopsy was negative.", mention="liver"):
    at = sentence.index(mention)
    return {"item_id": f"{report}-{at}-{role}", "report_id": report, "stratum": stratum, "language": "en",
            "mention": mention, "sentence": sentence, "at": at, "start": at, "role": role, "structure": structure,
            "answers": {}}


def test_an_external_endpoint_needs_public_texts():
    with pytest.raises(SystemExit):
        rp.check_endpoint("https://openrouter.ai/api/v1/chat/completions", public_data=False)
    rp.check_endpoint("https://openrouter.ai/api/v1/chat/completions", public_data=True)
    rp.check_endpoint("http://localhost:8000/v1/chat/completions", public_data=False)


def test_the_gold_offset_is_moved_into_the_sentence():
    report = "Intro. The liver biopsy was negative."
    row = {"report_id": "r", "sentence": "The liver biopsy was negative.", "mention": "liver", "start": report.index("liver")}
    assert rp.place(row, {"r": report}) == 4


def test_each_llm_arm_asks_each_model_and_reader_lines_go_only_to_llm_plus_reader():
    seen = []

    def chat(prompt):
        seen.append(prompt)
        return json.dumps({"phrase": "liver biopsy", "head": "biopsy", "type": "procedure", "role": "procedure_site"})

    it = item("procedure_site")
    out = rp.ask_item(it, {"m": chat}, ["llm", "llm+reader"], ["- reader, readings: structure 0.60"], [])
    assert set(out) == {"llm:m", "llm+reader:m"} and all(a["valid"] for a in out.values())
    assert "reader, readings" not in seen[0] and "reader, readings" in seen[1]


def test_scores_count_uncertain_apart_and_critical_errors():
    items = [item("structure"), item("procedure_site"), item("not_a_body_site"), item("structure", report="r2")]
    for it, said in zip(items, ["structure", "structure", "structure", "uncertain"], strict=True):
        it["answers"]["a"] = said
    s = rp.score(items, "a")
    assert s["items"] == 4 and s["uncertain"] == 0.25
    assert s["accuracy"] == round(1 / 3, 4) and s["critical"] == 1 and s["said_structure_wrongly"] == 2


def test_examples_come_from_other_reports_and_include_a_non_structure_role():
    me = item("structure", report="r1")
    pool = [me, item("structure", report="r2"), item("structure", report="r3"),
            item("procedure_site", report="r4", sentence="A CT of the liver.")]
    ex = rp.examples_for(me, pool, 2)
    assert len(ex) == 2 and any(e["role"] == "procedure_site" for e in ex)
    assert all("<w>liver</w>" in e["marked"] for e in ex)


def test_answers_are_cached_by_model_and_prompt(tmp_path):
    calls = []
    cache = rp.Cache(tmp_path / "c.jsonl")
    chat = cache.wrap("m", lambda p: calls.append(p) or "{}")
    chat("q")
    chat("q")
    assert len(calls) == 1 and rp.Cache(tmp_path / "c.jsonl").data


def test_lora_folds_are_grouped_by_report_and_the_target_is_the_role_only(tmp_path):
    import lora_roles as lo

    assert lo.fold_of("r1") == lo.fold_of("r1") and 0 <= lo.fold_of("x") < lo.FOLDS
    gold = tmp_path / "g.jsonl"
    rows = [{"report_id": f"r{i % 7}", "sentence": "The liver biopsy was negative.", "mention": "liver", "start": 4,
             "item_id": f"i{i}", "stratum": "e3c_en", "language": "en", "structure": "liver",
             "role": "procedure_site" if i % 2 else "structure"} for i in range(20)]
    gold.write_text("\n".join(json.dumps(r) for r in rows), "utf-8")
    out = tmp_path / "d"
    assert lo.main(["prepare", "--gold", str(gold), "--out", str(out)]) == 0
    items = [json.loads(x) for x in (out / "items.jsonl").read_text().split("\n") if x]
    by_report = {}
    for it in items:
        by_report.setdefault(it["report_id"], set()).add(it["fold"])
        assert json.loads(it["completion"][0]["content"]) == {"role": it["role"]}
        assert '{"role": "<one of' in it["prompt"][0]["content"] and "<w>liver</w>" in it["prompt"][0]["content"]
    assert all(len(f) == 1 for f in by_report.values())


def test_the_role_is_read_from_the_last_json_object():
    from melampo.evaluation import llm_reading as lr

    assert lr.read_role('analysis {"x": 1} ... final {"role": "procedure_site"}') == "procedure_site"
    assert lr.read_role('{"role": "site"}') == "uncertain" and lr.read_role("nothing") == "uncertain"


def test_quantization_is_a_hard_filter_on_openrouter_and_recorded_in_the_name():
    q = rp.parse_quantization("openai/gpt-oss-120b=fp8")
    assert q == {"openai/gpt-oss-120b": "fp8"}
    with pytest.raises(SystemExit):
        rp.parse_quantization("openai/gpt-oss-120b=fp7")
    import os

    os.environ.setdefault("OPENROUTER_API_KEY", "test")
    chats = rp.make_chats(["openai/gpt-oss-120b", "mistralai/mistral-small-2603"],
                          "https://openrouter.ai/api/v1/chat/completions", rp.Cache(None), 100, q)
    assert set(chats) == {"openai/gpt-oss-120b@fp8", "mistralai/mistral-small-2603"}


def test_the_chat_client_sends_the_provider_filter():
    from melampo.evaluation import linking_bench as lb

    sent = {}

    class Response:
        def __init__(self, body):
            self.body = body

        def read(self):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(request, timeout=0):
        sent.update(json.loads(request.data))
        return Response(json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode())

    import urllib.request

    chat = lb.OpenRouterChat("openai/gpt-oss-120b", "k", provider={"quantizations": ["fp8"], "allow_fallbacks": False})
    original = urllib.request.urlopen
    urllib.request.urlopen = fake_urlopen
    try:
        chat("q")
    finally:
        urllib.request.urlopen = original
    assert sent["provider"] == {"quantizations": ["fp8"], "allow_fallbacks": False}
