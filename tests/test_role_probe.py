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


def test_the_cascade_decides_by_the_lattice_only_when_its_reading_is_exact():
    exact = item("procedure_site") | {"evidence": {"lattice_detail": {"exact": True}}}
    inferred = item("structure") | {"evidence": {"lattice_detail": {"exact": False}}}
    exact["answers"] = {"lattice": "procedure_site", "reader": "structure", "llm+reader:m": "structure",
                        "llm:m": "structure"}
    inferred["answers"] = {"lattice": "inherent_location", "reader": "structure", "llm+reader:m": "structure",
                           "llm:m": "procedure_site"}
    rp.cascade([exact, inferred])
    assert exact["answers"]["exact->reader"] == "procedure_site"
    assert exact["answers"]["exact->agree:m"] == "procedure_site"
    assert inferred["answers"]["exact->reader"] == "structure"
    assert inferred["answers"]["exact->llm+reader:m"] == "structure"
    assert inferred["answers"]["exact->agree:m"] == "uncertain"   # the reader and the model alone differ


def test_a_refused_call_is_unreachable_not_uncertain_and_is_not_scored():
    ok, refused = item("structure"), item("structure", report="r2")
    ok["answers"] = {"lattice": "structure", "llm+reader:m": "structure", "llm:m": "structure", "reader": "structure"}
    refused["answers"] = {"lattice": "structure", "llm+reader:m": rp.UNREACHABLE, "llm:m": rp.UNREACHABLE,
                          "reader": "structure"}
    rp.cascade([ok, refused])
    assert refused["answers"]["exact->agree:m"] == rp.UNREACHABLE
    assert refused["answers"]["exact->llm+reader:m"] == rp.UNREACHABLE
    s = rp.score([ok, refused], "llm+reader:m")
    assert s["items"] == 1 and s["uncertain"] == 0 and s["accuracy"] == 1.0
    assert rp.score([ok, refused], "lattice")["items"] == 2


def test_the_agreement_arms_use_the_answer_that_did_not_see_the_reader():
    a, b = item("structure"), item("structure", report="r2")
    a["answers"] = {"lattice": "structure", "reader": "structure", "llm:m": "structure", "llm:n": "structure",
                    "llm+reader:m": "inherent_location"}
    b["answers"] = {"lattice": "structure", "reader": "structure", "llm:m": "structure", "llm:n": "procedure_site"}
    rp.cascade([a, b])
    assert a["answers"]["exact->agree:m"] == "structure"
    assert a["answers"]["exact->agree_all"] == "structure"
    assert b["answers"]["exact->agree:m"] == "structure"
    assert b["answers"]["exact->agree_all"] == "uncertain"


def test_one_training_procedure_for_every_profile(tmp_path, capsys):
    import lora_roles as lo

    from melampo.evaluation import model_profiles as mp

    rows = [{"item_id": f"i{k}", "report_id": f"r{k}", "stratum": "e3c_en", "language": "en", "mention": "liver",
             "sentence": "The liver.", "structure": "liver", "role": "structure", "fold": k % lo.FOLDS,
             "prompt": [{"role": "user", "content": "q"}],
             "completion": [{"role": "assistant", "content": "{\"role\": \"structure\"}"}]} for k in range(10)]
    data = tmp_path / "data"
    data.mkdir()
    (data / "items.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert lo.main(["export", "--data", str(data), "--fold", "0", "--out", str(tmp_path / "f0")]) == 0
    train = [json.loads(x) for x in (tmp_path / "f0" / "train.jsonl").read_text().splitlines()]
    test = [json.loads(x) for x in (tmp_path / "f0" / "test.jsonl").read_text().splitlines()]
    assert len(train) == 8 and len(test) == 2
    assert [m["role"] for m in train[0]["messages"]] == ["user", "assistant"]
    for name in mp.PROFILES:
        assert lo.main(["plan", "--profile", name]) == 0
        out = capsys.readouterr().out
        assert mp.PROFILES[name].hf_id in out and "evaluate" in out and "export" in out
    with pytest.raises(ValueError):
        mp.profile("no-such-model")


def test_agreement_on_the_outcome_and_roles_as_a_set():
    a, b, c = item("structure"), item("structure", report="r2"), item("structure", report="r3")
    a["answers"] = {"lattice": "structure", "reader": "structure", "llm:m": "source_of"}
    a["roles"] = {"llm:m": ["source_of", "structure"]}
    a["evidence"] = {"reader_detail": {"shares": {"structure": 0.9, "source_of": 0.1}}}
    b["answers"] = {"lattice": "structure", "reader": "structure", "llm:m": "not_a_body_site"}
    c["answers"] = {"lattice": "structure", "reader": "inherent_location", "llm:m": "procedure_site"}
    c["roles"] = {"llm:m": ["procedure_site"]}
    got = rp.site_agreement([a, b, c])["m"]
    assert got["items"] == 3 and got["to_a_person"] == 1 and got["to_a_person_items"] == [b["item_id"]]
    assert got["agreed_with_a_shared_role"] == 1 and got["agreed_without_a_shared_role"] == 1
    assert got["agreed_but_wrong_outcome"] == 0


def test_the_model_may_give_several_roles():
    from melampo.evaluation import llm_reading as lr

    s = "No pleural effusion."
    got = lr.check({"phrase": "pleural effusion", "head": "effusion", "type": "disease or finding",
                    "roles": ["structure", "source_of"]}, s, 3, 10)
    assert got.valid and got.answer["roles"] == ["structure", "source_of"] and got.role == "structure"
    old = lr.check({"phrase": "pleural effusion", "head": "effusion", "type": "disease or finding",
                    "role": "structure"}, s, 3, 10)
    assert old.valid and old.answer["roles"] == ["structure"]
