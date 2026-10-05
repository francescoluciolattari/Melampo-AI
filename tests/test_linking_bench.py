"""Offline tests for the linking bench: cases, sparse index, contextual re-ranker, LLM stages."""

import importlib.util
import io
import json
import urllib.error
from pathlib import Path

import pytest

from melampo.evaluation import linking_bench as lb
from melampo.evaluation.encoder_bench import EncoderError, load_gold

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_linking_bench.py"


def _hash_embedder(texts):
    """Deterministic bag of character trigrams, enough to make similar strings close."""
    vectors = []
    for text in texts:
        vector = [0.0] * 512
        padded = f" {lb._fold(text)} "
        for i in range(max(1, len(padded) - 2)):
            vector[hash(padded[i : i + 3]) % 512] += 1.0
        vectors.append(vector)
    return vectors


@pytest.fixture(scope="module")
def gold():
    return load_gold()


@pytest.fixture(scope="module")
def cases(gold):
    return lb.build_cases(gold, lb.load_contexts())


def test_every_query_gets_a_sentence_containing_it(cases, gold):
    assert len(cases) == len(gold.queries) == 189
    assert all(case.query in case.sentence for case in cases)


def test_every_structure_has_exactly_one_context_group(gold):
    contexts = lb.load_contexts()
    members = [m for g in contexts["groups"].values() for m in g["members"]]
    assert sorted(members) == sorted(entry["id"] for entry in gold.pool)


def test_context_templates_never_name_a_structure(gold):
    """A template that spelled out the answer would measure nothing."""
    labels = {_t for e in gold.pool for _t in (lb._fold(e["it"]), lb._fold(e["en"]))}
    for group in lb.load_contexts()["groups"].values():
        for template in group["templates"]:
            folded = lb._fold(template.replace("{m}", " "))
            assert not any(f" {label} " in f" {folded} " for label in labels), template


def test_build_cases_rejects_a_structure_in_two_groups(gold):
    bad = {
        "groups": {
            "a": {"members": ["spleen"], "templates": ["{m}"]},
            "b": {"members": ["spleen"], "templates": ["{m}"]},
        }
    }
    with pytest.raises(ValueError, match="two context groups"):
        lb.build_cases(gold, bad)


def test_build_cases_rejects_an_unmapped_structure(gold):
    with pytest.raises(ValueError, match="no context group"):
        lb.build_cases(
            gold, {"groups": {"a": {"members": ["spleen"], "templates": ["{m}"]}}}
        )


def test_sparse_index_finds_the_obvious_label(gold):
    index = lb.SparseIndex(gold.pool)
    assert index.top_k("rene destro", 3)[0] == "kidney_right"
    assert index.top_k("milza", 1) == ["spleen"]


def test_evaluate_sparse_reports_nested_recalls(gold, cases):
    result = lb.evaluate_sparse(gold, cases)
    assert result["n"] == 189
    assert (
        result["recall_at_1"]
        <= result["recall_at_3"]
        <= result["recall_at_5"]
        <= result["recall_at_10"]
        <= 1.0
    )


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("3", 3),
        ("Numero: 2.", 2),
        ("0", 0),
        ("12", None),
        ("boh", None),
        ("", None),
        ("-1", None),
    ],
)
def test_parse_choice(answer, expected):
    assert lb.parse_choice(answer, 10) == expected


def test_contextual_lambda_zero_is_the_no_context_baseline(gold, cases):
    result = lb.evaluate_contextual(
        _hash_embedder, gold, cases[:40], lambdas=(0.0, 1.0)
    )
    assert set(result) == {"lambda_0", "lambda_1"}
    for row in result.values():
        assert row["correct"] + row["wrong"] == pytest.approx(1.0)
        assert row["abstain"] == 0.0


def test_contextual_rejects_inconsistent_embedding_dimensions(gold, cases):
    with pytest.raises(EncoderError):
        lb.evaluate_contextual(
            lambda t: [[1.0]] + [[1.0, 2.0]] * (len(t) - 1), gold, cases[:5]
        )


def _scripted_chat(cases, *, choice_for=None, verdict="9999", describe="una struttura"):
    """A chat double: describes, then picks the option whose label matches the case's target."""
    by_query = {c.query: c for c in cases}

    def chat(prompt):
        if prompt.startswith("Sei un radiologo"):
            return describe
        if prompt.startswith("Controllo di coerenza"):
            return verdict
        case = next(c for q, c in by_query.items() if f"Espressione: {q}\n" in prompt)
        if choice_for is not None:
            return choice_for
        options = [
            line for line in prompt.splitlines() if line[:1].isdigit() and ". " in line
        ]
        for line in options:
            number, label = line.split(". ", 1)
            if case.target.replace("_", " ").split()[0] in lb._fold(label):
                return number
        return "0"

    return chat


def test_deepel_style_records_each_stage_and_counts_unparsed(gold, cases):
    subset = cases[:12]
    chat = _scripted_chat(subset, choice_for="non so")
    result = lb.evaluate_deepel_style(chat, _hash_embedder, gold, subset, workers=2)
    assert result["stage2_unparsed"] == 12
    assert result["stage2"]["abstain"] == 1.0
    assert result["stage3"]["abstain"] == 1.0


def test_deepel_style_validation_can_veto_a_choice(gold, cases):
    subset = cases[:12]
    chat = _scripted_chat(subset, choice_for="1", verdict="0")
    result = lb.evaluate_deepel_style(chat, _hash_embedder, gold, subset, workers=2)
    assert result["stage2"]["abstain"] == 0.0
    assert result["stage3"]["abstain"] == 1.0
    assert result["stage3_changed"] == 12


def test_agreement_abstains_when_two_models_disagree(gold, cases):
    subset = cases[:12]
    first = lb.evaluate_deepel_style(
        _scripted_chat(subset, choice_for="1"), _hash_embedder, gold, subset, workers=2
    )
    second = lb.evaluate_deepel_style(
        _scripted_chat(subset, choice_for="2"), _hash_embedder, gold, subset, workers=2
    )
    result = lb.agreement(first, second, subset)
    assert result["abstain"] == 1.0
    assert lb.agreement(first, first, subset)["abstain"] == 0.0


def test_render_markdown_lists_every_section(gold, cases):
    subset = cases[:12]
    deepel = lb.evaluate_deepel_style(
        _scripted_chat(subset), _hash_embedder, gold, subset, workers=2
    )
    deepel["encoder"] = "x"
    report = {
        "sparse": lb.evaluate_sparse(gold, subset),
        "contextual": {
            "enc": lb.evaluate_contextual(_hash_embedder, gold, subset, lambdas=(0.0,)),
            "bad": {"error": "boom"},
        },
        "deepel": {"m": deepel},
        "agreement": {"stage3": lb.agreement(deepel, deepel, subset)},
    }
    text = lb.render_markdown(report)
    assert (
        "Sparse candidates" in text
        and "CIFSYN-style" in text
        and "DeepEL-style" in text
        and "boom" in text
    )


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code):
    return urllib.error.HTTPError("u", code, "err", {}, io.BytesIO(b"{}"))


def test_chat_client_drops_the_reasoning_hint_after_a_400(monkeypatch):
    bodies = []

    def fake_urlopen(request, timeout):
        body = json.loads(request.data)
        bodies.append(body)
        if "reasoning" in body:
            raise _http_error(400)
        return _Response(
            json.dumps({"choices": [{"message": {"content": "3"}}]}).encode()
        )

    monkeypatch.setattr(lb.urllib.request, "urlopen", fake_urlopen)
    chat = lb.OpenRouterChat("m", "k", sleep=lambda s: None)
    assert chat("ciao") == "3"
    assert "reasoning" in bodies[0] and "reasoning" not in bodies[1]


def test_chat_client_retries_429_then_gives_up_without_leaking_the_key(monkeypatch):
    def always_429(request, timeout):
        raise _http_error(429)

    monkeypatch.setattr(lb.urllib.request, "urlopen", always_429)
    chat = lb.OpenRouterChat("m", "SECRET-KEY", sleep=lambda s: None, retries=1)
    with pytest.raises(EncoderError) as error:
        chat("x")
    assert "SECRET-KEY" not in str(error.value)


def test_chat_client_rejects_an_unreadable_payload(monkeypatch):
    monkeypatch.setattr(
        lb.urllib.request, "urlopen", lambda r, timeout: _Response(b'{"nope": 1}')
    )
    with pytest.raises(EncoderError, match="unreadable"):
        lb.OpenRouterChat("m", "k")("x")


def test_script_sparse_mode_runs_without_a_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    spec = importlib.util.spec_from_file_location("linking_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "r.json"
    assert module.main(["--mode", "all", "--out", str(out), "--limit", "20"]) == 0
    report = json.loads(out.read_text())
    assert (
        report["settings"]["phrases"] == 20
        and "sparse" in report
        and "contextual" not in report
    )


def test_script_rejects_unknown_names():
    spec = importlib.util.spec_from_file_location("linking_script2", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit):
        module._names("nope", module.ENCODERS, module.DEFAULT_ENCODERS)


def test_chain_counts_what_the_agent_does_to_the_reranker_top_one(gold, cases):
    subset = cases[:20]
    # An agent that always answers 0 abstains everywhere: it can only "catch" or "lose".
    abstaining = lb.evaluate_chain(
        _scripted_chat(subset, choice_for="0"), _hash_embedder, gold, subset, workers=2
    )
    transitions = abstaining["agent_vs_rerank"]
    assert sum(transitions.values()) == 20
    assert set(transitions) <= {"abstained_on_a_correct", "caught_a_wrong"}
    assert abstaining["stage3"]["abstain"] == 1.0
    assert abstaining["gold_in_agent_options"] <= abstaining["gold_in_kept"] <= 1.0


def test_chain_agent_only_sees_the_best_three_candidates(gold, cases):
    subset = cases[:6]
    seen = []

    def chat(prompt):
        if prompt.startswith("Sei un radiologo"):
            return "x"
        if prompt.startswith("Collega"):
            seen.append(
                sum(
                    1
                    for line in prompt.splitlines()
                    if line[:1].isdigit() and ". " in line and "nessuna" not in line
                )
            )
        return "0"

    lb.evaluate_chain(chat, _hash_embedder, gold, subset, workers=1)
    assert seen and set(seen) == {3}


def test_chain_validation_can_change_the_choice_and_is_counted(gold, cases):
    subset = cases[:8]
    result = lb.evaluate_chain(
        _scripted_chat(subset, choice_for="1", verdict="2"),
        _hash_embedder,
        gold,
        subset,
        workers=2,
    )
    assert result["stage2"]["abstain"] == 0.0
    assert all(r["stage3_choice"] != r["stage2_choice"] for r in result["_records"])


def test_render_includes_the_chain_section(gold, cases):
    subset = cases[:6]
    chain = lb.evaluate_chain(
        _scripted_chat(subset), _hash_embedder, gold, subset, workers=1
    )
    text = lb.render_markdown(
        {
            "chain": {"m": chain, "bad": {"error": "boom"}},
            "chain_agreement": {"stage3": lb.agreement(chain, chain, subset)},
        }
    )
    assert "Chain: CIFSYN-style" in text and "boom" in text and "fixed" in text


def test_gemini_is_paced_and_other_encoders_are_not():
    spec = importlib.util.spec_from_file_location("linking_script3", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gemini = module._embedder("google-gemini-embedding-001", "k")
    other = module._embedder("voyage-4-large", "k")
    assert gemini.pause > 0 and gemini.batch_size < other.batch_size
    assert other.pause == 0 and other.retries == 6
