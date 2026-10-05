"""Tests for the text-encoder bench: its measurements, its client, and its script.

All offline. The embedders here are hand-built so the expected scores can be
worked out by eye; the network client is exercised against a stand-in for
`urllib.request.urlopen`.
"""

import importlib.util
import io
import json
import urllib.error
from pathlib import Path

import pytest

from melampo.evaluation import encoder_bench as eb

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_encoder_bench.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("encoder_bench_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_gold() -> eb.Gold:
    pool = [
        {
            "id": "kidney_left",
            "task": "total",
            "it": "rene sinistro",
            "en": "left kidney",
        },
        {
            "id": "kidney_right",
            "task": "total",
            "it": "rene destro",
            "en": "right kidney",
        },
    ]
    queries = [
        {"query": "rene sn", "target": "kidney_left", "kind": "short"},
        {"query": "rene dx", "target": "kidney_right", "kind": "short"},
    ]
    triplets = [
        {
            "anchor": "rene dx",
            "positive": "rene destro",
            "negative": "rene sinistro",
            "category": "laterality",
        },
        {
            "anchor": "nessun nodulo",
            "positive": "assenza di nodulo",
            "negative": "presenza di nodulo",
            "category": "negation",
        },
    ]
    return eb.Gold(pool=pool, queries=queries, triplets=triplets)


# Two orthogonal axes: left and right. Phrases map onto them by hand.
_VECTORS = {
    "rene sinistro": [1.0, 0.0],
    "left kidney": [1.0, 0.0],
    "rene sinistro / left kidney": [1.0, 0.0],
    "rene destro": [0.0, 1.0],
    "right kidney": [0.0, 1.0],
    "rene destro / right kidney": [0.0, 1.0],
    "rene sn": [0.9, 0.1],
    "rene dx": [0.1, 0.9],
    "nessun nodulo": [1.0, 0.2],
    "assenza di nodulo": [1.0, 0.0],
    "presenza di nodulo": [0.0, 1.0],
}


def _dictionary_embedder(table):
    def embed(texts):
        return [list(table[text]) for text in texts]

    return embed


# --------------------------------------------------------------------------
# Gold data
# --------------------------------------------------------------------------


def test_shipped_gold_data_is_consistent_and_covers_the_hard_categories():
    gold = eb.load_gold()
    assert len(gold.pool) >= 50 and len(gold.queries) >= 150
    categories = {row["category"] for row in gold.triplets}
    assert {
        "laterality",
        "negation",
        "size_and_unit",
        "temporal_comparison",
    } <= categories
    # Every structure is queried in three different ways, so no single phrasing decides its score.
    per_target = {}
    for row in gold.queries:
        per_target.setdefault(row["target"], set()).add(row["kind"])
    assert all(len(kinds) == 3 for kinds in per_target.values())
    # Every identifier is a real TotalSegmentator class of the task it claims, so a typo cannot
    # quietly make a structure unreachable. The class lists were read from the package, not typed.
    classes = json.loads(
        (eb.DEFAULT_DATA_DIR / "totalsegmentator_classes.json").read_text()
    )["tasks"]
    for entry in gold.pool:
        assert entry["id"] in classes[entry["task"]], entry["id"]


def test_validation_rejects_a_query_for_an_unknown_structure():
    gold = _tiny_gold()
    bad = eb.Gold(
        gold.pool, [{"query": "x", "target": "nonexistent", "kind": "k"}], gold.triplets
    )
    with pytest.raises(ValueError, match="unknown id"):
        eb.validate_gold(bad)


def test_validation_rejects_a_triplet_whose_negative_equals_its_positive():
    gold = _tiny_gold()
    bad = eb.Gold(
        gold.pool,
        gold.queries,
        [{"anchor": "a", "positive": "b", "negative": "b", "category": "c"}],
    )
    with pytest.raises(ValueError, match="equals its negative"):
        eb.validate_gold(bad)


# --------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------


def test_a_perfect_encoder_scores_perfectly():
    result = eb.evaluate_encoder(_dictionary_embedder(_VECTORS), _tiny_gold())
    assert result["dimension"] == 2
    for mode in eb.POOL_MODES:
        assert result["retrieval"][mode]["recall_at_1"] == 1.0
        assert result["retrieval"][mode]["mrr"] == 1.0
    assert result["triplets"]["accuracy"] == 1.0
    assert result["screening_score"] == 1.0


def test_an_encoder_blind_to_laterality_is_caught_per_category():
    """Left and right collapse onto one axis: the failure the bench exists to find."""
    blind = dict(_VECTORS)
    blind.update(
        {
            "rene destro": [1.0, 0.0],
            "right kidney": [1.0, 0.0],
            "rene destro / right kidney": [1.0, 0.0],
        }
    )
    result = eb.evaluate_encoder(_dictionary_embedder(blind), _tiny_gold())
    categories = result["triplets"]["by_category"]
    assert categories["laterality"]["accuracy"] == 0.0
    assert categories["negation"]["accuracy"] == 1.0
    assert result["triplets"]["failures"][0]["category"] == "laterality"
    assert result["retrieval"]["it"]["recall_at_1"] < 1.0
    assert result["retrieval"]["it"]["misses"]


def test_ties_count_against_the_target():
    """An encoder that scores everything the same must not look good by luck of ordering."""
    flat = {text: [1.0, 1.0] for text in _VECTORS}
    result = eb.evaluate_encoder(_dictionary_embedder(flat), _tiny_gold())
    assert result["retrieval"]["it"]["recall_at_1"] == 0.0
    assert result["triplets"]["accuracy"] == 0.0


def test_query_prefix_is_applied_to_retrieval_queries_only():
    seen = []

    def spy(texts):
        seen.extend(texts)
        return [
            list(_VECTORS.get(text.removeprefix("PFX "), [1.0, 1.0])) for text in texts
        ]

    eb.evaluate_encoder(spy, _tiny_gold(), query_prefix="PFX ")
    assert "PFX rene sn" in seen
    assert "rene sn" not in seen
    assert "rene destro" in seen  # a document text, untouched
    assert (
        "PFX rene dx" in seen and "PFX nessun nodulo" not in seen
    )  # triplet anchors stay raw


def test_inconsistent_dimensions_are_rejected_not_scored():
    """`cosine_similarity` returns 0.0 on a length mismatch; that must not pass for a result."""
    table = dict(_VECTORS)
    table["rene sn"] = [1.0, 0.0, 0.0]
    with pytest.raises(eb.EncoderError, match="dimensions"):
        eb.evaluate_encoder(_dictionary_embedder(table), _tiny_gold())


def test_wrong_number_of_embeddings_is_rejected():
    with pytest.raises(eb.EncoderError, match="received"):
        eb.evaluate_encoder(lambda texts: [[1.0, 0.0]], _tiny_gold())


def test_wilson_interval_behaves_at_the_edges():
    assert eb.wilson_interval(0, 0) == (0.0, 0.0)
    low, high = eb.wilson_interval(10, 10)
    assert high == 1.0 and 0.6 < low < 1.0
    low, high = eb.wilson_interval(0, 10)
    assert low == 0.0 and 0.0 < high < 0.4
    narrow = eb.wilson_interval(500, 1000)
    wide = eb.wilson_interval(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


# --------------------------------------------------------------------------
# OpenRouter client
# --------------------------------------------------------------------------


class _Response:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code, body=b"{}"):
    return urllib.error.HTTPError(
        "https://example.invalid", code, "x", {}, io.BytesIO(body)
    )


def test_client_restores_input_order_and_batches(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        sent = json.loads(request.data)
        calls.append(sent["input"])
        # Answer out of order, with explicit indices, as providers are allowed to.
        data = [
            {"index": i, "embedding": [float(len(text)), 0.0]}
            for i, text in enumerate(sent["input"])
        ]
        return _Response({"data": list(reversed(data))})

    monkeypatch.setattr(eb.urllib.request, "urlopen", fake_urlopen)
    embedder = eb.OpenRouterEmbedder(
        "vendor/model", "key", batch_size=2, sleep=lambda s: None
    )
    vectors = embedder(["a", "bb", "ccc"])
    assert vectors == [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    assert calls == [["a", "bb"], ["ccc"]]


def test_client_retries_transient_errors_then_succeeds(monkeypatch):
    attempts = []

    def fake_urlopen(request, timeout):
        attempts.append(1)
        if len(attempts) < 3:
            raise _http_error(429)
        return _Response({"data": [{"index": 0, "embedding": [1.0]}]})

    monkeypatch.setattr(eb.urllib.request, "urlopen", fake_urlopen)
    slept = []
    embedder = eb.OpenRouterEmbedder("m", "key", sleep=slept.append)
    assert embedder(["x"]) == [[1.0]]
    assert len(attempts) == 3 and slept == [1, 2]


def test_client_does_not_retry_a_client_error(monkeypatch):
    attempts = []

    def fake_urlopen(request, timeout):
        attempts.append(1)
        raise _http_error(404, b'{"error": "no such model"}')

    monkeypatch.setattr(eb.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(eb.EncoderError, match="HTTP 404"):
        eb.OpenRouterEmbedder("m", "key", sleep=lambda s: None)(["x"])
    assert len(attempts) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"error": {"message": "quota"}},
        {"data": [{"index": 0}]},
        {"data": []},
        {"nothing": True},
    ],
)
def test_client_rejects_unreadable_payloads(monkeypatch, payload):
    monkeypatch.setattr(
        eb.urllib.request, "urlopen", lambda request, timeout: _Response(payload)
    )
    with pytest.raises(eb.EncoderError):
        eb.OpenRouterEmbedder("m", "key", sleep=lambda s: None)(["x"])


def test_the_api_key_never_appears_in_an_error_message(monkeypatch):
    def fake_urlopen(request, timeout):
        raise _http_error(401, b"unauthorized")

    monkeypatch.setattr(eb.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(eb.EncoderError) as caught:
        eb.OpenRouterEmbedder("m", "sk-secret-value", sleep=lambda s: None)(["x"])
    assert "sk-secret-value" not in str(caught.value)


# --------------------------------------------------------------------------
# Script
# --------------------------------------------------------------------------


@pytest.fixture
def script():
    return _load_script()


def test_results_file_is_written_when_no_key_is_set(script, tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    out = tmp_path / "result.json"
    assert script.main(["--out", str(out), "--markdown", str(tmp_path / "s.md")]) == 1
    report = json.loads(out.read_text())
    assert report["results"] == []
    assert set(report["preflight"].values()) == {"OPENROUTER_API_KEY not set"}
    assert "No encoder produced" in report["verdict"]
    assert (tmp_path / "s.md").exists()


def test_list_candidates_prints_names_and_needs_no_key(script, capsys, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert script.main(["--list-candidates"]) == 0
    names = json.loads(capsys.readouterr().out)
    assert "qwen3-embedding-8b" in names and len(names) == len(set(names))


def test_unknown_roster_name_is_refused(script, tmp_path):
    assert (
        script.main(["--roster", "nonexistent", "--out", str(tmp_path / "r.json")]) == 1
    )


def test_one_unreachable_candidate_does_not_stop_the_others(
    script, tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    gold = eb.load_gold()
    # An encoder that knows nothing: hash each text to a position on a few axes.
    import zlib

    def toy(texts):
        out = []
        for text in texts:
            vector = [0.0] * 16
            for token in text.lower().replace("/", " ").split():
                vector[zlib.crc32(token.encode()) % 16] += 1.0
            out.append(vector)
        return out

    class FakeEmbedder:
        def __init__(self, slug, api_key):
            self.slug = slug

        def __call__(self, texts):
            if self.slug == "mistralai/mistral-embed-2312":
                raise eb.EncoderError(
                    "HTTP 404 from mistralai/mistral-embed-2312: no such model"
                )
            return toy(texts)

    monkeypatch.setattr(script, "OpenRouterEmbedder", FakeEmbedder)
    out = tmp_path / "r.json"
    code = script.main(
        [
            "--roster",
            "qwen3-embedding-8b,mistral-embed",
            "--out",
            str(out),
            "--markdown",
            str(tmp_path / "s.md"),
        ]
    )
    assert code == 0
    report = json.loads(out.read_text())
    assert [row["name"] for row in report["results"]] == ["qwen3-embedding-8b"]
    assert report["preflight"]["mistral-embed"].startswith("HTTP 404")
    assert report["settings"]["pool_size"] == len(gold.pool)
    markdown = (tmp_path / "s.md").read_text()
    assert (
        "qwen3-embedding-8b" in markdown and "mistral-embed" in markdown
    )  # the skipped one is listed with its reason


def test_extra_slugs_join_the_roster_without_replacing_it(script):
    chosen, unknown = script._select("bge-m3", "vendor/some-embed")
    assert [entry[0] for entry in chosen] == [
        "bge-m3",
        "vendor-some-embed",
    ] and unknown == []


# --------------------------------------------------------------------------
# Document/triplet prefixes and the local backend
# --------------------------------------------------------------------------


def test_document_and_triplet_prefixes_reach_the_right_texts():
    seen = []

    def spy(texts):
        seen.extend(texts)
        return [[1.0, 1.0] for _ in texts]

    eb.evaluate_encoder(
        spy, _tiny_gold(), query_prefix="Q ", document_prefix="D ", triplet_prefix="T "
    )
    assert "D rene destro" in seen and "rene destro" not in seen
    assert "Q rene sn" in seen
    assert "T rene dx" in seen and "rene dx" not in seen


def test_local_embedder_wraps_a_loader_and_returns_plain_floats():
    class FakeModel:
        def __init__(self, model_id, trust_remote_code=False):
            self.model_id = model_id
            self.trust_remote_code = trust_remote_code

        def encode(self, texts, **kwargs):
            return [[float(len(text)), 1.0] for text in texts]

    embedder = eb.LocalEmbedder("org/model", loader=FakeModel, trust_remote_code=True)
    assert embedder(["ab", "abc"]) == [[2.0, 1.0], [3.0, 1.0]]
    assert embedder._model.trust_remote_code is True


def test_local_embedder_reports_a_load_failure_as_an_encoder_error():
    def broken(model_id, trust_remote_code=False):
        raise OSError("gated repo, no token")

    with pytest.raises(eb.EncoderError, match="could not load org/model"):
        eb.LocalEmbedder("org/model", loader=broken)


def test_local_candidates_run_through_the_script_and_a_failure_is_isolated(
    script, tmp_path, monkeypatch
):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import zlib

    def toy(texts):
        out = []
        for text in texts:
            vector = [0.0] * 16
            for token in text.lower().replace("/", " ").split():
                vector[zlib.crc32(token.encode()) % 16] += 1.0
            out.append(vector)
        return out

    class FakeLocal:
        def __init__(self, model_id, trust_remote_code=False):
            if "gemma" in model_id:
                raise eb.EncoderError(f"could not load {model_id}: gated")

        def __call__(self, texts):
            return toy(texts)

    monkeypatch.setattr(script, "LocalEmbedder", FakeLocal)
    out = tmp_path / "r.json"
    code = script.main(
        [
            "--backend",
            "local",
            "--roster",
            "granite-97m-multilingual-r2,embeddinggemma-300m",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    report = json.loads(out.read_text())
    assert [row["name"] for row in report["results"]] == ["granite-97m-multilingual-r2"]
    assert "gated" in report["preflight"]["embeddinggemma-300m"]


def test_backend_local_without_roster_runs_only_local_candidates(
    script, tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "key")
    calls = []

    class Spy:
        def __init__(self, *args, **kwargs):
            calls.append(args)
            raise eb.EncoderError("stop")

    monkeypatch.setattr(script, "OpenRouterEmbedder", Spy)
    monkeypatch.setattr(script, "LocalEmbedder", Spy)
    out = tmp_path / "r.json"
    script.main(["--backend", "local", "--out", str(out)])
    report = json.loads(out.read_text())
    assert set(report["preflight"]) == {
        e["name"] for e in script.LOCAL_CANDIDATES if not e.get("opt_in")
    }


def test_every_candidate_name_is_unique_across_both_backends(script):
    names = [entry[0] for entry in script.CANDIDATE_ENCODERS] + [
        e["name"] for e in script.LOCAL_CANDIDATES
    ]
    assert len(names) == len(set(names))


def test_a_local_only_roster_makes_the_openrouter_backend_a_no_op(script, tmp_path):
    out = tmp_path / "r.json"
    code = script.main(
        [
            "--backend",
            "openrouter",
            "--roster",
            "granite-97m-multilingual-r2",
            "--out",
            str(out),
        ]
    )
    assert code == 0 and not out.exists()


# --------------------------------------------------------------------------
# Cohere
# --------------------------------------------------------------------------


def test_cohere_embedder_sends_the_role_of_each_text_and_keeps_order(monkeypatch):
    sent = []

    def fake_urlopen(request, timeout):
        body = json.loads(request.data)
        sent.append((body["input_type"], body["texts"]))
        vectors = [[float(len(text)), 1.0] for text in body["texts"]]
        return _Response({"embeddings": {"float": vectors}})

    monkeypatch.setattr(eb.urllib.request, "urlopen", fake_urlopen)
    embedder = eb.CohereEmbedder("embed-v5.0-pro", "key", sleep=lambda s: None)
    texts = [
        eb.COHERE_ROLE_QUERY + "abc",
        "doc",
        eb.COHERE_ROLE_DOCUMENT + "de",
        eb.COHERE_ROLE_SYMMETRIC + "f",
    ]
    vectors = embedder(texts)
    assert [v[0] for v in vectors] == [3.0, 3.0, 2.0, 1.0]  # input order restored
    assert ("search_query", ["abc"]) in sent
    assert ("search_document", ["doc", "de"]) in sent
    assert ("clustering", ["f"]) in sent


def test_cohere_unreadable_payload_is_an_encoder_error(monkeypatch):
    monkeypatch.setattr(
        eb.urllib.request,
        "urlopen",
        lambda request, timeout: _Response({"message": "invalid model"}),
    )
    with pytest.raises(eb.EncoderError, match="unreadable"):
        eb.CohereEmbedder("m", "key", sleep=lambda s: None)(["x"])


def test_cohere_candidates_need_their_own_key_and_are_reported_without_it(
    script, tmp_path, monkeypatch
):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    out = tmp_path / "r.json"
    assert script.main(["--backend", "cohere", "--out", str(out)]) == 1
    report = json.loads(out.read_text())
    assert set(report["preflight"].values()) == {"COHERE_API_KEY not set"}
    assert set(report["preflight"]) == {"cohere-embed-v5-pro", "cohere-embed-v5-fast"}


def test_nv_embed_v2_runs_only_when_named(script):
    assert "nv-embed-v2" not in [e["name"] for e in script._select_local(None, "local")]
    named = script._select_local("nv-embed-v2", "local")
    assert [e["name"] for e in named] == ["nv-embed-v2"]


def test_pause_is_applied_between_batches_only():
    sleeps = []
    embedder = eb.OpenRouterEmbedder(
        "m", "k", batch_size=2, pause=3.0, sleep=sleeps.append
    )
    embedder._embed_batch = lambda batch: [[1.0]] * len(batch)
    assert len(embedder(["a", "b", "c", "d", "e"])) == 5
    assert sleeps == [3.0, 3.0]


def test_no_pause_by_default():
    sleeps = []
    embedder = eb.OpenRouterEmbedder("m", "k", batch_size=1, sleep=sleeps.append)
    embedder._embed_batch = lambda batch: [[1.0]] * len(batch)
    embedder(["a", "b", "c"])
    assert sleeps == []
