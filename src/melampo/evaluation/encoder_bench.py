"""Which text encoder should Melampo pin for normalised clinical terms?

The decision (D17/D21 in docs/imaging_decision_record.md) is made by
measuring, not by reading leaderboards. General multilingual benchmarks say
little about the failure that matters here: an encoder that places "rene
destro" next to "rene sinistro", or "nodulo presente" next to "assenza di
nodulo", turns a similarity search into a silent clinical error.

Two measurements, both on synthetic Italian phrases written by hand. No
patient data is involved, so the phrases can safely be sent to a hosted
embedding endpoint for the comparison.

**Concept linking.** An Italian anatomical phrase (formal, abbreviated,
colloquial) must retrieve the right anatomical structure from a pool whose
identifiers are TotalSegmentator class names (verified against
totalsegmentator 2.18.0, `map_to_binary.py`, tasks `total` and
`liver_segments`). Linking to those names is what the later agreement check
between VoxTell and TotalSegmentator needs, so an encoder that cannot do it
is ruled out early. The pool is embedded three ways -- Italian labels,
English labels, both -- because D20 asks whether VoxTell should be prompted
in Italian or in English taken from a fixed vocabulary.

**Hard triplets.** For an anchor phrase, a paraphrase (positive) must score
higher than a near-miss (negative) that differs in exactly the way that
matters clinically: laterality, an adjacent organ with a similar name, a
liver segment, a vertebral level, negation, size units, comparison with a
prior exam, consistency. Accuracy is reported per category because a good
average can hide a category at chance.

What this does not establish. The phrase sets are small (see the counts in
the report) and the 95% Wilson intervals are printed next to the headline
rates: a difference smaller than the interval is not a difference. The
phrases are general-purpose anatomical Italian, not real report text, so the
result is a screen that removes unsuitable encoders, not a clinical
validation. It says nothing about speed under load, cost, or licence.

Pure computation lives here so it can be tested without a network; the
OpenRouter client is the only part that talks to anything.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..memory.concept_normalisation import cosine_similarity

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "encoder_bench"
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
POOL_MODES = ("it", "en", "both")

Embedder = Callable[[Sequence[str]], list[list[float]]]

# Qwen3-Embedding is instruction-aware: the model card recommends a task line
# in front of the *query* side only. Benchmarked as a separate candidate so
# the gain (or loss) from the instruction is visible instead of assumed.
QWEN_QUERY_INSTRUCTION = "Instruct: Given an Italian clinical anatomical phrase, retrieve the name of the matching anatomical structure\nQuery: "

_MAX_RECORDED_FAILURES = 10


class EncoderError(RuntimeError):
    """The embedding endpoint failed or returned something unusable."""


@dataclass(frozen=True)
class Gold:
    pool: list[dict[str, Any]]
    queries: list[dict[str, Any]]
    triplets: list[dict[str, Any]]


# --------------------------------------------------------------------------
# Gold data
# --------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_gold(gold: Gold) -> None:
    """Fail loudly on a malformed gold set: a typo here would silently skew every candidate."""
    ids = [entry["id"] for entry in gold.pool]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ids in the anatomy pool")
    for entry in gold.pool:
        if not entry.get("it") or not entry.get("en"):
            raise ValueError(
                f"pool entry {entry.get('id')!r} lacks an Italian or English label"
            )
    known = set(ids)
    for row in gold.queries:
        if row["target"] not in known:
            raise ValueError(
                f"query {row['query']!r} targets unknown id {row['target']!r}"
            )
        if not row["query"].strip():
            raise ValueError("empty query")
    if not gold.triplets:
        raise ValueError("no triplets")
    for row in gold.triplets:
        if not (
            row["anchor"].strip()
            and row["positive"].strip()
            and row["negative"].strip()
        ):
            raise ValueError(f"triplet with an empty field: {row!r}")
        if row["positive"] == row["negative"]:
            raise ValueError(f"triplet whose positive equals its negative: {row!r}")
        if not row.get("category"):
            raise ValueError(f"triplet without a category: {row!r}")


def load_gold(data_dir: Path = DEFAULT_DATA_DIR) -> Gold:
    gold = Gold(
        pool=_read_jsonl(data_dir / "anatomy_pool.jsonl"),
        queries=_read_jsonl(data_dir / "queries_it.jsonl"),
        triplets=_read_jsonl(data_dir / "hard_triplets.jsonl"),
    )
    validate_gold(gold)
    return gold


def pool_text(entry: dict[str, Any], mode: str) -> str:
    if mode == "it":
        return entry["it"]
    if mode == "en":
        return entry["en"]
    if mode == "both":
        return f"{entry['it']} / {entry['en']}"
    raise ValueError(f"unknown pool mode {mode!r}")


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion; sensible at small n and at 0 or 1."""
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _interval(successes: int, total: int) -> list[float]:
    low, high = wilson_interval(successes, total)
    return [round(low, 4), round(high, 4)]


# --------------------------------------------------------------------------
# Embedding and scoring
# --------------------------------------------------------------------------


def _embed_unique(embedder: Embedder, texts: Sequence[str]) -> dict[str, list[float]]:
    """Embed each distinct text once and require one consistent dimension.

    A length mismatch is rejected here rather than scored: `cosine_similarity`
    returns 0.0 for vectors of different lengths, which would turn a broken
    endpoint into a plausible-looking (and bad) result.
    """
    unique = list(dict.fromkeys(texts))
    vectors = embedder(unique)
    if len(vectors) != len(unique):
        raise EncoderError(
            f"asked for {len(unique)} embeddings, received {len(vectors)}"
        )
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1 or 0 in dimensions:
        raise EncoderError(
            f"inconsistent or empty embedding dimensions: {sorted(dimensions)}"
        )
    return dict(zip(unique, vectors, strict=True))


def _rank_of_target(scores: list[float], target_index: int) -> int:
    """1-based rank, counting ties against the target (the conservative reading)."""
    target = scores[target_index]
    return 1 + sum(
        1
        for index, score in enumerate(scores)
        if index != target_index and score >= target
    )


def _retrieval(
    lookup: dict[str, list[float]],
    gold: Gold,
    mode: str,
    query_prefix: str,
    document_prefix: str = "",
) -> dict[str, Any]:
    pool_vectors = [
        lookup[document_prefix + pool_text(entry, mode)] for entry in gold.pool
    ]
    index_of = {entry["id"]: position for position, entry in enumerate(gold.pool)}
    hits1 = hits3 = 0
    reciprocal = 0.0
    by_kind: dict[str, list[int]] = defaultdict(list)
    misses: list[dict[str, Any]] = []
    for row in gold.queries:
        query_vector = lookup[query_prefix + row["query"]]
        scores = [cosine_similarity(query_vector, vector) for vector in pool_vectors]
        rank = _rank_of_target(scores, index_of[row["target"]])
        hits1 += rank == 1
        hits3 += rank <= 3
        reciprocal += 1.0 / rank
        by_kind[row["kind"]].append(int(rank == 1))
        if rank != 1 and len(misses) < _MAX_RECORDED_FAILURES:
            best = max(range(len(scores)), key=scores.__getitem__)
            misses.append(
                {
                    "query": row["query"],
                    "expected": row["target"],
                    "got": gold.pool[best]["id"],
                    "rank": rank,
                }
            )
    total = len(gold.queries)
    return {
        "n": total,
        "recall_at_1": round(hits1 / total, 4),
        "recall_at_1_ci": _interval(hits1, total),
        "recall_at_3": round(hits3 / total, 4),
        "mrr": round(reciprocal / total, 4),
        "recall_at_1_by_kind": {
            kind: round(sum(v) / len(v), 4) for kind, v in sorted(by_kind.items())
        },
        "misses": misses,
    }


def _triplets(
    lookup: dict[str, list[float]], gold: Gold, triplet_prefix: str = ""
) -> dict[str, Any]:
    by_category: dict[str, list[float]] = defaultdict(list)
    failures: list[dict[str, Any]] = []
    margins: list[float] = []
    for row in gold.triplets:
        anchor = lookup[triplet_prefix + row["anchor"]]
        margin = cosine_similarity(
            anchor, lookup[triplet_prefix + row["positive"]]
        ) - cosine_similarity(anchor, lookup[triplet_prefix + row["negative"]])
        margins.append(margin)
        by_category[row["category"]].append(margin)
        if margin <= 0 and len(failures) < _MAX_RECORDED_FAILURES:
            failures.append({**row, "margin": round(margin, 4)})
    correct = sum(1 for margin in margins if margin > 0)
    total = len(margins)
    return {
        "n": total,
        "accuracy": round(correct / total, 4),
        "accuracy_ci": _interval(correct, total),
        "mean_margin": round(sum(margins) / total, 4),
        "by_category": {
            category: {
                "n": len(values),
                "accuracy": round(sum(1 for v in values if v > 0) / len(values), 4),
                "mean_margin": round(sum(values) / len(values), 4),
            }
            for category, values in sorted(by_category.items())
        },
        "failures": failures,
    }


def evaluate_encoder(
    embedder: Embedder,
    gold: Gold,
    query_prefix: str = "",
    document_prefix: str = "",
    triplet_prefix: str = "",
) -> dict[str, Any]:
    """Run both measurements for one encoder.

    `query_prefix` goes on retrieval queries, `document_prefix` on the pool
    texts they are matched against (models such as multilingual-e5 and
    EmbeddingGemma are trained with a marker on each side), and
    `triplet_prefix` on all three texts of a triplet. Triplets are
    paraphrase against near-miss, a symmetric task, so a retrieval
    instruction would not mean the same thing there; the default is no
    prefix, and a model that documents a symmetric-task prefix gets that one.
    """
    texts: list[str] = []
    for mode in POOL_MODES:
        texts.extend(document_prefix + pool_text(entry, mode) for entry in gold.pool)
    texts.extend(query_prefix + row["query"] for row in gold.queries)
    for row in gold.triplets:
        texts.extend(
            triplet_prefix + row[key] for key in ("anchor", "positive", "negative")
        )
    lookup = _embed_unique(embedder, texts)
    retrieval = {
        mode: _retrieval(lookup, gold, mode, query_prefix, document_prefix)
        for mode in POOL_MODES
    }
    triplets = _triplets(lookup, gold, triplet_prefix)
    mean_recall = sum(retrieval[mode]["recall_at_1"] for mode in POOL_MODES) / len(
        POOL_MODES
    )
    return {
        "dimension": len(next(iter(lookup.values()))),
        "retrieval": retrieval,
        "triplets": triplets,
        # A screening aid, not a decision: equal weight to linking and to the
        # hard triplets. The components are always reported beside it.
        "screening_score": round((mean_recall + triplets["accuracy"]) / 2, 4),
    }


# --------------------------------------------------------------------------
# Local backend (sentence-transformers), for encoders OpenRouter does not serve
# --------------------------------------------------------------------------
class LocalEmbedder:
    """Embeddings computed in-process with sentence-transformers.

    The import is lazy so that the rest of the module (and the offline test
    suite) never needs torch. Weights are downloaded from Hugging Face on
    first use; a gated model (EmbeddingGemma) needs HF_TOKEN in the
    environment from an account that accepted its terms.
    """

    def __init__(
        self,
        model_id: str,
        *,
        batch_size: int = 16,
        trust_remote_code: bool = False,
        loader: Callable[..., Any] | None = None,
    ) -> None:
        self.model_id = model_id
        self.batch_size = batch_size
        try:
            if loader is None:
                from sentence_transformers import SentenceTransformer as loader
            self._model = loader(model_id, trust_remote_code=trust_remote_code)
        except Exception as error:  # noqa: BLE001 - any load failure is one EncoderError
            raise EncoderError(
                f"could not load {model_id}: {type(error).__name__}: {error}"
            ) from None

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            matrix = self._model.encode(
                list(texts),
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as error:  # noqa: BLE001
            raise EncoderError(
                f"{self.model_id} failed to embed: {type(error).__name__}: {error}"
            ) from None
        return [[float(value) for value in row] for row in matrix]


# --------------------------------------------------------------------------
# OpenRouter client
# --------------------------------------------------------------------------


class OpenRouterEmbedder:
    """Embeddings through OpenRouter's OpenAI-compatible endpoint, stdlib only."""

    _RETRYABLE = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        slug: str,
        api_key: str,
        *,
        endpoint: str = OPENROUTER_EMBEDDINGS_URL,
        batch_size: int = 64,
        timeout: float = 60.0,
        retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.slug = slug
        self._api_key = api_key
        self.endpoint = endpoint
        self.batch_size = batch_size
        self.timeout = timeout
        self.retries = retries
        self._sleep = sleep

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(
                self._embed_batch(list(texts[start : start + self.batch_size]))
            )
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        body = json.dumps(
            {"model": self.slug, "input": batch, "encoding_format": "float"}
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                self.endpoint, data=body, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                if error.code in self._RETRYABLE and attempt < self.retries:
                    self._sleep(2**attempt)
                    continue
                raise EncoderError(
                    f"HTTP {error.code} from {self.slug}: {_error_detail(error)}"
                ) from error
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                if attempt < self.retries:
                    self._sleep(2**attempt)
                    continue
                raise EncoderError(
                    f"{type(error).__name__} calling {self.slug}: {error}"
                ) from error
            return self._parse(payload, len(batch))
        raise EncoderError(f"no response from {self.slug}")  # pragma: no cover

    def _parse(self, payload: Any, expected: int) -> list[list[float]]:
        if isinstance(payload, dict) and payload.get("error"):
            raise EncoderError(
                f"{self.slug} returned an error: {str(payload['error'])[:300]}"
            )
        try:
            items = sorted(
                enumerate(payload["data"]),
                key=lambda pair: pair[1].get("index", pair[0]),
            )
            vectors = [[float(x) for x in item["embedding"]] for _, item in items]
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise EncoderError(
                f"{self.slug} returned an unreadable embedding payload"
            ) from error
        if len(vectors) != expected:
            raise EncoderError(
                f"{self.slug} returned {len(vectors)} embeddings for {expected} inputs"
            )
        return vectors


def _error_detail(error: urllib.error.HTTPError) -> str:
    try:
        return error.read().decode("utf-8", errors="replace")[:300]
    except Exception:  # noqa: BLE001 -- the detail is a courtesy; the status code already says what failed
        return "no detail"


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def screening_verdict(results: list[dict[str, Any]]) -> str:
    scored = [row for row in results if "screening_score" in row]
    if not scored:
        return "No encoder produced a usable result."
    ordered = sorted(scored, key=lambda row: row["screening_score"], reverse=True)
    order = " > ".join(
        f"{row['name']} ({row['screening_score']:.3f})" for row in ordered
    )
    text = f"Screening order, not a decision: {order}."
    if len(ordered) >= 2:
        first, second = ordered[0], ordered[1]
        low_first = first["triplets"]["accuracy_ci"][0]
        high_second = second["triplets"]["accuracy_ci"][1]
        if low_first <= high_second:
            text += " The top two are within each other's 95% interval on the hard triplets: not separable on this data."
    return text


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def render_markdown(report: dict[str, Any]) -> str:
    results = [row for row in report.get("results", []) if "screening_score" in row]
    lines = [
        "### Text encoder bench (synthetic Italian phrases)",
        "",
        f"**{report.get('verdict', '')}**",
        "",
    ]
    if results:
        sample = results[0]
        lines.append(
            f"Linking: {sample['retrieval']['it']['n']} phrases over a pool of {report['settings']['pool_size']} structures. "
            f"Triplets: {sample['triplets']['n']}. Intervals are 95% Wilson; a gap smaller than the interval is not a gap."
        )
        lines += [
            "",
            "| Encoder | Origin / access | Dim | Link R@1 it | en | both | Triplets | 95% interval | Laterality | Negation | Size/unit | Temporal | Score |",
            "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
        for row in sorted(results, key=lambda r: r["screening_score"], reverse=True):
            retrieval, triplets = row["retrieval"], row["triplets"]
            categories = triplets["by_category"]

            def category(name: str, categories: dict[str, Any] = categories) -> str:
                return (
                    _pct(categories[name]["accuracy"]) if name in categories else "n/a"
                )

            low, high = triplets["accuracy_ci"]
            lines.append(
                f"| {row['name']} | {row.get('origin', '')}; {row.get('access', '')} | {row['dimension']} "
                f"| {_pct(retrieval['it']['recall_at_1'])} | {_pct(retrieval['en']['recall_at_1'])} | {_pct(retrieval['both']['recall_at_1'])} "
                f"| {_pct(triplets['accuracy'])} | {_pct(low)}-{_pct(high)} "
                f"| {category('laterality')} | {category('negation')} | {category('size_and_unit')} | {category('temporal_comparison')} "
                f"| {row['screening_score']:.3f} |"
            )
    preflight = report.get("preflight") or {}
    skipped = {name: reason for name, reason in preflight.items() if reason != "ok"}
    if skipped:
        lines += [
            "",
            "<details><summary>Candidates that did not run</summary>",
            "",
            "| Candidate | Reason |",
            "|---|---|",
        ]
        lines += [
            f"| {name} | {str(reason)[:200]} |"
            for name, reason in sorted(skipped.items())
        ]
        lines += ["", "</details>"]
    return "\n".join(lines) + "\n"
