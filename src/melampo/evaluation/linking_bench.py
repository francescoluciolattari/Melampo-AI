"""Entity-linking bench: a CIFSYN-style contextual re-ranker and a DeepEL-style LLM linker.

Both are measured on the same 63 structures and 189 Italian phrases as
`encoder_bench`, so the numbers are comparable with the encoder results.

**What this is not.** Neither published system is run. CIFSYN (J. Biomed.
Inform. 2025) is a trained model for English; no public code was found and it
would need Italian annotated mentions to train. DeepEL (arXiv 2511.14181) is
general-domain, built on GPT-4 and BLINK. What is tested here is the one idea
each paper contributes, zero-shot, on this project's data:

- **CIFSYN's idea**: a mention has context and a candidate does not, so put
  the candidate *into the mention's sentence* and compare the two sentences.
  `contextual` scores ``cos(mention, candidate) + lambda * cos(sentence with
  the mention, sentence with the candidate)`` for lambda in `LAMBDAS`, with
  lambda = 0 as the no-context baseline.
- **DeepEL's idea**: use an LLM at every stage. (1) describe the mention and
  retrieve with both the mention and the description; (2) multiple choice among
  the merged candidates; (3) validate the choice by putting it back in the
  sentence. Stage 2 gets an explicit "none of these" option, which the paper
  does not have, because here an unresolved mention is the safe outcome.

**Outcomes are three-way, never two.** A linker that answers wrongly in silence
is the failure this project is built against, so every run reports correct,
wrong and abstained separately. An LLM answer that cannot be parsed counts as
an abstention and is tallied on its own.

**The contexts are ours.** `linking_contexts.json` holds two carrier sentences
per anatomical region (abdomen, lung, spine...). They name the region, never
the structure, but they were written for this bench, so the context benefit is
an upper bound for real reports. DeepEL's global-consistency stage is about
several entities in one sentence; each carrier sentence has one, so stage 3
here exercises its validation prompt, not inter-entity reasoning.
"""

import json
import math
import re
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .encoder_bench import (
    DEFAULT_DATA_DIR,
    EncoderError,
    Gold,
    _embed_unique,
    pool_text,
)

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
LAMBDAS = (0.0, 0.5, 1.0, 2.0)
TOP_K = 10
OUTCOME_CORRECT = "correct"
OUTCOME_WRONG = "wrong"
OUTCOME_ABSTAIN = "abstain"

ChatFn = Callable[[str], str]
Embedder = Callable[[Sequence[str]], list[list[float]]]


# --------------------------------------------------------------------------
# Cases: a phrase, its sentence, its gold structure
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    query: str
    target: str
    kind: str
    template: str
    sentence: str


def load_contexts(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, Any]:
    return json.loads((data_dir / "linking_contexts.json").read_text("utf-8"))


def build_cases(gold: Gold, contexts: dict[str, Any]) -> list[Case]:
    """One sentence per query: the region's templates, taken in turn."""
    group_of: dict[str, str] = {}
    for name, group in contexts["groups"].items():
        for member in group["members"]:
            if member in group_of:
                raise ValueError(f"{member!r} is in two context groups")
            group_of[member] = name
    unmapped = [entry["id"] for entry in gold.pool if entry["id"] not in group_of]
    if unmapped:
        raise ValueError(f"no context group for: {unmapped}")
    seen: Counter[str] = Counter()
    cases: list[Case] = []
    for query in gold.queries:
        templates = contexts["groups"][group_of[query["target"]]]["templates"]
        template = templates[seen[query["target"]] % len(templates)]
        seen[query["target"]] += 1
        cases.append(
            Case(
                query=query["query"],
                target=query["target"],
                kind=query["kind"],
                template=template,
                sentence=template.format(m=query["query"]),
            )
        )
    return cases


# --------------------------------------------------------------------------
# Sparse candidate generation (character n-gram TF-IDF, as in the BioSyn family)
# --------------------------------------------------------------------------


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _ngrams(text: str, n: int = 3) -> Counter[str]:
    padded = f" {_fold(text)} "
    return Counter(padded[i : i + n] for i in range(max(1, len(padded) - n + 1)))


class SparseIndex:
    """TF-IDF over character trigrams of every label (Italian and English)."""

    def __init__(self, pool: Sequence[dict[str, Any]]) -> None:
        self.entries: list[tuple[str, Counter[str]]] = []
        for entry in pool:
            for language in ("it", "en"):
                self.entries.append((entry["id"], _ngrams(entry[language])))
        document_frequency: Counter[str] = Counter()
        for _, grams in self.entries:
            document_frequency.update(grams.keys())
        total = len(self.entries)
        self.idf = {
            g: math.log((1 + total) / (1 + df)) + 1
            for g, df in document_frequency.items()
        }
        self.vectors = [(cid, self._weigh(grams)) for cid, grams in self.entries]

    def _weigh(self, grams: Counter[str]) -> dict[str, float]:
        weights = {g: c * self.idf.get(g, 1.0) for g, c in grams.items()}
        norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
        return {g: w / norm for g, w in weights.items()}

    def top_k(self, mention: str, k: int = TOP_K) -> list[str]:
        query = self._weigh(_ngrams(mention))
        best: dict[str, float] = {}
        for cid, vector in self.vectors:
            score = sum(w * vector.get(g, 0.0) for g, w in query.items())
            if score > best.get(cid, -1.0):
                best[cid] = score
        return [cid for cid, _ in sorted(best.items(), key=lambda p: (-p[1], p[0]))[:k]]


def _tally(cases: Sequence[Case], outcomes: Sequence[str]) -> dict[str, Any]:
    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    for case, outcome in zip(cases, outcomes, strict=True):
        by_kind[case.kind][outcome] += 1
    total = Counter(outcomes)
    n = len(outcomes)

    def rate(counter: Counter[str], key: str, size: int) -> float:
        return round(counter[key] / size, 4) if size else 0.0

    return {
        "n": n,
        "correct": rate(total, OUTCOME_CORRECT, n),
        "wrong": rate(total, OUTCOME_WRONG, n),
        "abstain": rate(total, OUTCOME_ABSTAIN, n),
        "by_kind": {
            kind: {
                "n": sum(c.values()),
                "correct": rate(c, OUTCOME_CORRECT, sum(c.values())),
                "wrong": rate(c, OUTCOME_WRONG, sum(c.values())),
                "abstain": rate(c, OUTCOME_ABSTAIN, sum(c.values())),
            }
            for kind, c in sorted(by_kind.items())
        },
    }


def evaluate_sparse(gold: Gold, cases: Sequence[Case]) -> dict[str, Any]:
    """Recall of the gold structure among the sparse candidates (no network)."""
    index = SparseIndex(gold.pool)
    ranks = []
    for case in cases:
        candidates = index.top_k(case.query, 50)
        ranks.append(
            candidates.index(case.target) + 1 if case.target in candidates else None
        )
    n = len(ranks)
    return {
        "n": n,
        **{
            f"recall_at_{k}": round(
                sum(1 for r in ranks if r is not None and r <= k) / n, 4
            )
            for k in (1, 3, 5, 10)
        },
    }


# --------------------------------------------------------------------------
# CIFSYN-style: put the candidate into the mention's sentence
# --------------------------------------------------------------------------


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def evaluate_contextual(
    embedder: Embedder,
    gold: Gold,
    cases: Sequence[Case],
    *,
    lambdas: Sequence[float] = LAMBDAS,
    k: int = TOP_K,
    pool_mode: str = "both",
) -> dict[str, Any]:
    """Rank candidates by mention similarity plus lambda times sentence similarity."""
    labels = {entry["id"]: pool_text(entry, pool_mode) for entry in gold.pool}
    texts: list[str] = list(labels.values())
    for case in cases:
        texts.append(case.query)
        texts.append(case.sentence)
    for template in {case.template for case in cases}:
        texts.extend(template.format(m=label) for label in labels.values())
    vectors = _embed_unique(embedder, texts)

    results: dict[str, Any] = {}
    for lam in lambdas:
        outcomes: list[str] = []
        in_top_k = 0
        for case in cases:
            entity = {
                cid: cosine(vectors[case.query], vectors[label])
                for cid, label in labels.items()
            }
            shortlist = sorted(entity, key=lambda c: (-entity[c], c))[:k]
            in_top_k += case.target in shortlist
            scored = {
                cid: entity[cid]
                + lam
                * cosine(
                    vectors[case.sentence], vectors[case.template.format(m=labels[cid])]
                )
                for cid in shortlist
            }
            winner = max(scored, key=lambda c: (scored[c], c))
            outcomes.append(OUTCOME_CORRECT if winner == case.target else OUTCOME_WRONG)
        results[f"lambda_{lam:g}"] = {
            **_tally(cases, outcomes),
            "recall_at_k_of_shortlist": round(in_top_k / len(cases), 4),
        }
    return results


# --------------------------------------------------------------------------
# DeepEL-style: LLM at every stage
# --------------------------------------------------------------------------

_NUMBER = re.compile(r"-?\d+")


def parse_choice(answer: str, n_options: int) -> int | None:
    """The first integer in the reply, or None when it is not an option."""
    match = _NUMBER.search(answer or "")
    if not match:
        return None
    value = int(match.group())
    return value if 0 <= value <= n_options else None


def _describe_prompt(case: Case) -> str:
    return (
        "Sei un radiologo. In una frase breve (massimo 25 parole), in italiano, "
        f"dì a quale struttura anatomica si riferisce l'espressione «{case.query}» "
        f"nel testo seguente, senza ripetere il testo.\n\nTesto: {case.sentence}"
    )


def _choice_prompt(case: Case, description: str, labels: Sequence[str]) -> str:
    options = "\n".join(f"{i}. {label}" for i, label in enumerate(labels, start=1))
    return (
        "Collega l'espressione a una struttura anatomica. Rispondi solo con il numero "
        "dell'opzione; rispondi 0 se nessuna opzione è corretta.\n\n"
        f"Testo: {case.sentence}\nEspressione: {case.query}\n"
        f"Interpretazione: {description}\n\nOpzioni:\n{options}\n0. nessuna di queste\n\nNumero:"
    )


def _validate_prompt(case: Case, chosen: str, labels: Sequence[str]) -> str:
    options = "\n".join(f"{i}. {label}" for i, label in enumerate(labels, start=1))
    replaced = case.template.format(m=chosen)
    return (
        "Controllo di coerenza. Nel testo l'espressione è stata sostituita con la "
        f"struttura scelta.\n\nTesto originale: {case.sentence}\nTesto con la sostituzione: "
        f"{replaced}\n\nLa sostituzione conserva il significato del testo originale? "
        "Rispondi con il numero di un'altra opzione se ritieni che sia sbagliata, "
        "con 0 se nessuna opzione è corretta, oppure con 9999 se la sostituzione è corretta.\n\n"
        f"Opzioni:\n{options}\n0. nessuna di queste\n\nNumero:"
    )


CONFIRM = 9999


def _outcome(chosen: str | None, case: Case) -> str:
    if chosen is None:
        return OUTCOME_ABSTAIN
    return OUTCOME_CORRECT if chosen == case.target else OUTCOME_WRONG


def link_one(
    case: Case,
    description: str,
    chat: ChatFn,
    labels: dict[str, str],
    vectors: dict[str, list[float]],
    *,
    k: int = TOP_K,
) -> dict[str, Any]:
    """Stages 2 and 3 of the DeepEL-style linker for one phrase, given stage 1's description."""

    def ranked(text_vector: list[float]) -> list[str]:
        scores = {cid: cosine(text_vector, vectors[labels[cid]]) for cid in labels}
        return sorted(scores, key=lambda c: (-scores[c], c))

    by_mention = ranked(vectors[case.query])
    by_description = ranked(vectors[description]) if description else []
    merged: list[str] = []
    for pair in zip(by_mention, by_description or by_mention, strict=False):
        for cid in pair:
            if cid not in merged:
                merged.append(cid)
    merged = merged[:k]
    option_ids = merged

    record: dict[str, Any] = {
        "query": case.query,
        "target": case.target,
        "kind": case.kind,
        "description": description,
        "gold_in_mention_top_k": case.target in by_mention[:k],
        "gold_in_merged_top_k": case.target in merged,
    }

    record.update(_agent(case, description, chat, option_ids, labels))
    return record


def _agent(
    case: Case,
    description: str,
    chat: ChatFn,
    option_ids: Sequence[str],
    labels: dict[str, str],
) -> dict[str, Any]:
    """Multiple choice among the options, then validation; what each step decided."""
    option_labels = [labels[cid] for cid in option_ids]
    record: dict[str, Any] = {}
    reply = chat(_choice_prompt(case, description, option_labels))
    choice = parse_choice(reply, len(option_ids))
    record["stage2_unparsed"] = choice is None
    chosen = option_ids[choice - 1] if choice else None
    record["stage2"] = _outcome(chosen, case)
    record["stage2_choice"] = chosen

    final = chosen
    if chosen is not None:
        verdict = parse_choice(
            chat(_validate_prompt(case, labels[chosen], option_labels)), CONFIRM
        )
        record["stage3_unparsed"] = verdict is None
        if verdict is None or verdict == 0:
            final = None
        elif verdict != CONFIRM and verdict <= len(option_ids):
            final = option_ids[verdict - 1]
        record["stage3_changed"] = final != chosen
    record["stage3"] = _outcome(final, case)
    record["stage3_choice"] = final
    return record


def evaluate_deepel_style(
    chat: ChatFn,
    embedder: Embedder,
    gold: Gold,
    cases: Sequence[Case],
    *,
    workers: int = 6,
    pool_mode: str = "both",
    k: int = TOP_K,
) -> dict[str, Any]:
    """Per-stage correct/wrong/abstain for one chat model."""
    labels = {entry["id"]: pool_text(entry, pool_mode) for entry in gold.pool}
    # Stage 1 first, for every phrase: the descriptions must be embedded before retrieval.
    described = _run(
        workers, lambda c: (chat(_describe_prompt(c)) or "").strip(), cases
    )
    texts = (
        list(labels.values()) + [c.query for c in cases] + [d for d in described if d]
    )
    vectors = _embed_unique(embedder, texts)
    pairs = list(zip(cases, described, strict=True))
    records = _run(
        workers,
        lambda pair: link_one(pair[0], pair[1], chat, labels, vectors, k=k),
        pairs,
    )

    summary: dict[str, Any] = {
        "records": len(records),
        "gold_in_mention_top_k": round(
            sum(r["gold_in_mention_top_k"] for r in records) / len(records), 4
        ),
        "gold_in_merged_top_k": round(
            sum(r["gold_in_merged_top_k"] for r in records) / len(records), 4
        ),
        "stage2_unparsed": sum(r["stage2_unparsed"] for r in records),
        "stage3_unparsed": sum(r.get("stage3_unparsed", False) for r in records),
        "stage3_changed": sum(r.get("stage3_changed", False) for r in records),
    }
    for stage in ("stage2", "stage3"):
        summary[stage] = _tally(cases, [r[stage] for r in records])
    summary["_records"] = records
    return summary


def evaluate_chain(
    chat: ChatFn,
    embedder: Embedder,
    gold: Gold,
    cases: Sequence[Case],
    *,
    lam: float = 2.0,
    keep: int = 5,
    to_agent: int = 3,
    k: int = TOP_K,
    workers: int = 6,
    pool_mode: str = "both",
) -> dict[str, Any]:
    """The pipeline as designed: a CIFSYN-style re-ranker keeps `keep` candidates,
    the best `to_agent` go to a DeepEL-style agent that chooses and validates.

    Stage 1 here is the re-ranker, stage 2 the agent's choice, stage 3 its
    validation. The re-ranker's own top-1 is the baseline the agent must beat, so
    the report counts what the agent fixed, broke, and caught.
    """
    labels = {entry["id"]: pool_text(entry, pool_mode) for entry in gold.pool}
    texts: list[str] = list(labels.values())
    for case in cases:
        texts.extend((case.query, case.sentence))
    for template in {case.template for case in cases}:
        texts.extend(template.format(m=label) for label in labels.values())
    vectors = _embed_unique(embedder, texts)

    def reranked(case: Case) -> list[str]:
        entity = {
            cid: cosine(vectors[case.query], vectors[label])
            for cid, label in labels.items()
        }
        shortlist = sorted(entity, key=lambda c: (-entity[c], c))[:k]
        scored = {
            cid: entity[cid]
            + lam
            * cosine(
                vectors[case.sentence], vectors[case.template.format(m=labels[cid])]
            )
            for cid in shortlist
        }
        return sorted(scored, key=lambda c: (-scored[c], c))

    def one(case: Case) -> dict[str, Any]:
        order = reranked(case)
        description = (chat(_describe_prompt(case)) or "").strip()
        record = {
            "query": case.query,
            "target": case.target,
            "kind": case.kind,
            "rerank_top1": order[0],
            "rerank_top1_outcome": _outcome(order[0], case),
            "gold_in_kept": case.target in order[:keep],
            "gold_in_agent_options": case.target in order[:to_agent],
            "description": description,
        }
        record.update(_agent(case, description, chat, order[:to_agent], labels))
        return record

    records = _run(workers, one, cases)
    n = len(records)
    summary: dict[str, Any] = {
        "records": n,
        "lambda": lam,
        "gold_in_kept": round(sum(r["gold_in_kept"] for r in records) / n, 4),
        "gold_in_agent_options": round(
            sum(r["gold_in_agent_options"] for r in records) / n, 4
        ),
        "stage2_unparsed": sum(r["stage2_unparsed"] for r in records),
        "stage3_unparsed": sum(r.get("stage3_unparsed", False) for r in records),
        "rerank_top1": _tally(cases, [r["rerank_top1_outcome"] for r in records]),
    }
    for stage in ("stage2", "stage3"):
        summary[stage] = _tally(cases, [r[stage] for r in records])
    summary["agent_vs_rerank"] = _transitions(records, "stage3")
    summary["_records"] = records
    return summary


def _transitions(records: Sequence[dict[str, Any]], stage: str) -> dict[str, int]:
    """What the agent did to the re-ranker's top-1, case by case."""
    counts: Counter[str] = Counter()
    for record in records:
        before = record["rerank_top1_outcome"]
        after = record[stage]
        if before == OUTCOME_CORRECT:
            counts[
                {
                    OUTCOME_CORRECT: "kept_correct",
                    OUTCOME_WRONG: "broke_a_correct",
                    OUTCOME_ABSTAIN: "abstained_on_a_correct",
                }[after]
            ] += 1
        else:
            counts[
                {
                    OUTCOME_CORRECT: "fixed_a_wrong",
                    OUTCOME_WRONG: "kept_wrong",
                    OUTCOME_ABSTAIN: "caught_a_wrong",
                }[after]
            ] += 1
    return dict(counts)


def _run(
    workers: int, function: Callable[[Any], Any], items: Sequence[Any]
) -> list[Any]:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(function, items))


def agreement(
    first: dict[str, Any],
    second: dict[str, Any],
    cases: Sequence[Case],
    stage: str = "stage3",
) -> dict[str, Any]:
    """Accept an answer only when two models give the same one; otherwise abstain."""
    outcomes = []
    for a, b, case in zip(first["_records"], second["_records"], cases, strict=True):
        same = (
            a[f"{stage}_choice"] is not None
            and a[f"{stage}_choice"] == b[f"{stage}_choice"]
        )
        outcomes.append(_outcome(a[f"{stage}_choice"] if same else None, case))
    return _tally(cases, outcomes)


# --------------------------------------------------------------------------
# Chat client (stdlib only)
# --------------------------------------------------------------------------


class OpenRouterChat:
    _RETRYABLE = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        slug: str,
        api_key: str,
        *,
        endpoint: str = OPENROUTER_CHAT_URL,
        timeout: float = 90.0,
        retries: int = 2,
        sleep: Callable[[float], None] = time.sleep,
        max_tokens: int = 200,
    ) -> None:
        self.slug = slug
        self._api_key = api_key
        self.endpoint = endpoint
        self.timeout = timeout
        self.retries = retries
        self._sleep = sleep
        self.max_tokens = max_tokens

    def __call__(self, prompt: str) -> str:
        hint = True
        for attempt in range(self.retries + 2):
            body = {
                "model": self.slug,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": self.max_tokens,
                **({"reasoning": {"enabled": False}} if hint else {}),
            }
            request = urllib.request.Request(
                self.endpoint,
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                if error.code == 400 and hint:
                    hint = (
                        False  # the model rejects the reasoning hint: retry without it
                    )
                    continue
                if error.code in self._RETRYABLE and attempt < self.retries + 1:
                    self._sleep(2**attempt)
                    continue
                raise EncoderError(f"HTTP {error.code} from {self.slug}") from error
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                if attempt < self.retries + 1:
                    self._sleep(2**attempt)
                    continue
                raise EncoderError(
                    f"{type(error).__name__} calling {self.slug}"
                ) from error
            try:
                return str(payload["choices"][0]["message"]["content"] or "")
            except (KeyError, IndexError, TypeError) as error:
                raise EncoderError(
                    f"{self.slug} returned an unreadable chat payload"
                ) from error
        raise EncoderError(f"no response from {self.slug}")  # pragma: no cover


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Linking bench (CIFSYN-style and DeepEL-style, zero-shot)", ""]
    sparse = report.get("sparse")
    if sparse:
        lines += [
            "## Sparse candidates (trigram TF-IDF, no network)",
            "",
            f"Recall of the gold structure: @1 {_pct(sparse['recall_at_1'])}, @3 {_pct(sparse['recall_at_3'])}, "
            f"@5 {_pct(sparse['recall_at_5'])}, @10 {_pct(sparse['recall_at_10'])}.",
            "",
        ]
    contextual = report.get("contextual", {})
    if contextual:
        lines += [
            "## CIFSYN-style contextual re-ranking",
            "",
            "| Encoder | lambda | correct | wrong | shortlist recall |",
            "|---|---|---|---|---|",
        ]
        for name, per_lambda in contextual.items():
            if isinstance(per_lambda, dict) and "error" in per_lambda:
                lines.append(f"| {name} | - | error: {per_lambda['error']} | | |")
                continue
            for key, row in per_lambda.items():
                lines.append(
                    f"| {name} | {key.removeprefix('lambda_')} | {_pct(row['correct'])} | {_pct(row['wrong'])} | {_pct(row['recall_at_k_of_shortlist'])} |"
                )
        lines.append("")
    chain = report.get("chain", {})
    if chain:
        lines += [
            "## Chain: CIFSYN-style re-ranker, then DeepEL-style agent",
            "",
            "| Model | gold in kept 5 / agent's 3 | re-ranker top-1 correct/wrong | agent choice c/w/a | after validation c/w/a | fixed / broke / abstained on correct / caught wrong |",
            "|---|---|---|---|---|---|",
        ]
        for name, row in chain.items():
            if "error" in row:
                lines.append(f"| {name} | error: {row['error']} | | | | |")
                continue
            r1, s2, s3, t = (
                row["rerank_top1"],
                row["stage2"],
                row["stage3"],
                row["agent_vs_rerank"],
            )
            lines.append(
                f"| {name} | {_pct(row['gold_in_kept'])} / {_pct(row['gold_in_agent_options'])} | "
                f"{_pct(r1['correct'])} / {_pct(r1['wrong'])} | "
                f"{_pct(s2['correct'])} / {_pct(s2['wrong'])} / {_pct(s2['abstain'])} | "
                f"{_pct(s3['correct'])} / {_pct(s3['wrong'])} / {_pct(s3['abstain'])} | "
                f"{t.get('fixed_a_wrong', 0)} / {t.get('broke_a_correct', 0)} / {t.get('abstained_on_a_correct', 0)} / {t.get('caught_a_wrong', 0)} |"
            )
        lines.append("")
        for name, row in report.get("chain_agreement", {}).items():
            lines.append(
                f"Catena, accordo fra i due modelli ({name}): corretti {_pct(row['correct'])}, errati {_pct(row['wrong'])}, astenuti {_pct(row['abstain'])}."
            )
        lines.append("")
    deepel = report.get("deepel", {})
    if deepel:
        lines += [
            "## DeepEL-style (LLM at every stage)",
            "",
            "| Model | encoder | gold in mention top-k | gold in merged top-k | stage 2 correct/wrong/abstain | stage 3 correct/wrong/abstain | unparsed 2/3 |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, row in deepel.items():
            if "error" in row:
                lines.append(f"| {name} | | error: {row['error']} | | | | |")
                continue
            s2, s3 = row["stage2"], row["stage3"]
            lines.append(
                f"| {name} | {row.get('encoder', '')} | {_pct(row['gold_in_mention_top_k'])} | {_pct(row['gold_in_merged_top_k'])} | "
                f"{_pct(s2['correct'])} / {_pct(s2['wrong'])} / {_pct(s2['abstain'])} | "
                f"{_pct(s3['correct'])} / {_pct(s3['wrong'])} / {_pct(s3['abstain'])} | {row['stage2_unparsed']}/{row['stage3_unparsed']} |"
            )
        lines.append("")
        for name, row in report.get("agreement", {}).items():
            lines.append(
                f"Accordo fra i due modelli ({name}): corretti {_pct(row['correct'])}, errati {_pct(row['wrong'])}, astenuti {_pct(row['abstain'])}."
            )
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The full anatomy linker: lexicon -> checks -> two models -> checks -> abstain
# --------------------------------------------------------------------------


def _describe_en(mention: str, sentence: str) -> str:
    return (
        "You are a radiologist. In one short sentence (max 25 words), in English, say which "
        f"anatomical structure the expression «{mention}» refers to in this report sentence. "
        f"Do not repeat the sentence.\n\nSentence: {sentence}"
    )


class EmbeddingRetriever:
    """DeepEL-style candidates: rank the pool by the mention and by a model's description, merged."""

    def __init__(
        self, embedder: Embedder, pool: Sequence[Any], describe: ChatFn | None = None
    ) -> None:
        self._embedder = embedder
        self._pool = list(pool)
        self._describe = describe
        self._cache: dict[str, list[float]] = {}
        self._pool_vectors: list[list[float]] | None = None

    def _vectors(self, texts: Sequence[str]) -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            self._cache.update(zip(missing, self._embedder(missing), strict=True))
        return [self._cache[t] for t in texts]

    def _rank(self, vector: list[float]) -> list[str]:
        if self._pool_vectors is None:
            self._pool_vectors = self._vectors([c.label for c in self._pool])
        scores = [
            (cosine(vector, v), c.cid)
            for c, v in zip(self._pool, self._pool_vectors, strict=True)
        ]
        return [cid for _, cid in sorted(scores, key=lambda p: (-p[0], p[1]))]

    def __call__(self, mention: str, sentence: str, k: int) -> list[str]:
        lists = [self._rank(self._vectors([mention])[0])[:k]]
        if self._describe is not None:
            description = (
                self._describe(_describe_en(mention, sentence)) or ""
            ).strip()
            if description:
                lists.append(self._rank(self._vectors([description])[0])[:k])
        merged: list[str] = []
        for group in zip(*lists, strict=False):
            for cid in group:
                if cid not in merged:
                    merged.append(cid)
        return merged[:k]


def upper_error_bound(errors: int, n: int, confidence: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound on the error rate among n accepted links."""
    if n == 0:
        return 1.0
    if errors == 0:
        return 1 - (1 - confidence) ** (1 / n)
    if errors >= n:
        return 1.0
    lo, hi = errors / n, 1.0
    for _ in range(60):  # bisection on the binomial tail
        mid = (lo + hi) / 2
        tail = sum(
            math.comb(n, i) * mid**i * (1 - mid) ** (n - i) for i in range(errors + 1)
        )
        lo, hi = (mid, hi) if tail > 1 - confidence else (lo, mid)
    return hi


def evaluate_anatomy_linker(
    linker: Any, rows: Sequence[dict[str, Any]], workers: int = 4
) -> dict[str, Any]:
    """Four outcomes per mention, never two: correct, wrong (silent), other concept, abstained."""
    from ..memory.anatomy_linker import ACCEPTED

    results = _run(workers, lambda r: linker.link(r["mention"], r["sentence"]), rows)
    tally: Counter[str] = Counter()
    fallback: Counter[str] = Counter()
    by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    reasons: Counter[str] = Counter()
    details: list[dict[str, Any]] = []
    for row, result in zip(rows, results, strict=True):
        target = row["target"]
        if result.status == ACCEPTED:
            is_class = result.cid in getattr(linker.lexicon, "classes", {})
            if result.cid == target:
                outcome = "correct"
            elif is_class:
                outcome = "wrong"
            else:
                outcome = "other_concept"
                # What the graph's parent proposal would have been worth (not applied).
                proposed = getattr(result, "fallback", "")
                if proposed:
                    fallback[
                        "matches_target"
                        if proposed == target
                        else "differs_from_target"
                    ] += 1
        else:
            outcome = "abstained_as_expected" if target is None else "abstained"
            reasons[result.reason] += 1
        tally[outcome] += 1
        by_stage[result.stage][outcome] += 1
        if outcome not in ("correct", "abstained_as_expected"):
            details.append(
                {
                    "mention": row["mention"],
                    "sentence": row["sentence"],
                    "target": target,
                    "kind": row.get("kind"),
                    "outcome": outcome,
                    "chosen": result.cid,
                    "relation": getattr(result, "relation", "equal"),
                    "part": getattr(result, "part", ""),
                    "fallback": getattr(result, "fallback", ""),
                    "stage": result.stage,
                    "reason": result.reason,
                    "votes": result.votes,
                }
            )
    accepted_class = tally["correct"] + tally["wrong"]
    with_target = sum(1 for r in rows if r["target"] is not None)
    return {
        "n": len(rows),
        "with_target": with_target,
        "outcomes": dict(tally),
        "graph_fallback_proposals": dict(fallback),
        "by_stage": {k: dict(v) for k, v in by_stage.items()},
        "abstention_reasons": dict(reasons),
        "precision_of_accepted": round(tally["correct"] / accepted_class, 4)
        if accepted_class
        else None,
        "error_rate_upper_95": round(
            upper_error_bound(tally["wrong"], accepted_class), 4
        ),
        "coverage": round(tally["correct"] / with_target, 4) if with_target else None,
        "details": details,
    }


def render_linker_markdown(report: dict[str, Any]) -> str:
    lines = [
        "## Anatomy linker (lexicon, checks, two models, abstention)",
        "",
        "| Set | n | correct | wrong (silent) | other concept | abstained | of which expected | precision of accepted | error-rate upper bound (95%) | coverage |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, row in report.items():
        o = row["outcomes"]
        precision = (
            "-"
            if row["precision_of_accepted"] is None
            else _pct(row["precision_of_accepted"])
        )
        coverage = "-" if row["coverage"] is None else _pct(row["coverage"])
        lines.append(
            f"| {name} | {row['n']} | {o.get('correct', 0)} | {o.get('wrong', 0)} | {o.get('other_concept', 0)} | "
            f"{o.get('abstained', 0) + o.get('abstained_as_expected', 0)} | {o.get('abstained_as_expected', 0)} | "
            f"{precision} | {_pct(row['error_rate_upper_95'])} | {coverage} |"
        )
    lines.append("")
    for name, row in report.items():
        flagged = [
            d for d in row["details"] if d["outcome"] in ("wrong", "other_concept")
        ]
        if flagged:
            lines.append(
                f"**{name}: links to check** (other_concept is not a pass: each one needs a human look, in the 2026-10-05 run 3 of 38 were wrong)"
            )
            for d in flagged:
                lines.append(
                    f"- {d['outcome']}: «{d['mention']}» expected {d['target']}, got {d['chosen']} ({d['stage']})"
                )
            lines.append("")
    return "\n".join(lines)
