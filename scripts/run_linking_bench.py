"""Run the CIFSYN-style and DeepEL-style linking bench on the synthetic Italian anatomy set.

Modes: ``linker`` (the full anatomy linker: lexicon, checks, two models,
abstention; its deterministic part runs offline), ``sparse`` (no network), ``contextual`` (embeddings via OpenRouter),
``deepel`` (embeddings plus two chat models), ``chain`` (re-ranker then agent, the
pipeline as designed), ``all``.
Slugs are declared here and nowhere else. Needs OPENROUTER_API_KEY except for
``sparse``. Synthetic phrases only: no patient data leaves the machine.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from melampo.evaluation import linking_bench as lb  # noqa: E402
from melampo.memory import anatomy_linker as al  # noqa: E402
from melampo.memory import anatomy_parts as ap  # noqa: E402
from melampo.memory import form_ambiguity as fa  # noqa: E402
from melampo.memory.anatomy_graph import AnatomyGraph  # noqa: E402
from melampo.memory.blind_reader import BlindReader  # noqa: E402
from melampo.evaluation.encoder_bench import (  # noqa: E402
    DEFAULT_DATA_DIR,
    EncoderError,
    OpenRouterEmbedder,
    load_gold,
)

ENCODERS = {
    "google-gemini-embedding-001": "google/gemini-embedding-001",
    "voyage-4-large": "voyageai/voyage-4-large",
    "nemotron-3-embed-1b": "nvidia/nemotron-3-embed-1b:free",
    "qwen3-embedding-8b": "qwen/qwen3-embedding-8b",
}
# Gemini answers HTTP 429 (per-minute quota) when sent ~25 large batches back to
# back, as the contextual stage does: smaller batches, a pause between them.
PACING = {"google-gemini-embedding-001": {"batch_size": 16, "pause": 4.0, "retries": 8}}
DEFAULT_ENCODERS = (
    "google-gemini-embedding-001",
    "voyage-4-large",
    "nemotron-3-embed-1b",
)
CHAT_MODELS = {
    "nemotron-3-super": "nvidia/nemotron-3-super-120b-a12b",
    "gemma-3-27b": "google/gemma-3-27b-it",
}


# The chat models answer HTTP 429 when several rows ask at once (the 2026-10-07 run lost every
# model answer after 7 seconds of retries): wait as long as the server says, up to a minute,
# and keep at least half a second between two requests of the same model.
CHAT_PACING = {"retries": 8, "min_interval": 0.5, "max_wait": 60.0}


def _chat(slug: str, key: str) -> lb.OpenRouterChat:
    return lb.OpenRouterChat(slug, key, **CHAT_PACING)


def _embedder(name: str, key: str) -> OpenRouterEmbedder:
    return OpenRouterEmbedder(
        ENCODERS[name], key, **({"retries": 6} | PACING.get(name, {}))
    )


def _linker_sets(data_dir: Path, cases) -> dict:
    sets = {
        "dev_it": [
            {
                "mention": c.query,
                "sentence": c.sentence,
                "target": c.target,
                "kind": c.kind,
            }
            for c in cases
        ]
    }
    for stem in ("heldout", "heldout2"):
        for language in ("it", "en"):
            path = data_dir.parent / "linking" / f"{stem}_{language}.jsonl"
            if not path.exists():
                continue
            sets[f"{stem}_{language}"] = [
                json.loads(line)
                for line in path.read_text("utf-8").splitlines()
                if line.strip()
            ]
    return sets


def _run_linker(args, gold, cases, key: str):
    data_dir = Path(args.data_dir)
    lexicon = al.Lexicon.from_json(
        json.loads(
            (data_dir.parent / "linking" / "anatomy_lexicon.json").read_text("utf-8")
        )
    )
    terms = []
    graph = None
    if Path(args.uberon).exists():
        with open(args.uberon, encoding="utf-8") as handle:
            terms = al.load_obo_terms(handle)
        if not args.no_graph:
            with open(args.uberon, encoding="utf-8") as handle:
                graph = AnatomyGraph.from_obo(handle)
    else:
        print(
            f"{args.uberon} not found: the pool has no ontology distractors",
            file=sys.stderr,
        )
    pool, equivalent = al.build_pool(lexicon, terms)
    sets = _linker_sets(data_dir, cases)
    parts = ap.PartTable.from_json(
        json.loads(
            (data_dir.parent / "linking" / "anatomy_parts.json").read_text("utf-8")
        ),
        lexicon,
    )

    parts_json = json.loads(
        (data_dir.parent / "linking" / "anatomy_parts.json").read_text("utf-8")
    )
    blind = BlindReader.from_sources(lexicon, terms, equivalent, graph, parts_json)
    deterministic = al.AnatomyLinker(
        lexicon,
        pool,
        equivalent,
        parts=parts,
        graph=graph,
        blind=blind,
        blind_veto=args.blind_veto,
    )
    det = {
        name: lb.evaluate_anatomy_linker(deterministic, rows)
        for name, rows in sets.items()
    }
    for row in det.values():
        row["pool_size"] = len(pool)
        row["graph"] = graph is not None
    if not key:
        return det, {}

    chats = {name: _chat(slug, key) for name, slug in CHAT_MODELS.items()}
    retriever = lb.EmbeddingRetriever(
        _embedder(args.deepel_encoder, key), pool, describe=chats["nemotron-3-super"]
    )
    flags = fa.audit(
        lexicon,
        [
            (name, entry["whole"])
            for entry in json.loads(
                (data_dir.parent / "linking" / "anatomy_parts.json").read_text("utf-8")
            )["direct"]
            for name in entry["names"]
        ],
    )
    full = al.AnatomyLinker(
        lexicon,
        pool,
        equivalent,
        retriever=retriever,
        chats=chats,
        parts=parts,
        graph=graph,
        ambiguous=fa.keys(flags),
        verify=args.verify,
        blind=blind,
        blind_veto=args.blind_veto,
    )
    try:
        report = {
            name: lb.evaluate_anatomy_linker(full, rows, workers=args.workers)
            for name, rows in sets.items()
        }
    except EncoderError as error:
        return det, {
            "error": {
                "n": 0,
                "outcomes": {},
                "details": [],
                "error": str(error)[:300],
                "precision_of_accepted": None,
                "error_rate_upper_95": 1.0,
                "coverage": None,
            }
        }
    for row in report.values():
        row["pool_size"] = len(pool)
        row["graph"] = graph is not None
        row["verify"] = bool(args.verify)
    return det, report


def _names(
    value: str | None, allowed: dict[str, str], default: tuple[str, ...]
) -> list[str]:
    names = (
        [n.strip() for n in value.split(",") if n.strip()] if value else list(default)
    )
    unknown = [n for n in names if n not in allowed]
    if unknown:
        raise SystemExit(f"unknown names {unknown}; choose from {sorted(allowed)}")
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--mode",
        choices=("sparse", "contextual", "deepel", "chain", "linker", "all"),
        default="all",
    )
    parser.add_argument("--out", default="linking_results.json")
    parser.add_argument("--markdown")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--encoders", help="comma-separated; default " + ",".join(DEFAULT_ENCODERS)
    )
    parser.add_argument("--deepel-encoder", default="voyage-4-large")
    parser.add_argument("--chain-lambda", type=float, default=2.0)
    parser.add_argument(
        "--uberon",
        default="data/linking/uberon-basic.obo",
        help="UBERON OBO file for the distractor pool (the workflow downloads a pinned release)",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="do not use the UBERON graph (neighbour check, fallback to the parent): to compare",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="both models read the sentence for every link made from the name alone to a form the "
        "data flags as able to mean more than one thing (form_ambiguity); extra calls",
    )
    parser.add_argument(
        "--blind-veto",
        action="store_true",
        help="the blind reader (no models) may stop an accepted link it confidently contradicts; "
        "without it the reader only writes its verdict in the trace",
    )
    parser.add_argument("--chat-models", help="comma-separated; default both")
    parser.add_argument(
        "--limit", type=int, help="use only the first N phrases (smoke test)"
    )
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    gold = load_gold(data_dir)
    cases = lb.build_cases(gold, lb.load_contexts(data_dir))
    if args.limit:
        cases = cases[: args.limit]
    report: dict = {"settings": {"phrases": len(cases), "pool": len(gold.pool)}}

    report["sparse"] = lb.evaluate_sparse(gold, cases)

    key = os.environ.get("OPENROUTER_API_KEY", "")
    if args.mode != "sparse" and not key:
        print(
            "OPENROUTER_API_KEY is not set: only the offline parts ran "
            "(sparse candidates and the linker's deterministic stages)",
            file=sys.stderr,
        )
        args.mode = "linker" if args.mode in ("linker", "all") else "sparse"

    if args.mode in ("contextual", "all"):
        report["contextual"] = {}
        for name in _names(args.encoders, ENCODERS, DEFAULT_ENCODERS):
            try:
                report["contextual"][name] = lb.evaluate_contextual(
                    _embedder(name, key), gold, cases
                )
            except EncoderError as error:
                report["contextual"][name] = {"error": str(error)[:300]}

    if args.mode in ("deepel", "all"):
        embedder = _embedder(args.deepel_encoder, key)
        report["deepel"] = {}
        for name in _names(args.chat_models, CHAT_MODELS, tuple(CHAT_MODELS)):
            try:
                result = lb.evaluate_deepel_style(
                    _chat(CHAT_MODELS[name], key),
                    embedder,
                    gold,
                    cases,
                    workers=args.workers,
                )
                result["encoder"] = args.deepel_encoder
                report["deepel"][name] = result
            except EncoderError as error:
                report["deepel"][name] = {"error": str(error)[:300]}
        ok = [r for r in report["deepel"].values() if "error" not in r]
        if len(ok) == 2:
            report["agreement"] = {
                "stage2": lb.agreement(ok[0], ok[1], cases, "stage2"),
                "stage3": lb.agreement(ok[0], ok[1], cases, "stage3"),
            }

    if args.mode in ("chain", "all"):
        embedder = _embedder(args.deepel_encoder, key)
        report["chain"] = {}
        for name in _names(args.chat_models, CHAT_MODELS, tuple(CHAT_MODELS)):
            try:
                result = lb.evaluate_chain(
                    _chat(CHAT_MODELS[name], key),
                    embedder,
                    gold,
                    cases,
                    lam=args.chain_lambda,
                    workers=args.workers,
                )
                result["encoder"] = args.deepel_encoder
                report["chain"][name] = result
            except EncoderError as error:
                report["chain"][name] = {"error": str(error)[:300]}
        ok = [r for r in report["chain"].values() if "error" not in r]
        if len(ok) == 2:
            report["chain_agreement"] = {
                "stage3": lb.agreement(ok[0], ok[1], cases, "stage3")
            }

    if args.mode in ("linker", "all"):
        report["linker_deterministic"], report["linker"] = _run_linker(
            args, gold, cases, key
        )

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
    text = lb.render_markdown(report)
    for key_name, title in (
        ("linker_deterministic", "deterministic stages only"),
        ("linker", "with the two models"),
    ):
        if report.get(key_name):
            text += f"\n### {title}\n\n" + lb.render_linker_markdown(report[key_name])
    if args.markdown:
        Path(args.markdown).write_text(text, "utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
