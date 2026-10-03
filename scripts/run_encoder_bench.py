"""Measure candidate text encoders on synthetic Italian anatomical phrases, through OpenRouter.

Manual, billed (cents), never on every commit: see .github/workflows/encoder-bench.yml.
The phrases are synthetic, so nothing patient-related leaves the machine.

What is measured and why is in src/melampo/evaluation/encoder_bench.py. This
script only selects candidates, checks that each one answers at all, runs the
measurement, and writes the files.

    OPENROUTER_API_KEY=... uv run python scripts/run_encoder_bench.py --out encoder_results.json

The results file is written on every path, including when no candidate
could be reached, because the reason a run produced nothing is exactly what
the artifact is for.

Candidates are (name, OpenRouter slug, query prefix, origin, access). The
slugs are the ones OpenRouter listed when this was written; a slug it no
longer serves is reported by the preflight, not fatal to the others. Anything
else can be added for one run with --extra-slugs. Encoders that are not on
OpenRouter (IBM Granite R2, EmbeddingGemma, Snowflake Arctic, Nomic) are in
LOCAL_CANDIDATES and run with --backend local: computed on this machine with
sentence-transformers, which must be installed (CPU torch is enough).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from melampo.evaluation.encoder_bench import (
    DEFAULT_DATA_DIR,
    QWEN_QUERY_INSTRUCTION,
    EncoderError,
    LocalEmbedder,
    OpenRouterEmbedder,
    evaluate_encoder,
    load_gold,
    render_markdown,
    screening_verdict,
)

# OpenRouter candidates:
# (bench_name, openrouter_slug, query_prefix, origin, access, options)
# `options` may carry `document_prefix` and `triplet_prefix` for models trained
# with a marker on each side (multilingual-e5).
CANDIDATE_ENCODERS = [
    (
        "qwen3-embedding-8b",
        "qwen/qwen3-embedding-8b",
        "",
        "China (Alibaba)",
        "open weights, Apache-2.0",
        {},
    ),
    (
        "qwen3-embedding-8b-instruct",
        "qwen/qwen3-embedding-8b",
        QWEN_QUERY_INSTRUCTION,
        "China (Alibaba)",
        "open weights, Apache-2.0",
        {},
    ),
    (
        "qwen3-embedding-4b",
        "qwen/qwen3-embedding-4b",
        "",
        "China (Alibaba)",
        "open weights, Apache-2.0",
        {},
    ),
    ("bge-m3", "baai/bge-m3", "", "China (BAAI)", "open weights, MIT", {}),
    (
        "mistral-embed",
        "mistralai/mistral-embed-2312",
        "",
        "France (Mistral)",
        "API only",
        {},
    ),
    (
        "openai-text-embedding-3-large",
        "openai/text-embedding-3-large",
        "",
        "USA (OpenAI)",
        "API only",
        {},
    ),
    (
        "google-gemini-embedding-001",
        "google/gemini-embedding-001",
        "",
        "USA (Google)",
        "API only",
        {},
    ),
    # Added 2026-10-03: everything else OpenRouter serves that can plausibly
    # handle Italian. English-only models (bge-*-en, e5-*-v2, gte-*, MiniLM,
    # mpnet, ada-002) are left out on purpose; --extra-slugs adds any of them.
    (
        "google-gemini-embedding-2",
        "google/gemini-embedding-2",
        "",
        "USA (Google)",
        "API only",
        {},
    ),
    (
        "openai-text-embedding-3-small",
        "openai/text-embedding-3-small",
        "",
        "USA (OpenAI)",
        "API only",
        {},
    ),
    ("voyage-4", "voyageai/voyage-4", "", "USA (Voyage)", "API only", {}),
    (
        "voyage-4-large",
        "voyageai/voyage-4-large",
        "",
        "USA (Voyage)",
        "API only",
        {},
    ),
    ("voyage-4-lite", "voyageai/voyage-4-lite", "", "USA (Voyage)", "API only", {}),
    (
        "nemotron-3-embed-1b",
        "nvidia/nemotron-3-embed-1b:free",
        "",
        "USA (NVIDIA)",
        "open weights, licence to verify; free tier, rate-limited",
        {},
    ),
    (
        "pplx-embed-v1-4b",
        "perplexity/pplx-embed-v1-4b",
        "",
        "USA (Perplexity)",
        "licence to verify",
        {},
    ),
    (
        "pplx-embed-v1-0.6b",
        "perplexity/pplx-embed-v1-0.6b",
        "",
        "USA (Perplexity)",
        "licence to verify",
        {},
    ),
    (
        "liquid-lfm-2.5-embedding-350m",
        "liquid/lfm-2.5-embedding-350m:free",
        "",
        "USA (Liquid AI)",
        "open weights, licence to verify; free tier, rate-limited",
        {},
    ),
    (
        "multilingual-e5-large",
        "intfloat/multilingual-e5-large",
        "query: ",
        "China (Microsoft Research Asia)",
        "open weights, MIT",
        {"document_prefix": "passage: ", "triplet_prefix": "query: "},
    ),
]

# Local candidates, computed on the machine that runs the script with
# sentence-transformers (see LocalEmbedder). They are the open, non-Chinese
# encoders OpenRouter does not serve. Model ids are Hugging Face ids; the
# prefixes are the ones each model card documents.
LOCAL_CANDIDATES = [
    {
        "name": "granite-311m-multilingual-r2",
        "model_id": "ibm-granite/granite-embedding-311m-multilingual-r2",
        "origin": "USA (IBM)",
        "access": "open weights, Apache-2.0",
    },
    {
        "name": "granite-97m-multilingual-r2",
        "model_id": "ibm-granite/granite-embedding-97m-multilingual-r2",
        "origin": "USA (IBM)",
        "access": "open weights, Apache-2.0",
    },
    {
        "name": "embeddinggemma-300m",
        "model_id": "google/embeddinggemma-300m",
        "origin": "USA (Google)",
        "access": "open weights, Gemma terms; gated, needs HF_TOKEN",
        "query_prefix": "task: search result | query: ",
        "document_prefix": "title: none | text: ",
        "triplet_prefix": "task: sentence similarity | query: ",
    },
    {
        "name": "arctic-embed-l-v2",
        "model_id": "Snowflake/snowflake-arctic-embed-l-v2.0",
        "origin": "USA (Snowflake)",
        "access": "open weights, Apache-2.0",
        "query_prefix": "query: ",
    },
    {
        "name": "nomic-embed-text-v2-moe",
        "model_id": "nomic-ai/nomic-embed-text-v2-moe",
        "origin": "USA (Nomic)",
        "access": "open weights, Apache-2.0",
        "query_prefix": "search_query: ",
        "document_prefix": "search_document: ",
        "triplet_prefix": "search_query: ",
        "trust_remote_code": True,
    },
]

PREFLIGHT_PROBE = ["prova di raggiungibilità"]


def _local_names() -> list[str]:
    return [entry["name"] for entry in LOCAL_CANDIDATES]


def _select(
    roster: str | None, extra_slugs: str | None
) -> tuple[list[tuple], list[str]]:
    """OpenRouter candidates to run, and roster names that exist nowhere."""
    wanted = (
        {name.strip() for name in roster.split(",") if name.strip()} if roster else None
    )
    known = {entry[0] for entry in CANDIDATE_ENCODERS} | set(_local_names())
    unknown = sorted(wanted - known) if wanted else []
    chosen = [
        entry for entry in CANDIDATE_ENCODERS if wanted is None or entry[0] in wanted
    ]
    for slug in (part.strip() for part in (extra_slugs or "").split(",")):
        if slug:
            chosen.append(
                (slug.replace("/", "-"), slug, "", "unspecified", "unspecified", {})
            )
    return chosen, unknown


def _select_local(roster: str | None, backend: str) -> list[dict]:
    """Local candidates for this backend: the roster's, else all of them."""
    if backend == "openrouter":
        return []
    if roster:
        wanted = {name.strip() for name in roster.split(",") if name.strip()}
        return [entry for entry in LOCAL_CANDIDATES if entry["name"] in wanted]
    return list(LOCAL_CANDIDATES)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="encoder_results.json")
    parser.add_argument(
        "--markdown", help="also write a markdown summary here (for the job summary)"
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--roster", help="comma-separated candidate names; default all")
    parser.add_argument(
        "--backend",
        choices=("openrouter", "local", "all"),
        default="openrouter",
        help="which candidates run when --roster is empty; local ones are computed "
        "here with sentence-transformers (needs torch, downloads weights)",
    )
    parser.add_argument(
        "--extra-slugs", help="comma-separated OpenRouter slugs to add for this run"
    )
    parser.add_argument(
        "--list-candidates",
        action="store_true",
        help="print the candidate names as JSON and exit",
    )
    args = parser.parse_args(argv)

    chosen, unknown = _select(args.roster, args.extra_slugs)
    if args.list_candidates:
        print(json.dumps([entry[0] for entry in CANDIDATE_ENCODERS] + _local_names()))
        return 0
    if unknown:
        print(f"Unknown candidate name(s) in --roster: {unknown}", file=sys.stderr)
        return 1
    if args.backend == "local":
        chosen = [entry for entry in chosen if entry[4] == "unspecified"]
    local = _select_local(args.roster, args.backend)
    if not chosen and not local:
        print("Nothing to run for this backend and roster.")
        return 0

    gold = load_gold(Path(args.data_dir))
    api_key = os.environ.get("OPENROUTER_API_KEY")
    report: dict = {
        "settings": {
            "pool_size": len(gold.pool),
            "queries": len(gold.queries),
            "triplets": len(gold.triplets),
            "generated_unix": int(time.time()),
        },
        "results": [],
        "preflight": {},
    }

    def measure(name, slug, embedder, origin, access, prefixes):
        started = time.monotonic()
        try:
            measured = evaluate_encoder(embedder, gold, **prefixes)
        except EncoderError as error:
            report["preflight"][name] = f"failed during the run: {error}"
            return
        report["results"].append(
            {
                "name": name,
                "slug": slug,
                "origin": origin,
                "access": access,
                "query_prefix": bool(prefixes.get("query_prefix")),
                "seconds": round(time.monotonic() - started, 1),
                **measured,
            }
        )

    for name, slug, prefix, origin, access, options in chosen:
        if not api_key:
            report["preflight"][name] = "OPENROUTER_API_KEY not set"
            continue
        embedder = OpenRouterEmbedder(slug, api_key)
        try:
            embedder(PREFLIGHT_PROBE)
        except EncoderError as error:
            report["preflight"][name] = str(error)
            continue
        report["preflight"][name] = "ok"
        measure(
            name, slug, embedder, origin, access, {"query_prefix": prefix, **options}
        )

    for entry in local:
        name = entry["name"]
        try:
            embedder = LocalEmbedder(
                entry["model_id"],
                trust_remote_code=entry.get("trust_remote_code", False),
            )
            embedder(PREFLIGHT_PROBE)
        except EncoderError as error:
            report["preflight"][name] = str(error)
            continue
        report["preflight"][name] = "ok"
        prefixes = {
            key: entry[key]
            for key in ("query_prefix", "document_prefix", "triplet_prefix")
            if key in entry
        }
        measure(
            name,
            entry["model_id"],
            embedder,
            entry["origin"],
            entry["access"],
            prefixes,
        )

    report["verdict"] = screening_verdict(report["results"])
    Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.markdown:
        Path(args.markdown).write_text(render_markdown(report), encoding="utf-8")
    print(report["verdict"])
    return 0 if report["results"] else 1


if __name__ == "__main__":
    sys.exit(main())
