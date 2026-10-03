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
OpenRouter (IBM Granite R2, EmbeddingGemma) need a local backend and are not
covered here.
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
    OpenRouterEmbedder,
    evaluate_encoder,
    load_gold,
    render_markdown,
    screening_verdict,
)

# (bench_name, openrouter_slug, query_prefix, origin, access)
CANDIDATE_ENCODERS = [
    (
        "qwen3-embedding-8b",
        "qwen/qwen3-embedding-8b",
        "",
        "China (Alibaba)",
        "open weights, Apache-2.0",
    ),
    (
        "qwen3-embedding-8b-instruct",
        "qwen/qwen3-embedding-8b",
        QWEN_QUERY_INSTRUCTION,
        "China (Alibaba)",
        "open weights, Apache-2.0",
    ),
    (
        "qwen3-embedding-4b",
        "qwen/qwen3-embedding-4b",
        "",
        "China (Alibaba)",
        "open weights, Apache-2.0",
    ),
    ("bge-m3", "baai/bge-m3", "", "China (BAAI)", "open weights, MIT"),
    (
        "mistral-embed",
        "mistralai/mistral-embed-2312",
        "",
        "France (Mistral)",
        "API only",
    ),
    (
        "openai-text-embedding-3-large",
        "openai/text-embedding-3-large",
        "",
        "USA (OpenAI)",
        "API only",
    ),
    (
        "google-gemini-embedding-001",
        "google/gemini-embedding-001",
        "",
        "USA (Google)",
        "API only",
    ),
]

PREFLIGHT_PROBE = ["prova di raggiungibilità"]


def _select(
    roster: str | None, extra_slugs: str | None
) -> tuple[list[tuple], list[str]]:
    wanted = (
        {name.strip() for name in roster.split(",") if name.strip()} if roster else None
    )
    unknown = (
        sorted(wanted - {entry[0] for entry in CANDIDATE_ENCODERS}) if wanted else []
    )
    chosen = [
        entry for entry in CANDIDATE_ENCODERS if wanted is None or entry[0] in wanted
    ]
    for slug in (part.strip() for part in (extra_slugs or "").split(",")):
        if slug:
            chosen.append(
                (slug.replace("/", "-"), slug, "", "unspecified", "unspecified")
            )
    return chosen, unknown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="encoder_results.json")
    parser.add_argument(
        "--markdown", help="also write a markdown summary here (for the job summary)"
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--roster", help="comma-separated candidate names; default all")
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
        print(json.dumps([entry[0] for entry in CANDIDATE_ENCODERS]))
        return 0
    if unknown:
        print(f"Unknown candidate name(s) in --roster: {unknown}", file=sys.stderr)
        return 1

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

    for name, slug, prefix, origin, access in chosen:
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
        started = time.monotonic()
        try:
            measured = evaluate_encoder(embedder, gold, query_prefix=prefix)
        except EncoderError as error:
            report["preflight"][name] = f"failed during the run: {error}"
            continue
        report["results"].append(
            {
                "name": name,
                "slug": slug,
                "origin": origin,
                "access": access,
                "query_prefix": bool(prefix),
                "seconds": round(time.monotonic() - started, 1),
                **measured,
            }
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
