"""Run the CIFSYN-style and DeepEL-style linking bench on the synthetic Italian anatomy set.

Modes: ``sparse`` (no network), ``contextual`` (embeddings via OpenRouter),
``deepel`` (embeddings plus two chat models via OpenRouter), ``all``.
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
DEFAULT_ENCODERS = (
    "google-gemini-embedding-001",
    "voyage-4-large",
    "nemotron-3-embed-1b",
)
CHAT_MODELS = {
    "nemotron-3-super": "nvidia/nemotron-3-super-120b-a12b",
    "gemma-3-27b": "google/gemma-3-27b-it",
}


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
        "--mode", choices=("sparse", "contextual", "deepel", "all"), default="all"
    )
    parser.add_argument("--out", default="linking_results.json")
    parser.add_argument("--markdown")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--encoders", help="comma-separated; default " + ",".join(DEFAULT_ENCODERS)
    )
    parser.add_argument("--deepel-encoder", default="google-gemini-embedding-001")
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
            "OPENROUTER_API_KEY is not set: only the sparse part ran", file=sys.stderr
        )
        args.mode = "sparse"

    if args.mode in ("contextual", "all"):
        report["contextual"] = {}
        for name in _names(args.encoders, ENCODERS, DEFAULT_ENCODERS):
            try:
                report["contextual"][name] = lb.evaluate_contextual(
                    OpenRouterEmbedder(ENCODERS[name], key), gold, cases
                )
            except EncoderError as error:
                report["contextual"][name] = {"error": str(error)[:300]}

    if args.mode in ("deepel", "all"):
        embedder = OpenRouterEmbedder(ENCODERS[args.deepel_encoder], key)
        report["deepel"] = {}
        for name in _names(args.chat_models, CHAT_MODELS, tuple(CHAT_MODELS)):
            try:
                result = lb.evaluate_deepel_style(
                    lb.OpenRouterChat(CHAT_MODELS[name], key),
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

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
    if args.markdown:
        Path(args.markdown).write_text(lb.render_markdown(report), "utf-8")
    print(lb.render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
