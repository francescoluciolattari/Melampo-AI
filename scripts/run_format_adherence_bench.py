#!/usr/bin/env python3
"""Run the root-model format-adherence bench against real endpoints.

Reads model API keys from environment variables — populated from GitHub
Actions secrets in CI, exported locally otherwise — and never accepts a key as
a command-line argument or literal, for the same reason the PMC connector
does not: a key is a value that must never appear in a diff, a log, or a shell
history file.

A model whose secret is absent is skipped rather than causing the run to fail.
Partial results are useful — knowing that three of four candidates were
reachable is better than no result because the fourth key was never set — and
the report says explicitly which were skipped and why.

Usage:
    python scripts/run_format_adherence_bench.py [--out results.json]

Environment variables consulted, all optional:
    MISTRAL_API_KEY, OPENROUTER_API_KEY

Candidates that need a key with no variable set are reported as skipped, not
silently dropped.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from melampo.evaluation.format_adherence_bench import (
    BenchCase,
    bench_models,
)
from melampo.memory.context_environment import EnvironmentDocument

# Synthetic only. This bench measures format adherence, not clinical reasoning,
# and phase-one data-class discipline (see rlm_engine.py) applies here as
# everywhere else the environment is populated.
BENCH_DOCUMENT = EnvironmentDocument(
    document_id="report_1",
    text=(
        "Chest radiograph shows bibasilar opacities. Prednisone 40 mg daily was "
        "started. The patient reports progressive dyspnoea over three weeks with "
        "no fever."
    ),
    source="synthetic_bench_fixture",
    metadata={"data_class": "synthetic"},
)

BENCH_CASES = (
    BenchCase("dose", (BENCH_DOCUMENT,), "What steroid dose was started, and from which document?"),
    BenchCase("finding", (BENCH_DOCUMENT,), "What imaging finding is documented?"),
    BenchCase("symptom_duration", (BENCH_DOCUMENT,), "How long has the dyspnoea been present?"),
)

ACTION_GRAMMAR = (
    "describe() | grep(pattern) | slice(document_id, start, end) | "
    "search(query) | expand(concept) | final(answer)"
)


def _http_chat_completion(endpoint: str, api_key: str, model: str, prompt: str) -> str:
    """Minimal OpenAI-compatible chat completion call, stdlib only.

    Mistral, OpenRouter and most inference gateways implement this shape.
    Google's native Gemini endpoint does not, and is handled separately below.
    """
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You navigate a document environment by emitting exactly one action per line, "
                        f"chosen from: {ACTION_GRAMMAR}. Emit nothing else -- no prose, no explanation. "
                        "Call final(answer) once you can answer the question."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 256,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


def build_candidates() -> tuple[dict[str, "callable"], list[str]]:
    """Construct callables for every candidate whose key is present.

    Returns (candidates, skipped) rather than raising on a missing key, so a
    partial run still produces a usable comparison.
    """
    candidates: dict[str, object] = {}
    skipped: list[str] = []

    mistral_key = os.environ.get("MISTRAL_API_KEY")
    if mistral_key:
        for name, model in (("mistral-small-3.1", "mistral-small-latest"),):
            candidates[name] = _bind(_http_chat_completion, "https://api.mistral.ai/v1/chat/completions", mistral_key, model)
    else:
        skipped.append("mistral-small-3.1 (MISTRAL_API_KEY not set)")

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        # Every remaining candidate, Claude included, is reached through
        # OpenRouter's own catalogue rather than a first-party endpoint: one
        # key covers all four instead of requiring a separate credential per
        # provider. OpenRouter is a named, established aggregator that proxies
        # to the real provider -- unlike an unverified gateway once considered
        # and rejected for this bench (see recursive_engine_decision_record.md).
        for name, model in (
            ("claude", "anthropic/claude-sonnet-4.6"),
            ("qwen-3.5", "qwen/qwen-3.5-72b-instruct"),
            ("llama-3.3-70b", "meta-llama/llama-3.3-70b-instruct"),
            ("gemma-3-27b", "google/gemma-3-27b-it"),
        ):
            candidates[name] = _bind(
                _http_chat_completion, "https://openrouter.ai/api/v1/chat/completions", openrouter_key, model
            )
    else:
        skipped.append("claude, qwen-3.5, llama-3.3-70b, gemma-3-27b (OPENROUTER_API_KEY not set)")

    return candidates, skipped


def _bind(fn, endpoint, key, model):
    def _call(prompt: str) -> str:
        try:
            return fn(endpoint, key, model, prompt)
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError) as error:
            # A provider error becomes empty text, which the engine already
            # treats as model_emitted_no_action -- consistent with how the
            # adapter treats a refused SafeModelClient call.
            print(f"  [warn] {model}: {error}", file=sys.stderr)
            return ""

    return _call


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("bench_results.json"))
    args = parser.parse_args()

    candidates, skipped = build_candidates()
    if not candidates:
        print("No API keys found in the environment; nothing to bench.", file=sys.stderr)
        print("Set at least one of MISTRAL_API_KEY, OPENROUTER_API_KEY.", file=sys.stderr)
        return 1

    print(f"Benching: {', '.join(sorted(candidates))}")
    if skipped:
        print(f"Skipped (no key): {'; '.join(skipped)}")

    report = bench_models(candidates, BENCH_CASES, adherence_target=0.95)
    payload = report.as_dict()
    payload["skipped"] = skipped

    args.out.write_text(json.dumps(payload, indent=2))

    print(f"\nVerdict: {payload['verdict']}\n")
    print(f"{'model':<20}{'adherence':>11}{'completion':>12}{'near-miss share':>18}")
    for row in payload["results"]:
        print(f"{row['model_name']:<20}{row['adherence']:>10.0%}{row['completion_rate']:>12.0%}{row['near_miss_share']:>17.0%}")
    print(f"\nFull report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
