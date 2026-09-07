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
from melampo.reasoning.rlm_engine import Budget

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

# The run that motivated these constants took 10.5 minutes and ended in
# failure because most or all candidates were unreachable (bad key, wrong
# model slug) -- and every one of them was still given the full 3 cases x up
# to 12 iterations x 60s-per-call budget before the failure became visible.
# A single bad candidate should fail in seconds, not minutes.
PREFLIGHT_TIMEOUT_SECONDS = 15
CALL_TIMEOUT_SECONDS = 30
# Six iterations and a 40s wall clock are plenty to describe, look, and
# answer three one-fact questions about a two-sentence document; this bench
# measures format adherence, not how far a model can be pushed. A factory,
# not a shared instance: Budget carries mutable per-run state (iteration
# count, start time), and reusing one instance across cases would corrupt
# both.
def _bench_budget() -> Budget:
    return Budget(max_iterations=6, max_wall_clock_seconds=40.0)


def _http_chat_completion(endpoint: str, api_key: str, model: str, prompt: str, *, timeout: int) -> str:
    """Minimal OpenAI-compatible chat completion call, stdlib only.

    Mistral, OpenRouter and most inference gateways implement this shape.
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
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


def build_candidates() -> tuple[dict[str, "callable"], list[str], dict[str, str]]:
    """Preflight every candidate whose key is present, then wrap the survivors.

    Returns (candidates, skipped, preflight_detail) rather than raising on a
    missing key or a failed preflight, so a partial run still produces a
    usable comparison and a failed one still explains itself.
    """
    candidates: dict[str, object] = {}
    skipped: list[str] = []
    preflight_detail: dict[str, str] = {}
    to_check: list[tuple[str, str, str, str]] = []  # (name, endpoint, key, model)

    mistral_key = os.environ.get("MISTRAL_API_KEY")
    if mistral_key:
        to_check.append(("mistral-small-3.1", "https://api.mistral.ai/v1/chat/completions", mistral_key, "mistral-small-latest"))
    else:
        skipped.append("mistral-small-3.1 (MISTRAL_API_KEY not set)")
        preflight_detail["mistral-small-3.1"] = "MISTRAL_API_KEY not set"

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    openrouter_models = (
        ("claude-sonnet-5", "anthropic/claude-sonnet-5"),
        ("claude-opus-5", "anthropic/claude-opus-5"),
        ("claude-fable-5.1", "anthropic/claude-fable-5.1"),
        ("gpt-6-astra", "openai/gpt-6-astra"),
        ("qwen-3.5", "qwen/qwen-3.5-72b-instruct"),
        ("glm-5", "z-ai/glm-5"),
        ("llama-3.3-70b", "meta-llama/llama-3.3-70b-instruct"),
        ("gemma-3-27b", "google/gemma-3-27b-it"),
    )
    if openrouter_key:
        # Every candidate here, Claude included, is reached through
        # OpenRouter's own catalogue rather than a first-party endpoint: one
        # key covers all eight instead of requiring a separate credential per
        # provider. OpenRouter is a named, established aggregator that proxies
        # to the real provider -- unlike an unverified gateway once considered
        # and rejected for this bench (see recursive_engine_decision_record.md).
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        for name, model in openrouter_models:
            to_check.append((name, endpoint, openrouter_key, model))
    else:
        skipped.append(
            "claude (all tiers), gpt-6-astra, qwen-3.5, glm-5, llama-3.3-70b, gemma-3-27b "
            "(OPENROUTER_API_KEY not set)"
        )
        for name, _ in openrouter_models:
            preflight_detail[name] = "OPENROUTER_API_KEY not set"

    if to_check:
        print(f"Preflighting {len(to_check)} candidate(s) (timeout {PREFLIGHT_TIMEOUT_SECONDS}s each)...")
    for name, endpoint, key, model in to_check:
        reachable, reason = _preflight(name, endpoint, key, model)
        preflight_detail[name] = reason
        if reachable:
            candidates[name] = _bind(_http_chat_completion, endpoint, key, model)
        else:
            skipped.append(f"{name} (preflight failed: {reason})")

    return candidates, skipped, preflight_detail


def _bind(fn, endpoint, key, model):
    def _call(prompt: str) -> str:
        try:
            return fn(endpoint, key, model, prompt, timeout=CALL_TIMEOUT_SECONDS)
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, json.JSONDecodeError) as error:
            # A provider error becomes empty text, which the engine already
            # treats as model_emitted_no_action -- consistent with how the
            # adapter treats a refused SafeModelClient call.
            print(f"  [warn] {model}: {error}", file=sys.stderr)
            return ""

    return _call


def _preflight(name: str, endpoint: str, key: str, model: str) -> tuple[bool, str]:
    """One short, short-timeout call per candidate before committing to the full bench.

    The run that motivated this function spent 10.5 minutes discovering that
    most candidates were unreachable, because each one was given the full
    per-case budget before its failure became visible. A bad key or an
    unrecognised model slug almost always fails fast (an auth or not-found
    response arrives in well under a second); a preflight call with a short
    timeout catches that in seconds per candidate instead of minutes.

    Returns (reachable, reason) so the reason survives into the results file
    rather than existing only as a line on stderr -- when every candidate
    fails, that reason is the entire useful output of the run.
    """
    try:
        response = _http_chat_completion(
            endpoint, key, model, "final(preflight check -- respond with exactly this action)",
            timeout=PREFLIGHT_TIMEOUT_SECONDS,
        )
        if not response.strip():
            print(f"  [preflight] {model}: reachable but returned empty text", file=sys.stderr)
            return True, "reachable, empty response to preflight"
        return True, "reachable"
    except urllib.error.HTTPError as error:
        # Distinguished from a generic URLError because the status code is the
        # single most useful diagnostic: 401/403 means the key, 404 means the
        # model slug, 429 means rate limiting. Guessing between those from a
        # generic message is what makes a failed run hard to act on.
        detail = f"HTTP {error.code} {error.reason}"
        print(f"  [preflight] {model}: unreachable ({detail}), skipping the full bench for it", file=sys.stderr)
        return False, detail
    except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError) as error:
        detail = f"{type(error).__name__}: {error}"
        print(f"  [preflight] {model}: unreachable ({detail}), skipping the full bench for it", file=sys.stderr)
        return False, detail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("bench_results.json"))
    args = parser.parse_args()

    candidates, skipped, preflight_detail = build_candidates()

    if not candidates:
        # Write the results file even here. This is the case where the
        # artifact matters most: the run failed, and the per-candidate reason
        # is the only thing that tells the operator whether to fix a key, a
        # model slug, or nothing at all. Exiting without writing leaves them
        # with an empty artifact and a red X.
        payload = {
            "status": "no_candidates",
            "verdict": (
                "no candidate survived preflight; check the per-candidate reasons below -- "
                "401/403 indicates the API key, 404 indicates the model slug, 429 indicates rate limiting"
            ),
            "models": 0,
            "results": [],
            "skipped": skipped,
            "preflight": preflight_detail,
        }
        _write_results(args.out, payload)
        print("\nNo candidate survived preflight; nothing to bench.", file=sys.stderr)
        for name, detail in sorted(preflight_detail.items()):
            print(f"  {name}: {detail}", file=sys.stderr)
        print(f"\nDiagnostics written to {args.out}", file=sys.stderr)
        return 1

    print(f"\nBenching: {', '.join(sorted(candidates))}")
    if skipped:
        print(f"Skipped: {'; '.join(skipped)}")

    report = bench_models(candidates, BENCH_CASES, adherence_target=0.95, budget_factory=_bench_budget)
    payload = report.as_dict()
    payload["status"] = "completed"
    payload["skipped"] = skipped
    payload["preflight"] = preflight_detail

    _write_results(args.out, payload)

    print(f"\nVerdict: {payload['verdict']}\n")
    print(f"{'model':<20}{'adherence':>11}{'completion':>12}{'near-miss share':>18}")
    for row in payload["results"]:
        print(f"{row['model_name']:<20}{row['adherence']:>10.0%}{row['completion_rate']:>12.0%}{row['near_miss_share']:>17.0%}")
    print(f"\nFull report written to {args.out}")
    return 0


def _write_results(path: Path, payload: dict) -> None:
    """Write results, creating the parent directory if the caller named one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    sys.exit(main())
