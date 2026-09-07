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
import time
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
    # Added after a first live run showed only 33% completion across seven
    # candidates with n=3 -- too small a sample to tell a genuine limitation
    # from noise (one case is 33 percentage points). Six cases quantise
    # completion_rate into ~17% steps instead of ~33%, and cover facts a
    # single-question bench could not distinguish: an absence rather than a
    # presence, and a qualifier on an already-asked symptom.
    BenchCase("fever_status", (BENCH_DOCUMENT,), "Is fever present according to the report?"),
    BenchCase("onset_pattern", (BENCH_DOCUMENT,), "Is the dyspnoea described as progressive or sudden?"),
    BenchCase("treatment_frequency", (BENCH_DOCUMENT,), "How often is the prednisone dose taken?"),
)

ACTION_GRAMMAR = (
    "describe() | grep(pattern) | slice(document_id, start, end) | "
    "search(query) | expand(concept) | final(answer)"
)

# Rewritten after a first live run showed three distinct failure patterns that
# a short instruction left room for: two Claude tiers exhausted their full
# iteration budget without ever calling final() despite well-formed actions
# throughout (over-verification, not confusion); Claude Opus produced
# dialogue-style artifacts ("human error", "Assistantslice(...)") suggesting
# it was pattern-matching the rendered history onto a chat transcript; GPT-6
# Astra narrated tool output in prose ("[doc 0] note.txt (152 chars)") instead
# of emitting the next action. A worked example and explicit prohibitions
# target each pattern directly rather than hoping a longer budget alone fixes
# behaviour a longer budget cannot address.
_SYSTEM_PROMPT = (
    "You navigate a document environment by emitting exactly one action per line, "
    f"chosen from: {ACTION_GRAMMAR}. Emit nothing else: no prose, no explanation, no "
    "commentary on what an action returned, no role labels or dialogue formatting "
    "(never write \"human\" or \"Assistant\" or similar).\n\n"
    "One clean lookup that answers the question is enough. Call final(answer) as soon "
    "as you can answer -- do not re-verify with a second lookup if the first one already "
    "gave you the answer; every extra action spends part of a small, fixed budget.\n\n"
    "Example of a complete, correct exchange for a question like "
    "\"What dose was prescribed?\":\n"
    "grep(dose)\n"
    "final(40 mg daily)\n\n"
    "That is the whole exchange: one lookup, then final() on the next turn. Longer "
    "exchanges are for questions one lookup cannot answer, not the default."
)

# The run that motivated these constants took 10.5 minutes and ended in
# failure because most or all candidates were unreachable (bad key, wrong
# model slug) -- and every one of them was still given the full 3 cases x up
# to 12 iterations x 60s-per-call budget before the failure became visible.
# A single bad candidate should fail in seconds, not minutes.
PREFLIGHT_TIMEOUT_SECONDS = 15
CALL_TIMEOUT_SECONDS = 30
# A first live run showed two model families (Claude Sonnet and Fable) hitting
# max_iterations=6 on every single case without ever calling final() -- 100%
# adherence, 0% completion, always stopped by the ceiling rather than by
# choice. That is the signature of a budget too tight for those models'
# exploration style, not a comprehension failure: they were emitting
# well-formed actions the whole time. Raised to 10/60s; report.budget_bound
# on each result now says explicitly whether the ceiling was the limiting
# factor, so this number can be revisited with evidence instead of guessing
# again. A factory, not a shared instance: Budget carries mutable per-run
# state (iteration count, start time), and reusing one instance across cases
# would corrupt both.
def _bench_budget() -> Budget:
    return Budget(max_iterations=10, max_wall_clock_seconds=60.0)


RATE_LIMIT_MAX_RETRIES = 1
RATE_LIMIT_DEFAULT_WAIT_SECONDS = 5.0
RATE_LIMIT_MAX_WAIT_SECONDS = 20.0


def _http_chat_completion(
    endpoint: str, api_key: str, model: str, prompt: str, *, timeout: int, disable_reasoning: bool = False
) -> str:
    """Minimal OpenAI-compatible chat completion call, stdlib only.

    Mistral, OpenRouter and most inference gateways implement this shape.

    ``disable_reasoning`` sends OpenRouter's own ``reasoning: {enabled: false}``
    parameter (documented at openrouter.ai/docs/use-cases/reasoning-tokens),
    best-effort: models that always reason (GLM-5.3's listing states this
    explicitly) will ignore it, and models without a reasoning mode at all are
    unaffected either way. Scoped to OpenRouter calls only -- the direct
    Mistral endpoint's tolerance for unrecognised top-level fields is not
    verified from here, so the hint is not sent there.

    A 429 is retried once, honouring the provider's own ``Retry-After`` header
    when present rather than guessing a wait. Mistral's free evaluation tier in
    particular has conservative per-second limits and documents that it
    returns this header on every 429; ignoring it and failing immediately
    turns a transient, self-resolving condition into a permanently skipped
    candidate. One retry, not a backoff loop: this is a preflight-scale
    utility, not a production client, and a candidate that is rate-limited
    twice in a row is more informatively reported as such than retried
    indefinitely.
    """
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            # Raised from 256 after a first live run showed 0% completion on
            # two model families that used well-formed actions the entire
            # time (100% adherence). GLM-5.3's own listing states its
            # reasoning "is always on and cannot be disabled"; several other
            # candidates here are reasoning-capable by default. If hidden
            # reasoning tokens are consuming the completion budget before the
            # visible action line is ever written, 256 tokens may simply not
            # have left room for both -- and there is no way to tell that
            # apart from "genuinely stuck" without more room to see the whole
            # output. 1024 gives that room while staying a small fraction of
            # a cent per call at every candidate's pricing.
            "max_tokens": 1024,
            **({"reasoning": {"enabled": False}} if disable_reasoning else {}),
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )

    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return _extract_content(payload)
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt >= RATE_LIMIT_MAX_RETRIES:
                raise
            wait = _parse_retry_after(error.headers.get("Retry-After") if error.headers else None)
            print(f"  [rate limit] {model}: HTTP 429, waiting {wait:.0f}s before one retry", file=sys.stderr)
            time.sleep(wait)
            attempt += 1


def _parse_retry_after(header_value: str | None) -> float:
    """Read Retry-After as seconds, capped, falling back when absent or malformed.

    RFC 7231 also allows an HTTP-date in this header; that form is not parsed
    here -- on the small set of providers this script calls, a delta-seconds
    value is what has been documented, and an unparseable value falls back to
    the default wait rather than raising, since a malformed header should not
    prevent the retry it announces.
    """
    if header_value is None:
        return RATE_LIMIT_DEFAULT_WAIT_SECONDS
    try:
        seconds = float(header_value)
    except ValueError:
        return RATE_LIMIT_DEFAULT_WAIT_SECONDS
    return max(0.0, min(seconds, RATE_LIMIT_MAX_WAIT_SECONDS))


class ProviderResponseError(Exception):
    """The provider answered (no network or HTTP-status failure) but the body
    was not a usable completion. Distinct from a connectivity failure because
    the two need different fixes: this one usually means the model is
    unavailable or was rejected, not that the key or the network is broken.
    """


def _extract_content(payload: object) -> str:
    """Pull the completion text out of a chat-completion response body.

    OpenRouter and similar aggregators return HTTP 200 even when the request
    could not be served -- no endpoint available for the model, content
    filtered, upstream provider error -- with the failure described inside
    the JSON body instead of the status code. ``payload["choices"][0]`` on an
    empty list is a real response shape, not a hypothetical one, and it must
    raise something the callers already catch rather than an IndexError or
    TypeError that was never in their except clause. Every failure mode here
    raises ProviderResponseError, so one exception type covers all of them.
    """
    if not isinstance(payload, dict):
        raise ProviderResponseError(f"response body is not a JSON object: {type(payload).__name__}")
    if error := payload.get("error"):
        raise ProviderResponseError(f"provider returned an error: {error}")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderResponseError(f"no choices in response: {json.dumps(payload)[:300]}")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise ProviderResponseError(f"no text content in first choice: {json.dumps(choices[0])[:300]}")
    return content


def build_candidates() -> tuple[dict[str, "callable"], list[str], dict[str, str]]:
    """Preflight every candidate whose key is present, then wrap the survivors.

    Returns (candidates, skipped, preflight_detail) rather than raising on a
    missing key or a failed preflight, so a partial run still produces a
    usable comparison and a failed one still explains itself.
    """
    candidates: dict[str, object] = {}
    skipped: list[str] = []
    preflight_detail: dict[str, str] = {}
    to_check: list[tuple[str, str, str, str, bool]] = []  # (name, endpoint, key, model, disable_reasoning)

    mistral_key = os.environ.get("MISTRAL_API_KEY")
    if mistral_key:
        to_check.append(
            ("mistral-small-3.1", "https://api.mistral.ai/v1/chat/completions", mistral_key, "mistral-small-latest", False)
        )
    else:
        skipped.append("mistral-small-3.1 (MISTRAL_API_KEY not set)")
        preflight_detail["mistral-small-3.1"] = "MISTRAL_API_KEY not set"

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    openrouter_models = (
        ("claude-sonnet-5", "anthropic/claude-sonnet-5"),
        ("claude-opus-5", "anthropic/claude-opus-5"),
        ("claude-fable-5.1", "anthropic/claude-fable-5.1"),
        ("gpt-6-astra", "openai/gpt-6-astra"),
        # qwen-3.5 previously used the invented slug "qwen-3.5-72b-instruct",
        # which never existed -- there is no 72B-parameter Qwen 3.5 variant.
        # Verified against OpenRouter's own listing: qwen/qwen3.5-plus-02-15.
        ("qwen-3.5", "qwen/qwen3.5-plus-02-15"),
        ("qwen-3.7", "qwen/qwen3.7-max"),
        ("qwen-3.8", "qwen/qwen3.8-max"),
        ("glm-5", "z-ai/glm-5"),
        ("glm-5.3", "z-ai/glm-5.3"),
        ("llama-3.3-70b", "meta-llama/llama-3.3-70b-instruct"),
        ("llama-4-maverick", "meta-llama/llama-4-maverick"),
        ("llama-4-scout", "meta-llama/llama-4-scout"),
        ("gemma-3-27b", "google/gemma-3-27b-it"),
        # Gemma 4 (April 2026) shipped under Apache 2.0, replacing Gemma 3's
        # more restrictive terms -- newer and licence-cleared in one move.
        # Both sizes benched rather than assuming the larger one wins: 31B is
        # dense (#3 on the Arena text leaderboard at release), 26B-A4B is a
        # cheaper MoE with only 4B active parameters (#6).
        ("gemma-4-31b", "google/gemma-4-31b-it"),
        ("gemma-4-26b-a4b", "google/gemma-4-26b-a4b-it"),
        # Mistral via OpenRouter as well as the direct API: the direct path
        # failed with HTTP 429 on the first live run, a genuine rate limit on
        # Mistral's own free evaluation tier (conservative RPS caps,
        # documented as intended for prototyping) rather than a wrong slug or
        # a bug. OpenRouter's Mistral pass-through has its own, separate
        # limits, so it is a real fallback rather than hitting the same wall
        # twice, and one of the two paths surviving is enough for a result.
        # ("Mistral 3.6" does not exist -- checked against Mistral's full,
        # dated release history; the current lineup is Large 3 and Small 4.)
        ("mistral-large-openrouter", "mistralai/mistral-large-2512"),
        ("mistral-small-openrouter", "mistralai/mistral-small-2603"),
        # Three additions outside the families already covered, from a survey
        # of what else exists as of September 2026 rather than only extending
        # families already in the registry.
        #
        # Kimi K2.6 (Moonshot): reported to sustain the longest correct
        # tool-calling sequences of any open-weight model, which is close to
        # this bench's actual task -- a multi-step, format-constrained loop --
        # rather than a general capability score.
        ("kimi-k2.6", "moonshotai/kimi-k2.6"),
        # DeepSeek V4 Flash: the cheapest capable candidate here by a wide
        # margin. Included with a caveat rather than assumed reliable:
        # independent reports describe its predecessor's structured
        # tool-calling as unreliable and note V4 was too new for a settled
        # verdict at time of writing. This bench measures exactly that
        # question on our specific six-verb grammar rather than inheriting
        # the reputation either way. Flash rather than Pro: one integration
        # report describes Pro hitting a thinking-mode protocol
        # incompatibility in some harnesses; Flash is also the cheaper of the
        # two and was independently described as "the real star" of the V4
        # release.
        ("deepseek-v4-flash", "deepseek/deepseek-v4-flash"),
        # Grok 4 Fast (xAI): the verified slug for xAI's current
        # cost-efficient tier; "Grok 4.5" is referenced in press coverage but
        # its exact OpenRouter slug was not confirmed, so it is not guessed at.
        ("grok-4-fast", "x-ai/grok-4-fast"),
    )
    # Kimi and DeepSeek are Chinese-developed models; on OpenRouter this bench
    # reaches them through OpenRouter's own infrastructure rather than a
    # China-hosted endpoint directly, and every document this bench sends is
    # synthetic (enforced by RlmEngine's data_class check, independent of this
    # list) -- so there is no live data-residency exposure here. The
    # consideration is recorded because it becomes relevant the moment any
    # candidate here is considered for production use on real case content,
    # where the existing PHI/data-class discipline would need to account for
    # where each provider actually processes the request.
    reasoning_capable_via_openrouter = True
    if openrouter_key:
        # Every candidate here is reached through OpenRouter's own catalogue
        # rather than a first-party endpoint: one key covers all of them
        # instead of requiring a separate credential per provider. OpenRouter
        # is a named, established aggregator that proxies to the real
        # provider -- unlike an unverified gateway once considered and
        # rejected for this bench (see recursive_engine_decision_record.md).
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        for name, model in openrouter_models:
            to_check.append((name, endpoint, openrouter_key, model, reasoning_capable_via_openrouter))
    else:
        skipped.append(f"{', '.join(name for name, _ in openrouter_models)} (OPENROUTER_API_KEY not set)")
        for name, _ in openrouter_models:
            preflight_detail[name] = "OPENROUTER_API_KEY not set"

    if to_check:
        print(f"Preflighting {len(to_check)} candidate(s) (timeout {PREFLIGHT_TIMEOUT_SECONDS}s each)...")
    for name, endpoint, key, model, disable_reasoning in to_check:
        reachable, reason = _preflight(name, endpoint, key, model, disable_reasoning=disable_reasoning)
        preflight_detail[name] = reason
        if reachable:
            candidates[name] = _bind(_http_chat_completion, endpoint, key, model, disable_reasoning=disable_reasoning)
        else:
            skipped.append(f"{name} (preflight failed: {reason})")

    return candidates, skipped, preflight_detail


def _bind(fn, endpoint, key, model, *, disable_reasoning=False):
    def _call(prompt: str) -> str:
        try:
            return fn(endpoint, key, model, prompt, timeout=CALL_TIMEOUT_SECONDS, disable_reasoning=disable_reasoning)
        except urllib.error.HTTPError as error:
            print(f"  [warn] {model}: HTTP {error.code} {error.reason}", file=sys.stderr)
            return ""
        except (urllib.error.URLError, ProviderResponseError, KeyError, json.JSONDecodeError, TimeoutError) as error:
            # A provider error becomes empty text, which the engine already
            # treats as model_emitted_no_action -- consistent with how the
            # adapter treats a refused SafeModelClient call.
            print(f"  [warn] {model}: {type(error).__name__}: {error}", file=sys.stderr)
            return ""
        except Exception as error:  # noqa: BLE001 - see module docstring: any failure here
            # degrades to an empty response, it never crashes the script. A
            # provider integration talks to code we do not control, and its
            # failure modes are open-ended -- the case that motivated this
            # clause was OpenRouter returning HTTP 200 with an empty choices
            # list, which is neither a network error nor a KeyError.
            print(f"  [warn] {model}: unexpected {type(error).__name__}: {error}", file=sys.stderr)
            return ""

    return _call


def _preflight(name: str, endpoint: str, key: str, model: str, *, disable_reasoning: bool = False) -> tuple[bool, str]:
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
            timeout=PREFLIGHT_TIMEOUT_SECONDS, disable_reasoning=disable_reasoning,
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
    except (urllib.error.URLError, ProviderResponseError, KeyError, json.JSONDecodeError, TimeoutError) as error:
        detail = f"{type(error).__name__}: {error}"
        print(f"  [preflight] {model}: unreachable ({detail}), skipping the full bench for it", file=sys.stderr)
        return False, detail
    except Exception as error:  # noqa: BLE001 - see _bind: any unexpected provider
        # behaviour must degrade to a reported, catchable outcome, never a
        # script crash. This is the same defensive posture as _bind's final
        # clause, kept here too because _preflight has its own except chain
        # rather than calling through _bind.
        detail = f"unexpected {type(error).__name__}: {error}"
        print(f"  [preflight] {model}: unreachable ({detail}), skipping the full bench for it", file=sys.stderr)
        return False, detail


def main() -> int:
    """Top-level safety net: every path below writes a results file before returning.

    _bind and _preflight already convert provider failures into reported
    outcomes, but that only covers calls made through them. Anything else
    unexpected -- a bug in this script, a change in a dependency's behaviour,
    a JSON payload shaped in a way nothing here anticipated -- must still
    leave a diagnostic file behind rather than exiting via an uncaught
    traceback with nothing written. The run that motivated this function did
    exactly that: an IndexError from an empty ``choices`` list, raised two
    calls below _bind's except clause at the time, crashed the whole script
    before a single byte was written.
    """
    out_path = Path("bench_results.json")
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--out", type=Path, default=out_path)
        args = parser.parse_args()
        out_path = args.out
        return _run(args.out)
    except Exception as error:  # noqa: BLE001 - last-resort net, see docstring
        import traceback

        trace = traceback.format_exc()
        print(f"\nUnexpected error: {type(error).__name__}: {error}", file=sys.stderr)
        print(trace, file=sys.stderr)
        try:
            _write_results(
                out_path,
                {
                    "status": "crashed",
                    "verdict": f"the bench script crashed with an unhandled {type(error).__name__}; see traceback",
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "traceback": trace,
                    "results": [],
                },
            )
            print(f"Crash diagnostics written to {out_path}", file=sys.stderr)
        except Exception as write_error:  # noqa: BLE001 - do not let the handler itself crash
            print(f"Could not write crash diagnostics either: {write_error}", file=sys.stderr)
        return 1


def _run(out: Path) -> int:
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
        _write_results(out, payload)
        print("\nNo candidate survived preflight; nothing to bench.", file=sys.stderr)
        for name, detail in sorted(preflight_detail.items()):
            print(f"  {name}: {detail}", file=sys.stderr)
        print(f"\nDiagnostics written to {out}", file=sys.stderr)
        return 1

    print(f"\nBenching: {', '.join(sorted(candidates))}")
    if skipped:
        print(f"Skipped: {'; '.join(skipped)}")

    report = bench_models(candidates, BENCH_CASES, adherence_target=0.95, budget_factory=_bench_budget)
    payload = report.as_dict()
    payload["status"] = "completed"
    payload["skipped"] = skipped
    payload["preflight"] = preflight_detail

    _write_results(out, payload)

    print(f"\nVerdict: {payload['verdict']}\n")
    print(f"{'model':<20}{'adherence':>11}{'completion':>12}{'near-miss share':>18}")
    for row in payload["results"]:
        print(f"{row['model_name']:<20}{row['adherence']:>10.0%}{row['completion_rate']:>12.0%}{row['near_miss_share']:>17.0%}")
    print(f"\nFull report written to {out}")
    return 0


def _write_results(path: Path, payload: dict) -> None:
    """Write results, creating the parent directory if the caller named one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    sys.exit(main())
