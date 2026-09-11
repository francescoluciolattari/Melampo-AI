#!/usr/bin/env python3
"""Run the vetting bench against real providers -- the runnable counterpart to
melampo.evaluation.vetting_bench, which is a library with no way to execute
itself.

That gap was found by a direct question ("how do I run this") after the
library was built and merged with no script and no workflow behind it -- the
same shape of gap found repeatedly across this project: a module built,
tested, and never connected to anything that would call it. This closes it
for the vetting bench specifically, by reusing this file's sibling rather
than re-implementing HTTP calling, retries, and preflight a second time.

Reuse, not duplication. `run_format_adherence_bench.py` already has the
HTTP-calling machinery this needs -- `_http_chat_completion` (with its 429
retry and 400 reasoning-hint fallback already debugged against real
providers), `_preflight`, `build_candidates`, and the full `CANDIDATE_MODELS`
registry, the single source of truth for every model slug in this project.
This script imports them directly as a sibling module rather than copying a
second version that could drift from the first the way a duplicated
verdict-ranking or answer-matching function did earlier in this project.

One candidate per invocation, `--candidate NAME`, exactly like
`run_format_adherence_bench.py` -- so a workflow can fan this out into a
parallel matrix the same proven way, rather than inventing a second
orchestration pattern for what is structurally the same problem.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_format_adherence_bench import (
    MISTRAL_DIRECT_MODEL,
    _bind,
    _http_chat_completion,
    _preflight,
    all_candidate_names,
)

from melampo.evaluation.vetting_bench import (
    VETTING_CASES,
    bench_vetting,
    rank_vetting_results,
    vetting_graph,
    vetting_table,
)

DEFAULT_ROSTER = ("claude-opus-5", "gpt-oss-120b")


def _find_candidate(name: str) -> tuple[str, str, bool] | None:
    """Look up one candidate's (name, slug, disable_reasoning) from the shared registry.

    Duplicating no table: `run_format_adherence_bench.CANDIDATE_MODELS` is
    read directly, so a slug correction made there (the project has needed
    one more than once) is picked up here automatically rather than needing
    a second edit.
    """
    import run_format_adherence_bench as base

    if name == "mistral-small-3.1":
        return (name, MISTRAL_DIRECT_MODEL, False)
    for candidate_name, model, disable_reasoning in base.CANDIDATE_MODELS:
        if candidate_name == name:
            return (candidate_name, model, disable_reasoning)
    return None


def _build_one_candidate(name: str) -> tuple[dict, str]:
    """Preflight and bind exactly one candidate, mirroring build_candidates(only=name).

    A thin, single-candidate version rather than calling build_candidates
    itself: that function's job is the full-roster preflight-everyone sweep
    used by the format-adherence bench, and reimplementing its single-name
    filter here keeps this script simple to read on its own.
    """
    import os

    entry = _find_candidate(name)
    if entry is None:
        return {}, f"unknown candidate name {name!r}"
    candidate_name, model, disable_reasoning = entry

    if candidate_name == "mistral-small-3.1":
        key = os.environ.get("MISTRAL_API_KEY")
        endpoint = "https://api.mistral.ai/v1/chat/completions"
    else:
        key = os.environ.get("OPENROUTER_API_KEY")
        endpoint = "https://openrouter.ai/api/v1/chat/completions"

    if not key:
        return {}, "API key not set"

    reachable, reason = _preflight(candidate_name, endpoint, key, model, disable_reasoning=disable_reasoning)
    if not reachable:
        return {}, reason
    fn = _bind(_http_chat_completion, endpoint, key, model, disable_reasoning=disable_reasoning)
    return {candidate_name: fn}, "reachable"


def _write_results(out: Path, payload: dict) -> None:
    out.write_text(json.dumps(payload, indent=2))


def main() -> int:
    out_path = Path("vetting_results.json")
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--out", type=Path, default=out_path)
        parser.add_argument("--candidate", default=None, help="Run only this one candidate (by bench name).")
        parser.add_argument(
            "--list-candidates",
            action="store_true",
            help="Print every known candidate name as a JSON array and exit.",
        )
        parser.add_argument(
            "--roster",
            default=None,
            help="Comma-separated candidate names, used with --list-candidates to restrict the printed list.",
        )
        args = parser.parse_args()
        out_path = args.out

        if args.list_candidates:
            names = list(DEFAULT_ROSTER) if args.roster is None else [
                item.strip() for item in args.roster.split(",") if item.strip()
            ]
            known = set(all_candidate_names())
            unknown = [name for name in names if name not in known and name != "gpt-oss-120b"]
            # gpt-oss-120b is allowed even though it is not yet in the shared
            # registry -- see the note in _find_candidate's caller below.
            if unknown:
                print(f"Unknown candidate name(s): {unknown}", file=sys.stderr)
                return 1
            print(json.dumps(names))
            return 0

        if not args.candidate:
            print("--candidate is required outside --list-candidates", file=sys.stderr)
            return 1

        candidates, reason = _build_one_candidate(args.candidate)
        if not candidates:
            payload = {
                "status": "no_candidates",
                "candidate": args.candidate,
                "reason": reason,
                "results": [],
            }
            _write_results(out_path, payload)
            print(f"{args.candidate}: {reason}", file=sys.stderr)
            return 1

        graph, table = vetting_graph(), vetting_table()
        results = [
            bench_vetting(name, model, VETTING_CASES, graph, table=table) for name, model in candidates.items()
        ]
        ranked = rank_vetting_results(results)
        payload = {
            "status": "completed",
            "candidate": args.candidate,
            "results": [item.as_dict() for item in ranked],
        }
        _write_results(out_path, payload)
        print(f"{args.candidate}: grounding_rate={ranked[0].grounding_rate:.0%}")
        return 0

    except Exception as error:  # noqa: BLE001 - last-resort net, same discipline as run_format_adherence_bench.py
        payload = {
            "status": "crashed",
            "verdict": f"the vetting bench script crashed with an unhandled {type(error).__name__}; see traceback",
            "error": str(error),
            "error_type": type(error).__name__,
            "traceback": __import__("traceback").format_exc(),
            "results": [],
        }
        try:
            _write_results(out_path, payload)
        except OSError as write_error:
            # The crash report itself must not raise: printing the failure to
            # stderr is the only remaining way to surface it.
            print(f"Could not write crash diagnostics: {write_error}", file=sys.stderr)
        print(f"\nUnexpected error: {type(error).__name__}: {error}", file=sys.stderr)
        print(payload["traceback"], file=sys.stderr)
        print(f"\nCrash diagnostics written to {out_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
