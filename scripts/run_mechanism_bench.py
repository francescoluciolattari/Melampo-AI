"""Run the vetting bench, version 3 (melampo.evaluation.mechanism_bench): one candidate's mechanisms judged by
the medical sources and by two judge models of other families that must agree.

The calling machinery is the sibling run_format_adherence_bench.py's, as for version 2 (run_vetting_bench.py).
The sources are OBO files given with --sources (the workflow downloads HPO, GO, MONDO and ChEBI); what each case's
accepted mechanisms resolve to is written into the results before any model is asked (--resolution-only prints
it and asks nothing).
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import run_format_adherence_bench as base  # noqa: E402
from melampo.evaluation.mechanism_bench import (  # noqa: E402
    ERRORS,
    JUDGE_SYSTEM_PROMPT,
    MECHANISM_CASES,
    Sources,
    bench_mechanisms,
    resolution_report,
)

CANDIDATE_SYSTEM_PROMPT = (
    "You answer one clinical question with one JSON object, exactly in the format the question gives. "
    "No prose before or after it."
)
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_JUDGES = ("mistral-large-openrouter", "gpt-oss-120b")


def _entry(name: str) -> tuple[str, str, object] | None:
    return next((e for e in base.CANDIDATE_MODELS if e[0] == name), None)


def _family(slug: str) -> str:
    return slug.split("/", 1)[0]


def _bound(name: str, system_prompt: str):
    entry = _entry(name)
    if entry is None:
        return None, f"unknown model {name!r}"
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None, "OPENROUTER_API_KEY not set"
    _, slug, mode = entry
    reachable, reason = base._preflight(name, ENDPOINT, key, slug, disable_reasoning=mode)
    if not reachable:
        return None, reason
    return base._bind(base._http_chat_completion, ENDPOINT, key, slug, disable_reasoning=mode,
                      system_prompt=system_prompt), "reachable"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate")
    parser.add_argument("--judges", default=",".join(DEFAULT_JUDGES))
    parser.add_argument("--sources", nargs="*", default=[])
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("result.json"))
    parser.add_argument("--resolution-only", action="store_true")
    parser.add_argument("--list-candidates", action="store_true")
    parser.add_argument("--roster", default="nemotron-3-ultra,gemma-4-31b")
    args = parser.parse_args()

    if args.list_candidates:
        names = [n.strip() for n in args.roster.split(",") if n.strip()]
        unknown = [n for n in names if _entry(n) is None]
        if unknown:
            print(f"Unknown candidate name(s): {unknown}", file=sys.stderr)
            return 1
        print(json.dumps(names))
        return 0

    sources = Sources.from_files(args.sources) if args.sources else None
    resolution = resolution_report(MECHANISM_CASES, sources) if sources else {}
    payload = {"status": "started", "candidate": args.candidate, "sources": sources.versions if sources else {},
               "resolution": resolution}
    if args.resolution_only:
        print(json.dumps(resolution, indent=2))
        args.out.write_text(json.dumps(payload, indent=2))
        return 0

    judge_names = [n.strip() for n in args.judges.split(",") if n.strip()]
    candidate_entry = _entry(args.candidate or "")
    if candidate_entry is None:
        payload.update(status="no_candidates", reason=f"unknown candidate {args.candidate!r}")
        args.out.write_text(json.dumps(payload, indent=2))
        return 1
    # the judges are of other families than the candidate: a model never judges its own family's answers
    for name in judge_names:
        entry = _entry(name)
        if entry is None or _family(entry[1]) == _family(candidate_entry[1]):
            payload.update(status="no_candidates", reason=f"judge {name!r} unknown or of the candidate's family")
            args.out.write_text(json.dumps(payload, indent=2))
            return 1
    if len(judge_names) < 2:
        payload.update(status="no_candidates", reason="two judges are needed: they must agree")
        args.out.write_text(json.dumps(payload, indent=2))
        return 1

    model, reason = _bound(args.candidate, CANDIDATE_SYSTEM_PROMPT)
    judges = {}
    for name in judge_names:
        judge, judge_reason = _bound(name, JUDGE_SYSTEM_PROMPT)
        if judge is None:
            payload.update(status="no_candidates", reason=f"judge {name}: {judge_reason}")
            args.out.write_text(json.dumps(payload, indent=2))
            return 1
        judges[name] = judge
    if model is None:
        payload.update(status="no_candidates", reason=reason)
        args.out.write_text(json.dumps(payload, indent=2))
        return 1

    result = bench_mechanisms(args.candidate, model, MECHANISM_CASES, sources, judges, trials_per_case=args.trials)
    payload.update(status="completed", judges=judge_names, results=[result.as_dict()])
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"{args.candidate}: right {result.count('right')}/{result.runs}, errors {result.errors} "
          f"({', '.join(f'{o} {result.count(o)}' for o in ERRORS)}), doubts {result.count('doubt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
