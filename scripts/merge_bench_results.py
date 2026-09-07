#!/usr/bin/env python3
"""Combine per-candidate result files from a parallel matrix run into one report.

The parallel workflow runs each candidate in its own job via
``run_format_adherence_bench.py --candidate NAME``, each writing its own small
results file. This script reads all of those files back and produces the same
top-level shape a single sequential run would have written -- one ``results``
list, one merged ``skipped`` list, one merged ``preflight`` dict, and a verdict
recomputed over the combined set via
``melampo.evaluation.format_adherence_bench.compute_verdict`` -- the same
function a live run calls, so a parallel run and a sequential run reach the
same conclusion from the same underlying numbers.

Usage:
    python scripts/merge_bench_results.py INPUT_DIR --out bench_results.json

INPUT_DIR should contain one JSON file per candidate, each written by
``--candidate NAME --out <path>``. Files that do not parse as JSON, or that do
not have the expected shape, are recorded as merge failures rather than
aborting the whole merge -- one malformed artifact from one matrix job should
not erase the twenty results that did arrive cleanly.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from melampo.evaluation.format_adherence_bench import (
    compute_verdict,
    rank_result_dicts,
)

ADHERENCE_TARGET = 0.95


def merge(input_dir: Path) -> dict:
    """Read every JSON file in input_dir and combine them into one report.

    A file contributing zero usable results (a crashed or no-candidate matrix
    job) still contributes its skipped/preflight entries, so the merged
    output accounts for every candidate the roster named, not only the ones
    that produced a benchable result.
    """
    results: list[dict] = []
    skipped: list[str] = []
    preflight: dict[str, str] = {}
    merge_failures: list[str] = []

    files = sorted(input_dir.glob("*.json"))
    if not files:
        merge_failures.append(f"no JSON files found in {input_dir}")

    for path in files:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            merge_failures.append(f"{path.name}: could not read/parse ({type(error).__name__}: {error})")
            continue
        if not isinstance(payload, dict):
            merge_failures.append(f"{path.name}: top-level JSON is not an object")
            continue

        results.extend(item for item in payload.get("results", []) if isinstance(item, dict))
        skipped.extend(str(item) for item in payload.get("skipped", []))
        for name, reason in (payload.get("preflight") or {}).items():
            # A candidate that was "not requested" in one job's restricted
            # view is expected to be genuinely reachable or skipped in the
            # job that was actually assigned it; prefer any non-restriction
            # reason already recorded over a later "not requested" note from
            # a job that was not responsible for that candidate.
            if name in preflight and "not requested" in preflight[name] and "not requested" not in reason or name not in preflight:
                preflight[name] = reason

    verdict = compute_verdict(results, ADHERENCE_TARGET) if not merge_failures or results else (
        "merge produced no usable results; see merge_failures"
    )

    return {
        "status": "completed" if results else "no_candidates",
        "verdict": verdict,
        "models": len(results),
        "results": rank_result_dicts(results),
        "skipped": sorted(set(skipped)),
        "preflight": dict(sorted(preflight.items())),
        "merge_failures": merge_failures,
        "source_files": [path.name for path in files],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory containing one JSON file per candidate")
    parser.add_argument("--out", type=Path, default=Path("bench_results.json"))
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Not a directory: {args.input_dir}", file=sys.stderr)
        args.out.write_text(
            json.dumps(
                {
                    "status": "crashed",
                    "verdict": f"input directory {args.input_dir} does not exist",
                    "results": [],
                    "merge_failures": [f"{args.input_dir} is not a directory"],
                },
                indent=2,
            )
        )
        return 1

    payload = merge(args.input_dir)
    args.out.write_text(json.dumps(payload, indent=2))

    print(f"Merged {len(payload['source_files'])} file(s) into {len(payload['results'])} result(s)")
    if payload["merge_failures"]:
        print(f"Merge failures: {payload['merge_failures']}", file=sys.stderr)
    print(f"Verdict: {payload['verdict']}")
    print(f"Written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
