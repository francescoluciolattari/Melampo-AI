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

INPUT_DIR is searched recursively for ``*.json`` files -- point it directly at
what ``actions/download-artifact@v4`` produces (with ``merge-multiple: false``,
its default, every artifact lands in its own subdirectory named after the
artifact, e.g. ``INPUT_DIR/candidate-result-0/result.json``,
``INPUT_DIR/candidate-result-1/result.json``). No flattening step is needed or
wanted: since every candidate job writes the same internal filename
(``result.json``), copying them into one flat directory first would collide on
that repeated name and silently keep only the last one copied. Each file
should be one candidate's output, written by ``--candidate NAME --out <path>``.
Files that do not parse as JSON, or that do
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

    # Recursive: actions/download-artifact@v4 with merge-multiple: false (the
    # default) downloads each artifact into its own subdirectory named after
    # the artifact -- all_results/candidate-result-0/result.json,
    # all_results/candidate-result-1/result.json, and so on -- since every
    # candidate job writes the same internal filename ("result.json"). A
    # flat glob() would only see files directly in input_dir and silently
    # find nothing; a shell step that copies them all into one flat
    # directory first would collide on that same repeated filename and
    # overwrite all but the last one copied, which is the defect this
    # comment replaces: rglob() reads the artifacts directly from their
    # nested layout, so no flattening step -- and no collision -- is needed
    # at all.
    files = sorted(input_dir.rglob("*.json"))
    if not files:
        merge_failures.append(f"no JSON files found in {input_dir}")

    for path in files:
        # Relative to input_dir, not path.name alone: with the nested
        # artifact layout, two files both named "result.json" in different
        # subdirectories are indistinguishable by bare name, and a merge
        # failure report naming the wrong one is worse than unhelpful.
        label = str(path.relative_to(input_dir))
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            merge_failures.append(f"{label}: could not read/parse ({type(error).__name__}: {error})")
            continue
        if not isinstance(payload, dict):
            merge_failures.append(f"{label}: top-level JSON is not an object")
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
        "source_files": [str(path.relative_to(input_dir)) for path in files],
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
