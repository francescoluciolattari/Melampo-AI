"""Gold set tool for the anatomy linker. Subcommands, in the order a study runs.

  sample    reports.jsonl  -> sheets/annotator_A.csv, annotator_B.csv, valid_structures.txt, items.jsonl
  check     annotator_A.csv                      -> lists empty or invalid cells
  agree     annotator_A.csv annotator_B.csv      -> agreement, kappa, adjudication.csv for the third reviewer
  merge     A.csv B.csv [adjudication.csv]       -> gold.jsonl (final labels)
  evaluate  gold.jsonl                           -> error bound on accepted links, coverage, verdict
  size                                           -> how many accepted links certify a target error

reports.jsonl: one JSON per line, {"report_id", "text", optional "language", "site"}. The text must be
pseudonymised at the source (names, dates of birth, identifiers). No raw report leaves the hospital.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from melampo.evaluation import gold_set as gs  # noqa: E402
from melampo.memory import anatomy_linker as al  # noqa: E402
from melampo.memory import anatomy_parts as ap  # noqa: E402
from melampo.memory.anatomy_graph import AnatomyGraph  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data" / "linking"


def _load():
    lexicon = al.Lexicon.from_json(
        json.loads((DATA / "anatomy_lexicon.json").read_text("utf-8"))
    )
    parts = ap.PartTable.from_json(
        json.loads((DATA / "anatomy_parts.json").read_text("utf-8")), lexicon
    )
    return lexicon, parts


def _jsonl(path):
    return [
        json.loads(x) for x in Path(path).read_text("utf-8").splitlines() if x.strip()
    ]


def _count(text: str) -> int:
    """A whole number; "600", " 600 " and the pasted "n=600" all mean 600 (the form field is the name)."""
    value = text.strip().split("=")[-1].strip()
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a whole number such as 600, got {text!r}"
        ) from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"expected a positive number, got {text!r}")
    return number


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("reports")
    s.add_argument("--out", default="gold_study")
    s.add_argument("--n", type=_count, default=None)
    s.add_argument("--seed", type=int, default=20261006)
    s.add_argument(
        "--cap-per-mention",
        type=_count,
        default=None,
        help="at most this many items per written mention (not population-weighted)",
    )
    c = sub.add_parser("check")
    c.add_argument("sheet")
    a = sub.add_parser("agree")
    a.add_argument("first")
    a.add_argument("second")
    a.add_argument("--out", default="adjudication.csv")
    m = sub.add_parser("merge")
    m.add_argument("first")
    m.add_argument("second")
    m.add_argument("adjudicated", nargs="?")
    m.add_argument("--out", default="gold.jsonl")
    e = sub.add_parser("evaluate")
    e.add_argument("gold")
    e.add_argument("--uberon", default=None)
    e.add_argument(
        "--reports",
        default=None,
        help="reports.jsonl the items were sampled from: the linker then also gets the state of the report",
    )
    e.add_argument("--out", default="gold_report.json")
    z = sub.add_parser("size")
    z.add_argument("--target", type=float, default=0.01)
    z.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args(argv)

    lexicon, parts = _load()
    if args.cmd == "sample":
        items = gs.sample_items(
            _jsonl(args.reports),
            lexicon,
            parts,
            n=args.n,
            seed=args.seed,
            cap_per_mention=args.cap_per_mention,
        )
        out = Path(args.out)
        gs.write_sheets(items, out, class_ids=sorted(lexicon.classes))
        (out / "items.jsonl").write_text(
            "\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n",
            encoding="utf-8",
        )
        print(
            f"{len(items)} items from {len({i['report_id'] for i in items})} reports -> {out}/"
        )
        return 0
    if args.cmd == "check":
        problems = gs.validate_sheet(gs.read_sheet(Path(args.sheet)), lexicon.classes)
        print("\n".join(problems) or "ok")
        return 1 if problems else 0
    if args.cmd == "agree":
        first, second = (
            gs.read_sheet(Path(args.first)),
            gs.read_sheet(Path(args.second)),
        )
        print(json.dumps(gs.agreement(first, second), indent=1))
        queue = gs.adjudication_queue(first, second)
        if queue:
            import csv

            with open(args.out, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(queue[0]))
                writer.writeheader()
                writer.writerows(queue)
        print(f"{len(queue)} items for the third reviewer -> {args.out}")
        return 0
    if args.cmd == "merge":
        first, second = (
            gs.read_sheet(Path(args.first)),
            gs.read_sheet(Path(args.second)),
        )
        adjudicated = (
            gs.read_sheet(Path(args.adjudicated)) if args.adjudicated else None
        )
        gold = gs.merge_gold(first, second, adjudicated)
        Path(args.out).write_text(
            "\n".join(json.dumps(g, ensure_ascii=False) for g in gold) + "\n",
            encoding="utf-8",
        )
        print(
            f"{len(gold)} final labels of {len(set(first) & set(second))} items -> {args.out}"
        )
        return 0
    if args.cmd == "evaluate":
        terms = []
        graph = None
        if args.uberon and Path(args.uberon).exists():
            with open(args.uberon, encoding="utf-8") as handle:
                terms = al.load_obo_terms(handle)
            with open(args.uberon, encoding="utf-8") as handle:
                graph = AnatomyGraph.from_obo(handle)
        pool, equivalent = al.build_pool(lexicon, terms)
        linker = al.AnatomyLinker(lexicon, pool, equivalent, parts=parts, graph=graph)
        reports = (
            {r["report_id"]: r["text"] for r in _jsonl(args.reports)}
            if args.reports
            else None
        )
        report = gs.evaluate(linker, _jsonl(args.gold), reports=reports)
        Path(args.out).write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(report["verdict"])
        print(
            {
                k: report[k]
                for k in ("n", "accepted", "wrong", "error_upper_bound", "coverage")
            }
        )
        return 0
    if args.cmd == "size":
        for errors in (0, 2, 5, 12):
            print(
                f"<= {errors} errors: {gs.cases_needed(args.target, args.confidence, errors)} accepted links for <= {args.target:.1%} at {args.confidence:.0%}"
            )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
