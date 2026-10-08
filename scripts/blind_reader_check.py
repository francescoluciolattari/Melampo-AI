"""How good is the blind reader at catching a wrong class? (no models, no network)

For every mention of the labelled sets (the dev queries and the held-out files), with its true
class, it asks the reader twice: against the **true** class (it must not disagree: false alarm) and
against **another class** chosen at random from the same lexicon (it should disagree: power).
A wrong class is not a real error pattern; real errors are look-alike classes (the other side, a
sister organ), so the second probe is also run against the graph's neighbours of the true class.

    python scripts/blind_reader_check.py --uberon data/linking/uberon-basic.obo --out blind_check.json

It measures the reader, not the linker. The numbers are rates on our own mentions (written by us), so
they say whether the mechanism works, not how the system performs on real text.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from melampo.memory import anatomy_linker as al  # noqa: E402
from melampo.memory.anatomy_graph import AnatomyGraph  # noqa: E402
from melampo.memory.blind_reader import AGAINST, SILENT, SUPPORT, BlindReader  # noqa: E402

DATA = ROOT / "data"


def _rows():
    rows = []
    for path in sorted((DATA / "encoder_bench").glob("queries_*.jsonl")):
        for line in path.read_text("utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows.append((path.stem, row["query"], row["target"]))
    for path in sorted((DATA / "linking").glob("heldout*_*.jsonl")):
        for line in path.read_text("utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("target") and not str(row["target"]).startswith(
                    ("NONE", "NOT", "AMBIG")
                ):
                    rows.append((path.stem, row["mention"], row["target"]))
    return rows


def _rate(counter: Counter, total: int) -> dict:
    return {
        k: {"n": counter[k], "rate": round(counter[k] / total, 3) if total else None}
        for k in (SUPPORT, SILENT, AGAINST)
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uberon", default=str(DATA / "linking/uberon-basic.obo"))
    parser.add_argument("--out", default="blind_check.json")
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args(argv)

    lexicon = al.Lexicon.from_json(
        json.loads((DATA / "linking/anatomy_lexicon.json").read_text("utf-8"))
    )
    terms, graph = [], None
    if Path(args.uberon).exists():
        with open(args.uberon, encoding="utf-8") as handle:
            terms = al.load_obo_terms(handle)
        with open(args.uberon, encoding="utf-8") as handle:
            graph = AnatomyGraph.from_obo(handle)
    _, equivalent = al.build_pool(lexicon, terms)
    parts = json.loads((DATA / "linking/anatomy_parts.json").read_text("utf-8"))
    reader = BlindReader.from_sources(lexicon, terms, equivalent, graph, parts)
    classes = sorted(lexicon.classes)
    rng = random.Random(args.seed)

    true, other, near = Counter(), Counter(), Counter()
    by_set: dict[str, Counter] = {}
    n = n_near = 0
    for name, mention, target in _rows():
        if target not in lexicon.classes:
            continue
        n += 1
        verdict = reader.read(mention, target).verdict
        true[verdict] += 1
        by_set.setdefault(name, Counter())[verdict] += 1
        wrong = rng.choice([c for c in classes if c != target])
        other[reader.read(mention, wrong).verdict] += 1
        if graph is not None:
            sides = [
                c
                for c in (
                    target.replace("_left", "_right"),
                    target.replace("_right", "_left"),
                )
                if c != target and c in lexicon.classes
            ]
            if sides:
                n_near += 1
                near[reader.read(mention, sides[0]).verdict] += 1
    report = {
        "mentions": n,
        "against_the_true_class": _rate(true, n),
        "against_a_random_other_class": _rate(other, n),
        "against_the_other_side": _rate(near, n_near),
        "by_set": {k: dict(v) for k, v in by_set.items()},
    }
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "by_set"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
