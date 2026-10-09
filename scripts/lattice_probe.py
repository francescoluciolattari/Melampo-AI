#!/usr/bin/env python
"""Measure the chunk lattice on the rows of the external check (CRAFT, MedMentions).

The external check judges every accepted link by the corpus label. This script reads the same rows
with ``melampo.memory.chunk_lattice`` and answers four questions:

1. *Plain links.* If a link is made only when the lattice reads the structure as the block's subject
   (or does not read the phrase), how many errors remain among them, and how many right links
   changed reading? (``agrees`` / ``error`` rows.)
2. *Roles against the annotators.* For MedMentions links whose label has a semantic type that names
   a role (procedure, device, process, measurement, molecule, organism...), is the role the lattice
   records the same one?
3. *Conventions.* The rows the project rule counts as conventions (imaging measure of the structure,
   site of a device, origin of cells) -- what does the lattice say?
4. *Stability.* The same numbers over a grid of the lattice's costs: the result must not depend on
   one setting.

The costs of the lattice are fixed by principle (remembered < composed < word), not fitted here. The
rules were written after the 33 errors of the first external check were read, so question 1 is
optimistic; questions 2 and 3 use rows the rules were not written from (about 1,800 accepted
MedMentions links with a label type), which is why they are reported separately.

    python scripts/lattice_probe.py --rows external_check.rows.json --out lattice_probe.json \
        --markdown lattice_probe.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ncit_kinds import ROLE_OF_TYPE  # noqa: E402

from melampo.memory import chunk_lattice as cl  # noqa: E402
from melampo.memory.longer_names import LongerNames  # noqa: E402

DISEASE_TYPES = frozenset(
    "T047 T191 T046 T048 T019 T020 T033 T184 T037 T049 T190 T050 T029".split()
)
# A molecule label ("liver enzymes") and a recorded source of the molecule are the same role.
EQUIVALENT = {("source_of", "inside_a_name")}


def gold_role(row) -> str:
    """The role the label's semantic types name; "disease" or "structure" when they name none."""
    types = {
        t
        for label in row.get("labels", ())
        if "|" in label
        for t in label.split("|", 1)[1].split(",")
    }
    roles = sorted({ROLE_OF_TYPE[t] for t in types if t in ROLE_OF_TYPE})
    if roles:
        return roles[0]
    return "disease" if types & DISEASE_TYPES else "structure"


def agrees(predicted: str, gold: str) -> bool:
    return predicted == gold or (predicted, gold) in EQUIVALENT


def read_all(lattice, rows):
    out = []
    for row in rows:
        reading = lattice.read(row["mention"], row["sentence"])
        out.append((row, reading))
    return out


def summarise(rows, lattice) -> dict:
    accepted = [
        r for r in rows if r["status"] == "accepted" and r.get("by_project_rule")
    ]
    read = read_all(lattice, accepted)
    judged = Counter()
    for row, reading in read:
        rule = row["by_project_rule"]
        if rule not in ("agrees", "error"):
            continue
        shape = "plain" if reading.outcome in ("link", "unread") else reading.outcome
        judged[(row["corpus"], rule, shape)] += 1
    report: dict = {
        "judged": {},
        "roles": {},
        "conventions": {},
        "errors": [],
        "lost": [],
    }
    for corpus in ("craft", "medmentions"):
        errors = sum(
            v for (c, r, _), v in judged.items() if c == corpus and r == "error"
        )
        right = sum(
            v for (c, r, _), v in judged.items() if c == corpus and r == "agrees"
        )
        plain_err = judged[(corpus, "error", "plain")]
        plain_right = judged[(corpus, "agrees", "plain")]
        report["judged"][corpus] = {
            "errors": errors,
            "right": right,
            "errors_by_reading": {
                s: judged[(corpus, "error", s)]
                for s in ("plain", "role", "not_a_site", "underspecified")
            },
            "right_by_reading": {
                s: judged[(corpus, "agrees", s)]
                for s in ("plain", "role", "not_a_site", "underspecified")
            },
            "plain_error_rate_before": errors / (errors + right)
            if errors + right
            else None,
            "plain_error_rate_after": plain_err / (plain_err + plain_right)
            if plain_err + plain_right
            else None,
        }
    confusion = defaultdict(Counter)
    for row, reading in read:
        if row["corpus"] != "medmentions" or not row.get("labels"):
            continue
        confusion[gold_role(row)][reading.role or reading.outcome] += 1
    for gold, counts in confusion.items():
        total = sum(counts.values())
        if gold in ("structure", "disease"):
            # the label names a structure or a disease: no role should be added
            same = sum(v for k, v in counts.items() if k in ("link", "unread"))
        else:
            same = sum(v for k, v in counts.items() if agrees(k, gold))
        report["roles"][gold] = {
            "n": total,
            "same_role": same,
            "read_as": dict(counts.most_common()),
        }
    conventions = defaultdict(Counter)
    for row, reading in read:
        rule = row["by_project_rule"]
        if rule.startswith("convention:"):
            conventions[rule][reading.role or reading.outcome] += 1
    report["conventions"] = {k: dict(v.most_common()) for k, v in conventions.items()}
    for row, reading in read:
        item = {
            "corpus": row["corpus"],
            "mention": row["mention"],
            "outcome": reading.outcome,
            "role": reading.role,
            "block": reading.block,
            "why": reading.why,
            "gold": gold_role(row),
            "sentence": row["sentence"][:160],
        }
        if row["by_project_rule"] == "error":
            report["errors"].append(item)
        elif row["by_project_rule"] == "agrees" and reading.outcome in (
            "not_a_site",
            "underspecified",
        ):
            report["lost"].append(item)
    return report


def grid(rows, memory) -> list[dict]:
    saved = (cl.C_SHARE, cl.C_OPEN, cl.MARGIN)
    out = []
    try:
        for share, opened, margin in itertools.product(
            (0.5, 1.0, 2.0), (1.0, 2.0, 3.0), (0.15, 0.25, 0.4)
        ):
            cl.C_SHARE, cl.C_OPEN, cl.MARGIN = share, opened, margin
            s = summarise(rows, cl.ChunkLattice(memory))
            out.append(
                {
                    "c_share": share,
                    "c_open": opened,
                    "margin": margin,
                    "errors_read": sum(
                        v
                        for c in s["judged"].values()
                        for k, v in c["errors_by_reading"].items()
                        if k != "plain"
                    ),
                    "right_lost": sum(
                        c["right_by_reading"]["not_a_site"]
                        + c["right_by_reading"]["underspecified"]
                        for c in s["judged"].values()
                    ),
                    "right_role": sum(
                        c["right_by_reading"]["role"] for c in s["judged"].values()
                    ),
                }
            )
    finally:
        cl.C_SHARE, cl.C_OPEN, cl.MARGIN = saved
    return out


def _pct(value) -> str:
    return "-" if value is None else f"{100 * value:.2f}%"


def markdown(report: dict, settings: list[dict]) -> str:
    lines = ["# Experiment lattice-probe: the phrase read as blocks", ""]
    lines += [
        "## 1. Judged links: plain links and readings",
        "",
        "| corpus | errors | right | errors read (role / not a site / underspecified) | right links read (role / lost) | error rate of plain links |",
        "|---|---|---|---|---|---|",
    ]
    for corpus, j in report["judged"].items():
        e, r = j["errors_by_reading"], j["right_by_reading"]
        before, after = j["plain_error_rate_before"], j["plain_error_rate_after"]
        lines.append(
            f"| {corpus} | {j['errors']} | {j['right']} | {e['role']} / {e['not_a_site']} / {e['underspecified']} | "
            f"{r['role']} / {r['not_a_site'] + r['underspecified']} | "
            f"{_pct(before)} -> {_pct(after)} |"
        )
    lines += [
        "",
        "## 2. Role against the label's semantic type (MedMentions, accepted links with a label)",
        "",
        "| role of the label | n | same role (structure, disease: no role added) | read as |",
        "|---|---|---|---|",
    ]
    for gold, v in sorted(report["roles"].items(), key=lambda kv: -kv[1]["n"]):
        reads = ", ".join(f"{k} {n}" for k, n in list(v["read_as"].items())[:6])
        lines.append(f"| {gold} | {v['n']} | {v['same_role']} | {reads} |")
    lines += ["", "## 3. Conventions of the project rule", ""]
    for rule, counts in report["conventions"].items():
        lines.append(f"- {rule}: " + ", ".join(f"{k} {n}" for k, n in counts.items()))
    lines += [
        "",
        "## 4. Stability over the costs",
        "",
        "The reading is stable when the open-node cost is 2 or more; with 1 the block swallows"
        " the heads after the first and right links are lost.",
        "",
        "| share cost | open-node cost | margin | errors read | right links lost | right links with a role |",
        "|---|---|---|---|---|---|",
    ]
    for s in settings:
        lines.append(
            f"| {s['c_share']} | {s['c_open']} | {s['margin']} | {s['errors_read']} | {s['right_lost']} | {s['right_role']} |"
        )
    lines += ["", "## Errors, as the lattice reads them", ""]
    for e in report["errors"]:
        lines.append(
            f"- {e['corpus']} `{e['mention']}` -> {e['outcome']} {e['role']} (label role: {e['gold']}); block `{e['block']}`: {e['sentence']}"
        )
    lines += [
        "",
        "## Right links the lattice would lose (not a site / underspecified)",
        "",
    ]
    for e in report["lost"]:
        lines.append(
            f"- {e['corpus']} `{e['mention']}` -> {e['outcome']} ({e['why']}); block `{e['block']}`"
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--rows", type=Path, required=True, help="external_check rows json"
    )
    parser.add_argument("--memory", type=Path, default=cl.DEFAULT_PATH)
    parser.add_argument("--out", type=Path, default=Path("lattice_probe.json"))
    parser.add_argument("--markdown", type=Path, default=Path("lattice_probe.md"))
    args = parser.parse_args(argv)
    rows = json.loads(args.rows.read_text("utf-8"))
    memory = cl.BlockMemory.load(args.memory, LongerNames.load().names)
    report = summarise(rows, cl.ChunkLattice(memory))
    settings = grid(rows, memory)
    report["stability"] = settings
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", "utf-8"
    )
    args.markdown.write_text(markdown(report, settings), "utf-8")
    print(args.markdown.read_text("utf-8")[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
