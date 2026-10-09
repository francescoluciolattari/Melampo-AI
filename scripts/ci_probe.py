#!/usr/bin/env python
"""E5: measure the construction-integration reader (``melampo.memory.ci_reader``) on the external check.

The reader builds every reading of the mention (structure, site of a procedure, of a device, thing a
measurement is about, source of a molecule, piece of the name of something that is not a body site)
and integrates the evidence (lattice blocks, head, predication in a semantic space, bag of context
words, explicit traces from training documents, gist of the document) by spreading activation.
This script reads every judged link with it and answers, against the corpus labels:

1. *Errors and right links.* How many of the errors does it read as something other than the
   structure, and how many right links change reading? The AUC of ``1 - share(structure)``.
2. *Roles.* For MedMentions links whose label names a role, is the reading the same? Per label class.
3. *Which evidence matters.* The same numbers removing one family of evidence at a time.
4. *Stability.* The same numbers for several strengths of inhibition.

The memory (traces, gist) is written from MedMentions *training* documents only and the document
being read is never in it. The semantic space is built from the text of the corpora and the NCIt
definitions (no label). The constants are fixed by principle; nothing here is fitted to the errors.

    python scripts/ci_probe.py --rows external_check.rows.json --craft ext/craft --medmentions ext/mm \
        --ncit ncit.obo --out ci_probe.json --markdown ci_probe.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import external_check as ec  # noqa: E402
import phrase_knowledge as pk  # noqa: E402
from head_probe import auc  # noqa: E402
from lattice_probe import gold_role  # noqa: E402
from ncit_kinds import Kinds  # noqa: E402

from melampo.memory import chunk_lattice as cl  # noqa: E402
from melampo.memory import ci_reader as ci  # noqa: E402
from melampo.memory.longer_names import LongerNames  # noqa: E402

FAMILIES = ("blocks", "head", "predication", "traces", "domain", "gist")


def say(start: float, text: str) -> None:
    print(f"[{time.monotonic() - start:7.1f}s] {text}", file=sys.stderr, flush=True)


def build_resources(args, start: float):
    """Texts, space, prototypes, traces."""
    mm_docs = list(ec.medmentions_documents(Path(args.medmentions))) if args.medmentions else []
    craft_docs = list(ec.craft_documents(Path(args.craft))) if args.craft else []
    splits = pk.medmentions_splits(Path(args.medmentions)) if args.medmentions else {}
    say(start, f"corpora: {len(mm_docs)} MedMentions, {len(craft_docs)} CRAFT documents")
    kinds = Kinds.from_obo(Path(args.ncit))
    definitions = pk._ncit_definitions(Path(args.ncit))
    kind_defs = [(kinds.kind_of_id[i], d) for i, d in definitions.items() if i in kinds.kind_of_id and d]
    say(start, f"NCIt definitions: {len(kind_defs)}")
    if args.space and Path(args.space).exists():
        space = ci.SemanticSpace.load(Path(args.space))
    else:
        texts = [t for _, t, _ in mm_docs] + [t for _, t, _ in craft_docs] + [d for _, d in kind_defs]
        space = ci.SemanticSpace.build(texts, dim=args.dim)
        if args.space:
            space.save(Path(args.space))
    say(start, f"semantic space: {len(space.words)} words, {space.vectors.shape[1]} dimensions")
    protos = ci.prototypes(space, kind_defs)
    say(start, f"prototypes: {sorted(protos)}")
    return mm_docs, craft_docs, splits, space, protos


def doc_keys(docs, corpus, lattice) -> dict:
    """The ambito of each whole document."""
    return {(corpus, d): ci.ambito(lattice.memory, t)[0] for d, t, _ in docs}


def write_traces(mm_docs, splits, space, lattice, keys=None) -> ci.TraceMemory:
    memory = ci.TraceMemory(space)
    for doc, text, labels in mm_docs:
        if splits.get(doc) != "trng":
            continue
        vector = space.vector(ci.content_words(text))
        key = (keys or {}).get(("medmentions", doc), "")
        for s, e, (cui, types) in labels:
            role = ci.role_to_reading(gold_role({"labels": [f"{cui}|{','.join(sorted(types))}"]}))
            memory.add(doc, text, s, e, role, lattice, vector, key)
    return memory


def predicted(reading: ci.CIReading) -> str:
    return reading.decision


def summarise(rows, readings, lattice_outcomes, splits=None) -> dict:
    """Numbers per corpus for one configuration of the reader."""
    report: dict = {}
    for corpus in ("craft", "medmentions"):
        idx = [i for i, r in enumerate(rows) if r["corpus"] == corpus]
        if not idx:
            continue
        label = np.array([1 if rows[i]["by_project_rule"] == "error" else 0 for i in idx])
        score = np.array([readings[i].p_not_structure for i in idx])
        changed = np.array([readings[i].decision not in (ci.STRUCTURE,) for i in idx])
        moved = np.array([readings[i].decision not in (ci.STRUCTURE, "underspecified") for i in idx])
        lat = np.array([lattice_outcomes[i] not in ("link", "unread") for i in idx])
        item = {
            "errors": int(label.sum()),
            "right": int((1 - label).sum()),
            "auc": auc(score, label),
            "errors_not_structure": int((changed & (label == 1)).sum()),
            "right_not_structure": int((changed & (label == 0)).sum()),
            "errors_with_a_role_or_non_site": int((moved & (label == 1)).sum()),
            "right_with_a_role_or_non_site": int((moved & (label == 0)).sum()),
            "lattice_errors": int((lat & (label == 1)).sum()),
            "lattice_right": int((lat & (label == 0)).sum()),
        }
        if corpus == "medmentions":
            # false alarms by split: dev and test documents were never in the memory, whatever the leave-out
            by_split: dict = defaultdict(Counter)
            for i in idx:
                if rows[i]["by_project_rule"] == "agrees":
                    by_split[(splits or {}).get(rows[i].get("doc"), "?")]["right"] += 1
                    by_split[(splits or {}).get(rows[i].get("doc"), "?")]["changed"] += int(
                        readings[i].decision != ci.STRUCTURE)
            item["right_changed_by_split"] = {k: dict(v) for k, v in sorted(by_split.items())}
            table: dict = defaultdict(Counter)
            for i in idx:
                gold = ci.role_to_reading(gold_role(rows[i]))
                table[gold][readings[i].decision] += 1
            item["by_gold"] = {g: dict(c) for g, c in sorted(table.items())}
            item["role_agreement"] = {
                g: {"n": sum(c.values()), "same": c.get(g, 0)} for g, c in sorted(table.items())
            }
            err_roles = [(ci.role_to_reading(gold_role(rows[i])), readings[i].decision)
                         for i in idx if rows[i]["by_project_rule"] == "error"]
            item["errors_role_same"] = sum(g == p for g, p in err_roles)
        report[corpus] = item
    return report


def run_all(reader, rows, texts_vector, keys) -> list[ci.CIReading]:
    out = []
    for r in rows:
        k = (r["corpus"], r.get("doc"))
        out.append(reader.read(r["mention"], r["sentence"], r.get("at"), r.get("doc"),
                               texts_vector.get(k), keys.get(k, "")))
    return out


def markdown(report: dict) -> str:
    lines = ["# E5: the construction-integration reader on the external check", "",
             f"{report['n']} judged links ({report['errors']} errors). Space: {report['space']}. "
             f"Traces: {report['traces']} annotations of MedMentions training documents.", ""]
    for name, block in report["configurations"].items():
        lines += [f"## {name}", "", "| corpus | errors | not structure (errors) | not structure (right) | role or non-site (errors) | role or non-site (right) | lattice (errors / right) | AUC |",
                  "|---|---|---|---|---|---|---|---|"]
        for corpus, v in block.items():
            lines.append(
                f"| {corpus} | {v['errors']} | {v['errors_not_structure']} | {v['right_not_structure']} / {v['right']} "
                f"| {v['errors_with_a_role_or_non_site']} | {v['right_with_a_role_or_non_site']} "
                f"| {v['lattice_errors']} / {v['lattice_right']} | {v['auc'] if v['auc'] is None else round(v['auc'], 3)} |")
        mm = block.get("medmentions")
        if mm and name == "all evidence":
            lines += ["", f"False alarms by split of MedMentions (right links changed / right): {mm['right_changed_by_split']}",
                      "", "Roles against the label (MedMentions): gold class -> same reading / n", ""]
            for g, v in mm["role_agreement"].items():
                lines.append(f"- {g}: {v['same']} / {v['n']}  (readings: {mm['by_gold'][g]})")
            lines.append(f"- errors whose reading is the gold role: {mm['errors_role_same']} of {mm['errors']}")
        lines.append("")
    lines += ["## Errors as the reader reads them (all evidence)", ""]
    for e in report["error_readings"]:
        lines.append(f"- {e['corpus']} `{e['mention']}` gold {e['gold']}; reader {e['decision']} "
                     f"(structure {e['structure']:.2f}, margin {e['margin']:.2f}); {e['sentence'][:110]}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", required=True)
    ap.add_argument("--craft")
    ap.add_argument("--medmentions")
    ap.add_argument("--ncit", required=True)
    ap.add_argument("--space", help="semantic space file (built and saved if absent)")
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--memory", type=Path, default=cl.DEFAULT_PATH)
    ap.add_argument("--out", default="ci_probe.json")
    ap.add_argument("--markdown", default="ci_probe.md")
    args = ap.parse_args(argv)
    start = time.monotonic()
    rows = [r for r in json.loads(Path(args.rows).read_text("utf-8"))
            if r.get("status") == "accepted" and r.get("by_project_rule") in ("agrees", "error")]
    mm_docs, craft_docs, splits, space, protos = build_resources(args, start)
    lattice = cl.ChunkLattice(cl.BlockMemory.load(args.memory, LongerNames.load().names))
    keys = {**doc_keys(mm_docs, "medmentions", lattice), **doc_keys(craft_docs, "craft", lattice)}
    traces = write_traces(mm_docs, splits, space, lattice, keys)
    say(start, f"traces: {traces.size}")
    vectors = {("medmentions", d): space.vector(ci.content_words(t)) for d, t, _ in mm_docs}
    vectors.update({("craft", d): space.vector(ci.content_words(t)) for d, t, _ in craft_docs})
    outcomes = [lattice.read(r["mention"], r["sentence"], r.get("at")).outcome for r in rows]
    say(start, "lattice read")
    configs: dict[str, ci.CIReader] = {
        "all evidence": ci.CIReader(lattice, space, protos, traces),
    }
    for family in FAMILIES:
        configs[f"without {family}"] = ci.CIReader(
            lattice, space, protos, traces, use=frozenset(FAMILIES) - {family})
    for inh in (0.25, 1.0):
        configs[f"inhibition {inh}"] = ci.CIReader(lattice, space, protos, traces, inhibition=inh)
    configs["without memory (no traces, no domain, no gist)"] = ci.CIReader(
        lattice, space, protos, None, use=frozenset(FAMILIES) - {"traces", "domain", "gist"})
    configs["with the bag of context words (v3 context)"] = ci.CIReader(
        lattice, space, protos, traces, use=frozenset(FAMILIES) | {"context"})
    report = {"n": len(rows), "errors": sum(r["by_project_rule"] == "error" for r in rows),
              "space": f"{len(space.words)} words x {space.vectors.shape[1]}", "traces": traces.size,
              "configurations": {}, "error_readings": []}
    first = None
    for name, reader in configs.items():
        readings = run_all(reader, rows, vectors, keys)
        report["configurations"][name] = summarise(rows, readings, outcomes, splits)
        if first is None:
            first = readings
        say(start, f"{name} done")
    for r, rd in zip(rows, first, strict=True):
        if r["by_project_rule"] == "error":
            report["error_readings"].append({
                "corpus": r["corpus"], "mention": r["mention"], "sentence": r["sentence"],
                "gold": ci.role_to_reading(gold_role(r)) if r["corpus"] == "medmentions" else "(CRAFT: structure)",
                "decision": rd.decision, "structure": rd.shares[ci.STRUCTURE], "margin": rd.margin,
                "evidence": rd.nodes[:12]})
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), "utf-8")
    Path(args.markdown).write_text(markdown(report), "utf-8")
    print(markdown(report)[:8000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
