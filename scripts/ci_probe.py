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

from melampo.evaluation.frozen_split import DEFAULT_PATH as FROZEN_PATH  # noqa: E402
from melampo.evaluation.frozen_split import FrozenSplit, development_rows  # noqa: E402
from melampo.memory import chunk_lattice as cl  # noqa: E402
from melampo.memory import ci_reader as ci  # noqa: E402
from melampo.memory.grammar import Grammar  # noqa: E402
from melampo.memory.head_typing import UmlsHeadTyper  # noqa: E402
from melampo.memory.longer_names import LongerNames  # noqa: E402

FAMILIES = ("blocks", "head", "predication", "traces", "domain", "gist", "rules", "discourse")


def say(start: float, text: str) -> None:
    print(f"[{time.monotonic() - start:7.1f}s] {text}", file=sys.stderr, flush=True)


def load_frozen() -> FrozenSplit | None:
    return FrozenSplit.load(FROZEN_PATH) if FROZEN_PATH.exists() else None


def build_resources(args, start: float, frozen: FrozenSplit | None = None):
    """Texts, space, prototypes, traces. The frozen test documents are not in the semantic space."""
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
        open_mm = [t for d, t, _ in mm_docs if not (frozen and frozen.is_frozen(d, "medmentions"))]
        texts = open_mm + [t for _, t, _ in craft_docs] + [d for _, d in kind_defs]
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


def write_traces(mm_docs, splits, space, lattice, keys=None, frozen: FrozenSplit | None = None) -> ci.TraceMemory:
    memory = ci.TraceMemory(space)
    for doc, text, labels in mm_docs:
        if splits.get(doc) != "trng":
            continue
        if frozen is not None:
            frozen.refuse([doc], "the trace memory", "medmentions")
        vector = space.vector(ci.content_words(text))
        key = (keys or {}).get(("medmentions", doc), "")
        for s, e, (cui, types) in labels:
            role = ci.role_to_reading(gold_role({"labels": [f"{cui}|{','.join(sorted(types))}"]}))
            memory.add(doc, text, s, e, role, lattice, vector, key)
    return memory


def lattice_variant(rows, outcomes) -> dict:
    """Per corpus: errors the lattice acts on, right links it acts on (role, not a site, underspecified)."""
    out = {}
    for corpus in ("craft", "medmentions"):
        idx = [i for i, r in enumerate(rows) if r["corpus"] == corpus]
        acts = [outcomes[i] not in ("link", "unread") for i in idx]
        err = [rows[i]["by_project_rule"] == "error" for i in idx]
        out[corpus] = {"errors": sum(err), "errors_acted_on": sum(a and e for a, e in zip(acts, err, strict=True)),
                       "right": len(idx) - sum(err),
                       "right_acted_on": sum(a and not e for a, e in zip(acts, err, strict=True))}
    return out


def gated_stream(rows, readings, outcomes) -> dict:
    """What the reader would do as a stream of the linker (``ci_reader_mode`` record/review): it speaks
    against a link only if it reads the word as no body site at all and the chunk lattice has not already
    acted on the phrase. The linker also leaves out procedure neighbours and ambiguous forms, which this
    count does not know: so the right links here are an upper bound of those the linker would send to review."""
    out = {}
    for corpus in ("craft", "medmentions"):
        idx = [i for i, r in enumerate(rows) if r["corpus"] == corpus]
        spoke = [i for i in idx if readings[i].decision in ("not_a_body_site", "inside_a_name")
                 and outcomes[i] in ("link", "unread")]
        gated = [i for i in idx if readings[i].decision in ("not_a_body_site", "inside_a_name")
                 and outcomes[i] not in ("link", "unread")]
        err = lambda i: rows[i]["by_project_rule"] == "error"  # noqa: E731
        out[corpus] = {
            "errors": sum(err(i) for i in idx), "right": sum(not err(i) for i in idx),
            "reads_against": len(spoke), "errors_among_them": sum(err(i) for i in spoke),
            "right_among_them": sum(not err(i) for i in spoke),
            "left_to_the_lattice": len(gated),
        }
    return out


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
            # what the reader adds: it moves a link that the lattice leaves alone (a role the lattice also
            # records is not a false alarm of the reader, the product keeps the link and writes the role)
            "errors_moved_reader_only": int((moved & ~lat & (label == 1)).sum()),
            "right_moved_reader_only": int((moved & ~lat & (label == 0)).sum()),
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


def run_all(reader, rows, texts_vector, keys, texts=None) -> list[ci.CIReading]:
    """Every judged link read with its document: the gist, the ambito and what the document has settled."""
    out = []
    discourses: dict = {}
    for r in rows:
        k = (r["corpus"], r.get("doc"))
        text = (texts or {}).get(k)
        discourse, doc_at = None, None
        if text is not None:
            if k not in discourses:
                discourses[k] = reader.discourse(text)
            discourse = discourses[k]
            found = text.find(r["sentence"])
            if found >= 0 and r.get("at") is not None:
                doc_at = found + r["at"]
        out.append(reader.read(r["mention"], r["sentence"], r.get("at"), r.get("doc"),
                               texts_vector.get(k), keys.get(k, ""), discourse, doc_at))
    return out


def markdown(report: dict) -> str:
    lines = ["# E5: the construction-integration reader on the external check", "",
             f"{report['n']} judged links ({report['errors']} errors). Space: {report['space']}. "
             f"Traces: {report['traces']} annotations of MedMentions training documents. "
             + ("FINAL run: frozen test documents included." if report.get("final") else
                f"{report.get('held_back_frozen', 0)} judged links of frozen test documents held back."), ""]
    for name, block in report["configurations"].items():
        lines += [f"## {name}", "", "| corpus | errors | not structure (errors) | not structure (right) | role or non-site (errors) | role or non-site (right) | lattice (errors / right) | reader only (errors / right) | AUC |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for corpus, v in block.items():
            lines.append(
                f"| {corpus} | {v['errors']} | {v['errors_not_structure']} | {v['right_not_structure']} / {v['right']} "
                f"| {v['errors_with_a_role_or_non_site']} | {v['right_with_a_role_or_non_site']} "
                f"| {v['lattice_errors']} / {v['lattice_right']} "
                f"| {v['errors_moved_reader_only']} / {v['right_moved_reader_only']} | {v['auc'] if v['auc'] is None else round(v['auc'], 3)} |")
        mm = block.get("medmentions")
        if mm and name == "all evidence":
            lines += ["", f"False alarms by split of MedMentions (right links changed / right): {mm['right_changed_by_split']}",
                      "", "Roles against the label (MedMentions): gold class -> same reading / n", ""]
            for g, v in mm["role_agreement"].items():
                lines.append(f"- {g}: {v['same']} / {v['n']}  (readings: {mm['by_gold'][g]})")
            lines.append(f"- errors whose reading is the gold role: {mm['errors_role_same']} of {mm['errors']}")
        lines.append("")
    lines += ["## The chunk lattice alone, by variant (acted on = role, not a site, or underspecified)", ""]
    for name, v in report.get("lattice_variants", {}).items():
        if name == "umls_lookups":
            lines.append(f"- UMLS lookups for heads: {v}")
            continue
        for corpus, c in v.items():
            lines.append(f"- {name} / {corpus}: errors acted on {c['errors_acted_on']} of {c['errors']}; "
                         f"right links acted on {c['right_acted_on']} of {c['right']}")
    lines += ["", "## The reader as a stream of the linker (record / review mode; upper bound, see `gated_stream`)", ""]
    for corpus, g in (report.get("gated_stream") or {}).items():
        lines.append(f"- {corpus}: reads a word as no body site where the lattice is silent in {g['reads_against']} links: "
                     f"{g['errors_among_them']} of {g['errors']} errors, {g['right_among_them']} of {g['right']} right links "
                     f"({g['left_to_the_lattice']} more left to the lattice)")
    lines += ["", "## Errors as the reader reads them (all evidence)", ""]
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
    ap.add_argument("--umls", action="store_true", help="type the head words the memory does not know from UMLS (UMLS_API_KEY)")
    ap.add_argument("--umls-cache", default="umls_cache.json")
    ap.add_argument("--collocations", help="collocation counts (json.gz, collocations.Collocations) for a lattice variant")
    ap.add_argument("--frames", help="procedure frames (json, frames.ProcedureFrames) for a lattice variant")
    ap.add_argument("--final", action="store_true",
                    help="the one run that writes the certificate: keeps the frozen test documents")
    args = ap.parse_args(argv)
    start = time.monotonic()
    frozen = load_frozen()
    rows = [r for r in json.loads(Path(args.rows).read_text("utf-8"))
            if r.get("status") == "accepted" and r.get("by_project_rule") in ("agrees", "error")]
    rows, held = development_rows(rows, frozen, args.final)
    say(start, f"{held} judged links of frozen test documents held back" if held else "no frozen test links held back")
    mm_docs, craft_docs, splits, space, protos = build_resources(args, start, frozen)
    lattice = cl.ChunkLattice(cl.BlockMemory.load(args.memory, LongerNames.load().names))
    keys = {**doc_keys(mm_docs, "medmentions", lattice), **doc_keys(craft_docs, "craft", lattice)}
    traces = write_traces(mm_docs, splits, space, lattice, keys, frozen)
    say(start, f"traces: {traces.size}")
    vectors = {("medmentions", d): space.vector(ci.content_words(t)) for d, t, _ in mm_docs}
    vectors.update({("craft", d): space.vector(ci.content_words(t)) for d, t, _ in craft_docs})
    texts = {("medmentions", d): t for d, t, _ in mm_docs}
    texts.update({("craft", d): t for d, t, _ in craft_docs})
    outcomes = [lattice.read(r["mention"], r["sentence"], r.get("at")).outcome for r in rows]
    say(start, "lattice read")
    # the lattice with the grammatical filter, and with the heads the memory does not know typed from UMLS
    memory = lattice.memory
    lattices = {"memory only": lattice,
                "grammar filter": cl.ChunkLattice(memory, grammar=Grammar.for_language("en"))}
    typer = None
    if args.umls:
        from umls_lookup import UmlsLookup  # noqa: E402

        lookup = UmlsLookup(cache_path=Path(args.umls_cache))
        if lookup.available:
            typer = UmlsHeadTyper(lookup.exact)
            lattices["grammar filter + UMLS-typed head"] = cl.ChunkLattice(
                memory, grammar=Grammar.for_language("en"), typer=typer)
        else:
            say(start, "UMLS_API_KEY not set: the UMLS-typed head arm is off")
    with_names = cl.ChunkLattice(cl.BlockMemory.load(args.memory, LongerNames.load().names, cl.load_anatomy_names()))
    lattices["anatomy names"] = with_names
    lattices["grammar filter + anatomy names"] = cl.ChunkLattice(with_names.memory, grammar=Grammar.for_language("en"))
    if args.collocations:
        from melampo.memory.collocations import Collocations  # noqa: E402

        colloc = Collocations.load(Path(args.collocations))
        lattices["collocations"] = cl.ChunkLattice(memory, collocations=colloc)
        lattices["grammar filter + collocations"] = cl.ChunkLattice(
            memory, grammar=Grammar.for_language("en"), collocations=colloc)
    if args.frames:
        from melampo.memory.frames import ProcedureFrames  # noqa: E402

        frames = ProcedureFrames.load(Path(args.frames))
        lattices["frames"] = cl.ChunkLattice(memory, frames=frames)
        lattices["anatomy names + frames" + (" + collocations" if args.collocations else "")] = cl.ChunkLattice(
            with_names.memory, frames=frames, collocations=colloc if args.collocations else None)
    variants = {}
    for name, lat in lattices.items():
        outs = outcomes if lat is lattice else [lat.read(r["mention"], r["sentence"], r.get("at")).outcome for r in rows]
        variants[name] = lattice_variant(rows, outs)
        say(start, f"lattice variant {name}: {variants[name]}")
    if typer is not None:
        variants["umls_lookups"] = {"asked": typer.asked, "lost": typer.lost}
        lookup.save()
    configs: dict[str, ci.CIReader] = {
        "all evidence": ci.CIReader(lattice, space, protos, traces),
    }
    for name, lat in lattices.items():
        if lat is not lattice:
            configs[f"all evidence, lattice: {name}"] = ci.CIReader(lat, space, protos, traces)
    for family in FAMILIES:
        configs[f"without {family}"] = ci.CIReader(
            lattice, space, protos, traces, use=frozenset(FAMILIES) - {family})
    for inh in (0.25, 1.0):
        configs[f"inhibition {inh}"] = ci.CIReader(lattice, space, protos, traces, inhibition=inh)
    configs["as before (no rules, no discourse)"] = ci.CIReader(
        lattice, space, protos, traces, use=frozenset(FAMILIES) - {"rules", "discourse"})
    configs["without memory (no traces, no domain, no gist)"] = ci.CIReader(
        lattice, space, protos, None, use=frozenset(FAMILIES) - {"traces", "domain", "gist"})
    configs["with the bag of context words (v3 context)"] = ci.CIReader(
        lattice, space, protos, traces, use=frozenset(FAMILIES) | {"context"})
    report = {"gated_stream": None, "lattice_variants": variants, "held_back_frozen": held, "final": args.final, "n": len(rows), "errors": sum(r["by_project_rule"] == "error" for r in rows),
              "space": f"{len(space.words)} words x {space.vectors.shape[1]}", "traces": traces.size,
              "configurations": {}, "error_readings": []}
    first = None
    for name, reader in configs.items():
        readings = run_all(reader, rows, vectors, keys, texts)
        report["configurations"][name] = summarise(rows, readings, outcomes, splits)
        if first is None:
            first = readings
        say(start, f"{name} done")
    report["gated_stream"] = gated_stream(rows, first, outcomes)
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
